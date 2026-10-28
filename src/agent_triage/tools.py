"""Restricted local tools and the single execution path used by the runner.

There is no generic shell tool.  Every path is re-resolved against the verified
project root immediately before the file is opened, sensitive paths are
rejected, and every result carries a bounded snippet plus an evidence id.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Protocol

from pydantic import Field, ValidationError, field_validator

from .budget import Budget
from .contracts import (
    Evidence,
    EvidenceKind,
    LessonSummary,
    LineNumber,
    NonEmptyStr,
    RelativePath,
    ShortStr,
    StrictInt,
    StrictModel,
    ToolResult,
    ToolSpec,
    ToolStatus,
    TraceKind,
    TraceRecord,
)
from .errors import ToolFailure
from .ingest import MAX_INPUT_BYTES
from .lifecycle import Decision, LifecycleEvent
from .store import LessonStore, SENSITIVE_DIRS, SENSITIVE_PARTS, SENSITIVE_SUFFIXES, hash_file

READ_MAX_BYTES = 8 * 1024
SCAN_MAX_BYTES = 8 * 1024 * 1024
SCAN_MAX_FILES = 5000
SCAN_FILE_BYTES = 2 * 1024 * 1024
SNIPPET_MAX_BYTES = 8 * 1024
SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".cache",
}

SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9_-]{10,}|gh[pousr]_[A-Za-z0-9]{10,}|github_pat_[A-Za-z0-9_]{10,}"
    r"|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|Bearer\s+[A-Za-z0-9._-]{10,})"
)


# ---------------------------------------------------------------------------
# Argument contracts
# ---------------------------------------------------------------------------


class ReadFileArgs(StrictModel):
    path: RelativePath
    start_line: LineNumber = 1
    max_lines: StrictInt = Field(default=80, ge=1, le=120)


class SearchWorkspaceArgs(StrictModel):
    query: ShortStr
    paths: list[RelativePath] = Field(default_factory=list, max_length=5)
    limit: StrictInt = Field(default=10, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def _literal_query(cls, value: str) -> str:
        if "\x00" in value or "\n" in value or "\r" in value:
            raise ValueError("query must be a single-line literal")
        return value


class GitContextArgs(StrictModel):
    paths: list[RelativePath] = Field(default_factory=list, max_length=5)


class SearchLessonsArgs(StrictModel):
    query: ShortStr
    limit: StrictInt = Field(default=3, ge=1, le=5)


class GithubSearchIssuesArgs(StrictModel):
    query: ShortStr
    limit: StrictInt = Field(default=3, ge=1, le=5)


class GithubReadIssueArgs(StrictModel):
    issue_ref: ShortStr = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+#[1-9][0-9]*$")


TOOL_DEFINITIONS: dict[str, tuple[type[StrictModel], str]] = {
    "read_file": (
        ReadFileArgs,
        "Read a bounded UTF-8 snippet from a project-relative text file. "
        "Returns line numbers, a sha256 of the whole file, an evidence id, and continuation metadata.",
    ),
    "search_workspace": (
        SearchWorkspaceArgs,
        "Literal, case-insensitive search inside at most five project-relative paths. "
        "No shell and no regular expressions; returns at most 20 short matches.",
    ),
    "get_git_context": (
        GitContextArgs,
        "Read commit, dirty state and a bounded diff for up to five project-relative paths. "
        "Non-Git directories return commit=null and per-file hashes.",
    ),
    "search_lessons": (
        SearchLessonsArgs,
        "Search local lessons by keyword. Verified lessons are listed first; candidates are "
        "reference material, not facts.",
    ),
    "github_search_issues": (
        GithubSearchIssuesArgs,
        "Search issues in the configured GitHub repository only. Returns candidate summaries; "
        "an empty list means the query completed with no matches.",
    ),
    "github_read_issue": (
        GithubReadIssueArgs,
        "Read one issue and bounded comments. issue_ref must come from github_search_issues in "
        "this run and belong to the configured repository.",
    ),
}

TOOL_NAMES = tuple(TOOL_DEFINITIONS)


def tool_specs() -> list[ToolSpec]:
    specs: list[ToolSpec] = []
    for name, (model, description) in TOOL_DEFINITIONS.items():
        schema = model.model_json_schema()
        schema.pop("title", None)
        schema["additionalProperties"] = False
        specs.append(ToolSpec(name=name, description=description, input_schema=schema))
    return specs


def tool_definitions_text() -> str:
    return json.dumps(
        [spec.model_dump(mode="json") for spec in tool_specs()],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


# ---------------------------------------------------------------------------
# Paths and bounded reads
# ---------------------------------------------------------------------------


def is_sensitive(relative: str) -> bool:
    parts = PurePosixPath(relative).parts
    for part in parts:
        lowered = part.lower()
        if lowered in SENSITIVE_PARTS:
            return True
        if lowered.startswith(".env.") and lowered != ".env.example":
            return True
        if lowered in SENSITIVE_DIRS:
            return True
        if any(lowered.endswith(suffix) for suffix in SENSITIVE_SUFFIXES):
            return True
    return False


class Workspace:
    def __init__(self, root: Path) -> None:
        try:
            self.root = root.resolve(strict=True)
        except OSError as exc:
            raise ToolFailure("invalid_workspace", "project root does not exist") from exc
        if not self.root.is_dir():
            raise ToolFailure("invalid_workspace", "project root is not a directory")

    def resolve(self, relative: str, *, must_exist: bool = True) -> Path:
        if relative in ("", "."):
            candidate = self.root
        else:
            candidate = (self.root / relative).resolve(strict=False)
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ToolFailure("path_escape", "path escapes the verified project root") from exc
        rel = "." if candidate == self.root else candidate.relative_to(self.root).as_posix()
        if is_sensitive(rel):
            raise ToolFailure("sensitive_path", f"{rel} is excluded from tool reads")
        if must_exist and not candidate.exists():
            raise ToolFailure("missing", f"{rel} does not exist in the workspace")
        return candidate

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    def read_bytes(self, relative: str) -> tuple[bytes, str]:
        path = self.resolve(relative, must_exist=True)
        if not path.is_file():
            raise ToolFailure("not_a_file", f"{relative} is not a regular file")
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError as exc:
            raise ToolFailure("read_failed", f"cannot open {relative}: {exc.strerror}") from exc
        try:
            before = os.fstat(descriptor)
            if before.st_size > MAX_INPUT_BYTES:
                raise ToolFailure(
                    "file_too_large",
                    f"{relative} exceeds 2 MiB; choose a smaller file or read a specific range",
                )
            chunks: list[bytes] = []
            remaining = MAX_INPUT_BYTES + 1
            while remaining > 0:
                chunk = os.read(descriptor, min(262_144, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        payload = b"".join(chunks)
        if len(payload) > MAX_INPUT_BYTES or len(payload) != before.st_size:
            raise ToolFailure("file_changed", f"{relative} changed while it was being read", retryable=True)
        if (before.st_mtime_ns, before.st_size, before.st_ino) != (after.st_mtime_ns, after.st_size, after.st_ino):
            raise ToolFailure("file_changed", f"{relative} changed while it was being read", retryable=True)
        if b"\x00" in payload[:8192]:
            raise ToolFailure("unsupported_binary", f"{relative} looks binary; text files only")
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolFailure("invalid_utf8", f"{relative} is not valid UTF-8") from exc
        return payload, text


def truncate_utf8(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    cut = encoded[:max_bytes]
    while cut:
        try:
            return cut.decode("utf-8")
        except UnicodeDecodeError:
            cut = cut[:-1]
    return ""


def build_evidence(
    *,
    kind: EvidenceKind,
    project_id: str,
    snippet: str,
    path: str | None = None,
    line_start: int | None = None,
    line_end: int | None = None,
    sha256: str | None = None,
    total_lines: int | None = None,
    commit: str | None = None,
    url: str | None = None,
    truncated: bool = False,
) -> Evidence:
    snippet = truncate_utf8(snippet, SNIPPET_MAX_BYTES)
    digest = hashlib.sha256(snippet.encode("utf-8")).hexdigest()
    identity = "|".join(
        [
            kind.value,
            path or "",
            url or "",
            str(line_start or ""),
            str(line_end or ""),
            sha256 or "",
            digest,
        ]
    )
    evidence_id = f"ev-{hashlib.sha256(identity.encode()).hexdigest()[:12]}"
    return Evidence(
        evidence_id=evidence_id,
        kind=kind,
        project_id=project_id,
        path=path,
        line_start=line_start,
        line_end=line_end,
        url=url,
        sha256=sha256 or hashlib.sha256(snippet.encode()).hexdigest(),
        snippet=snippet,
        captured_at=datetime.now(timezone.utc),
        truncated=truncated,
        total_lines=total_lines,
        commit=commit,
    )


# ---------------------------------------------------------------------------
# GitHub backend boundary
# ---------------------------------------------------------------------------


@dataclass
class IssueSearchOutcome:
    issues: list[dict[str, Any]]
    total_count: int | None
    complete: bool
    query_scope: str


@dataclass
class IssueReadOutcome:
    issue: dict[str, Any]
    comments: list[dict[str, Any]]
    url: str
    updated_at: str | None
    comments_error: str | None = None


class GithubBackend(Protocol):
    available: bool
    unavailable_reason: str | None

    def search_issues(self, query: str, limit: int) -> IssueSearchOutcome: ...

    def read_issue(self, issue_ref: str) -> IssueReadOutcome: ...


class DisabledGithubBackend:
    available = False
    unavailable_reason = "mcp_not_configured"

    def search_issues(self, query: str, limit: int) -> IssueSearchOutcome:
        raise ToolFailure(
            "mcp_not_configured",
            "GitHub MCP is disabled; local analysis can continue but remote history is not checked",
        )

    def read_issue(self, issue_ref: str) -> IssueReadOutcome:
        raise ToolFailure(
            "mcp_not_configured",
            "GitHub MCP is disabled; remote issue bodies were not read",
        )


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class ToolExecutor:
    def __init__(
        self,
        *,
        workspace: Workspace,
        project_id: str,
        budget: Budget,
        evidence_sink: Callable[[Evidence], Evidence] | None = None,
        lesson_store: LessonStore | None = None,
        github: GithubBackend | None = None,
        trace_sink: Callable[[TraceRecord], None] | None = None,
        run_id: str = "run-unknown",
        policy: Any | None = None,
        evidence_lookup: Callable[[str], bool] | None = None,
    ) -> None:
        self.workspace = workspace
        self.project_id = project_id
        self.budget = budget
        self.evidence_sink = evidence_sink
        self.lesson_store = lesson_store
        self.github = github or DisabledGithubBackend()
        self.trace_sink = trace_sink
        self.run_id = run_id
        self.policy = policy
        self.evidence_lookup = evidence_lookup
        self._tool_scope = threading.local()

    def execute(self, tool_name: str, arguments: dict[str, Any], *, call_id: str, turn: int) -> ToolResult:
        if tool_name not in TOOL_DEFINITIONS:
            result = ToolResult(
                status=ToolStatus.ERROR,
                error_code="unknown_tool",
                data={"tool": tool_name},
                retryable=False,
                duration_ms=0,
            )
            self._trace(tool_name, arguments, call_id, turn, result, 1)
            return result
        model, _ = TOOL_DEFINITIONS[tool_name]
        try:
            parsed = model.model_validate(arguments)
        except ValidationError as exc:
            details = []
            for error in exc.errors()[:6]:
                location = ".".join(str(item) for item in error["loc"]) or "<root>"
                details.append(f"{location}: {error['msg']}")
            result = ToolResult(
                status=ToolStatus.ERROR,
                error_code="invalid_arguments",
                data={"errors": details},
                retryable=False,
                duration_ms=0,
            )
            self._trace(tool_name, arguments, call_id, turn, result, 1)
            return result

        attempts = 0
        while True:
            attempts += 1
            can, reason = self.budget.can_attempt_tool()
            if not can:
                result = ToolResult(
                    status=ToolStatus.ERROR,
                    error_code=reason or "budget_exhausted",
                    data=None,
                    retryable=False,
                    duration_ms=0,
                )
                self._trace(tool_name, arguments, call_id, turn, result, attempts)
                return result
            denial = self._policy_before_tool(tool_name, parsed, call_id=call_id, turn=turn)
            if denial is not None:
                # A rejected request is still bounded: it consumes one attempt
                # when any budget remains, so repeated denials cannot loop free.
                if self.budget.can_attempt_tool()[0]:
                    self.budget.consume_tool_attempt()
                self._trace(tool_name, arguments, call_id, turn, denial, attempts, denied=True)
                return denial
            self.budget.consume_tool_attempt()
            result = self._run_with_timeout(tool_name, parsed)
            result = self._policy_after_tool(tool_name, parsed, result, call_id=call_id, turn=turn)
            self._trace(tool_name, arguments, call_id, turn, result, attempts)
            if result.status == ToolStatus.OK or not result.retryable or attempts >= 2:
                return result
            can_retry, _ = self.budget.can_attempt_tool()
            if not can_retry:
                return result

    # -- lifecycle gates ----------------------------------------------------
    def _policy_before_tool(
        self,
        tool_name: str,
        parsed: StrictModel,
        *,
        call_id: str,
        turn: int,
    ) -> ToolResult | None:
        if self.policy is None:
            return None
        payload = parsed.model_dump(mode="json")
        decision = self.policy.evaluate(
            LifecycleEvent.BEFORE_TOOL,
            data={
                "tool": tool_name,
                "workspace": self.workspace,
                "budget": self.budget,
                **payload,
            },
            turn=turn,
            call_id=call_id,
            budget=self.budget.snapshot().model_dump(mode="json"),
        )
        if decision.decision != Decision.DENY:
            return None
        code = decision.reason_code
        error_code = (
            code
            if code
            in {
                "tool_attempt_limit",
                "total_time_exhausted",
                "sensitive_path",
                "path_escape",
                "missing",
            }
            else f"policy_denied:{code}"
        )
        return ToolResult(
            status=ToolStatus.ERROR,
            error_code=error_code,
            data={
                "rule_id": decision.rule_id,
                "reason_code": decision.reason_code,
                "message": decision.message[:500],
            },
            retryable=False,
            duration_ms=decision.duration_ms,
        )

    def _policy_after_tool(
        self,
        tool_name: str,
        parsed: StrictModel,
        result: ToolResult,
        *,
        call_id: str,
        turn: int,
    ) -> ToolResult:
        if self.policy is None or result.status != ToolStatus.OK:
            return result
        persisted = None
        if self.evidence_lookup is not None and result.evidence_refs:
            persisted = all(self.evidence_lookup(evidence_id) for evidence_id in result.evidence_refs)
        decision = self.policy.evaluate(
            LifecycleEvent.AFTER_TOOL,
            data={
                "tool": tool_name,
                "workspace": self.workspace,
                "budget": self.budget,
                **parsed.model_dump(mode="json"),
                "evidence_refs": list(result.evidence_refs),
                "evidence_persisted": persisted,
            },
            turn=turn,
            call_id=call_id,
            budget=self.budget.snapshot().model_dump(mode="json"),
        )
        if decision.decision == Decision.ALLOW:
            return result
        payload = result.data if isinstance(result.data, dict) else {"result": result.data}
        return result.model_copy(
            update={
                "data": {
                    **payload,
                    "policy_warning": {
                        "decision": decision.decision.value,
                        "rule_id": decision.rule_id,
                        "reason_code": decision.reason_code,
                        "message": decision.message[:500],
                    },
                }
            }
        )

    # -- dispatch -----------------------------------------------------------
    def _run_with_timeout(self, tool_name: str, parsed: StrictModel) -> ToolResult:
        timeout = self.budget.per_tool_timeout()
        outcome: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)
        cancel = threading.Event()

        def work() -> None:
            self._tool_scope.cancel = cancel
            try:
                outcome.put(("ok", self._dispatch(tool_name, parsed)))
            except Exception as exc:  # noqa: BLE001 - handed back to the caller
                outcome.put(("error", exc))
            finally:
                self._tool_scope.cancel = None

        thread = threading.Thread(target=work, name="triage-tool", daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            cancel.set()
            # The worker is a daemon so a stuck C call cannot keep the CLI
            # process alive. The timed-out result is deliberately discarded.
            return ToolResult(
                status=ToolStatus.ERROR,
                error_code="tool_timeout",
                retryable=True,
                duration_ms=int(timeout * 1000),
            )
        try:
            kind, value = outcome.get_nowait()
        except queue.Empty:
            return ToolResult(
                status=ToolStatus.ERROR,
                error_code="tool_internal_error",
                data={"message": "worker produced no result"},
                retryable=False,
                duration_ms=0,
            )
        if kind == "ok":
            assert isinstance(value, ToolResult)
            return value
        if isinstance(value, ToolFailure):
            return ToolResult(
                status=ToolStatus.ERROR,
                error_code=value.code,
                data={"message": value.message},
                retryable=value.retryable,
                duration_ms=0,
            )
        return ToolResult(
            status=ToolStatus.ERROR,
            error_code="tool_internal_error",
            data={"message": type(value).__name__},
            retryable=False,
            duration_ms=0,
        )

    def _dispatch(self, tool_name: str, parsed: StrictModel) -> ToolResult:
        started = datetime.now(timezone.utc)
        if tool_name == "read_file":
            result = self._read_file(parsed)  # type: ignore[arg-type]
        elif tool_name == "search_workspace":
            result = self._search_workspace(parsed)  # type: ignore[arg-type]
        elif tool_name == "get_git_context":
            result = self._git_context(parsed)  # type: ignore[arg-type]
        elif tool_name == "search_lessons":
            result = self._search_lessons(parsed)  # type: ignore[arg-type]
        elif tool_name == "github_search_issues":
            result = self._github_search(parsed)  # type: ignore[arg-type]
        elif tool_name == "github_read_issue":
            result = self._github_read(parsed)  # type: ignore[arg-type]
        else:  # guarded by caller
            raise ToolFailure("unknown_tool", tool_name)
        elapsed = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
        return result.model_copy(update={"duration_ms": elapsed})

    # -- tools --------------------------------------------------------------
    def _read_file(self, args: ReadFileArgs) -> ToolResult:
        payload, text = self.workspace.read_bytes(args.path)
        sha = hashlib.sha256(payload).hexdigest()
        lines = text.splitlines()
        start = args.start_line
        if start > len(lines):
            raise ToolFailure(
                "line_out_of_range",
                f"{args.path} has {len(lines)} lines; start_line={start}",
            )
        selected: list[str] = []
        used = 0
        index = start - 1
        line_truncated = False
        skipped_long_line = False
        while index < len(lines) and len(selected) < args.max_lines:
            line = lines[index]
            encoded = (line + "\n").encode("utf-8")
            if len(encoded) > READ_MAX_BYTES - used:
                if not selected:
                    line = truncate_utf8(line, READ_MAX_BYTES)
                    selected.append(line)
                    line_truncated = True
                    index += 1
                else:
                    skipped_long_line = True
                break
            selected.append(line)
            used += len(encoded)
            index += 1
        end = start + len(selected) - 1 if selected else start - 1
        more = index < len(lines)
        continuation = {"start_line": index + 1} if more else None
        if line_truncated or skipped_long_line:
            continuation = {
                "reason": "line_too_long",
                "next": (
                    "the line exceeded the 8 KiB read limit; split the source, "
                    "read the next line range, or use search_workspace"
                ),
            }
            if more:
                continuation["next_line"] = index + 1
        snippet = "\n".join(selected)
        evidence = build_evidence(
            kind=EvidenceKind.LOCAL_FILE,
            project_id=self.project_id,
            path=args.path,
            line_start=start,
            line_end=end,
            sha256=sha,
            total_lines=len(lines),
            snippet=snippet,
            truncated=bool(continuation or line_truncated),
        )
        evidence = self._save_evidence(evidence)
        data = {
            "path": args.path,
            "start_line": start,
            "end_line": end,
            "total_lines": len(lines),
            "sha256": sha,
            "lines": [{"line": start + offset, "text": value} for offset, value in enumerate(selected)],
            "line_truncated": line_truncated,
            "skipped_long_line": skipped_long_line,
        }
        return ToolResult(
            status=ToolStatus.OK,
            data=data,
            evidence_refs=[evidence.evidence_id],
            truncated=bool(continuation),
            continuation=continuation,
            duration_ms=0,
        )

    def _search_workspace(self, args: SearchWorkspaceArgs) -> ToolResult:
        roots = args.paths or ["."]
        matches: list[dict[str, Any]] = []
        evidence_refs: list[str] = []
        files_scanned = 0
        bytes_scanned = 0
        limit_reached = False
        scan_truncated = False
        line_truncated = False
        needle = args.query.casefold()

        for root_relative in roots:
            root = self.workspace.resolve(root_relative, must_exist=True)
            candidates: list[Path]
            if root.is_file():
                candidates = [root]
            else:
                candidates = []
                for dirpath, dirnames, filenames in os.walk(root):
                    dirnames[:] = sorted(
                        name for name in dirnames if name not in SKIP_DIRS and not name.startswith(".git")
                    )
                    for filename in sorted(filenames):
                        candidates.append(Path(dirpath) / filename)
                        if len(candidates) >= SCAN_MAX_FILES:
                            scan_truncated = True
                            break
                    if scan_truncated:
                        break
            for path in candidates:
                if limit_reached:
                    break
                try:
                    relative = self.workspace.relative(path)
                except ValueError:
                    scan_truncated = True
                    continue
                if is_sensitive(relative):
                    continue
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                if size > SCAN_FILE_BYTES:
                    continue
                if bytes_scanned >= SCAN_MAX_BYTES:
                    scan_truncated = True
                    break
                try:
                    payload = path.read_bytes()
                except OSError:
                    continue
                bytes_scanned += len(payload)
                files_scanned += 1
                if b"\x00" in payload[:8192]:
                    continue
                try:
                    text = payload.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                sha = hashlib.sha256(payload).hexdigest()
                for line_number, line in enumerate(text.splitlines(), start=1):
                    if needle not in line.casefold():
                        continue
                    shortened = truncate_utf8(line, 500)
                    evidence = build_evidence(
                        kind=EvidenceKind.LOCAL_FILE,
                        project_id=self.project_id,
                        path=relative,
                        line_start=line_number,
                        line_end=line_number,
                        sha256=sha,
                        total_lines=text.count("\n") + (0 if text.endswith("\n") else 1),
                        snippet=shortened,
                        truncated=len(shortened) < len(line),
                    )
                    evidence = self._save_evidence(evidence)
                    evidence_refs.append(evidence.evidence_id)
                    line_truncated = line_truncated or len(shortened) < len(line)
                    matches.append(
                        {
                            "path": relative,
                            "line": line_number,
                            "text": shortened,
                            "evidence_id": evidence.evidence_id,
                            "line_truncated": len(shortened) < len(line),
                        }
                    )
                    if len(matches) >= args.limit:
                        limit_reached = True
                        break
        truncated = limit_reached or scan_truncated or line_truncated
        continuation = None
        if truncated:
            if limit_reached:
                reason = "limit_reached"
            elif scan_truncated:
                reason = "scan_limit_reached"
            else:
                reason = "line_truncated"
            continuation = {
                "reason": reason,
                "next": "narrow the query or paths to continue",
            }
        return ToolResult(
            status=ToolStatus.OK,
            data={
                "query": args.query,
                "matches": matches,
                "files_scanned": files_scanned,
                "bytes_scanned": bytes_scanned,
            },
            evidence_refs=evidence_refs,
            truncated=truncated,
            continuation=continuation,
            duration_ms=0,
        )

    def _git_context(self, args: GitContextArgs) -> ToolResult:
        env = {
            **os.environ,
            "GIT_PAGER": "cat",
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
        }
        root = str(self.workspace.root)
        try:
            probe = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            probe = None
        is_git = probe is not None and probe.returncode == 0 and probe.stdout.strip() == "true"
        file_hashes: list[dict[str, Any]] = []
        for relative in args.paths:
            path = self.workspace.resolve(relative, must_exist=False)
            exists = path.is_file()
            entry: dict[str, Any] = {"path": relative, "exists": exists, "sha256": None}
            if exists:
                try:
                    if path.stat().st_size > MAX_INPUT_BYTES:
                        entry["skipped"] = "file_too_large"
                    else:
                        entry["sha256"] = hash_file(path)
                except OSError:
                    entry["skipped"] = "unreadable"
            file_hashes.append(entry)
        if not is_git:
            snippet = json.dumps({"git": False, "files": file_hashes}, ensure_ascii=False, indent=2)
            path_for_evidence = args.paths[0] if args.paths else "."
            evidence = build_evidence(
                kind=EvidenceKind.GIT,
                project_id=self.project_id,
                path=path_for_evidence,
                sha256=hashlib.sha256(snippet.encode()).hexdigest(),
                snippet=snippet,
            )
            evidence = self._save_evidence(evidence)
            return ToolResult(
                status=ToolStatus.OK,
                data={"commit": None, "dirty": None, "files": file_hashes, "diff": None, "diff_truncated": False},
                evidence_refs=[evidence.evidence_id],
                duration_ms=0,
            )
        commit = (self._git(["rev-parse", "HEAD"]) or "").strip() or None
        status = self._git(["status", "--porcelain"]) or ""
        diff_args = ["diff", "--no-color", "--unified=3"]
        if args.paths:
            diff_args.extend(["--", *args.paths])
        diff_value, diff_truncated = self._git_bounded(diff_args, SNIPPET_MAX_BYTES)
        diff = diff_value or ""
        snippet_payload = {
            "commit": commit,
            "dirty": bool(status.strip()),
            "files": file_hashes,
            "diff": diff,
        }
        snippet = json.dumps(snippet_payload, ensure_ascii=False, indent=2)
        path_for_evidence = args.paths[0] if args.paths else "."
        evidence = build_evidence(
            kind=EvidenceKind.GIT,
            project_id=self.project_id,
            path=path_for_evidence,
            sha256=hashlib.sha256(snippet.encode()).hexdigest(),
            snippet=snippet,
            commit=commit,
            truncated=diff_truncated,
        )
        evidence = self._save_evidence(evidence)
        return ToolResult(
            status=ToolStatus.OK,
            data={
                "commit": commit,
                "dirty": bool(status.strip()),
                "files": file_hashes,
                "diff": diff,
                "diff_truncated": diff_truncated,
            },
            evidence_refs=[evidence.evidence_id],
            truncated=diff_truncated,
            continuation={"reason": "diff_truncated", "next": "request a narrower path list"} if diff_truncated else None,
            duration_ms=0,
        )

    def _git(self, arguments: list[str]) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=str(self.workspace.root),
                env={
                    **os.environ,
                    "GIT_PAGER": "cat",
                    "GIT_TERMINAL_PROMPT": "0",
                    "LC_ALL": "C",
                },
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout

    def _git_bounded(self, arguments: list[str], max_bytes: int) -> tuple[str | None, bool]:
        """Read at most ``max_bytes`` of git output; kill the child if it overflows."""
        try:
            process = subprocess.Popen(
                ["git", *arguments],
                cwd=str(self.workspace.root),
                env={
                    **os.environ,
                    "GIT_PAGER": "cat",
                    "GIT_TERMINAL_PROMPT": "0",
                    "LC_ALL": "C",
                },
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            return None, False
        try:
            assert process.stdout is not None
            data = process.stdout.read(max_bytes + 1)
            truncated = len(data) > max_bytes
            if truncated:
                process.kill()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
            if process.returncode not in (0, -9):
                return None, truncated
            return data[:max_bytes].decode("utf-8", errors="replace"), truncated
        finally:
            if process.stdout is not None:
                process.stdout.close()

    def _search_lessons(self, args: SearchLessonsArgs) -> ToolResult:
        if self.lesson_store is None:
            raise ToolFailure("lessons_unavailable", "no local lesson store is configured")
        lessons = self.lesson_store.search(args.query, args.limit)
        evidence_refs: list[str] = []
        for lesson in lessons:
            snippet = json.dumps(lesson.model_dump(mode="json"), ensure_ascii=False, indent=2)
            evidence = build_evidence(
                kind=EvidenceKind.LESSON,
                project_id=self.project_id,
                snippet=snippet,
                sha256=hashlib.sha256(snippet.encode()).hexdigest(),
            )
            evidence = self._save_evidence(evidence)
            evidence_refs.append(evidence.evidence_id)
        return ToolResult(
            status=ToolStatus.OK,
            data={
                "query": args.query,
                "lessons": [lesson.model_dump(mode="json") for lesson in lessons],
                "note": "candidate lessons are reference material, not confirmed facts",
            },
            evidence_refs=evidence_refs,
            duration_ms=0,
        )

    def _github_search(self, args: GithubSearchIssuesArgs) -> ToolResult:
        if not self.github.available:
            raise ToolFailure(self.github.unavailable_reason or "mcp_unavailable", "GitHub MCP is unavailable")
        outcome = self.github.search_issues(args.query, args.limit)
        evidence_refs: list[str] = []
        issues: list[dict[str, Any]] = []
        for issue in outcome.issues:
            url = str(issue.get("html_url") or issue.get("url") or "")
            if not url:
                raise ToolFailure("unsupported_result", "GitHub issue result has no URL")
            snippet = json.dumps(issue, ensure_ascii=False, indent=2)
            evidence = build_evidence(
                kind=EvidenceKind.ISSUE,
                project_id=self.project_id,
                url=url,
                snippet=snippet,
                sha256=hashlib.sha256(snippet.encode()).hexdigest(),
            )
            evidence = self._save_evidence(evidence)
            evidence_refs.append(evidence.evidence_id)
            issues.append(
                {
                    "issue_ref": issue.get("issue_ref"),
                    "number": issue.get("number"),
                    "title": issue.get("title"),
                    "state": issue.get("state"),
                    "url": url,
                    "updated_at": issue.get("updated_at"),
                    "evidence_id": evidence.evidence_id,
                }
            )
        truncated = not outcome.complete
        return ToolResult(
            status=ToolStatus.OK,
            data={
                "query": args.query,
                "scope": outcome.query_scope,
                "total_count": outcome.total_count,
                "issues": issues,
                "complete": outcome.complete,
            },
            evidence_refs=evidence_refs,
            truncated=truncated,
            continuation={"reason": "page_limited", "next": "narrow the query"} if truncated else None,
            duration_ms=0,
        )

    def _github_read(self, args: GithubReadIssueArgs) -> ToolResult:
        if not self.github.available:
            raise ToolFailure(self.github.unavailable_reason or "mcp_unavailable", "GitHub MCP is unavailable")
        outcome = self.github.read_issue(args.issue_ref)
        snippet = json.dumps(
            {"issue": outcome.issue, "comments": outcome.comments},
            ensure_ascii=False,
            indent=2,
        )
        evidence = build_evidence(
            kind=EvidenceKind.ISSUE,
            project_id=self.project_id,
            url=outcome.url,
            snippet=snippet,
            sha256=hashlib.sha256(snippet.encode()).hexdigest(),
            truncated=len(snippet.encode()) > SNIPPET_MAX_BYTES,
        )
        evidence = self._save_evidence(evidence)
        return ToolResult(
            status=ToolStatus.OK,
            data={
                "issue": outcome.issue,
                "comments": outcome.comments,
                "updated_at": outcome.updated_at,
                "comments_error": outcome.comments_error,
                "evidence_id": evidence.evidence_id,
            },
            evidence_refs=[evidence.evidence_id],
            truncated=evidence.truncated,
            continuation={"reason": "snippet_truncated", "next": "read comments in smaller ranges"} if evidence.truncated else None,
            duration_ms=0,
        )

    # -- helpers ------------------------------------------------------------
    def _save_evidence(self, evidence: Evidence) -> Evidence:
        cancel = getattr(self._tool_scope, "cancel", None)
        if cancel is not None and cancel.is_set():
            # A timed-out worker may still finish in the background; do not let
            # it append evidence after the runner has moved on.
            return evidence
        if self.evidence_sink is None:
            return evidence
        return self.evidence_sink(evidence)

    def _trace(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        call_id: str,
        turn: int,
        result: ToolResult,
        attempt: int,
        *,
        denied: bool = False,
    ) -> None:
        if self.trace_sink is None:
            return
        finished = datetime.now(timezone.utc)
        started = finished - timedelta(milliseconds=max(0, result.duration_ms))
        rejected = denied or result.error_code in {"invalid_arguments", "unknown_tool"} or (
            result.error_code is not None and result.error_code.startswith("policy_denied:")
        )
        record = TraceRecord(
            kind=TraceKind.TOOL,
            run_id=self.run_id,
            turn=turn,
            call_id=call_id,
            tool_name=tool_name,
            arguments=redact_arguments(arguments),
            status="rejected" if rejected else result.status.value,
            error_code=result.error_code,
            duration_ms=result.duration_ms,
            evidence_refs=result.evidence_refs,
            attempt=attempt,
            started_at=started,
            finished_at=finished,
        )
        self.trace_sink(record)


def redact_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Keep trace arguments small and free of recognizable secrets."""

    def scrub(value: Any) -> Any:
        if isinstance(value, str):
            if SECRET_RE.search(value):
                return "[REDACTED]"
            return value[:500]
        if isinstance(value, list):
            return [scrub(item) for item in value[:20]]
        if isinstance(value, dict):
            return {str(key)[:100]: scrub(item) for key, item in list(value.items())[:20]}
        return value

    return scrub(arguments)


def evidence_snippet_from_store(evidence: Evidence) -> str:
    return evidence.snippet