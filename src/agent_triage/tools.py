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
from typing import Any, Callable, Literal, Protocol

from pydantic import Field, ValidationError, field_validator

from .budget import Budget
from .contracts import (
    Evidence,
    EvidenceId,
    EvidenceKind,
    IssueAssessment,
    IssueCandidate,
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
from .github_scope import ScopeViolation, SearchScope, compile_query
from .errors import InputError, StoreError, ToolFailure
from .ingest import MAX_INPUT_BYTES
from .lifecycle import Decision, LifecycleEvent
from .store import EvidenceStore, LessonStore, SENSITIVE_DIRS, SENSITIVE_PARTS, SENSITIVE_SUFFIXES, hash_file
from .cursors import make_cursor, read_cursor
from .windows import MAX_WINDOW_BYTES, make_window, select_windows

READ_MAX_BYTES = 8 * 1024
TOOL_RESULT_MAX_BYTES = 8 * 1024
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


ASSESSMENT_VALUE = Literal["match", "mismatch", "unknown"]


class RecordIssueAssessmentArgs(StrictModel):
    issue_ref: ShortStr = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+#[1-9][0-9]*$")
    component: ASSESSMENT_VALUE
    symptom: ASSESSMENT_VALUE
    version: ASSESSMENT_VALUE
    environment: ASSESSMENT_VALUE
    trigger: ASSESSMENT_VALUE
    fix: ASSESSMENT_VALUE
    evidence_refs: list[EvidenceId] = Field(min_length=1, max_length=6)
    note: ShortStr | None = None


class ReadEvidenceArgs(StrictModel):
    evidence_id: EvidenceId
    cursor: str | None = Field(default=None, min_length=1, max_length=1500)
    max_bytes: StrictInt = Field(default=8192, ge=4, le=8192)


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
        "Search issues in the program-controlled scope (configured repository by default; mapped "
        "dependency repositories or a public fallback only when layered research is enabled). "
        "Returns candidate summaries; an empty list means the query completed with no matches.",
    ),
    "github_read_issue": (
        GithubReadIssueArgs,
        "Read one issue and bounded comments. issue_ref must come from a persisted search "
        "receipt in this run and belong to an authorized search scope.",
    ),
    "record_issue_assessment": (
        RecordIssueAssessmentArgs,
        "Record a per-dimension applicability assessment for one issue candidate located and "
        "read in this run. evidence_refs must include the displayed evidence id from reading that "
        "candidate's body. Each dimension is match/mismatch/unknown; this never confirms a root cause.",
    ),
    "read_evidence": (
        ReadEvidenceArgs,
        "Read a bounded page from a snapshot registered in this run only. "
        "Use its opaque cursor to continue the same source version, including within a long line. "
        "No path, URL, shell or network access; uncaptured content remains unknown.",
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


def _casefold_match_start(text: str, needle: str) -> int | None:
    position = text.casefold().find(needle)
    if position < 0:
        return None
    folded_offset = 0
    for index, char in enumerate(text):
        folded_offset += len(char.casefold())
        if folded_offset > position:
            return index
    return None


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
    byte_offset: int | None = None,
) -> Evidence:
    original_snippet = snippet
    snippet = truncate_utf8(snippet, SNIPPET_MAX_BYTES)
    truncated = truncated or snippet != original_snippet
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
            str(byte_offset) if byte_offset is not None else "",
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
    query: str = ""
    layer: str = "current"
    incomplete: bool | None = None


@dataclass
class IssueReadOutcome:
    issue: dict[str, Any]
    comments: list[dict[str, Any]]
    url: str
    updated_at: str | None
    comments_error: str | None = None
    comments_complete: bool | None = None


class GithubBackend(Protocol):
    available: bool
    unavailable_reason: str | None

    def search_issues(
        self, query: str, limit: int, *, scope: SearchScope | None = None
    ) -> IssueSearchOutcome: ...

    def read_issue(self, issue_ref: str, *, allow_any_repo: bool = False) -> IssueReadOutcome: ...


class DisabledGithubBackend:
    available = False
    unavailable_reason = "mcp_not_configured"

    def search_issues(
        self, query: str, limit: int, *, scope: SearchScope | None = None
    ) -> IssueSearchOutcome:
        raise ToolFailure(
            "mcp_not_configured",
            "GitHub MCP is disabled; local analysis can continue but remote history is not checked",
        )

    def read_issue(self, issue_ref: str, *, allow_any_repo: bool = False) -> IssueReadOutcome:
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
        evidence_store: EvidenceStore | None = None,
        research: Any | None = None,
        displayed_lookup: Callable[[str], bool] | None = None,
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
        self.evidence_store = evidence_store
        self.research = research
        self.displayed_lookup = displayed_lookup
        self._tool_scope = threading.local()

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        call_id: str,
        turn: int,
        execution_context: Any | None = None,
    ) -> ToolResult:
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
            result = self._run_with_timeout(tool_name, parsed, execution_context)
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
    def _run_with_timeout(
        self,
        tool_name: str,
        parsed: StrictModel,
        execution_context: Any | None = None,
    ) -> ToolResult:
        timeout = self.budget.per_tool_timeout()
        outcome: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)
        cancel = threading.Event()

        def work() -> None:
            self._tool_scope.cancel = cancel
            self._tool_scope.evidence = {}
            self._tool_scope.context = execution_context
            try:
                outcome.put(("ok", self._dispatch(tool_name, parsed)))
            except Exception as exc:  # noqa: BLE001 - handed back to the caller
                outcome.put(("error", exc))
            finally:
                self._tool_scope.cancel = None
                self._tool_scope.context = None

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
                data={"message": truncate_utf8(value.message, 1500)},
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
        elif tool_name == "record_issue_assessment":
            result = self._record_assessment(parsed)  # type: ignore[arg-type]
        elif tool_name == "read_evidence":
            result = self._read_evidence(parsed)  # type: ignore[arg-type]
        else:  # guarded by caller
            raise ToolFailure("unknown_tool", tool_name)
        elapsed = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
        return self._bound_result(result.model_copy(update={"duration_ms": elapsed}))

    # -- tools --------------------------------------------------------------
    def _read_file(self, args: ReadFileArgs) -> ToolResult:
        payload, text = self.workspace.read_bytes(args.path)
        sha = hashlib.sha256(payload).hexdigest()
        lines = text.splitlines(keepends=True)
        start = args.start_line
        if start > len(lines):
            raise ToolFailure("line_out_of_range", f"{args.path} has {len(lines)} lines; start_line={start}")
        selected: list[str] = []
        used = 0
        index = start - 1
        line_truncated = False
        skipped_long_line = False
        while index < len(lines) and len(selected) < args.max_lines:
            line = lines[index]
            if len(line.encode()) > READ_MAX_BYTES - used:
                if not selected:
                    selected.append(truncate_utf8(line, READ_MAX_BYTES))
                    line_truncated = True
                    index += 1
                else:
                    skipped_long_line = True
                break
            selected.append(line)
            used += len(line.encode())
            index += 1
        end = start + len(selected) - 1
        snippet = "".join(selected)
        start_byte = len("".join(lines[:start - 1]).encode())
        more = start_byte + len(snippet.encode()) < len(payload)
        evidence = build_evidence(
            kind=EvidenceKind.LOCAL_FILE, project_id=self.project_id,
            path=args.path, line_start=start, line_end=end, sha256=sha,
            total_lines=len(lines), snippet=snippet, truncated=more,
        )
        evidence = self._save_evidence(evidence, snapshot=text, start_byte=start_byte,
                                       capture_scope="file", complete=True)
        continuation = {"start_line": index + 1} if more else None
        if line_truncated or skipped_long_line:
            continuation = {"reason": "line_too_long"}
            if index < len(lines):
                continuation["next_line"] = index + 1
        if more:
            continuation = {**(continuation or {}), **self._continuation(evidence, start_byte + len(snippet.encode()))}
        return ToolResult(
            status=ToolStatus.OK,
            data={"path": args.path, "start_line": start, "end_line": end,
                  "total_lines": len(lines), "sha256": sha,
                  "lines": [{"line": start + offset, "text": value.rstrip("\r\n")}
                            for offset, value in enumerate(selected)],
                  "line_truncated": line_truncated, "skipped_long_line": skipped_long_line,
                  **self._capture_data(evidence)},
            evidence_refs=[evidence.evidence_id], truncated=more,
            continuation=continuation, duration_ms=0,
        )

    def _read_evidence(self, args: ReadEvidenceArgs) -> ToolResult:
        if self.evidence_store is None:
            raise ToolFailure("evidence_unavailable", "this executor has no run evidence store")
        try:
            root = self.evidence_store.require(args.evidence_id)
            text = self.evidence_store.capture_text(root.evidence_id)
        except InputError as exc:
            raise ToolFailure("unknown_evidence", "evidence is not registered in this run") from exc
        except StoreError as exc:
            code = "snapshot_missing" if "missing" in exc.message else "snapshot_corrupt"
            raise ToolFailure(code, exc.message) from exc
        # Validate current local access policy, but never re-read a live source
        # to replace this object's bytes. Missing sources can still be historical.
        stale = root.stale
        stale_reason = root.stale_reason
        if root.path and root.kind == EvidenceKind.LOCAL_FILE:
            path = self.workspace.resolve(root.path, must_exist=False)
            try:
                current_hash = hash_file(path) if path.stat().st_size <= MAX_INPUT_BYTES else "source_too_large"
            except OSError:
                current_hash = None
            stale = current_hash != root.sha256
            stale_reason = ("source_missing" if current_hash is None else "source_changed") if stale else None
        raw = text.encode("utf-8")
        start = read_cursor(args.cursor, self.run_id, root, raw) if args.cursor else (root.snapshot_start_byte or 0)
        initial_end = root.snapshot_end_byte if root.snapshot_end_byte is not None else len(raw)
        stop = len(raw) if args.cursor else initial_end
        fragment = truncate_utf8(raw[start:stop].decode("utf-8"), min(args.max_bytes, MAX_WINDOW_BYTES))
        if not fragment:
            raise ToolFailure("empty_capture", "no captured bytes at this position")
        end = start + len(fragment.encode())
        base_line = root.snapshot_line_start or root.line_start or 1
        line_start = base_line + raw[:start].count(b"\n")
        # A range's identity includes offsets even when repeated text occurs.
        identity = "|".join([root.kind.value, root.path or root.url or "", root.sha256,
                             root.snapshot_sha256 or hashlib.sha256(raw).hexdigest(),
                             str(start), str(end), hashlib.sha256(fragment.encode()).hexdigest()])
        page = Evidence(
            schema_version=2, evidence_id="ev-" + hashlib.sha256(identity.encode()).hexdigest()[:12],
            kind=root.kind, project_id=root.project_id, path=root.path, url=root.url,
            line_start=line_start if root.kind == EvidenceKind.LOCAL_FILE else None,
            line_end=(line_start + fragment.count("\n") - int(fragment.endswith("\n"))) if root.kind == EvidenceKind.LOCAL_FILE else None,
            sha256=root.sha256, snippet=fragment, captured_at=root.captured_at,
            truncated=end < len(raw), total_lines=root.total_lines, commit=root.commit,
            stale=stale, stale_reason=stale_reason,
            parent_evidence_id=root.parent_evidence_id or root.evidence_id,
        )
        existing = next((item for item in self.evidence_store.all()
                         if item.snapshot_sha256 == root.snapshot_sha256
                         and item.snapshot_sha256 is not None
                         and item.sha256 == root.sha256 and item.kind == root.kind
                         and item.path == root.path and item.url == root.url
                         and item.capture_scope == root.capture_scope
                         and item.snapshot_line_start == root.snapshot_line_start
                         and item.snapshot_start_byte == start and item.snapshot_end_byte == end), None)
        page = existing or self._save_evidence(page, snapshot=text, start_byte=start,
                                              capture_scope=root.capture_scope,
                                              complete=root.complete_within_scope,
                                              notes=root.capture_notes)
        if stale:
            self.evidence_store.mark_source_stale(root, stale_reason or "source_changed")
            page = self.evidence_store.require(page.evidence_id)
        records = getattr(self._tool_scope, "evidence", None)
        if records is not None:
            records[page.evidence_id] = page
        next_cursor = make_cursor(self.run_id, root, end) if end < len(raw) else None
        return ToolResult(
            status=ToolStatus.OK,
            data={"parent_evidence_id": root.evidence_id, "evidence_id": page.evidence_id,
                  "source_sha256": root.sha256, "start_byte": start, "end_byte": end,
                  "stale": page.stale, "stale_reason": page.stale_reason,
                  "next_cursor": next_cursor, "cursor_evidence_id": root.evidence_id,
                  **self._capture_data(page)},
            evidence_refs=[page.evidence_id], truncated=end < len(raw),
            continuation={"tool": "read_evidence", "evidence_id": root.evidence_id,
                          "cursor": next_cursor} if next_cursor else None,
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
                    payload, text = self.workspace.read_bytes(relative)
                except ToolFailure as exc:
                    if exc.code not in {"invalid_utf8", "unsupported_binary"}:
                        scan_truncated = True
                    continue
                bytes_scanned += len(payload)
                files_scanned += 1
                sha = hashlib.sha256(payload).hexdigest()
                for line_number, line in enumerate(text.splitlines(), start=1):
                    match_start = _casefold_match_start(line, needle)
                    if match_start is None:
                        continue
                    # The search excerpt must contain the hit, including when
                    # it is well past the first 500 bytes of a long line.
                    anchor = max(0, match_start - 70)
                    shortened = truncate_utf8(line[anchor:], 500)
                    if args.query.casefold() not in shortened.casefold():
                        anchor = match_start
                        shortened = truncate_utf8(line[anchor:], 500)
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
                        byte_offset=len(line[:anchor].encode()),
                    )
                    evidence = self._save_evidence(
                        evidence, snapshot=line, start_byte=len(line[:anchor].encode()),
                        capture_scope="search_match", complete=True,
                        notes=["only this matched line was captured; other file content was not archived"],
                    )
                    evidence_refs.append(evidence.evidence_id)
                    line_truncated = line_truncated or len(shortened) < len(line)
                    matches.append(
                        {
                            "path": relative,
                            "line": line_number,
                            "text": shortened,
                            "evidence_id": evidence.evidence_id,
                            "line_truncated": len(shortened) < len(line),
                            **self._capture_data(evidence),
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
            evidence = self._save_evidence(evidence, snapshot=snippet, capture_scope="git_output", complete=True)
            return ToolResult(
                status=ToolStatus.OK,
                data={"commit": None, "dirty": None, "files": file_hashes, "diff": None, "diff_truncated": False, **self._capture_data(evidence)},
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
        evidence = self._save_evidence(evidence, snapshot=snippet, capture_scope="git_output", complete=not diff_truncated, notes=["git subprocess capture may be partial"] if diff_truncated else [])
        return ToolResult(
            status=ToolStatus.OK,
            data={
                "commit": commit,
                "dirty": bool(status.strip()),
                "files": file_hashes,
                "diff": diff,
                "diff_truncated": diff_truncated,
                **self._capture_data(evidence),
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
        windows: list[dict[str, Any]] = []
        for lesson in lessons:
            snippet = json.dumps(lesson.model_dump(mode="json"), ensure_ascii=False, indent=2)
            evidence = build_evidence(
                kind=EvidenceKind.LESSON,
                project_id=self.project_id,
                snippet=snippet,
                sha256=hashlib.sha256(snippet.encode()).hexdigest(),
            )
            evidence = self._save_evidence(evidence, snapshot=snippet, capture_scope="lesson", complete=True, notes=[])
            evidence_refs.append(evidence.evidence_id)
            windows.extend(self._capture_data(evidence)["evidence_windows"])
        return ToolResult(
            status=ToolStatus.OK,
            data={
                "query": args.query,
                "lessons": [lesson.model_dump(mode="json") for lesson in lessons],
                "note": "candidate lessons are reference material, not confirmed facts",
                "evidence_windows": windows,
            },
            evidence_refs=evidence_refs,
            duration_ms=0,
        )

    def _github_search(self, args: GithubSearchIssuesArgs) -> ToolResult:
        if not self.github.available:
            raise ToolFailure(self.github.unavailable_reason or "mcp_unavailable", "GitHub MCP is unavailable")
        context = getattr(self._tool_scope, "context", None)
        research = self.research
        action = context if getattr(context, "kind", None) == "search" else None
        scope: SearchScope | None = None
        query = args.query
        if research is not None:
            if action is not None:
                scope = action.scope
                query = action.query or args.query
            else:
                if research.state.status == "done":
                    raise ToolFailure("research_finished", "issue research already finished in this run")
                scope = research.model_scope()
                try:
                    compiled = compile_query([args.query], scope)
                except ScopeViolation as exc:
                    raise ToolFailure(exc.code, str(exc)[:500]) from exc
                query = compiled.rsplit(f" {scope.qualifier} is:issue", 1)[0]
                if research.state.searches_used >= research.config.max_searches:
                    raise ToolFailure("research_search_limit", "the configured layered search budget is exhausted")
            if not research.note_remote_request():
                raise ToolFailure("research_remote_limit", "the configured remote request budget is exhausted")
            outcome = self.github.search_issues(query, args.limit, scope=scope)
        else:
            outcome = self.github.search_issues(query, args.limit)
        evidence_refs: list[str] = []
        issues: list[dict[str, Any]] = []
        receipt_issues: list[Any] = []
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
            evidence = self._save_evidence(evidence, snapshot=snippet, capture_scope="github_response", complete=False, notes=["search candidate only; issue body and comment completeness are not established"])
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
                    **self._capture_data(evidence),
                }
            )
            receipt_issues.append((issue, evidence.evidence_id))
        truncated = not outcome.complete
        data: dict[str, Any] = {
            "query": query,
            "scope": outcome.query_scope,
            "layer": outcome.layer,
            "total_count": outcome.total_count,
            "issues": issues,
            "complete": outcome.complete,
        }
        if research is not None:
            receipt, candidates = self._register_search_receipt(
                research=research,
                action=action,
                scope=scope,
                query=query,
                call_id=context.call_id if action is not None else f"model-{len(research.state.receipts) + 1:03d}",
                receipt_issues=receipt_issues,
                complete=outcome.complete,
                limit=args.limit,
            )
            data["receipt_id"] = receipt.receipt_id
            data["candidates"] = [candidate.issue_ref for candidate in candidates]
            data["search_evidence_id"] = receipt.evidence_id
        return ToolResult(
            status=ToolStatus.OK,
            data=data,
            evidence_refs=evidence_refs,
            truncated=truncated,
            continuation={"reason": "page_limited", "next": "narrow the query"} if truncated else None,
            duration_ms=0,
        )

    def _register_search_receipt(
        self,
        *,
        research: Any,
        action: Any,
        scope: SearchScope,
        query: str,
        call_id: str,
        receipt_issues: list[tuple[dict[str, Any], str]],
        complete: bool,
        limit: int,
    ) -> tuple[Any, list[Any]]:
        from .contracts import IssueCandidate, ReceiptIssue, SearchReceipt
        from .issue_research import ResearchAction

        snapshot = json.dumps(
            [issue for issue, _evidence in receipt_issues], ensure_ascii=False, sort_keys=True
        )
        snapshot_sha = hashlib.sha256(snapshot.encode()).hexdigest()
        receipt_evidence = build_evidence(
            kind=EvidenceKind.ISSUE,
            project_id=self.project_id,
            url="https://github.com",
            snippet=snapshot,
            sha256=snapshot_sha,
        )
        receipt_evidence = self._save_evidence(
            receipt_evidence,
            snapshot=snapshot,
            capture_scope="github_response",
            complete=complete,
            notes=["search response snapshot bound to one receipt"],
        )
        digest = hashlib.sha256(
            f"{self.run_id}|{scope.layer.value}|{scope.label}|{query}|{call_id}|{receipt_evidence.evidence_id}".encode()
        ).hexdigest()[:12]
        receipt = SearchReceipt(
            receipt_id=f"rcpt-{digest}",
            run_id=self.run_id,
            layer=scope.layer,
            scope=scope.label,
            query_sha256=hashlib.sha256(query.encode()).hexdigest(),
            call_id=call_id,
            evidence_id=receipt_evidence.evidence_id,
            snapshot_sha256=snapshot_sha,
            issues=[
                ReceiptIssue(
                    issue_ref=str(issue.get("issue_ref")),
                    number=int(issue.get("number")),
                    title=str(issue.get("title") or "")[:500] or None,
                    state=_issue_state(issue.get("state")),
                    url=str(issue.get("html_url") or issue.get("url") or ""),
                    evidence_id=evidence_id,
                )
                for issue, evidence_id in receipt_issues
            ],
            captured_at=datetime.now(timezone.utc),
        )
        candidates = [
            IssueCandidate(
                issue_ref=str(issue.get("issue_ref")),
                layer=scope.layer,
                owner=str(issue.get("owner") or str(issue.get("issue_ref", "")).split("/", 1)[0]),
                repo=str(issue.get("repo") or str(issue.get("issue_ref", "")).split("/", 1)[1].split("#", 1)[0]),
                number=int(issue.get("number")),
                title=str(issue.get("title") or "")[:500] or None,
                state=_issue_state(issue.get("state")),
                url=str(issue.get("html_url") or issue.get("url") or ""),
                receipt_id=receipt.receipt_id,
                search_evidence_id=evidence_id,
                updated_at=_parse_timestamp(issue.get("updated_at")),
            )
            for issue, evidence_id in receipt_issues
        ]
        if action is None:
            action = ResearchAction(
                kind="search",
                call_id=f"model-{call_id}",
                layer=scope.layer,
                scope=scope,
                query=query,
                limit=limit,
            )
        research.accept_search(
            action,
            receipt=receipt,
            candidates=candidates,
            complete=complete,
            coverage=None if complete else "search response may be paginated or incomplete",
        )
        return receipt, candidates

    def _github_read(self, args: GithubReadIssueArgs) -> ToolResult:
        if not self.github.available:
            raise ToolFailure(self.github.unavailable_reason or "mcp_unavailable", "GitHub MCP is unavailable")
        context = getattr(self._tool_scope, "context", None)
        research = self.research
        action = context if getattr(context, "kind", None) == "read" else None
        if research is not None:
            if not research.can_read(args.issue_ref):
                raise ToolFailure(
                    "out_of_scope_issue",
                    "issue_ref has no persisted search receipt from an authorized layer in this run",
                )
            if any(
                candidate.issue_ref == args.issue_ref and candidate.read
                for candidate in research.state.candidates
            ):
                raise ToolFailure("already_read", "this candidate body was already read in this run")
            if research.state.status == "done":
                raise ToolFailure("research_finished", "issue research already finished in this run")
            if research.state.reads_used >= research.config.max_issue_reads:
                raise ToolFailure("research_read_limit", "the configured issue read budget is exhausted")
            if not research.note_remote_request():
                raise ToolFailure("research_remote_limit", "the configured remote request budget is exhausted")
            outcome = self.github.read_issue(args.issue_ref, allow_any_repo=True)
            if outcome.issue.get("issue_ref") != args.issue_ref:
                raise ToolFailure("ambiguous_issue_identity", "issue_read returned a different issue than requested")
        else:
            outcome = self.github.read_issue(args.issue_ref)
        snippet = json.dumps(
            {"issue": outcome.issue, "comments": outcome.comments, "comments_error": outcome.comments_error},
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
        evidence = self._save_evidence(evidence, snapshot=snippet, capture_scope="github_response", complete=False, notes=["captured backend response only; full comment pagination is not established", *([outcome.comments_error] if outcome.comments_error else [])])
        data: dict[str, Any] = {
            "issue": outcome.issue,
            "comments": outcome.comments,
            "updated_at": outcome.updated_at,
            "comments_error": outcome.comments_error,
            "comments_complete": outcome.comments_complete,
            "evidence_id": evidence.evidence_id,
            **self._capture_data(evidence),
        }
        if research is not None:
            if action is None:
                from .issue_research import ResearchAction

                action = ResearchAction(
                    kind="read",
                    call_id=f"model-{len(research.state.receipts) + research.state.reads_used + 1:03d}",
                    layer=research.state.layer,
                    scope=research.current_scope(),
                    issue_ref=args.issue_ref,
                )
            research.accept_read(action, evidence_id=evidence.evidence_id)
            data["layer"] = action.layer.value
        return ToolResult(
            status=ToolStatus.OK,
            data=data,
            evidence_refs=[evidence.evidence_id],
            truncated=evidence.truncated,
            continuation={"reason": "snippet_truncated", **self._continuation(evidence, len(evidence.snippet.encode()))} if evidence.truncated else None,
            duration_ms=0,
        )

    def _record_assessment(self, args: RecordIssueAssessmentArgs) -> ToolResult:
        research = self.research
        if research is None:
            raise ToolFailure("not_available", "layered issue research is not enabled for this run")
        candidate = research.candidate(args.issue_ref)
        if candidate is None:
            raise ToolFailure(
                "out_of_scope_issue",
                "assessment references an issue that this run did not locate and read",
            )
        if not candidate.read:
            raise ToolFailure("issue_not_read", "read the issue body before assessing it")
        if candidate.read_evidence_id not in args.evidence_refs:
            raise ToolFailure(
                "assessment_missing_issue_body_evidence",
                "assessment must cite the displayed body evidence from reading this issue",
            )
        for evidence_id in args.evidence_refs:
            if self.evidence_lookup is not None and not self.evidence_lookup(evidence_id):
                raise ToolFailure("unknown_evidence", "assessment cites evidence that is not registered in this run")
            if self.displayed_lookup is not None and not self.displayed_lookup(evidence_id):
                raise ToolFailure(
                    "assessment_evidence_not_displayed",
                    "assessment cites evidence the model has not received in a request",
                )
        assessment = IssueAssessment(
            issue_ref=args.issue_ref,
            component=args.component,
            symptom=args.symptom,
            version=args.version,
            environment=args.environment,
            trigger=args.trigger,
            fix=args.fix,
            evidence_refs=args.evidence_refs,
            note=args.note,
            recorded_at=datetime.now(timezone.utc),
        )
        return ToolResult(
            status=ToolStatus.OK,
            data={
                "status": "pending_commit",
                "assessment": assessment.model_dump(mode="json"),
                "applicable": assessment.applicable,
            },
            evidence_refs=[],
            duration_ms=0,
        )

    # -- helpers ------------------------------------------------------------
    def _save_evidence(
        self, evidence: Evidence, *, snapshot: str | None = None,
        start_byte: int = 0, capture_scope: str = "snippet", complete: bool = False,
        notes: list[str] | None = None,
    ) -> Evidence:
        cancel = getattr(self._tool_scope, "cancel", None)
        if cancel is not None and cancel.is_set():
            raise ToolFailure("tool_timeout", "capture cancelled after the tool deadline")
        try:
            if snapshot is not None and self.evidence_store is not None:
                saved = self.evidence_store.add_capture(
                    evidence, snapshot, start_byte=start_byte, capture_scope=capture_scope,
                    complete_within_scope=complete, notes=notes,
                )
            else:
                saved = self.evidence_sink(evidence) if self.evidence_sink else evidence
        except (StoreError, OSError) as exc:
            raise ToolFailure("archive_failed", "capture was not committed; no recoverable pointer was published; retry after storage is available") from exc
        records = getattr(self._tool_scope, "evidence", None)
        if records is not None:
            records[saved.evidence_id] = saved
        return saved

    def _capture_data(self, evidence: Evidence) -> dict[str, Any]:
        windows = select_windows(evidence, [], 16000)
        if evidence.snapshot_sha256 is not None:
            windows = [window.model_copy(update={
                "read_cursor": make_cursor(self.run_id, evidence, (evidence.snapshot_start_byte or 0) + window.start_byte)
            }) for window in windows]
        data = {
            "recoverable": evidence.snapshot_sha256 is not None,
            "snapshot_sha256": evidence.snapshot_sha256,
            "snapshot_bytes": evidence.snapshot_bytes,
            "capture_scope": evidence.capture_scope,
            "complete_within_scope": evidence.complete_within_scope,
            "capture_notes": evidence.capture_notes,
            "evidence_windows": [window.model_dump(mode="json") for window in windows],
        }
        if evidence.snapshot_sha256 and evidence.snapshot_start_byte:
            # A read beginning in the middle of a file must also expose a way
            # to read its captured prefix, without re-opening the live source.
            data["snapshot_start_cursor"] = make_cursor(self.run_id, evidence, 0)
        return data

    def _continuation(self, evidence: Evidence, offset: int) -> dict[str, Any]:
        if evidence.snapshot_sha256 is None:
            return {"reason": "capture_unavailable", "next": "re-read the source using the original tool"}
        return {"tool": "read_evidence", "evidence_id": evidence.evidence_id,
                "cursor": make_cursor(self.run_id, evidence, offset)}

    def _bound_result(self, result: ToolResult) -> ToolResult:
        def size(value: ToolResult) -> int:
            return len(json.dumps(value.model_dump(mode="json"), ensure_ascii=False).encode())
        if size(result) <= TOOL_RESULT_MAX_BYTES:
            return result
        records = getattr(self._tool_scope, "evidence", {})
        items = [records[eid] for eid in result.evidence_refs if eid in records]
        if not items:
            # No archived material means there is no safe offload pointer.
            raise ToolFailure("result_too_large", "result exceeds 8 KiB and no committed capture can be read back")
        original = result.data if isinstance(result.data, dict) else {}
        metadata_keys = {"path", "start_line", "end_line", "total_lines", "sha256",
                         "line_truncated", "skipped_long_line", "commit", "dirty",
                         "comments_error", "stale", "stale_reason"}
        metadata = {key: value for key, value in original.items() if key in metadata_keys}
        # Metadata, encoded JSON, cursors and all literal fragments share the
        # same hard result bound. More evidence remains indexed, not displayed.
        visible = items[:4]
        body_bytes = 3500
        while visible:
            windows = []
            for item in visible:
                chosen = select_windows(item, [], max(12, body_bytes // len(visible)))[:1]
                for window in chosen:
                    if item.snapshot_sha256:
                        window = window.model_copy(update={"read_cursor": make_cursor(
                            self.run_id, item, (item.snapshot_start_byte or 0) + window.start_byte)})
                    windows.append(window)
            first = visible[0]
            first_windows = [w for w in windows if w.evidence_id == first.evidence_id]
            end = (first.snapshot_start_byte or 0) + (first_windows[-1].end_byte if first_windows else 0)
            more = first.snapshot_bytes is not None and end < first.snapshot_bytes
            continuation = self._continuation(first, end if more else (first.snapshot_start_byte or 0))
            candidate = ToolResult(
                status=ToolStatus.OK,
                data={**metadata, "structured_result_omitted": True,
                      "unshown_fields_remain_unknown": True,
                      "capture_scope": first.capture_scope,
                      "complete_within_scope": first.complete_within_scope,
                      "capture_notes": first.capture_notes,
                      "snapshot_sha256": first.snapshot_sha256,
                      "snapshot_bytes": first.snapshot_bytes,
                      "recoverable": first.snapshot_sha256 is not None,
                      "next_cursor": continuation.get("cursor"),
                      "cursor_evidence_id": first.evidence_id,
                      **({"snapshot_start_cursor": make_cursor(self.run_id, first, 0)}
                         if first.snapshot_sha256 and first.snapshot_start_byte else {}),
                      "evidence_windows": [w.model_dump(mode="json") for w in windows],
                      "unshown_evidence_ids": [eid for eid in result.evidence_refs
                                               if eid not in {w.evidence_id for w in windows}]},
                evidence_refs=result.evidence_refs, truncated=True,
                continuation={**(result.continuation or {}), **continuation,
                              "reason": (result.continuation or {}).get("reason", "result_budget")},
                duration_ms=result.duration_ms,
            )
            if size(candidate) <= TOOL_RESULT_MAX_BYTES:
                return candidate
            body_bytes = body_bytes * 3 // 4
            if body_bytes < 40:
                visible = visible[:-1]
                body_bytes = 2000
        raise ToolFailure("result_too_large", "capture metadata does not fit the 8 KiB tool result limit")

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


def _issue_state(value: Any) -> Literal["open", "closed", "unknown"]:
    return value if value in {"open", "closed"} else "unknown"


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def evidence_snippet_from_store(evidence: Evidence) -> str:
    return evidence.snippet
