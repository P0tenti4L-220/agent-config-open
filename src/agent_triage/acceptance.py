"""Acceptance evidence contracts, provenance binding and profile aggregation.

This module is deliberately separate from ``eval.py``: ``eval`` executes tests and
writes evidence, while ``accept`` only inspects an existing evidence directory.
It never runs pytest, a model, a tool or a network request.  The distinction
matters because an executed check is not the same thing as an accepted result.

The authoritative rules are:

* only ``pass`` for the *current* target counts towards a profile;
* a missing, skipped, stale or unverifiable result is ``incomplete``, never pass;
* a required check cannot be waived with ``not_applicable``;
* an error outranks a failure, which outranks an incomplete result.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import ConfigDict, Field, ValidationError, field_validator, model_validator

from .case_manifest import CaseManifest, case_definition_hash, trusted_manifest
from .config import Config, config_fingerprint
from .contracts import (
    CheckStatus,
    NonEmptyStr,
    RelativePath,
    Sha256,
    StrictBool,
    StrictInt,
    StrictModel,
)
from .errors import InputError, ValidationFailure
from .store import atomic_write_json, atomic_write_text, read_json, utcnow

SCHEMA_VERSION = 1
CHECK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,63}$")
EVALUATION_ID_RE = re.compile(r"^eval-[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")

#: Directories whose contents are never part of the evaluated tree.  ``local``
#: holds run/evidence output, caches are host-specific, and ``.git`` has its own
#: commit identity.  Credential files are excluded by name as well.
EXCLUDED_DIRS = {
    ".git",
    "local",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "node_modules",
}
EXCLUDED_NAMES = {
    ".DS_Store",
    "auth.json",
    "auth.json.bak",
    ".env",
    ".netrc",
    "id_rsa",
    "id_ed25519",
    "credentials.json",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".pem", ".key", ".p12", ".pfx"}

MAX_ARTIFACT_BYTES = 8 * 1024 * 1024


# ---------------------------------------------------------------------------
# Capability catalog (H01)
# ---------------------------------------------------------------------------


class Capability(StrictModel):
    id: str
    behavior: NonEmptyStr
    requirements: list[NonEmptyStr] = Field(min_length=1, max_length=20)
    check_ids: list[str] = Field(min_length=1, max_length=60)
    out_of_scope: list[NonEmptyStr] = Field(default_factory=list, max_length=20)

    @field_validator("id")
    @classmethod
    def _capability_id(cls, value: str) -> str:
        if not re.fullmatch(r"F0[1-6]", value):
            raise ValueError("capability id must be F01..F06")
        return value

    @field_validator("check_ids")
    @classmethod
    def _check_id_shape(cls, value: list[str]) -> list[str]:
        for item in value:
            if not CHECK_ID_RE.fullmatch(item):
                raise ValueError(f"invalid check id: {item}")
        return value


class CapabilityCatalog(StrictModel):
    schema_version: Literal[1] = 1
    kind: Literal["capability_catalog"] = "capability_catalog"
    capabilities: list[Capability] = Field(min_length=6, max_length=6)


def load_capabilities(path: Path) -> CapabilityCatalog:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise InputError(f"cannot read capability catalog: {path.name}", detail=str(exc)) from exc
    if len(raw) > 256 * 1024:
        raise InputError("capability catalog is larger than 256 KiB")
    try:
        return CapabilityCatalog.model_validate_json(raw)
    except ValidationError as exc:
        raise ValidationFailure("capability catalog does not match the expected shape", detail=str(exc)) from exc


def validate_capabilities(catalog: CapabilityCatalog, *, known_check_ids: Iterable[str]) -> list[str]:
    """Return problems; an empty list means the declaration itself is consistent.

    This validates the *declaration*.  It never turns a capability into a pass:
    execution results live in an evidence directory and are aggregated by
    :func:`accept_profile`.
    """
    problems: list[str] = []
    ids = [item.id for item in catalog.capabilities]
    if len(ids) != len(set(ids)):
        problems.append("capability ids must be unique")
    if set(ids) != {f"F0{index}" for index in range(1, 7)}:
        problems.append("capability ids must be exactly F01..F06")
    known = set(known_check_ids)
    requirements: list[str] = []
    for capability in catalog.capabilities:
        requirements.extend(capability.requirements)
        for check_id in capability.check_ids:
            if check_id not in known:
                problems.append(f"{capability.id} references unknown check {check_id}")
        if len(capability.check_ids) != len(set(capability.check_ids)):
            problems.append(f"{capability.id} repeats a check id")
    if len(requirements) != len(set(requirements)):
        problems.append("requirements must be unique across capabilities")
    return problems


# ---------------------------------------------------------------------------
# Target provenance
# ---------------------------------------------------------------------------


class Target(StrictModel):
    schema_version: Literal[1] = 1
    code_commit: str | None = None
    relevant_tree_digest: Sha256
    dirty: bool | None = None
    # ``config_hash`` is the fixed offline configuration.  Live runs add
    # ``live_config_hash`` so switching a connected configuration does not make
    # every offline result stale, while live results still bind to the actual
    # configuration used.
    config_hash: Sha256
    live_config_hash: Sha256 | None = None
    versions: dict[str, str]
    corpus_digest: Sha256
    check_definition_hash: Sha256

    @field_validator("code_commit")
    @classmethod
    def _commit_shape(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError("code_commit must be a full 40-character git commit")
        return value


class NodeResult(StrictModel):
    nodeid: str
    outcome: Literal["passed", "failed", "skipped", "error", "missing"]
    duration_ms: StrictInt | None = None
    message: str | None = None


class Execution(StrictModel):
    started_at: datetime | None = None
    finished_at: datetime | None = None
    argv: list[str] | None = None
    exit_code: StrictInt | None = None
    interpreter_version: str | None = None
    import_root_checked: bool | None = None
    executor_id: str | None = None


class Artifact(StrictModel):
    relative_path: RelativePath
    sha256: Sha256
    media_type: str = "application/octet-stream"
    size_bytes: StrictInt | None = None

    @field_validator("media_type")
    @classmethod
    def _media_type_shape(cls, value: str) -> str:
        if "/" not in value or len(value) > 100:
            raise ValueError("media_type must look like type/subtype")
        return value


FROZEN_QUALITY_CASES: tuple[str, ...] = ("QA01", "QA02", "QA03", "QA04", "QA05")
REQUIRED_PLAYBOOKS: tuple[str, ...] = (
    "structured-output-mismatch",
    "tool-failure-and-recovery",
    "context-and-handoff-gap",
)
HUMAN_SUMMARY_CHECKS = {"AC15.quality", "HC10"}
PLAYBOOK_WALKTHROUGH_CHECKS = {"NC20.manual"}
HANDOFF_WALKTHROUGH_CHECKS = {"HC13"}
HUMAN_SCHEMA_VERSION = 3
CASE_EVIDENCE_ROLES: tuple[str, ...] = ("event", "run", "analysis", "display", "reference")
CASE_HASH_KEYS: tuple[str, ...] = (*CASE_EVIDENCE_ROLES, "analysis_content")


class HumanCaseScore(StrictModel):
    """One frozen case judged by a human reviewer.

    Schema 3 does not accept bare hash strings as proof.  Each case names the
    bounded bundle that was actually reviewed, lists every file in that bundle
    and records the role of the event, run metadata, analysis, display records
    and reference material.  ``acceptance_review`` re-hashes those files on
    import and again on every ``accept`` run.
    """

    case_id: str = Field(min_length=1, max_length=40)
    run_id: str = Field(min_length=1, max_length=80)
    provider_mode: Literal["replay", "live"]
    bundle_dir: RelativePath
    artifacts: list[Artifact] = Field(min_length=1, max_length=100)
    roles: dict[str, RelativePath]
    reviewed_subject_hashes: dict[str, Sha256]
    verdict: Literal["pass", "fail"]
    rationale: NonEmptyStr
    unsupported_fact: StrictBool = False
    unexecuted_check_reported_pass: StrictBool = False
    unauthorized_write: StrictBool = False

    @model_validator(mode="after")
    def _bundle_binding(self) -> "HumanCaseScore":
        if set(self.roles) != set(CASE_EVIDENCE_ROLES):
            raise ValueError("case roles must name event, run, analysis, display and reference")
        if set(self.reviewed_subject_hashes) != set(CASE_HASH_KEYS):
            raise ValueError("reviewed_subject_hashes must cover every case role and analysis_content")
        paths = [artifact.relative_path for artifact in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("case bundle repeats an artifact path")
        bundle = self.bundle_dir.rstrip("/")
        for path in paths:
            if path != bundle and not path.startswith(bundle + "/"):
                raise ValueError(f"artifact {path} is outside bundle_dir {self.bundle_dir}")
        for role, path in self.roles.items():
            if path not in paths:
                raise ValueError(f"role {role} is not covered by the case artifact list")
        return self


class PlaybookWalkthrough(StrictModel):
    playbook_id: Literal[
        "structured-output-mismatch",
        "tool-failure-and-recovery",
        "context-and-handoff-gap",
    ]
    sample: NonEmptyStr
    steps: NonEmptyStr
    outcome: NonEmptyStr
    blockers: list[NonEmptyStr] = Field(default_factory=list, max_length=10)
    verdict: Literal["pass", "fail"]


class HandoffWalkthrough(StrictModel):
    operator: NonEmptyStr
    no_chat_history: Literal[True]
    target_identified: StrictBool
    facts_identified: StrictBool
    gaps_identified: StrictBool
    next_step_identified: StrictBool
    result: NonEmptyStr
    blockers: list[NonEmptyStr] = Field(default_factory=list, max_length=10)
    verdict: Literal["pass", "fail"]


class HumanReviewRecord(StrictModel):
    """Structured external review used by acceptance-level human checks."""

    reviewer: NonEmptyStr
    method: Literal["human"] = "human"
    reviewed_at: datetime
    rubric_version: NonEmptyStr
    cases: list[HumanCaseScore] = Field(default_factory=list, max_length=20)
    playbooks: list[PlaybookWalkthrough] = Field(default_factory=list, max_length=5)
    handoff: HandoffWalkthrough | None = None
    summary: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _no_duplicate_items(self) -> "HumanReviewRecord":
        case_ids = [item.case_id for item in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("human review repeats a case_id")
        playbooks = [item.playbook_id for item in self.playbooks]
        if len(playbooks) != len(set(playbooks)):
            raise ValueError("human review repeats a playbook")
        return self


def human_review_problems(check_id: str, review: HumanReviewRecord | None) -> list[str]:
    if review is None:
        return ["structured human_review record is required"]
    problems: list[str] = []
    if check_id in HUMAN_SUMMARY_CHECKS:
        ids = [item.case_id for item in review.cases]
        if tuple(sorted(ids)) != tuple(sorted(FROZEN_QUALITY_CASES)):
            problems.append("quality review must cover exactly the five frozen case ids")
        if len(review.cases) != len(FROZEN_QUALITY_CASES):
            problems.append("quality review must not duplicate or omit a frozen case")
        passes = sum(1 for item in review.cases if item.verdict == "pass")
        if passes < 4:
            problems.append(f"quality review has {passes}/5 passing cases; at least 4 are required")
        for item in review.cases:
            if item.unsupported_fact:
                problems.append(f"{item.case_id} contains an unsupported fact")
            if item.unexecuted_check_reported_pass:
                problems.append(f"{item.case_id} reported an unexecuted check as pass")
            if item.unauthorized_write:
                problems.append(f"{item.case_id} performed an unauthorized write")
    elif check_id in PLAYBOOK_WALKTHROUGH_CHECKS:
        ids = {item.playbook_id for item in review.playbooks}
        if ids != set(REQUIRED_PLAYBOOKS):
            problems.append("manual walkthrough must cover all three playbooks exactly once")
        if any(item.verdict != "pass" for item in review.playbooks):
            problems.append("a playbook walkthrough did not pass")
        for item in review.playbooks:
            if item.verdict == "pass" and item.blockers:
                # A recorded blocker is a legitimate observation, but the
                # observation is a fail until a verifiable resolution exists.
                problems.append(f"{item.playbook_id} reports pass while unresolved blockers remain")
    elif check_id in HANDOFF_WALKTHROUGH_CHECKS:
        if review.handoff is None:
            problems.append("handoff walkthrough record is missing")
        else:
            handoff = review.handoff
            if not all(
                [
                    handoff.target_identified,
                    handoff.facts_identified,
                    handoff.gaps_identified,
                    handoff.next_step_identified,
                ]
            ):
                problems.append("handoff walkthrough did not identify target/facts/gaps/next step")
            if handoff.verdict != "pass":
                problems.append("handoff walkthrough did not pass")
            if handoff.verdict == "pass" and handoff.blockers:
                problems.append("handoff walkthrough reports pass while unresolved blockers remain")
    return problems


class ExecutionResult(StrictModel):
    schema_version: Literal[1] = 1
    evaluation_id: str
    check_id: str
    attempt_id: str
    status: CheckStatus
    reason_code: str | None = None
    reason: str | None = None
    source_type: Literal["pytest", "run_artifact", "human_review"]
    case_definition_hash: Sha256 | None = None
    target: Target | None = None
    execution: Execution = Field(default_factory=Execution)
    related_run_ids: list[str] = Field(default_factory=list, max_length=20)
    artifacts: list[Artifact] = Field(default_factory=list, max_length=200)
    nodes: list[NodeResult] = Field(default_factory=list, max_length=200)
    human_review: HumanReviewRecord | None = None

    @model_validator(mode="after")
    def _human_binding(self) -> "ExecutionResult":
        if self.human_review is not None and self.source_type != "human_review":
            raise ValueError("human_review is only valid for a human_review result")
        if self.source_type == "human_review" and self.status == CheckStatus.PASS:
            problems = human_review_problems(self.check_id, self.human_review)
            if problems:
                raise ValueError("; ".join(problems))
        return self

    @field_validator("check_id")
    @classmethod
    def _check_id_shape(cls, value: str) -> str:
        if not CHECK_ID_RE.fullmatch(value):
            raise ValueError("check_id contains unsupported characters")
        return value

    @field_validator("evaluation_id")
    @classmethod
    def _evaluation_id_shape(cls, value: str) -> str:
        if not EVALUATION_ID_RE.fullmatch(value):
            raise ValueError("evaluation_id must look like eval-YYYYMMDD-HHMMSS-abcdef")
        return value

    @model_validator(mode="after")
    def _status_shape(self) -> "ExecutionResult":
        if self.source_type == "human_review":
            if self.execution.argv is not None or self.execution.exit_code is not None:
                raise ValueError("human review must not invent a command or exit code")
            if self.target is None and self.status in (CheckStatus.PASS, CheckStatus.FAIL):
                raise ValueError("human review must declare the target it reviewed")
        if self.status == CheckStatus.PASS:
            if self.source_type == "run_artifact":
                if not self.related_run_ids:
                    raise ValueError("run_artifact pass requires the run id it came from")
                if not self.artifacts:
                    raise ValueError("run_artifact pass requires imported artifact hashes")
            if self.source_type == "human_review":
                if self.execution.argv is not None or self.execution.exit_code is not None:
                    raise ValueError("human review pass must not invent a command or exit code")
            elif self.execution.exit_code != 0:
                raise ValueError("pass requires an actual exit code 0")
            if self.source_type == "pytest":
                if not self.nodes:
                    raise ValueError("pytest pass requires structured node results")
                if any(node.outcome != "passed" for node in self.nodes):
                    raise ValueError("pytest pass requires every collected node to pass")
        if self.status == CheckStatus.FAIL and self.execution.exit_code in (None, 0):
            # A pytest failure can also be a collection error with a non-zero
            # exit; the caller maps that to ``error``.  A human fail has no
            # exit code, so only pytest is constrained here.
            if self.source_type == "pytest":
                raise ValueError("pytest fail requires a non-zero exit code")
        if self.status == CheckStatus.NOT_APPLICABLE and not (self.reason or "").strip():
            raise ValueError("not_applicable requires a reason")
        return self


class AcceptanceSummary(StrictModel):
    required: StrictInt
    pass_count: StrictInt
    fail_count: StrictInt
    error_count: StrictInt
    incomplete_count: StrictInt


class CheckAcceptance(StrictModel):
    check_id: str
    execution_status: CheckStatus
    effective_state: Literal["current", "stale", "unknown"]
    result_ref: str | None
    source_type: str | None
    reasons: list[str] = Field(default_factory=list)


class AcceptanceResult(StrictModel):
    schema_version: Literal[1] = 1
    profile: Literal["offline", "connected", "reviewed"]
    evaluated_target: Target
    profile_definition_hash: Sha256
    overall: Literal["pass", "fail", "error", "incomplete"]
    exit_code: StrictInt
    checks: list[CheckAcceptance]
    summary: AcceptanceSummary
    evidence_issues: list[str] = Field(default_factory=list, max_length=50)
    claim_scope: NonEmptyStr
    not_established: list[NonEmptyStr] = Field(min_length=1)


PROFILE_ORDER: tuple[str, ...] = ("offline", "connected", "reviewed")

_OFFLINE_REQUIRED = [
    *(f"AC{index:02d}" for index in range(1, 15)),
    *(f"NC{index:02d}" for index in range(1, 20)),
    "NC20.routing",
    *(f"NC{index:02d}" for index in range(21, 27)),
    *(f"HC{index:02d}" for index in range(1, 9)),
    "HC11",
    "HC12",
    "HC14",
    "HC15",
    "HC16",
    "HC17",
]

PROFILES: dict[str, dict[str, Any]] = {
    "offline": {
        "inherits": [],
        "required": _OFFLINE_REQUIRED,
        "claim_scope": (
            "当前版本的离线契约、回放数据流、本地 MCP 协议适配、验收来源绑定与开发交接检查"
        ),
        "not_established": [
            "真实模型与真实 GitHub MCP 联调（AC15.live / HC09）",
            "五例人工诊断质量与接手走查（AC15.quality / NC20.manual / HC10 / HC13）",
            "分析 complete 不等于根因已复现，诊断结论的语义正确性仍未证明",
        ],
    },
    "connected": {
        "inherits": ["offline"],
        "required": ["AC15.live", "HC09"],
        "claim_scope": "指定模型、配置和测试仓库的真实只读联调链路",
        "not_established": [
            "诊断结果普遍正确或生产可用",
            "五例人工试点与三类 playbook 人工走查",
        ],
    },
    "reviewed": {
        "inherits": ["connected"],
        "required": ["AC15.quality", "NC20.manual", "HC10", "HC13"],
        "claim_scope": "这组冻结小样本达到 MVP 试点标准",
        "not_established": [
            "普遍提升、统计显著性或生产可靠性",
            "未参与本轮评分的其他项目与错误类型",
        ],
    },
}


def profile_required(profile: str) -> list[str]:
    if profile not in PROFILES:
        raise InputError(f"unknown acceptance profile: {profile}")
    required: list[str] = []
    for inherited in PROFILES[profile]["inherits"]:
        required.extend(profile_required(inherited))
    required.extend(PROFILES[profile]["required"])
    # Preserve first-seen order while de-duplicating.
    return list(dict.fromkeys(required))


def profile_definition_hash() -> str:
    payload = {
        name: {
            "inherits": spec["inherits"],
            "required": spec["required"],
            "claim_scope": spec["claim_scope"],
            "not_established": spec["not_established"],
        }
        for name, spec in PROFILES.items()
    }
    return _sha256_text(_canonical(payload))


# ---------------------------------------------------------------------------
# Canonical hashing and target construction
# ---------------------------------------------------------------------------


def _canonical(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_excluded(path: Path) -> bool:
    if path.name in EXCLUDED_NAMES or path.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    return any(part in EXCLUDED_DIRS for part in path.parts)


def _tree_entries(root: Path, *, exclude: Iterable[Path] = ()) -> list[tuple[str, str]]:
    root = root.resolve()
    excluded = {item.resolve() for item in exclude}
    entries: list[tuple[str, str]] = []
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        directory = Path(current)
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in EXCLUDED_DIRS
            and not (directory / name).is_symlink()
            and (directory / name).resolve() not in excluded
        )
        for name in sorted(filenames):
            path = directory / name
            if path.is_symlink():
                resolved = path.resolve()
                if root not in resolved.parents:
                    continue
                path = resolved
            if path.resolve() in excluded or _is_excluded(path):
                continue
            relative = path.resolve().relative_to(root).as_posix()
            try:
                entries.append((relative, _sha256_file(path)))
            except OSError as exc:
                raise ValidationFailure(f"cannot hash evaluated file: {relative}", detail=str(exc)) from exc
    entries.sort()
    return entries


def tree_digest(root: Path, *, exclude: Iterable[Path] = ()) -> str:
    return _sha256_text(_canonical(_tree_entries(root, exclude=exclude)))


def _git(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def code_commit(root: Path) -> str | None:
    output = _git(root, "rev-parse", "HEAD")
    return output.strip() if output and re.fullmatch(r"[0-9a-f]{40}", output.strip()) else None


def tree_dirty(root: Path, *, exclude: Iterable[Path] = ()) -> bool | None:
    if code_commit(root) is None:
        return None
    output = _git(root, "status", "--porcelain", "--untracked-files=all")
    if output is None:
        return None
    excluded = {item.resolve() for item in exclude}
    for line in output.splitlines():
        if not line.strip():
            continue
        relative = line[3:].strip()
        if " -> " in relative:
            relative = relative.split(" -> ", 1)[1]
        relative = relative.strip('"')
        path = (root / relative).resolve()
        if path in excluded or _is_excluded(Path(relative)):
            continue
        return True
    return False


def corpus_digest(root: Path) -> str:
    from .case_manifest import load_manifest

    cases = root / "tests" / "cases"
    if not cases.is_dir():
        return _sha256_text("no-cases")
    try:
        return load_manifest(cases).digest
    except (InputError, ValidationFailure):
        # A broken manifest must not silently look like a valid corpus.  The
        # raw file digest keeps the target deterministic; accept will separately
        # refuse to treat results as current because no trusted definition can
        # be loaded.
        entries = [
            (path.name, _sha256_file(path))
            for path in sorted(cases.glob("*.json"))
            if not _is_excluded(path)
        ]
        return _sha256_text("invalid-manifest:" + _canonical(entries))


def versions_manifest() -> dict[str, str]:
    from . import __version__
    from .context import PROMPT_MANIFEST
    from .playbooks import catalog_manifest
    from .policy import version_manifest

    return {
        "package": __version__,
        **version_manifest(),
        **PROMPT_MANIFEST,
        **catalog_manifest(),
    }


def check_definition_hash(root: Path) -> str:
    capabilities = root / "docs" / "agent-capabilities.json"
    payload: dict[str, Any] = {
        "profiles": {
            name: spec for name, spec in PROFILES.items()
        },
        "capabilities": None,
    }
    if capabilities.is_file():
        try:
            payload["capabilities"] = json.loads(capabilities.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValidationFailure("cannot hash capability catalog", detail=str(exc)) from exc
    return _sha256_text(_canonical(payload))


def build_target(
    root: Path,
    *,
    config: Config | None = None,
    live_config: Config | None = None,
    exclude: Iterable[Path] = (),
) -> Target:
    root = root.resolve()
    if not root.is_dir():
        raise InputError(f"target root is not a directory: {root}")
    excluded = list(exclude)
    return Target(
        code_commit=code_commit(root),
        relevant_tree_digest=tree_digest(root, exclude=excluded),
        dirty=tree_dirty(root, exclude=excluded),
        config_hash=config_fingerprint(config or Config()),
        live_config_hash=config_fingerprint(live_config) if live_config is not None else None,
        versions=versions_manifest(),
        corpus_digest=corpus_digest(root),
        check_definition_hash=check_definition_hash(root),
    )


# ---------------------------------------------------------------------------
# Result loading and artifact verification
# ---------------------------------------------------------------------------


def effective_state(
    result: ExecutionResult,
    current: Target,
    *,
    live: bool = False,
) -> Literal["current", "stale", "unknown"]:
    target = result.target
    if target is None:
        return "unknown"
    if target.schema_version != current.schema_version:
        return "stale"
    if target.code_commit != current.code_commit:
        # Both None (not a git tree) still needs the digest comparison below.
        if not (target.code_commit is None and current.code_commit is None):
            return "stale"
    if target.relevant_tree_digest != current.relevant_tree_digest:
        return "stale"
    if live:
        if target.live_config_hash != current.live_config_hash:
            if target.live_config_hash is None and current.live_config_hash is None:
                # Both sides predate the live-config dimension; the case
                # definition hash still has to match.
                pass
            elif target.live_config_hash is None or current.live_config_hash is None:
                return "unknown"
            else:
                return "stale"
    elif target.config_hash != current.config_hash:
        return "stale"
    if target.corpus_digest != current.corpus_digest:
        return "stale"
    if target.check_definition_hash != current.check_definition_hash:
        return "stale"
    if target.versions != current.versions:
        return "stale"
    if target.dirty is False and current.dirty is True:
        return "stale"
    return "current"


def _safe_artifact_path(evidence_dir: Path, relative: str) -> Path:
    root = evidence_dir.resolve()
    candidate = root / relative
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValidationFailure(f"artifact must not be a symbolic link: {relative}")
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValidationFailure(f"artifact escapes the evidence directory: {relative}")
    if not resolved.is_file():
        raise ValidationFailure(f"artifact is missing: {relative}")
    return resolved


def _verify_artifacts(evidence_dir: Path, result: ExecutionResult) -> ExecutionResult:
    for artifact in result.artifacts:
        path = _safe_artifact_path(evidence_dir, artifact.relative_path)
        size = path.stat().st_size
        if size > MAX_ARTIFACT_BYTES:
            raise ValidationFailure(
                f"artifact is larger than {MAX_ARTIFACT_BYTES} bytes: {artifact.relative_path}"
            )
        if artifact.size_bytes is not None and artifact.size_bytes != size:
            raise ValidationFailure(f"artifact size changed: {artifact.relative_path}")
        actual = _sha256_file(path)
        if actual != artifact.sha256:
            raise ValidationFailure(f"artifact hash does not match: {artifact.relative_path}")
    return result


def _result_from_case(
    case: dict[str, Any],
    *,
    evaluation_id: str,
    target: Target | None,
    interpretation: str | None,
) -> ExecutionResult:
    source = case.get("source") or "pytest"
    source_type = "pytest" if source in {"pytest", "aggregate"} else source
    status_value = case.get("status") or "not_run"
    status = CheckStatus(status_value) if status_value in CheckStatus._value2member_map_ else CheckStatus.ERROR
    execution: dict[str, Any] = {
        "exit_code": case.get("exit_code"),
        "argv": case.get("argv"),
        "started_at": case.get("started_at"),
        "finished_at": case.get("finished_at"),
        "interpreter_version": case.get("interpreter_version"),
        "import_root_checked": case.get("import_root_checked"),
        "executor_id": "agent-triage eval",
    }
    if source_type == "human_review":
        execution = {key: value for key, value in execution.items() if value is not None and key == "executor_id"}
    payload = {
        "evaluation_id": evaluation_id,
        "check_id": case["id"],
        "attempt_id": str(case.get("attempt_id") or "1"),
        "status": status,
        "reason_code": case.get("reason_code") or ("not_run" if status == CheckStatus.NOT_RUN else None),
        "reason": case.get("reason") or interpretation,
        "source_type": source_type,
        "case_definition_hash": case.get("case_definition_hash"),
        "target": target.model_dump(mode="json") if target is not None else None,
        "execution": execution,
        "artifacts": case.get("artifacts") or [],
        "related_run_ids": case.get("related_run_ids") or [],
        "nodes": case.get("nodes") or [],
    }
    try:
        return ExecutionResult.model_validate(payload)
    except ValidationError as exc:
        raise ValidationFailure(f"invalid result for {case.get('id')}", detail=str(exc)) from exc


def load_execution_results(
    evidence_dir: Path,
    *,
    current: Target | None = None,
    manifest: Any = None,
) -> tuple[list[ExecutionResult], list[str]]:
    """Load results from a directory produced by ``eval`` or a synthetic fixture.

    Legacy ``eval.json`` (schema 1) is accepted but its results have no target
    provenance; callers therefore treat them as ``unknown`` and incomplete.
    """
    evidence_dir = evidence_dir.resolve()
    if not evidence_dir.is_dir():
        raise InputError(f"evidence directory not found: {evidence_dir}")
    issues: list[str] = []
    results: list[ExecutionResult] = []
    payload = None
    envelope: EvaluationEnvelope | None = None
    legacy_v1 = False
    eval_path = evidence_dir / "eval.json"
    provenance: Target | None = None
    provenance_id: str | None = None
    if (evidence_dir / "provenance.json").is_file():
        try:
            provenance_payload = read_json(evidence_dir / "provenance.json")
            provenance_id = provenance_payload.get("evaluation_id")
            provenance = Target.model_validate(provenance_payload.get("target", provenance_payload))
        except (InputError, ValidationError) as exc:
            issues.append(f"provenance.json is invalid: {exc}")

    if eval_path.is_file():
        try:
            payload = read_json(eval_path)
        except InputError as exc:
            issues.append(f"eval.json is unreadable: {exc}")
        if payload is not None:
            if payload.get("schema_version") == 1:
                legacy_v1 = True
                for case in payload.get("cases", []):
                    results.append(
                        ExecutionResult(
                            evaluation_id="eval-19700101-000000-000000",
                            check_id=case["id"],
                            attempt_id="legacy",
                            status=CheckStatus.NOT_RUN,
                            reason=(
                                f"legacy v1 result ({case.get('status', 'unknown')}) has no provenance; "
                                "it cannot count as current"
                            ),
                            source_type="pytest",
                            target=None,
                            execution={},
                            nodes=[],
                        )
                    )
            elif payload.get("schema_version") == 2:
                try:
                    envelope = EvaluationEnvelope.model_validate(payload)
                except ValidationError as exc:
                    issues.append(f"eval.json has no valid evaluation envelope: {exc}")
                if provenance_id is not None and envelope is not None and provenance_id != envelope.evaluation_id:
                    issues.append(
                        "provenance.json evaluation_id does not match eval.json; "
                        "the root identity is ambiguous"
                    )
                if envelope is not None:
                    for case in payload.get("cases", []):
                        if case.get("source") == "aggregate":
                            continue
                        try:
                            result = _result_from_case(
                                case,
                                evaluation_id=envelope.evaluation_id,
                                target=envelope.target,
                                interpretation=None,
                            )
                            results.append(result)
                        except (ValidationFailure, ValidationError) as exc:
                            issues.append(f"invalid eval case {case.get('id')}: {exc}")
            else:
                issues.append("unsupported eval.json schema_version")
    elif not legacy_v1:
        issues.append(
            "evidence directory has no root eval.json envelope; result ownership cannot be established"
        )

    result_files = sorted((evidence_dir / "results").glob("*.json")) if (evidence_dir / "results").is_dir() else []
    for path in result_files:
        try:
            payload = read_json(path)
        except InputError as exc:
            issues.append(f"{path.name} is unreadable: {exc}")
            continue
        try:
            if payload.get("source_type") == "human_review":
                # Acceptance-level human evidence lives in human/<check>.json
                # with its own schema.  An embedded human_review under
                # results/ is a historical run-level record, not proof.
                results.append(
                    ExecutionResult(
                        evaluation_id=payload.get("evaluation_id", "eval-19700101-000000-000000"),
                        check_id=payload.get("check_id", "UNKNOWN"),
                        attempt_id=str(payload.get("attempt_id", "legacy")),
                        status=CheckStatus.NOT_RUN,
                        reason_code="legacy_human_record",
                        reason=(
                            "human evidence under results/ is historical; import a schema 3 "
                            "record with 'acceptance-review import'"
                        ),
                        source_type="human_review",
                        case_definition_hash=payload.get("case_definition_hash"),
                        target=None,
                    )
                )
                continue
            result = ExecutionResult.model_validate(payload)
            if envelope is None:
                result = result.model_copy(
                    update={
                        "status": CheckStatus.NOT_RUN,
                        "reason_code": "no_evaluation_envelope",
                        "reason": "no root eval.json envelope; this result cannot claim a current evaluation",
                        "target": None,
                    }
                )
            elif result.evaluation_id != envelope.evaluation_id:
                result = result.model_copy(
                    update={
                        "status": CheckStatus.ERROR,
                        "reason_code": "evaluation_mismatch",
                        "reason": (
                            f"record belongs to {result.evaluation_id}, but this directory is "
                            f"{envelope.evaluation_id}; import it explicitly instead of mixing files"
                        ),
                    }
                )
            results.append(result)
        except (InputError, ValidationError) as exc:
            issues.append(f"{path.name} is not a valid ExecutionResult: {exc}")

    human_files = (
        sorted(path for path in (evidence_dir / "human").glob("*.json") if CHECK_ID_RE.fullmatch(path.stem))
        if (evidence_dir / "human").is_dir()
        else []
    )
    for path in human_files:
        try:
            results.append(
                _human_result(
                    read_json(path),
                    evidence_dir,
                    envelope=envelope,
                    manifest=manifest,
                )
            )
        except (InputError, ValidationError, ValidationFailure) as exc:
            detail = getattr(exc, "detail", None)
            issues.append(f"{path.name} is not a valid human review: {exc}" + (f" ({detail})" if detail else ""))

    manifest_cases = getattr(manifest, "cases", None)
    verified: list[ExecutionResult] = []
    for result in results:
        if current is not None and manifest_cases is not None:
            spec = manifest_cases.get(result.check_id)
            if spec is not None and result.source_type != spec.source:
                result = result.model_copy(
                    update={
                        "status": CheckStatus.ERROR,
                        "reason_code": "source_mismatch",
                        "reason": (
                            f"source_type={result.source_type} but the current case definition "
                            f"requires {spec.source}"
                        ),
                    }
                )
        try:
            verified.append(_verify_artifacts(evidence_dir, result))
        except ValidationFailure as exc:
            issues.append(str(exc))
            verified.append(
                result.model_copy(
                    update={
                        "status": CheckStatus.ERROR,
                        "reason_code": "artifact_invalid",
                        "reason": str(exc),
                    }
                )
            )
    return verified, issues


def _human_result(
    payload: dict[str, Any],
    evidence_dir: Path,
    *,
    envelope: "EvaluationEnvelope | None",
    manifest: CaseManifest | None,
) -> ExecutionResult:
    """Acceptance-level human record (not a run-level claim review)."""
    schema_version = payload.get("schema_version", 1)
    if schema_version == 1:
        return ExecutionResult(
            evaluation_id=payload.get("evaluation_id", "eval-19700101-000000-000000"),
            check_id=payload.get("check_id", "UNKNOWN"),
            attempt_id=str(payload.get("record_id", "legacy")),
            status=CheckStatus.NOT_RUN,
            reason_code="legacy_human_record",
            reason="schema v1 human record has no typed per-case review; it cannot count as current",
            source_type="human_review",
            target=None,
        )
    if schema_version == 2:
        return ExecutionResult(
            evaluation_id=payload.get("evaluation_id", "eval-19700101-000000-000000"),
            check_id=payload.get("check_id", "UNKNOWN"),
            attempt_id=str(payload.get("record_id", "legacy")),
            status=CheckStatus.NOT_RUN,
            reason_code="legacy_human_record",
            reason=(
                "schema v2 human record has no case-to-bundle binding; re-import it with "
                "'acceptance-review import' before it can count"
            ),
            source_type="human_review",
            target=None,
        )
    record = HumanCheckRecord.model_validate(payload)
    from .acceptance_review import verify_human_record

    binding_problems = verify_human_record(
        record,
        evidence_dir,
        envelope=envelope,
        manifest=manifest,
    )
    if binding_problems:
        raise ValidationFailure(
            f"human review does not bind to verifiable evidence: {binding_problems[0]}",
            detail="; ".join(binding_problems[:8]),
        )
    review = record.human_review
    problems = human_review_problems(record.check_id, review)
    status = CheckStatus.PASS if not problems else CheckStatus.FAIL
    return ExecutionResult(
        evaluation_id=record.evaluation_id,
        check_id=record.check_id,
        attempt_id=record.record_id,
        status=status,
        reason_code=None if status == CheckStatus.PASS else "human_review_incomplete",
        reason="human review satisfies the check contract" if not problems else "; ".join(problems),
        source_type="human_review",
        case_definition_hash=record.case_definition_hash,
        target=record.target,
        execution={"executor_id": record.reviewer_id},
        artifacts=record.artifacts,
        nodes=[],
        human_review=review,
    )


class ImportOrigin(StrictModel):
    """Where a review crossed evaluation directories.

    The original record bytes are copied beside the imported record, so the
    source can be re-hashed later instead of being trusted as a label.
    """

    source_evaluation_id: str
    source_record_sha256: Sha256
    source_record_path: RelativePath
    imported_at: datetime

    @field_validator("source_evaluation_id")
    @classmethod
    def _source_id_shape(cls, value: str) -> str:
        if not EVALUATION_ID_RE.fullmatch(value):
            raise ValueError("source_evaluation_id must look like eval-YYYYMMDD-HHMMSS-abcdef")
        return value


class HumanCheckRecord(StrictModel):
    """Schema 3: the review plus the evidence it actually looked at."""

    schema_version: Literal[3] = HUMAN_SCHEMA_VERSION
    record_id: str = Field(min_length=1, max_length=120)
    evaluation_id: str
    check_id: str
    case_definition_hash: Sha256
    target: Target
    reviewer_id: NonEmptyStr
    review_method: Literal["human"] = "human"
    reviewed_at: datetime
    dataset_ref: Artifact | None = None
    dataset_digest: Sha256 | None = None
    dataset_id: str | None = None
    rubric_ref: Artifact | None = None
    rubric_digest: Sha256 | None = None
    artifacts: list[Artifact] = Field(default_factory=list, max_length=200)
    human_review: HumanReviewRecord
    imported_from: ImportOrigin | None = None

    @field_validator("check_id")
    @classmethod
    def _check_id_shape(cls, value: str) -> str:
        if not CHECK_ID_RE.fullmatch(value):
            raise ValueError("check_id contains unsupported characters")
        return value

    @field_validator("evaluation_id")
    @classmethod
    def _evaluation_id_shape(cls, value: str) -> str:
        if not EVALUATION_ID_RE.fullmatch(value):
            raise ValueError("evaluation_id must look like eval-YYYYMMDD-HHMMSS-abcdef")
        return value

    @model_validator(mode="after")
    def _review_identity(self) -> "HumanCheckRecord":
        review = self.human_review
        if review.reviewer != self.reviewer_id:
            raise ValueError("reviewer_id must equal human_review.reviewer")
        if review.method != self.review_method:
            raise ValueError("review_method must equal human_review.method")
        if review.reviewed_at != self.reviewed_at:
            raise ValueError("reviewed_at must equal human_review.reviewed_at")
        if self.check_id in HUMAN_SUMMARY_CHECKS:
            if self.dataset_ref is None or self.dataset_digest is None or self.dataset_id is None:
                raise ValueError("quality review requires dataset_ref, dataset_digest and dataset_id")
            if self.rubric_ref is None or self.rubric_digest is None:
                raise ValueError("quality review requires rubric_ref and rubric_digest")
        return self


class EvaluationEnvelope(StrictModel):
    """Root identity of an eval directory.

    Results are linked to the directory by this id, not by the directory name
    or by whichever file happens to be present.
    """

    model_config = ConfigDict(extra="ignore")

    schema_version: Literal[2] = 2
    evaluation_id: str
    target: Target
    # Kept opaque here: the envelope owns identity, not the executed case list.
    cases: list[Any] = Field(default_factory=list, max_length=1000)

    @field_validator("evaluation_id")
    @classmethod
    def _evaluation_id_shape(cls, value: str) -> str:
        if not EVALUATION_ID_RE.fullmatch(value):
            raise ValueError("evaluation_id must look like eval-YYYYMMDD-HHMMSS-abcdef")
        return value


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _latest(results: list[ExecutionResult]) -> ExecutionResult:
    def key(item: ExecutionResult) -> tuple[str, str]:
        finished = item.execution.finished_at
        return (finished.isoformat() if finished else "", item.attempt_id)

    return sorted(results, key=key)[-1]


def _judge_check(
    check_id: str,
    results: list[ExecutionResult],
    current: Target,
    *,
    manifest: CaseManifest | None = None,
    live: bool = False,
) -> CheckAcceptance:
    if not results:
        return CheckAcceptance(
            check_id=check_id,
            execution_status=CheckStatus.NOT_RUN,
            effective_state="unknown",
            result_ref=None,
            source_type=None,
            reasons=["no result was provided for this required check"],
        )
    spec = manifest.cases.get(check_id) if manifest is not None else None
    judged: list[tuple[ExecutionResult, str]] = []
    for result in results:
        state = effective_state(result, current, live=live)
        if state == "current":
            if spec is None:
                state = "unknown"
            elif result.case_definition_hash is None:
                state = "unknown"
            elif result.case_definition_hash != case_definition_hash(spec):
                state = "stale"
        judged.append((result, state))
    current_results = [(result, state) for result, state in judged if state == "current"]
    if any(result.status == CheckStatus.ERROR for result, _ in current_results):
        chosen = next(result for result, _ in current_results if result.status == CheckStatus.ERROR)
        return _entry(chosen, "current", [f"result execution failed: {chosen.reason or chosen.reason_code or 'unknown'}"])
    if any(result.status == CheckStatus.FAIL for result, _ in current_results):
        chosen = next(result for result, _ in current_results if result.status == CheckStatus.FAIL)
        return _entry(chosen, "current", [f"result failed for the current target: {chosen.reason or chosen.reason_code or 'unknown'}"])
    passes = [result for result, _ in current_results if result.status == CheckStatus.PASS]
    if passes:
        return _entry(_latest(passes), "current", [])
    if current_results:
        chosen = current_results[-1][0]
        return _entry(
            chosen,
            "current",
            [f"current result is {chosen.status.value}", chosen.reason or "not executed for the current target"],
        )
    # No current result: report the newest stale/unknown evidence and why it
    # cannot count.
    chosen, state = judged[-1]
    if state == "stale":
        reason = (
            "result was produced for a different code/config/corpus target "
            "or a different case definition"
        )
    else:
        reason = (
            "result has no provenance or case definition hash, so its target/definition "
            "cannot be verified"
        )
    return _entry(chosen, state, [reason])


def _entry(result: ExecutionResult, state: str, reasons: list[str]) -> CheckAcceptance:
    return CheckAcceptance(
        check_id=result.check_id,
        execution_status=result.status,
        effective_state=state,  # type: ignore[arg-type]
        result_ref=f"{result.evaluation_id}/{result.attempt_id}",
        source_type=result.source_type,
        reasons=reasons,
    )


def accept_profile(
    *,
    profile: str,
    evidence_dir: Path,
    project_root: Path,
    config: Config | None = None,
) -> AcceptanceResult:
    required = profile_required(profile)
    evidence_dir = evidence_dir.resolve()
    current = build_target(project_root, live_config=config, exclude=[evidence_dir])
    manifest = trusted_manifest(project_root)
    results, issues = load_execution_results(evidence_dir, current=current, manifest=manifest)
    if manifest is None:
        issues.append(
            "the current checkout has no trusted tests/cases manifest; no case definition can be verified"
        )
    by_check: dict[str, list[ExecutionResult]] = {}
    for result in results:
        by_check.setdefault(result.check_id, []).append(result)

    live_checks = {"AC15.live", "HC09"}
    entries: list[CheckAcceptance] = []
    for check_id in required:
        entry = _judge_check(
            check_id,
            by_check.get(check_id, []),
            current,
            manifest=manifest,
            live=check_id in live_checks,
        )
        if issues and entry.result_ref is None:
            entry.reasons.extend(issues[:5])
        entries.append(entry)

    counts = {
        "pass": sum(1 for entry in entries if entry.execution_status == CheckStatus.PASS and entry.effective_state == "current" and not entry.reasons),
        "fail": sum(1 for entry in entries if entry.execution_status == CheckStatus.FAIL and entry.effective_state == "current"),
        "error": sum(1 for entry in entries if entry.execution_status == CheckStatus.ERROR and entry.effective_state == "current"),
    }
    incomplete = len(entries) - counts["pass"] - counts["fail"] - counts["error"]
    if issues:
        overall, exit_code = "error", 4
    elif counts["error"]:
        overall, exit_code = "error", 4
    elif counts["fail"]:
        overall, exit_code = "fail", 1
    elif incomplete:
        overall, exit_code = "incomplete", 2
    else:
        overall, exit_code = "pass", 0

    spec = PROFILES[profile]
    result = AcceptanceResult(
        profile=profile,  # type: ignore[arg-type]
        evaluated_target=current,
        profile_definition_hash=profile_definition_hash(),
        overall=overall,  # type: ignore[arg-type]
        exit_code=exit_code,
        checks=entries,
        summary=AcceptanceSummary(
            required=len(entries),
            pass_count=counts["pass"],
            fail_count=counts["fail"],
            error_count=counts["error"],
            incomplete_count=incomplete,
        ),
        evidence_issues=issues,
        claim_scope=spec["claim_scope"],
        not_established=spec["not_established"],
    )
    return result


def write_acceptance(result: AcceptanceResult, evidence_dir: Path) -> list[Path]:
    payload = result.model_dump(mode="json")
    rendered = render_acceptance(result)
    written = [
        atomic_write_json(evidence_dir / f"acceptance-{result.profile}.json", payload),
        atomic_write_text(evidence_dir / f"acceptance-{result.profile}.md", rendered),
    ]
    # Keep a stable latest pointer for humans; the profile file is authoritative.
    written.append(atomic_write_json(evidence_dir / "acceptance.json", payload))
    written.append(atomic_write_text(evidence_dir / "acceptance.md", rendered))
    return written


def render_acceptance(result: AcceptanceResult) -> str:
    lines = [
        f"# 验收结果：{result.profile}",
        "",
        f"- 总体：`{result.overall}`（退出码 {result.exit_code}）",
        f"- 目标 commit：`{result.evaluated_target.code_commit or 'not-a-git-tree'}`",
        f"- 工作树 dirty：`{result.evaluated_target.dirty}`",
        f"- 目标摘要：`{result.evaluated_target.relevant_tree_digest[:16]}…`",
        f"- 必需检查：{result.summary.required}；通过 {result.summary.pass_count}；"
        f"失败 {result.summary.fail_count}；错误 {result.summary.error_count}；"
        f"不完整 {result.summary.incomplete_count}",
        "",
        "| 检查 | 执行状态 | 对当前目标 | 来源 | 原因 |",
        "|---|---|---|---|---|",
    ]
    for check in result.checks:
        reasons = "；".join(check.reasons[:2]) if check.reasons else "-"
        lines.append(
            f"| {check.check_id} | `{check.execution_status.value}` | `{check.effective_state}` | "
            f"{check.source_type or '-'} | {reasons} |"
        )
    if result.evidence_issues:
        lines.extend(["", "## 证据目录问题", ""])
        lines.extend([f"- {item}" for item in result.evidence_issues[:20]])
    lines.extend(
        [
            "",
            f"## 能证明什么\n\n{result.claim_scope}",
            "",
            "## 未证明什么",
            "",
            *[f"- {item}" for item in result.not_established],
            "",
        ]
    )
    return "\n".join(lines)