"""Strict data contracts shared by the CLI, runner, tools and renderers.

The models in this module deliberately reject unexpected fields.  An external
payload that does not match the declared shape must produce a readable error;
it must never be silently coerced into an empty list or a zero count.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EVIDENCE_ID_RE = re.compile(r"^ev-[0-9a-f]{12}$")
RUN_ID_RE = re.compile(r"^run-[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")
EVENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
CALL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
CHECK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,63}$")
CLAIM_ID_RE = re.compile(r"^claim-(fact|hyp)-[0-9]{3}$")
RULE_ID_RE = re.compile(r"^R(0[1-9]|00)$")


def _strict_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("must be an integer (bool and strings are rejected)")
    return value


def _strict_bool(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError("must be a boolean")
    return value


def _relative_path(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("path must be a non-empty string")
    if "\x00" in value:
        raise ValueError("path contains NUL")
    if "\\" in value:
        raise ValueError("use forward slashes in repository-relative paths")
    if value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        raise ValueError("path must be repository-relative, not absolute")
    parts = value.split("/")
    if ".." in parts:
        raise ValueError("path must not contain '..'")
    return value


def _hash(value: str) -> str:
    if not SHA256_RE.fullmatch(value):
        raise ValueError("expected a lowercase 64-character sha256 hex digest")
    return value


def _evidence_id(value: str) -> str:
    if not EVIDENCE_ID_RE.fullmatch(value):
        raise ValueError("expected an evidence id such as ev-0123456789ab")
    return value


StrictInt = Annotated[int, BeforeValidator(_strict_int)]
StrictBool = Annotated[bool, BeforeValidator(_strict_bool)]
NonEmptyStr = Annotated[str, StringConstraints(min_length=1, max_length=20000, strip_whitespace=True)]
ShortStr = Annotated[str, StringConstraints(min_length=1, max_length=500, strip_whitespace=True)]
RelativePath = Annotated[str, AfterValidator(_relative_path)]
Sha256 = Annotated[str, AfterValidator(_hash)]
EvidenceId = Annotated[str, AfterValidator(_evidence_id)]
LineNumber = Annotated[int, BeforeValidator(_strict_int), Field(ge=1, le=100_000_000)]


class StrictModel(BaseModel):
    """Base model: reject unknown fields, validate on assignment."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# ---------------------------------------------------------------------------
# Event input
# ---------------------------------------------------------------------------


class Attempt(StrictModel):
    action: ShortStr
    outcome: ShortStr | None = None
    command: ShortStr | None = None


class Environment(StrictModel):
    language: str | None = None
    runtime_version: str | None = None


class Source(StrictModel):
    kind: ShortStr
    session_id: str | None = None


class Event(StrictModel):
    schema_version: Literal[1]
    event_id: str
    occurred_at: datetime
    project_id: ShortStr
    rule_version: ShortStr
    goal: NonEmptyStr
    symptom: NonEmptyStr
    error_files: list[RelativePath] = Field(default_factory=list, max_length=20)
    related_files: list[RelativePath] = Field(default_factory=list, max_length=50)
    attempts: list[Attempt] = Field(default_factory=list, max_length=30)
    environment: Environment = Field(default_factory=Environment)
    source: Source
    note: str | None = None

    @field_validator("event_id")
    @classmethod
    def _event_id_shape(cls, value: str) -> str:
        if not EVENT_ID_RE.fullmatch(value):
            raise ValueError("event_id must match ^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
        return value

    @model_validator(mode="after")
    def _file_lists_are_consistent(self) -> "Event":
        # Duplicated entries between the two lists are tolerated: they are only
        # reading clues and the tools deduplicate evidence by source identity.
        return self


# ---------------------------------------------------------------------------
# Tool results
# ---------------------------------------------------------------------------


class ToolStatus(StrEnum):
    OK = "ok"
    ERROR = "error"


class ToolSpec(StrictModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class ToolResult(StrictModel):
    status: ToolStatus
    data: Any = None
    evidence_refs: list[EvidenceId] = Field(default_factory=list)
    truncated: StrictBool = False
    continuation: dict[str, Any] | None = None
    error_code: str | None = None
    retryable: StrictBool = False
    duration_ms: StrictInt = Field(ge=0, le=600_000)

    @model_validator(mode="after")
    def _status_shape(self) -> "ToolResult":
        if self.status == ToolStatus.ERROR:
            if not self.error_code:
                raise ValueError("error results must carry error_code")
            if self.evidence_refs:
                raise ValueError("error results cannot cite evidence")
        elif self.error_code is not None:
            raise ValueError("ok results must not carry error_code")
        if self.truncated and self.continuation is None:
            raise ValueError("truncated results must include continuation instructions")
        return self


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


class EvidenceKind(StrEnum):
    LOCAL_FILE = "local_file"
    GIT = "git"
    LESSON = "lesson"
    ISSUE = "issue"


class Evidence(StrictModel):
    schema_version: Literal[1, 2] = 1
    evidence_id: EvidenceId
    kind: EvidenceKind
    project_id: ShortStr
    path: RelativePath | None = None
    line_start: LineNumber | None = None
    line_end: LineNumber | None = None
    url: str | None = None
    sha256: Sha256
    snippet: str
    captured_at: datetime
    truncated: StrictBool = False
    total_lines: StrictInt | None = Field(default=None, ge=1)
    commit: str | None = None
    stale: StrictBool = False
    stale_reason: str | None = None
    snapshot_sha256: Sha256 | None = None
    snapshot_bytes: StrictInt | None = Field(default=None, ge=0, le=8 * 1024 * 1024)
    capture_scope: Literal["snippet", "file", "search_match", "github_response", "git_output", "lesson"] = "snippet"
    complete_within_scope: StrictBool = False
    capture_notes: list[ShortStr] = Field(default_factory=list, max_length=10)
    snapshot_start_byte: StrictInt | None = Field(default=None, ge=0)
    snapshot_end_byte: StrictInt | None = Field(default=None, ge=0)
    snapshot_line_start: LineNumber | None = None
    parent_evidence_id: EvidenceId | None = None

    @model_validator(mode="after")
    def _source_shape(self) -> "Evidence":
        if self.kind in (EvidenceKind.LOCAL_FILE, EvidenceKind.GIT) and not self.path:
            raise ValueError(f"{self.kind} evidence requires a path")
        if self.kind == EvidenceKind.ISSUE and not self.url:
            raise ValueError("issue evidence requires a url")
        if self.line_start is not None:
            if self.line_end is None or self.line_end < self.line_start:
                raise ValueError("line_end must be present and >= line_start")
            if self.total_lines is not None and self.line_end > self.total_lines:
                raise ValueError("line_end exceeds total_lines")
        if self.kind == EvidenceKind.LOCAL_FILE and self.line_start is None:
            raise ValueError("local file evidence requires a line range")
        if self.stale and not self.stale_reason:
            raise ValueError("stale evidence must state a reason")
        if self.snapshot_sha256 is not None:
            if self.schema_version < 2 or self.snapshot_bytes is None:
                raise ValueError("a snapshot requires evidence v2 and its byte length")
            if self.snapshot_start_byte is None or self.snapshot_end_byte is None:
                raise ValueError("a snapshot requires the snippet byte range")
            if not 0 <= self.snapshot_start_byte <= self.snapshot_end_byte <= self.snapshot_bytes:
                raise ValueError("snippet range is outside the snapshot")
            if self.snapshot_end_byte - self.snapshot_start_byte != len(self.snippet.encode("utf-8")):
                raise ValueError("snapshot range must equal the snippet byte length")
        return self


class EvidenceSummary(StrictModel):
    evidence_id: EvidenceId
    kind: EvidenceKind
    path: RelativePath | None = None
    line_start: LineNumber | None = None
    line_end: LineNumber | None = None
    url: str | None = None
    sha256: Sha256
    captured_at: datetime
    truncated: StrictBool = False
    stale: StrictBool = False


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


class CheckStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    NOT_RUN = "not_run"
    NOT_APPLICABLE = "not_applicable"


class Attribution(StrEnum):
    PUBLIC_WORKFLOW = "public_workflow"
    BUSINESS_DEFECT = "business_defect"
    UNDETERMINED = "undetermined"
    UNKNOWN = "unknown"


class AnalysisStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


class StopReason(StrEnum):
    COMPLETED = "completed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TOOL_UNAVAILABLE = "tool_unavailable"
    INVALID_OUTPUT = "invalid_output"
    INTERRUPTED = "interrupted"


class SupportReview(StrictModel):
    """Semantic support review, deliberately separate from structure checks.

    The program never sets this to ``pass`` on its own.  A missing review is
    ``not_run`` and must be rendered as such, even when schema and reference
    checks pass.
    """

    status: CheckStatus = CheckStatus.NOT_RUN
    reviewer: ShortStr | None = None
    method: ShortStr | None = None
    scope: ShortStr | None = None
    reviewed_at: datetime | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _review_shape(self) -> "SupportReview":
        if self.status == CheckStatus.NOT_RUN:
            if any(
                value is not None
                for value in (self.reviewer, self.method, self.scope, self.reviewed_at)
            ):
                raise ValueError("not_run support review must not look reviewed")
        else:
            missing = [
                name
                for name, value in (
                    ("reviewer", self.reviewer),
                    ("method", self.method),
                    ("scope", self.scope),
                    ("reviewed_at", self.reviewed_at),
                )
                if value is None
            ]
            if missing:
                raise ValueError(f"support review requires {', '.join(missing)}")
        return self


def _claim_id(value: str) -> str:
    if not CLAIM_ID_RE.fullmatch(value):
        raise ValueError("expected a claim id such as claim-fact-001 or claim-hyp-001")
    return value


ClaimId = Annotated[str, AfterValidator(_claim_id)]


class Fact(StrictModel):
    claim_id: ClaimId | None = None
    text: NonEmptyStr
    evidence_ids: list[EvidenceId] = Field(min_length=1, max_length=20)
    support_review: SupportReview = Field(default_factory=SupportReview)


class Hypothesis(StrictModel):
    claim_id: ClaimId | None = None
    text: NonEmptyStr
    supporting_evidence_ids: list[EvidenceId] = Field(default_factory=list, max_length=20)
    contradicting_evidence_ids: list[EvidenceId] = Field(default_factory=list, max_length=20)
    open_questions: list[ShortStr] = Field(default_factory=list, max_length=10)
    support_review: SupportReview = Field(default_factory=SupportReview)


class SuggestedCheck(StrictModel):
    check_id: str
    description: NonEmptyStr
    target_paths: list[RelativePath] = Field(default_factory=list, max_length=10)
    status: Literal["not_run"] = "not_run"

    @field_validator("check_id")
    @classmethod
    def _check_id_shape(cls, value: str) -> str:
        if not CHECK_ID_RE.fullmatch(value):
            raise ValueError("check_id must match ^[A-Za-z0-9][A-Za-z0-9._-]{1,63}$")
        return value


class DuplicateCandidate(StrictModel):
    source: Literal["local_event", "local_lesson", "github_issue"]
    reference: ShortStr
    title: ShortStr | None = None
    similarity_reason: ShortStr
    state: Literal["open", "closed", "unknown"] = "unknown"


class Analysis(StrictModel):
    schema_version: Literal[1, 2] = 2
    summary: NonEmptyStr
    attribution: Attribution
    facts: list[Fact] = Field(default_factory=list, max_length=40)
    hypotheses: list[Hypothesis] = Field(default_factory=list, max_length=3)
    missing_evidence: list[ShortStr] = Field(default_factory=list, max_length=20)
    suggested_checks: list[SuggestedCheck] = Field(default_factory=list, max_length=20)
    duplicate_candidates: list[DuplicateCandidate] = Field(default_factory=list, max_length=20)
    analysis_status: AnalysisStatus
    stop_reason: StopReason
    # Explicit v1 -> v2 adapter breadcrumb.  Old payloads keep loading, but the
    # missing claim/display/review fields become not_run instead of assumed pass.
    contract_upgraded_from: Literal[1] | None = None

    @model_validator(mode="before")
    @classmethod
    def _upgrade_v1_payload(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        version = value.get("schema_version", 1)
        if version not in (1, 2):
            return value
        payload = dict(value)
        if version == 1:
            payload["schema_version"] = 2
            payload["contract_upgraded_from"] = 1
        facts = []
        for index, fact in enumerate(payload.get("facts") or [], start=1):
            if isinstance(fact, dict) and not fact.get("claim_id"):
                fact = {**fact, "claim_id": f"claim-fact-{index:03d}"}
            facts.append(fact)
        payload["facts"] = facts
        hypotheses = []
        for index, hypothesis in enumerate(payload.get("hypotheses") or [], start=1):
            if isinstance(hypothesis, dict) and not hypothesis.get("claim_id"):
                hypothesis = {**hypothesis, "claim_id": f"claim-hyp-{index:03d}"}
            hypotheses.append(hypothesis)
        payload["hypotheses"] = hypotheses
        return payload

    @model_validator(mode="after")
    def _status_and_stop_reason(self) -> "Analysis":
        if self.analysis_status == AnalysisStatus.COMPLETE and self.stop_reason != StopReason.COMPLETED:
            raise ValueError("complete analysis requires stop_reason=completed")
        if self.analysis_status != AnalysisStatus.COMPLETE and self.stop_reason == StopReason.COMPLETED:
            raise ValueError("stop_reason=completed requires analysis_status=complete")
        if self.analysis_status != AnalysisStatus.COMPLETE and not self.missing_evidence:
            raise ValueError("partial/failed analysis must list at least one missing_evidence item")
        for check in self.suggested_checks:
            if check.status != "not_run":
                raise ValueError("suggested checks must remain not_run; run them via record-outcome")
        claim_ids = [claim.claim_id for claim in [*self.facts, *self.hypotheses] if claim.claim_id]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claim ids must be unique within one analysis")
        return self


# ---------------------------------------------------------------------------
# Checks and verification
# ---------------------------------------------------------------------------


class CheckResult(StrictModel):
    schema_version: Literal[1] = 1
    check_id: str
    status: CheckStatus
    description: NonEmptyStr
    command: str | None = None
    exit_code: StrictInt | None = None
    output_ref: str | None = None
    target_sha256: Sha256 | None = None
    scope: NonEmptyStr
    executed_by: str | None = None
    executed_at: datetime | None = None

    @field_validator("check_id")
    @classmethod
    def _check_id_shape(cls, value: str) -> str:
        if not CHECK_ID_RE.fullmatch(value):
            raise ValueError("check_id must match ^[A-Za-z0-9][A-Za-z0-9._-]{1,63}$")
        return value

    @model_validator(mode="after")
    def _status_shape(self) -> "CheckResult":
        if self.status == CheckStatus.NOT_RUN:
            if self.exit_code is not None:
                raise ValueError("not_run must not carry an exit_code")
            if self.executed_at is not None or self.executed_by is not None:
                raise ValueError("not_run must not look executed")
        if self.status in (CheckStatus.PASS, CheckStatus.FAIL):
            missing = [
                name
                for name, value in (
                    ("command", self.command),
                    ("exit_code", self.exit_code),
                    ("output_ref", self.output_ref),
                    ("target_sha256", self.target_sha256),
                    ("executed_at", self.executed_at),
                    ("executed_by", self.executed_by),
                )
                if value is None
            ]
            if missing:
                raise ValueError(f"{self.status} requires {', '.join(missing)}")
            if self.status == CheckStatus.PASS and self.exit_code != 0:
                raise ValueError("pass requires exit_code 0")
            if self.status == CheckStatus.FAIL and self.exit_code == 0:
                raise ValueError("fail requires a non-zero exit_code")
        if self.status == CheckStatus.ERROR:
            if self.exit_code is None and not self.output_ref:
                raise ValueError("error requires an exit_code or output_ref")
        if self.status == CheckStatus.NOT_APPLICABLE:
            if self.exit_code is not None:
                raise ValueError("not_applicable must not carry an exit_code")
            if len(self.scope.strip()) < 8:
                raise ValueError("not_applicable requires a scope justification")
        return self


class VerificationRecord(StrictModel):
    command: NonEmptyStr
    exit_code: StrictInt
    target_sha256: Sha256
    evidence_refs: list[EvidenceId] = Field(default_factory=list, max_length=20)
    executed_by: NonEmptyStr
    executed_at: datetime
    owner_confirmed: StrictBool
    note: str | None = None
    # v2 credentials: which planned checks were satisfied, every verified
    # target and the imported output snapshots.  Empty on legacy records.
    check_ids: list[str] = Field(default_factory=list, max_length=20)
    hypothesis_id: str | None = None
    target_hashes: dict[str, Sha256] = Field(default_factory=dict)
    output_refs: list[str] = Field(default_factory=list, max_length=20)
    contract: Literal["v1", "v2"] | None = None

    @model_validator(mode="after")
    def _must_be_successful_and_confirmed(self) -> "VerificationRecord":
        if self.exit_code != 0:
            raise ValueError("verification requires exit_code 0")
        if not self.owner_confirmed:
            raise ValueError("verification requires explicit owner confirmation")
        return self


class LessonStatus(StrEnum):
    CANDIDATE = "candidate"
    VERIFIED = "verified"
    DEPRECATED = "deprecated"


class Lesson(StrictModel):
    schema_version: Literal[1, 2] = 2
    lesson_id: str
    status: LessonStatus
    symptom_pattern: NonEmptyStr
    applicability: NonEmptyStr
    guidance: NonEmptyStr
    event_ids: list[str] = Field(default_factory=list, max_length=20)
    issue_urls: list[str] = Field(default_factory=list, max_length=20)
    verification: VerificationRecord | None = None
    applicable_versions: list[ShortStr] = Field(default_factory=list, max_length=20)
    not_applicable: list[ShortStr] = Field(default_factory=list, max_length=20)
    source_run_id: str | None = None
    source_check_ids: list[str] = Field(default_factory=list, max_length=20)
    promotion_reason: str | None = None

    @field_validator("lesson_id")
    @classmethod
    def _lesson_id_shape(cls, value: str) -> str:
        if not re.fullmatch(r"lesson-[0-9a-f]{12}", value):
            raise ValueError("lesson_id must look like lesson-0123456789ab")
        return value

    @model_validator(mode="after")
    def _verified_needs_verification(self) -> "Lesson":
        if self.status == LessonStatus.VERIFIED and self.verification is None:
            raise ValueError("verified lessons require a verification record")
        if self.status != LessonStatus.VERIFIED and self.verification is not None:
            raise ValueError("only verified lessons may carry a verification record")
        if self.status == LessonStatus.VERIFIED and self.schema_version == 2:
            if not self.verification.check_ids:  # type: ignore[union-attr]
                raise ValueError("v2 verified lessons require at least one verified check_id")
            if not self.applicable_versions:
                raise ValueError("v2 verified lessons require explicit applicable_versions")
        return self


class TargetHash(StrictModel):
    path: RelativePath
    sha256: Sha256


class OutcomeCommand(StrictModel):
    command: NonEmptyStr
    exit_code: StrictInt
    target_path: RelativePath
    target_sha256: Sha256
    output_ref: str | None = None
    output_sha256: Sha256 | None = None
    # v2 binding: the planned check this command is meant to satisfy and the
    # hypothesis it is meant to discriminate (usually a claim-hyp id).
    check_id: str | None = None
    hypothesis_id: str | None = None
    additional_targets: list[TargetHash] = Field(default_factory=list, max_length=20)

    @field_validator("check_id")
    @classmethod
    def _check_id_shape(cls, value: str | None) -> str | None:
        if value is not None and not CHECK_ID_RE.fullmatch(value):
            raise ValueError("check_id must match ^[A-Za-z0-9][A-Za-z0-9._-]{1,63}$")
        return value


class Outcome(StrictModel):
    schema_version: Literal[1, 2] = 2
    run_id: str
    commands: list[OutcomeCommand] = Field(min_length=1, max_length=10)
    executed_by: NonEmptyStr
    executed_at: datetime
    owner_confirmed: StrictBool
    note: str | None = None

    @field_validator("run_id")
    @classmethod
    def _run_id_shape(cls, value: str) -> str:
        if not RUN_ID_RE.fullmatch(value):
            raise ValueError("run_id must look like run-20261024-200000-abcdef")
        return value

    @model_validator(mode="after")
    def _v2_requires_binding(self) -> "Outcome":
        if self.schema_version != 2:
            return self
        for command in self.commands:
            if not command.check_id:
                raise ValueError("schema_version=2 outcome commands require check_id")
            if not command.output_ref:
                raise ValueError("schema_version=2 outcome commands require output_ref")
        return self


# ---------------------------------------------------------------------------
# Context packet
# ---------------------------------------------------------------------------


class SelectionDecision(StrEnum):
    INCLUDED = "included"
    TRUNCATED = "truncated"
    OMITTED = "omitted"


class CountMode(StrEnum):
    ESTIMATED = "estimated"
    MEASURED = "measured"


class SelectionRecord(StrictModel):
    layer: Literal[
        "constraints",
        "tool_definitions",
        "task_facts",
        "current_evidence",
        "history",
        "working_state",
        "protocol",
    ]
    source: ShortStr
    decision: SelectionDecision
    reason: ShortStr
    tokens: StrictInt = Field(ge=0, le=1_000_000)
    count_mode: CountMode = CountMode.ESTIMATED


class LayerBudget(StrictModel):
    layer: str
    limit_tokens: StrictInt = Field(ge=0)
    used_tokens: StrictInt = Field(ge=0)
    count_mode: CountMode = CountMode.ESTIMATED


class ContextBudget(StrictModel):
    total_limit_tokens: StrictInt = Field(ge=1)
    total_used_tokens: StrictInt = Field(ge=0)
    count_mode: CountMode = CountMode.ESTIMATED
    layers: list[LayerBudget]


class FailedAttempt(StrictModel):
    schema_version: Literal[1, 2] = 1
    action: ShortStr
    status: str
    reason: str | None = None
    call_id: str | None = None
    arguments_sha256: Sha256 | None = None
    source_ref: str | None = None
    retry_condition: str = "unknown"


class EvidenceIndexEntry(EvidenceSummary):
    pass


class LessonSummary(StrictModel):
    lesson_id: str
    status: LessonStatus
    symptom_pattern: ShortStr
    applicability: ShortStr
    guidance: ShortStr
    verified: StrictBool = False
    applicable_versions: list[ShortStr] = Field(default_factory=list)
    # current: verified for the event's rule version; out_of_scope: verified but
    # for other versions; legacy: v1 verification without version scope;
    # candidate: never verified.
    scope_status: Literal["current", "out_of_scope", "legacy", "candidate"] = "legacy"
    verification_contract: str | None = None


class DisplaySource(StrEnum):
    CONTEXT = "context"
    TOOL_RESULT = "tool_result"


class EvidenceWindow(StrictModel):
    """An exact UTF-8 slice; rendering separators are never part of the text."""

    schema_version: Literal[1] = 1
    window_id: str = Field(pattern=r"^win-[0-9a-f]{20}$")
    evidence_id: EvidenceId
    source_sha256: Sha256
    snapshot_sha256: Sha256 | None = None
    span_origin: Literal["evidence_snippet", "captured_snapshot"] = "evidence_snippet"
    start_byte: StrictInt = Field(ge=0)
    end_byte: StrictInt = Field(gt=0)
    line_start: LineNumber | None = None
    line_end: LineNumber | None = None
    text: str = Field(min_length=1, max_length=8192)
    text_sha256: Sha256
    selection_reason: ShortStr
    read_cursor: str | None = None

    @model_validator(mode="after")
    def _range_shape(self) -> "EvidenceWindow":
        import hashlib

        raw = self.text.encode("utf-8")
        if len(raw) > 8192 or self.end_byte - self.start_byte != len(raw):
            raise ValueError("window range must equal its bounded UTF-8 text length")
        if hashlib.sha256(raw).hexdigest() != self.text_sha256:
            raise ValueError("window text hash mismatch")
        return self


class DisplayedEvidence(StrictModel):
    """One bounded view of an evidence snippet that the model actually saw.

    Schema v2 binds the view to the provider request that carried it and to the
    evidence source hash.  Schema v1 records remain readable for old runs, but
    they are not proof that the body entered a model request.
    """

    schema_version: Literal[1, 2, 3] = 1
    evidence_id: EvidenceId
    turn: StrictInt = Field(ge=0)
    source: DisplaySource
    mode: Literal["full", "truncated"]
    visible_chars: StrictInt = Field(ge=0, le=1_000_000)
    visible_sha256: Sha256
    line_start: LineNumber | None = None
    line_end: LineNumber | None = None
    request_id: str | None = None
    source_sha256: Sha256 | None = None
    window: EvidenceWindow | None = None

    @model_validator(mode="after")
    def _v2_requires_request_binding(self) -> "DisplayedEvidence":
        if self.schema_version >= 2:
            if not self.request_id or not self.source_sha256:
                raise ValueError("display v2 requires request_id and source_sha256")
        if self.schema_version == 3:
            if self.window is None or self.window.evidence_id != self.evidence_id:
                raise ValueError("display v3 requires a matching exact window")
            if self.window.source_sha256 != self.source_sha256 or self.window.text_sha256 != self.visible_sha256:
                raise ValueError("display v3 window hashes must match the display record")
            if len(self.window.text) != self.visible_chars:
                raise ValueError("display v3 width must match its exact window")
        return self


class OmissionImpact(StrictModel):
    """A necessary item was dropped from the model context.

    The impact must travel to the affected conclusions and the handoff, not just
    stay in an internal trace field.
    """

    evidence_id: EvidenceId | None = None
    request: ShortStr
    reason: ShortStr
    state: Literal["not_captured", "stored_not_displayed", "partially_displayed", "stale"] = "stored_not_displayed"
    affected_claim_ids: list[str] = Field(default_factory=list, max_length=40)
    affected: list[Literal["report", "handoff", "analysis"]] = Field(default_factory=list, max_length=3)


class InvestigationPhase(StrEnum):
    LOCATE = "locate"
    COMPARE = "compare"
    HANDOFF = "handoff"


class AttemptLedgerEntry(StrictModel):
    schema_version: Literal[1, 2] = 1
    turn: StrictInt = Field(ge=0)
    action: ShortStr
    status: str
    new_information: StrictBool
    detail: str | None = None
    call_id: str | None = None
    arguments_sha256: Sha256 | None = None
    source_ref: str | None = None


class InvestigationState(StrictModel):
    schema_version: Literal[1, 2] = 1
    phase: InvestigationPhase = InvestigationPhase.LOCATE
    active_question: NonEmptyStr
    playbook_id: str | None = None
    playbook_version: str | None = None
    playbook_hash: str | None = None
    required_evidence_ids: list[EvidenceId] = Field(default_factory=list, max_length=50)
    counterevidence_ids: list[EvidenceId] = Field(default_factory=list, max_length=50)
    needed_materials: list[ShortStr] = Field(default_factory=list, max_length=20)
    attempt_ledger: list[AttemptLedgerEntry] = Field(default_factory=list, max_length=100)
    omissions: list[OmissionImpact] = Field(default_factory=list, max_length=50)
    next_action_reason: ShortStr


class RunPosition(StrictModel):
    run_id: str
    next_model_turn: StrictInt = Field(ge=0)
    completed_call_ids: list[str] = Field(default_factory=list, max_length=100)
    checkpoint_at: datetime


class ContextPacket(StrictModel):
    schema_version: Literal[1, 2] = 2
    run_id: str
    event_id: str
    project_id: ShortStr
    goal: NonEmptyStr
    facts: list[Fact] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    failed_attempts: list[FailedAttempt] = Field(default_factory=list)
    not_executed: list[ShortStr] = Field(default_factory=list)
    resume_from: RunPosition
    evidence_index: list[EvidenceIndexEntry] = Field(default_factory=list)
    lessons: list[LessonSummary] = Field(default_factory=list)
    selections: list[SelectionRecord] = Field(default_factory=list)
    budget: ContextBudget
    versions: dict[str, str] = Field(default_factory=dict)
    investigation: InvestigationState | None = None
    omissions: list[OmissionImpact] = Field(default_factory=list, max_length=50)
    playbook_id: str | None = None
    playbook_version: str | None = None
    playbook_hash: str | None = None
    displayed_count: StrictInt = Field(default=0, ge=0)
    evidence_windows: list[EvidenceWindow] = Field(default_factory=list, max_length=400)
    state_projection: dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime


# ---------------------------------------------------------------------------
# Trace records
# ---------------------------------------------------------------------------


class TraceKind(StrEnum):
    MODEL = "model"
    TOOL = "tool"


class TraceRecord(StrictModel):
    schema_version: Literal[1] = 1
    kind: TraceKind
    run_id: str
    turn: StrictInt = Field(ge=0)
    call_id: str | None = None
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    status: str
    error_code: str | None = None
    duration_ms: StrictInt = Field(ge=0, le=600_000)
    evidence_refs: list[EvidenceId] = Field(default_factory=list)
    attempt: StrictInt = Field(default=1, ge=1, le=3)
    started_at: datetime
    finished_at: datetime
    note: str | None = None

    @field_validator("call_id")
    @classmethod
    def _call_id_shape(cls, value: str | None) -> str | None:
        if value is not None and not CALL_ID_RE.fullmatch(value):
            raise ValueError("call_id contains unsupported characters")
        return value


# ---------------------------------------------------------------------------
# Budgets and configuration
# ---------------------------------------------------------------------------


class BudgetLimits(StrictModel):
    max_model_requests: StrictInt = Field(default=8, ge=1, le=50)
    max_tool_attempts: StrictInt = Field(default=12, ge=1, le=100)
    total_seconds: StrictInt = Field(default=180, ge=1, le=3600)
    per_tool_seconds: StrictInt = Field(default=15, ge=1, le=300)
    per_model_seconds: StrictInt = Field(default=45, ge=1, le=600)
    max_corrections: StrictInt = Field(default=1, ge=0, le=3)


class ContextLimits(StrictModel):
    constraints_tokens: StrictInt = Field(default=1000, ge=0, le=50_000)
    tool_definitions_tokens: StrictInt = Field(default=1200, ge=0, le=50_000)
    task_facts_tokens: StrictInt = Field(default=2000, ge=0, le=50_000)
    current_evidence_tokens: StrictInt = Field(default=4800, ge=0, le=100_000)
    history_tokens: StrictInt = Field(default=1500, ge=0, le=50_000)
    working_state_tokens: StrictInt = Field(default=1500, ge=0, le=50_000)
    total_input_tokens: StrictInt = Field(default=12_000, ge=1000, le=500_000)
    max_output_tokens: StrictInt = Field(default=3000, ge=256, le=64_000)
    reserve_tokens: StrictInt = Field(default=1000, ge=0, le=64_000)

    @model_validator(mode="after")
    def _layers_fit_total(self) -> "ContextLimits":
        layer_sum = (
            self.constraints_tokens
            + self.tool_definitions_tokens
            + self.task_facts_tokens
            + self.current_evidence_tokens
            + self.history_tokens
            + self.working_state_tokens
        )
        if layer_sum > self.total_input_tokens:
            raise ValueError("layer budgets exceed total_input_tokens")
        return self


class MCPConfig(StrictModel):
    enabled: StrictBool = False
    command: str | None = None
    args: list[str] = Field(default_factory=list, max_length=20)
    repo: str | None = None
    token_env: str = "GITHUB_PERSONAL_ACCESS_TOKEN"
    startup_timeout_seconds: StrictInt = Field(default=20, ge=1, le=120)
    call_timeout_seconds: StrictInt = Field(default=15, ge=1, le=120)

    @model_validator(mode="after")
    def _enabled_requires_settings(self) -> "MCPConfig":
        if self.enabled:
            if not self.command:
                raise ValueError("enabled MCP requires a command")
            if not self.repo or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.repo):
                raise ValueError("enabled MCP requires repo as owner/name")
        return self


class Config(StrictModel):
    schema_version: Literal[1] = 1
    provider: Literal["replay", "anthropic", "deepseek"] = "replay"
    model: str = "replay"
    credential_env: str = "ANTHROPIC_API_KEY"
    base_url: NonEmptyStr | None = None
    replay_script: str | None = None
    budgets: BudgetLimits = Field(default_factory=BudgetLimits)
    context: ContextLimits = Field(default_factory=ContextLimits)
    mcp: MCPConfig = Field(default_factory=MCPConfig)

    @model_validator(mode="before")
    @classmethod
    def _provider_defaults(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("provider") == "deepseek":
            value = dict(value)
            value.setdefault("model", "deepseek-flash")
            value.setdefault("credential_env", "DEEPSEEK_API_KEY")
            value.setdefault("base_url", "https://api.deepseek.com/anthropic")
        return value

    @field_validator("base_url")
    @classmethod
    def _nonsecret_endpoint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("base_url must be an HTTP(S) endpoint without credentials, query or fragment")
        return value.rstrip("/")
