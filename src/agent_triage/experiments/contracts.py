"""Typed contracts for local quality experiments.

An experiment is not an acceptance result.  It records what was executed on a
frozen dataset, what the deterministic evaluators observed, and which values
are missing.  A failed trial keeps its slot; it is never deleted or overwritten
by a later attempt.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from ..acceptance import Artifact, Target
from ..contracts import CheckStatus, NonEmptyStr, Sha256, StrictInt, StrictModel

EXPERIMENT_ID_RE = re.compile(r"^exp-[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")
TRIAL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,80}$")


class ExperimentLimits(StrictModel):
    max_trials: StrictInt = Field(default=100, ge=1, le=1000)
    max_model_requests: StrictInt = Field(default=1000, ge=1, le=100_000)
    max_duration_seconds: StrictInt = Field(default=3600, ge=1, le=86_400)


class ExperimentSpec(StrictModel):
    schema_version: Literal[1] = 1
    experiment_id: str
    dataset_id: NonEmptyStr
    dataset_digest: Sha256
    variant_id: NonEmptyStr
    target: Target
    provider_mode: Literal["replay", "live"]
    repetitions: StrictInt = Field(ge=1, le=10)
    max_concurrency: Literal[1] = 1
    limits: ExperimentLimits = Field(default_factory=ExperimentLimits)
    config_hash: Sha256
    python_version: NonEmptyStr
    created_at: datetime

    @field_validator("experiment_id")
    @classmethod
    def _experiment_id_shape(cls, value: str) -> str:
        if not EXPERIMENT_ID_RE.fullmatch(value):
            raise ValueError("experiment_id must look like exp-YYYYMMDD-HHMMSS-abcdef")
        return value


class TrialSlot(StrictModel):
    trial_id: str
    case_id: str
    repeat_index: StrictInt = Field(ge=0, le=9)
    attempt_id: str = "attempt-1"
    status: CheckStatus = CheckStatus.NOT_RUN
    reason: str | None = None

    @field_validator("trial_id")
    @classmethod
    def _trial_id_shape(cls, value: str) -> str:
        if not TRIAL_ID_RE.fullmatch(value):
            raise ValueError("trial_id contains unsupported characters")
        return value


class ExperimentPlan(StrictModel):
    schema_version: Literal[1] = 1
    experiment_id: str
    slots: list[TrialSlot] = Field(min_length=1, max_length=1000)
    created_at: datetime

    @model_validator(mode="after")
    def _unique_slots(self) -> "ExperimentPlan":
        ids = [slot.trial_id for slot in self.slots]
        if len(ids) != len(set(ids)):
            raise ValueError("plan repeats a trial_id")
        return self


class TrialRecord(StrictModel):
    schema_version: Literal[1] = 1
    trial_id: str
    experiment_id: str
    case_id: str
    repeat_index: StrictInt
    attempt_id: str
    run_id: str | None = None
    execution_status: CheckStatus
    reason_code: str | None = None
    reason: str | None = None
    result_available: bool = False
    duration_ms: StrictInt | None = Field(default=None, ge=0)
    artifacts: list[Artifact] = Field(default_factory=list, max_length=50)
    started_at: datetime | None = None
    ended_at: datetime | None = None


class MetricResult(StrictModel):
    schema_version: Literal[1] = 1
    metric_id: str
    version: str = "1"
    scope: Literal["claim", "request", "trial", "experiment"]
    subject_id: str
    producer: Literal["code", "human", "llm_judge"]
    execution_status: CheckStatus
    value: Any | None = None
    scale: str | None = None
    direction: Literal["higher", "lower", "neutral"] = "neutral"
    rationale: str | None = None
    evidence_refs: list[str] = Field(default_factory=list, max_length=50)
    input_digest: Sha256 | None = None
    rubric_digest: Sha256 | None = None
    created_at: datetime

    @model_validator(mode="after")
    def _value_requires_success(self) -> "MetricResult":
        if self.execution_status == CheckStatus.PASS and self.value is None:
            # Some registered metrics are counts; a zero count is a value, a
            # missing count is not.  The scorer must say which it means.
            raise ValueError("a passing metric must carry a value")
        if self.execution_status in {CheckStatus.ERROR, CheckStatus.NOT_RUN} and self.value is not None:
            raise ValueError("error/not_run metrics must not invent a value")
        return self


class TrialScore(StrictModel):
    schema_version: Literal[1] = 1
    trial_id: str
    experiment_id: str
    case_id: str
    repeat_index: StrictInt
    metrics: list[MetricResult] = Field(default_factory=list, max_length=100)


class ExperimentScore(StrictModel):
    schema_version: Literal[1] = 1
    experiment_id: str
    dataset_digest: Sha256
    variant_id: str
    trials: list[TrialScore] = Field(default_factory=list)
    created_at: datetime


class ComparisonSpec(StrictModel):
    schema_version: Literal[1] = 1
    baseline: str
    candidate: str
    comparable: bool
    differences: list[str] = Field(default_factory=list)
    created_at: datetime
