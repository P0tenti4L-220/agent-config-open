from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_triage.contracts import (
    Analysis,
    BudgetLimits,
    CheckResult,
    Config,
    ContextLimits,
    Event,
    Lesson,
    Outcome,
    ToolResult,
)
from helpers import base_analysis, make_event


def test_event_accepts_documented_sample() -> None:
    sample = Path(__file__).resolve().parents[1] / "examples" / "event.json"
    event = Event.model_validate_json(sample.read_bytes())
    assert event.event_id == "demo-event-001"
    assert event.related_files == ["src/check_reader.py", "data/checks.json"]


def test_unknown_schema_version_and_extra_fields_are_rejected() -> None:
    payload = make_event().model_dump(mode="json")
    payload["schema_version"] = 2
    with pytest.raises(ValidationError, match="schema_version"):
        Event.model_validate(payload)
    payload = make_event().model_dump(mode="json")
    payload["surprise"] = "silently ignored?"
    with pytest.raises(ValidationError, match="surprise"):
        Event.model_validate(payload)
    payload = make_event().model_dump(mode="json")
    payload["related_files"] = ["/etc/passwd"]
    with pytest.raises(ValidationError, match="repository-relative"):
        Event.model_validate(payload)


@pytest.mark.parametrize("bad", [[8], "8", True, 8.5])
def test_counts_reject_arrays_strings_bools_and_floats(bad) -> None:
    with pytest.raises(ValidationError):
        BudgetLimits(max_model_requests=bad)


def test_tool_result_strict_types_and_status_shape() -> None:
    with pytest.raises(ValidationError):
        ToolResult(status="ok", duration_ms=True)
    with pytest.raises(ValidationError):
        ToolResult(status="ok", duration_ms="10")
    with pytest.raises(ValidationError):
        ToolResult(status="error", duration_ms=1)
    with pytest.raises(ValidationError):
        ToolResult(status="ok", duration_ms=1, truncated=True)


def test_check_status_requires_evidence_of_execution() -> None:
    with pytest.raises(ValidationError, match="not_run"):
        CheckResult(
            check_id="tests-not-run",
            status="not_run",
            description="tests",
            exit_code=0,
            scope="should be rejected",
        )
    with pytest.raises(ValidationError, match="pass requires"):
        CheckResult(
            check_id="tests-pass",
            status="pass",
            description="tests",
            exit_code=0,
            scope="missing fields must fail",
        )
    valid = CheckResult(
        check_id="tests-pass",
        status="pass",
        description="tests",
        command="python -m pytest -q",
        exit_code=0,
        output_ref="pytest.log",
        target_sha256="a" * 64,
        scope="synthetic test run",
        executed_by="owner",
        executed_at=datetime.now(timezone.utc),
    )
    assert valid.status.value == "pass"
    with pytest.raises(ValidationError, match="pass requires exit_code 0"):
        CheckResult.model_validate({**valid.model_dump(mode="json"), "exit_code": 3})


def _concrete(data: dict) -> dict:
    counter = {"n": 0}

    def walk(value):
        if isinstance(value, str) and value.startswith("@path:"):
            counter["n"] += 1
            return f"ev-{counter['n']:012x}"
        if isinstance(value, list):
            return [walk(item) for item in value]
        if isinstance(value, dict):
            return {key: walk(item) for key, item in value.items()}
        return value

    return walk(data)


def test_analysis_requires_evidence_and_at_most_three_hypotheses() -> None:
    data = _concrete(base_analysis())
    data["facts"][0]["evidence_ids"] = []
    with pytest.raises(ValidationError):
        Analysis.model_validate(data)
    data = _concrete(base_analysis())
    data["hypotheses"] = data["hypotheses"] * 4
    with pytest.raises(ValidationError, match="at most 3"):
        Analysis.model_validate(data)
    data = _concrete(base_analysis())
    data["analysis_status"] = "partial"
    data["stop_reason"] = "insufficient_evidence"
    data["missing_evidence"] = []
    with pytest.raises(ValidationError, match="missing_evidence"):
        Analysis.model_validate(data)
    data = _concrete(base_analysis())
    data["suggested_checks"][0]["status"] = "pass"
    with pytest.raises(ValidationError):
        Analysis.model_validate(data)


def test_verified_lesson_requires_owner_confirmed_verification() -> None:
    with pytest.raises(ValidationError, match="verification"):
        Lesson(
            lesson_id="lesson-0123456789ab",
            status="verified",
            symptom_pattern="KeyError items",
            applicability="python",
            guidance="add an adapter",
        )


def test_outcome_rejects_strict_bool_and_bad_hash() -> None:
    payload = {
        "schema_version": 1,
        "run_id": "run-20261024-200000-abcdef",
        "commands": [
            {
                "command": "python -m pytest -q",
                "exit_code": 0,
                "target_path": "src/check_reader.py",
                "target_sha256": "a" * 64,
                "output_ref": "pytest.log",
            }
        ],
        "executed_by": "owner",
        "executed_at": "2026-10-24T20:00:00+00:00",
        "owner_confirmed": True,
    }
    assert Outcome.model_validate(payload).owner_confirmed is True
    payload["owner_confirmed"] = "true"
    with pytest.raises(ValidationError):
        Outcome.model_validate(payload)
    payload["owner_confirmed"] = True
    payload["commands"][0]["target_sha256"] = "not-a-hash"
    with pytest.raises(ValidationError):
        Outcome.model_validate(payload)


def test_default_layers_fit_total_budget() -> None:
    limits = ContextLimits()
    assert limits.total_input_tokens == 12_400
    with pytest.raises(ValidationError):
        ContextLimits(current_evidence_tokens=50_000)


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"schema_version": 1, "provider": "replay", "nope": 1})