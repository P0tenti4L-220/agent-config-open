from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agent_triage.contracts import Outcome
from agent_triage.errors import OutcomeRejected
from agent_triage.feedback import record_outcome
from agent_triage.store import EvidenceStore, LessonStore, RunStore, hash_file
from helpers import run_replay


def _setup(tmp_path: Path):
    result, project, output = run_replay(tmp_path)
    store = RunStore.load(output, result.run_id)
    evidence = EvidenceStore(store)
    lessons = LessonStore(output)
    return result, project, output, store, evidence, lessons


def _outcome(run_id: str, project: Path, *, check_id: str = "adapter-check", output_name: str = "pytest.log", extra: bool = True) -> dict:
    current = hash_file(project / "src/check_reader.py")
    (project / output_name).write_text("1 passed in 0.01s\n", encoding="utf-8")
    payload = {
        "schema_version": 2,
        "run_id": run_id,
        "commands": [
            {
                "command": "python -m pytest -q",
                "exit_code": 0,
                "target_path": "src/check_reader.py",
                "target_sha256": current,
                "output_ref": output_name,
                "output_sha256": hash_file(project / output_name),
                "check_id": check_id,
                "hypothesis_id": "claim-hyp-001",
            }
        ],
        "executed_by": "owner",
        "executed_at": "2026-10-24T20:00:00+00:00",
        "owner_confirmed": True,
    }
    if extra:
        payload["commands"][0]["additional_targets"] = [
            {"path": "data/checks.json", "sha256": hash_file(project / "data/checks.json")}
        ]
    return payload


def test_v2_missing_output_does_not_promote(tmp_path: Path) -> None:
    result, project, output, store, evidence, lessons = _setup(tmp_path)
    payload = _outcome(result.run_id, project)
    payload["commands"][0]["output_ref"] = "missing.log"
    with pytest.raises(OutcomeRejected, match="remains a candidate"):
        record_outcome(store=store, evidence=evidence, lesson_store=lessons, outcome=Outcome.model_validate(payload))
    assert store.meta.status == "awaiting_verification"
    assert all(lesson.status.value == "candidate" for lesson in lessons.load())
    promotion = json.loads((store.dir / "promotion-check.json").read_text(encoding="utf-8"))
    assert promotion["decision"] == "deny"
    assert any("does not exist" in reason for reason in promotion["reasons"])


def test_v2_output_hash_mismatch_does_not_promote(tmp_path: Path) -> None:
    result, project, output, store, evidence, lessons = _setup(tmp_path)
    payload = _outcome(result.run_id, project)
    payload["commands"][0]["output_sha256"] = "0" * 64
    with pytest.raises(OutcomeRejected):
        record_outcome(store=store, evidence=evidence, lesson_store=lessons, outcome=Outcome.model_validate(payload))
    assert all(lesson.status.value == "candidate" for lesson in lessons.load())


def test_v2_one_of_two_targets_mismatch_keeps_partial_hashes(tmp_path: Path) -> None:
    result, project, output, store, evidence, lessons = _setup(tmp_path)
    payload = _outcome(result.run_id, project)
    payload["commands"][0]["additional_targets"][0]["sha256"] = "0" * 64
    with pytest.raises(OutcomeRejected):
        record_outcome(store=store, evidence=evidence, lesson_store=lessons, outcome=Outcome.model_validate(payload))
    record = json.loads((store.dir / "outcome.json").read_text(encoding="utf-8"))
    assert set(record["current_target_hashes"]) == {"src/check_reader.py", "data/checks.json"}
    assert record["promoted"] is False


def test_v2_unplanned_check_is_rejected(tmp_path: Path) -> None:
    result, project, output, store, evidence, lessons = _setup(tmp_path)
    payload = _outcome(result.run_id, project, check_id="some-other-check")
    with pytest.raises(OutcomeRejected):
        record_outcome(store=store, evidence=evidence, lesson_store=lessons, outcome=Outcome.model_validate(payload))
    record = json.loads((store.dir / "outcome.json").read_text(encoding="utf-8"))
    assert any("not one of the analysis suggested checks" in reason for reason in record["reasons"])
    assert all(lesson.status.value == "candidate" for lesson in lessons.load())


def test_v1_valid_shape_still_cannot_promote(tmp_path: Path) -> None:
    result, project, output, store, evidence, lessons = _setup(tmp_path)
    payload = _outcome(result.run_id, project, extra=False)
    payload["schema_version"] = 1
    for key in ("check_id", "hypothesis_id", "output_sha256"):
        payload["commands"][0].pop(key, None)
    with pytest.raises(OutcomeRejected):
        record_outcome(store=store, evidence=evidence, lesson_store=lessons, outcome=Outcome.model_validate(payload))
    record = json.loads((store.dir / "outcome.json").read_text(encoding="utf-8"))
    assert any("legacy schema_version=1" in reason for reason in record["reasons"])
    assert all(lesson.status.value == "candidate" for lesson in lessons.load())


def test_v2_valid_outcome_promotes_with_scope_and_imported_output(tmp_path: Path) -> None:
    result, project, output, store, evidence, lessons = _setup(tmp_path)
    payload = _outcome(result.run_id, project)
    promoted = record_outcome(
        store=store, evidence=evidence, lesson_store=lessons, outcome=Outcome.model_validate(payload)
    )
    assert promoted["promoted"] is True
    assert promoted["status"] == "resolved"
    lesson = lessons.load()[0]
    assert lesson.status.value == "verified"
    assert lesson.applicable_versions == ["v0.1.0"]
    assert lesson.source_check_ids == ["adapter-check"]
    assert lesson.source_run_id == result.run_id
    assert lesson.verification is not None
    assert lesson.verification.contract == "v2"
    assert lesson.verification.check_ids == ["adapter-check"]
    assert "src/check_reader.py" in lesson.verification.target_hashes
    assert len(lesson.verification.output_refs) == 1
    snapshot = store.dir / lesson.verification.output_refs[0]
    assert snapshot.exists()
    assert snapshot.read_text(encoding="utf-8").startswith("1 passed")
    promotion = json.loads((store.dir / "promotion-check.json").read_text(encoding="utf-8"))
    assert promotion["decision"] == "allow"
    assert promotion["outputs"][0]["check_id"] == "adapter-check"
    handoff = (store.dir / "handoff.md").read_text(encoding="utf-8")
    assert "resolved" in handoff
    assert "adapter-check" in handoff