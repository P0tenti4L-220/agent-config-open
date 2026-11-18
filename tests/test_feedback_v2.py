from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agent_triage.cli import main
from agent_triage.contracts import Outcome
from agent_triage.errors import OutcomeRejected, StoreError
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


def test_rejected_feedback_keeps_verified_run_and_independent_audits(tmp_path: Path, capsys) -> None:
    result, project, output, store, evidence, lessons = _setup(tmp_path)
    outcome_path = tmp_path / "outcome.json"
    args = [
        "record-outcome", "--run-id", result.run_id, "--outcome", str(outcome_path),
        "--output", str(output),
    ]
    payload = _outcome(result.run_id, project)
    outcome_path.write_text(json.dumps(payload), encoding="utf-8")
    assert main(args) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "resolved"
    paths = [
        *(store.dir / name for name in (
            "checks.json", "promotion-check.json", "outcome.json", "run.json", "report.md", "handoff.md"
        )),
        output / "index.json",
        lessons.path,
        *list((store.dir / "evidence/outcomes").iterdir()),
    ]
    hashes = {path: hash_file(path) for path in paths}
    payload["owner_confirmed"] = False
    outcome_path.write_text(json.dumps(payload), encoding="utf-8")
    assert main(args) == 1
    assert "owner_confirmed is false" in capsys.readouterr().err
    payload["owner_confirmed"] = True
    payload["commands"][0]["target_sha256"] = "0" * 64
    outcome_path.write_text(json.dumps(payload), encoding="utf-8")
    assert main(args) == 1
    assert "old hash" in capsys.readouterr().err
    assert {path: hash_file(path) for path in paths} == hashes

    audit_dir = store.dir / "outcome-audits"
    audits = {path: json.loads(path.read_text(encoding="utf-8")) for path in audit_dir.glob("*.json")}
    assert len(audits) == 3
    assert audits[audit_dir / f"{receipt['outcome_id']}.json"]["receipt"] == receipt
    rejected = [audit for audit in audits.values() if audit["gate"]["decision"] == "deny"]
    assert len(rejected) == 2
    assert all(audit["receipt"] is None and not audit["record"]["promoted"] for audit in rejected)
    assert main(args) == 1
    assert {path: json.loads(path.read_text(encoding="utf-8")) for path in audit_dir.glob("*.json")} == audits


@pytest.mark.parametrize("failed_write", ["promotion-check.json", "outcome.json", "receipt"])
def test_retry_after_interrupted_promotion_returns_one_receipt(
    tmp_path: Path, monkeypatch, failed_write: str
) -> None:
    result, project, output, store, evidence, lessons = _setup(tmp_path)
    outcome = Outcome.model_validate(_outcome(result.run_id, project))
    write_json = store.write_json

    def interrupt(name, payload):
        if name == failed_write or (failed_write == "receipt" and payload.get("receipt")):
            raise StoreError("feedback write interrupted")
        return write_json(name, payload)

    with monkeypatch.context() as patch:
        patch.setattr(store, "write_json", interrupt)
        with pytest.raises(StoreError, match="feedback write interrupted"):
            record_outcome(store=store, evidence=evidence, lesson_store=lessons, outcome=outcome)
    pending_path = store.dir / "outcome-audits/.pending.json"
    pending = json.loads(pending_path.read_text(encoding="utf-8"))
    assert pending["gate"]["decision"] == "allow"
    assert pending["record"]["promoted"] is False
    assert "receipt" not in pending
    audit_path = pending_path.with_name(f"{pending['outcome_id']}.json")
    assert not audit_path.exists()
    if failed_write != "receipt":
        assert not (store.dir / "outcome.json").exists()

    # The next CLI process can finish from the pinned output snapshot.
    (project / "pytest.log").unlink()
    store = RunStore.load(output, result.run_id)
    receipt = record_outcome(store=store, evidence=EvidenceStore(store), lesson_store=lessons, outcome=outcome)
    assert receipt["promoted"] is True and receipt["status"] == "resolved"
    assert not pending_path.exists()
    report = store.dir / "report.md"
    assert report.read_text(encoding="utf-8").count("## 外部验证记录") == 1
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["receipt"] == receipt and audit["record"]["promoted"] is True
    paths = (audit_path, report, store.dir / "outcome.json", store.dir / "checks.json", lessons.path)
    hashes = {path: hash_file(path) for path in paths}
    assert record_outcome(store=store, evidence=EvidenceStore(store), lesson_store=lessons, outcome=outcome) == receipt
    assert {path: hash_file(path) for path in paths} == hashes
