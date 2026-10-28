"""Schema 3 acceptance reviews bind to concrete run files, not hash strings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_triage.acceptance import accept_profile
from agent_triage.acceptance_review import import_review
from agent_triage.errors import ValidationFailure
from agent_triage.store import atomic_write_json, read_json

from quality_fixtures import build_fixture, narrative_record, prepare_completed
from test_acceptance import EVALUATION_ID


def _accept(fixture: dict, profile: str = "reviewed"):
    return accept_profile(
        profile=profile,
        evidence_dir=fixture["evidence"],
        project_root=fixture["project"],
    )


def _quality(accepted):
    return next(entry for entry in accepted.checks if entry.check_id == "AC15.quality")


def test_prepare_does_not_create_a_current_review(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    prepare_completed(fixture)
    assert not (fixture["evidence"] / "human" / "AC15.quality.json").exists()
    accepted = _accept(fixture)
    assert _quality(accepted).effective_state == "unknown"


def test_import_binds_real_run_files(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    completed = prepare_completed(fixture)
    receipt = import_review(fixture["evidence"], completed)
    assert receipt["status"] == "imported"
    assert receipt["check_status"] == "pass"
    assert (fixture["evidence"] / "human" / "AC15.quality.json").is_file()
    assert (fixture["evidence"] / "human" / "receipts" / "AC15.quality.json").is_file()

    accepted = _accept(fixture)
    quality = _quality(accepted)
    assert quality.execution_status.value == "pass"
    assert quality.effective_state == "current"
    assert quality.reasons == []
    assert quality.source_type == "human_review"


def test_tampered_event_is_rejected(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    completed = prepare_completed(fixture)
    (fixture["evidence"] / "human" / "bundles" / "QA01" / "event.json").write_text(
        '{"tampered": true}\n', encoding="utf-8"
    )
    with pytest.raises(ValidationFailure, match="hash|size"):
        import_review(fixture["evidence"], completed)


def test_unregistered_bundle_file_is_rejected(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    completed = prepare_completed(fixture)
    (fixture["evidence"] / "human" / "bundles" / "QA02" / "notes.txt").write_text(
        "not part of the allowlist\n", encoding="utf-8"
    )
    with pytest.raises(ValidationFailure, match="allowlist"):
        import_review(fixture["evidence"], completed)


def test_role_outside_the_bundle_is_rejected(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    completed = prepare_completed(
        fixture,
        mutate=lambda payload: payload["human_review"]["cases"][0]["roles"].__setitem__(
            "event", "human/other/event.json"
        ),
    )
    with pytest.raises(ValidationFailure, match="schema 3"):
        import_review(fixture["evidence"], completed)


def test_run_id_mismatch_is_rejected(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    completed = prepare_completed(
        fixture,
        mutate=lambda payload: payload["human_review"]["cases"][0].__setitem__(
            "run_id", "run-19990101-000000-abcdef"
        ),
    )
    with pytest.raises(ValidationFailure, match="run_id"):
        import_review(fixture["evidence"], completed)


def test_dataset_change_after_prepare_is_rejected(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    completed = prepare_completed(fixture)
    reference = fixture["evidence"] / "human" / "dataset" / "references" / "QA03.md"
    reference.write_text("# changed after prepare\n", encoding="utf-8")
    with pytest.raises(ValidationFailure, match="dataset|hash|size"):
        import_review(fixture["evidence"], completed)


def test_foreign_evaluation_id_is_rejected_without_import(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    completed = prepare_completed(fixture)
    payload = read_json(completed)
    payload["evaluation_id"] = "eval-19990101-000000-abcdef"
    atomic_write_json(fixture["evidence"] / "human" / "AC15.quality.json", payload)
    accepted = _accept(fixture)
    assert accepted.overall == "error"
    assert any("belongs to" in item for item in accepted.evidence_issues)


def test_explicit_import_keeps_the_source_evaluation(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    completed = prepare_completed(fixture)
    payload = read_json(completed)
    payload["evaluation_id"] = "eval-19990101-000000-abcdef"
    atomic_write_json(completed, payload)
    receipt = import_review(fixture["evidence"], completed)
    assert receipt["check_status"] == "pass"
    stored = read_json(fixture["evidence"] / "human" / "AC15.quality.json")
    assert stored["evaluation_id"] == EVALUATION_ID
    assert stored["imported_from"]["source_evaluation_id"] == "eval-19990101-000000-abcdef"
    source = fixture["evidence"] / stored["imported_from"]["source_record_path"]
    assert source.is_file()
    assert _quality(_accept(fixture)).effective_state == "current"


def test_four_of_five_cases_is_fail_not_pass(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    completed = prepare_completed(
        fixture,
        mutate=lambda payload: payload["human_review"]["cases"].pop(),
    )
    receipt = import_review(fixture["evidence"], completed)
    assert receipt["check_status"] == "fail"
    accepted = _accept(fixture)
    assert _quality(accepted).execution_status.value == "fail"


def test_pass_with_unresolved_blockers_is_fail(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    record = narrative_record(
        "NC20.manual",
        fixture["target"],
        fixture["hashes"]["NC20.manual"],
        blockers=["synthetic unresolved blocker"],
    )
    path = fixture["evidence"].parent / "nc20.json"
    atomic_write_json(path, record)
    receipt = import_review(fixture["evidence"], path)
    assert receipt["check_status"] == "fail"
    accepted = _accept(fixture)
    entry = next(item for item in accepted.checks if item.check_id == "NC20.manual")
    assert entry.execution_status.value == "fail"
    assert entry.effective_state == "current"


def test_quality_checks_reuse_the_same_five_bundles(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    quality = prepare_completed(fixture)
    handoff = prepare_completed(fixture, check_id="HC10")
    assert import_review(fixture["evidence"], quality)["check_status"] == "pass"
    assert import_review(fixture["evidence"], handoff)["check_status"] == "pass"
    accepted = _accept(fixture)
    for check_id in ("AC15.quality", "HC10"):
        entry = next(item for item in accepted.checks if item.check_id == check_id)
        assert entry.execution_status.value == "pass"
        assert entry.effective_state == "current"


def test_accept_reverifies_files_after_import(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    completed = prepare_completed(fixture)
    import_review(fixture["evidence"], completed)
    assert _quality(_accept(fixture)).effective_state == "current"
    (fixture["evidence"] / "human" / "bundles" / "QA04" / "analysis.json").write_text(
        '{"tampered": true}\n', encoding="utf-8"
    )
    accepted = _accept(fixture)
    assert accepted.overall == "error"
    assert any("hash" in item or "size" in item for item in accepted.evidence_issues)


def test_schema_two_hashes_alone_are_not_current(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    record = {
        "schema_version": 2,
        "record_id": "review-legacy",
        "evaluation_id": EVALUATION_ID,
        "check_id": "AC15.quality",
        "target": fixture["target"].model_dump(mode="json"),
        "artifacts": [],
        "human_review": {
            "reviewer": "synthetic-reviewer",
            "method": "human",
            "reviewed_at": "2026-10-26T00:00:00+00:00",
            "rubric_version": "v1",
            "cases": [
                {
                    "case_id": case_id,
                    "run_id": "run-19990101-000000-abcdef",
                    "input_sha256": "a" * 64,
                    "analysis_sha256": "b" * 64,
                    "verdict": "pass",
                    "rationale": "hash strings only",
                }
                for case_id in ("QA01", "QA02", "QA03", "QA04", "QA05")
            ],
        },
    }
    atomic_write_json(fixture["evidence"] / "human" / "AC15.quality.json", record)
    quality = _quality(_accept(fixture))
    assert quality.execution_status.value == "not_run"
    assert quality.effective_state == "unknown"
    assert quality.reasons