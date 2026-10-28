"""Case definition and evidence-source binding (F02/F03).

A green pytest result for a human check, or a substituted case file with the
same id but different nodes, must never be counted as current evidence for the
checkout's declaration.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_triage.acceptance import accept_profile, build_target, profile_required
from agent_triage.eval import run_eval
from agent_triage.store import atomic_write_json

from test_acceptance import case_hashes, make_project, result_payload, seed_all, write_envelope, write_result


def _accept(root: Path, evidence: Path, profile: str = "offline"):
    return accept_profile(profile=profile, evidence_dir=evidence, project_root=root)


def test_pytest_record_cannot_satisfy_human_review_check(tmp_path: Path) -> None:
    root = make_project(tmp_path / "project")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    target = build_target(root)
    write_envelope(evidence, target)
    hashes = case_hashes(root)
    for check_id in profile_required("reviewed"):
        # Deliberately mislabel every check as a normal pytest result.
        write_result(
            evidence,
            result_payload(check_id, target, source="pytest", case_hash=hashes.get(check_id)),
        )
    accepted = _accept(root, evidence, "reviewed")
    assert accepted.overall == "error"
    assert accepted.exit_code == 4
    quality = next(entry for entry in accepted.checks if entry.check_id == "AC15.quality")
    assert quality.execution_status.value == "error"
    assert any("source_type" in item for item in quality.reasons)


def test_generic_human_pass_without_typed_review_is_not_current(tmp_path: Path) -> None:
    root = make_project(tmp_path / "project")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    target = build_target(root)
    hashes = case_hashes(root)
    seed_all(evidence, target, profile_required("reviewed"), hashes=hashes)
    generic = result_payload(
        "AC15.quality",
        target,
        source="human_review",
        case_hash=hashes["AC15.quality"],
    )
    generic.pop("human_review")
    write_result(evidence, generic)
    accepted = _accept(root, evidence, "reviewed")
    assert accepted.overall == "incomplete"
    quality = next(entry for entry in accepted.checks if entry.check_id == "AC15.quality")
    assert quality.execution_status.value == "not_run"
    assert quality.effective_state == "unknown"


def test_embedded_four_case_human_result_is_historical(tmp_path: Path) -> None:
    root = make_project(tmp_path / "project")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    target = build_target(root)
    hashes = case_hashes(root)
    seed_all(evidence, target, profile_required("reviewed"), hashes=hashes)
    broken = result_payload(
        "AC15.quality",
        target,
        source="human_review",
        case_hash=hashes["AC15.quality"],
    )
    broken["human_review"]["cases"] = broken["human_review"]["cases"][:4]
    write_result(evidence, broken)
    accepted = _accept(root, evidence, "reviewed")
    assert accepted.overall == "incomplete"
    quality = next(item for item in accepted.checks if item.check_id == "AC15.quality")
    assert quality.execution_status.value == "not_run"
    assert quality.effective_state == "unknown"


def test_three_of_five_human_score_cannot_pass(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from agent_triage.acceptance import human_review_problems

    cases = [
        SimpleNamespace(
            case_id=case_id,
            verdict="pass" if case_id in {"QA01", "QA02", "QA03"} else "fail",
            unsupported_fact=False,
            unexecuted_check_reported_pass=False,
            unauthorized_write=False,
        )
        for case_id in ("QA01", "QA02", "QA03", "QA04", "QA05")
    ]
    problems = human_review_problems(
        "HC10",
        SimpleNamespace(cases=cases, playbooks=[], handoff=None),
    )
    assert any("at least 4" in item for item in problems)


def test_substituted_case_definition_does_not_count_for_ac01(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    substituted = tmp_path / "substituted-cases"
    substituted.mkdir()
    (project / "tests" / "test_probe.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    original = json.loads((project / "tests" / "cases" / "AC01.json").read_text(encoding="utf-8"))
    original["pytest"] = ["tests/test_probe.py::test_ok"]
    atomic_write_json(substituted / "AC01.json", original)

    output = tmp_path / "eval"
    payload = run_eval(
        cases_dir=substituted,
        output_root=output,
        project_root=project,
    )
    assert payload["cases"][0]["status"] == "pass"
    evidence_dir = output / payload["evaluation_id"]

    accepted = _accept(project, evidence_dir, "offline")
    assert accepted.overall == "incomplete"
    entry = next(item for item in accepted.checks if item.check_id == "AC01")
    assert entry.execution_status.value == "pass"
    assert entry.effective_state == "stale"
    assert not (entry.effective_state == "current" and entry.execution_status.value == "pass")


def test_result_without_case_definition_hash_is_unknown(tmp_path: Path) -> None:
    root = make_project(tmp_path / "project")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    target = build_target(root)
    seed_all(evidence, target, profile_required("offline"), hashes=case_hashes(root))
    payload = result_payload("AC01", target, case_hash=case_hashes(root)["AC01"])
    payload["case_definition_hash"] = None
    write_result(evidence, payload)
    accepted = _accept(root, evidence, "offline")
    entry = next(item for item in accepted.checks if item.check_id == "AC01")
    assert entry.effective_state == "unknown"
    assert accepted.overall == "incomplete"

def test_schema_two_human_file_stays_historical(tmp_path: Path) -> None:
    from agent_triage.acceptance import load_execution_results
    from agent_triage.case_manifest import load_manifest
    from agent_triage.store import atomic_write_json

    project = make_project(tmp_path / "project")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    target = build_target(project)
    write_envelope(evidence, target)
    hashes = case_hashes(project)
    human = {
        "schema_version": 2,
        "record_id": "review-20261025-000001",
        "evaluation_id": "eval-20261025-000000-abcdef",
        "check_id": "AC15.quality",
        "case_definition_hash": hashes["AC15.quality"],
        "target": target.model_dump(mode="json"),
        "artifacts": [],
        "human_review": {
            "reviewer": "owner",
            "method": "human",
            "reviewed_at": "2026-10-25T12:00:00+00:00",
            "rubric_version": "v1",
            "cases": [
                {
                    "case_id": case_id,
                    "run_id": f"run-20261025-00000{index}-abcdef",
                    "input_sha256": "a" * 64,
                    "analysis_sha256": "b" * 64,
                    "verdict": "pass",
                    "rationale": "synthetic contract check",
                }
                for index, case_id in enumerate(("QA01", "QA02", "QA03", "QA04", "QA05"))
            ],
        },
    }
    atomic_write_json(evidence / "human" / "AC15.quality.json", human)
    results, issues = load_execution_results(
        evidence,
        current=target,
        manifest=load_manifest(project / "tests" / "cases"),
    )
    assert issues == []
    assert len(results) == 1
    result = results[0]
    assert result.source_type == "human_review"
    assert result.status.value == "not_run"
    assert result.reason_code == "legacy_human_record"
    assert result.case_definition_hash is None


def test_run_artifact_pass_without_imported_artifacts_is_rejected(tmp_path: Path) -> None:
    from agent_triage.acceptance import ExecutionResult

    project = make_project(tmp_path / "project")
    target = build_target(project)
    payload = result_payload(
        "HC09",
        target,
        source="run_artifact",
        case_hash=case_hashes(project)["HC09"],
    )
    payload["related_run_ids"] = ["run-20261025-000000-abcdef"]
    with pytest.raises(Exception, match="imported artifact hashes"):
        ExecutionResult.model_validate(payload)
