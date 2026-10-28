from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from agent_triage import cli
from agent_triage.acceptance import (
    PROFILE_ORDER,
    ExecutionResult,
    accept_profile,
    build_target,
    load_execution_results,
    profile_required,
    tree_digest,
)
from agent_triage.case_manifest import case_definition_hash, load_manifest
from agent_triage.eval import EvalCase, _run_pytest_case
from agent_triage.store import atomic_write_json

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ID = "eval-20261025-000000-abcdef"
_CASE_HASHES: dict[str, str] = {}


def write_envelope(evidence: Path, target, evaluation_id: str = EVALUATION_ID) -> None:
    """Write the root identity every result in this synthetic directory shares."""
    evidence.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        evidence / "eval.json",
        {
            "schema_version": 2,
            "evaluation_id": evaluation_id,
            "target": target.model_dump(mode="json"),
            "cases": [],
        },
    )


def make_project(root: Path) -> Path:
    global _CASE_HASHES
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    shutil.copytree(REPO_ROOT / "tests" / "cases", root / "tests" / "cases")
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "agent-capabilities.json").write_text('{"capabilities": []}\n', encoding="utf-8")
    _CASE_HASHES = case_hashes(root)
    return root


def case_hashes(root: Path) -> dict[str, str]:
    manifest = load_manifest(root / "tests" / "cases")
    return {check_id: case_definition_hash(case) for check_id, case in manifest.cases.items()}


def human_review_payload(check_id: str, *, verdict: str = "pass") -> dict:
    base = {
        "reviewer": "owner",
        "method": "human",
        "reviewed_at": "2026-10-25T12:00:00+00:00",
        "rubric_version": "rubric-v1",
        "summary": "synthetic contract fixture, not a real human review",
    }
    if check_id in {"AC15.quality", "HC10"}:
        base["cases"] = [
            {
                "case_id": f"QA{index:02d}",
                "run_id": f"run-20261025-12000{index}-abcdef",
                "input_sha256": f"{index:064x}",
                "analysis_sha256": f"{index + 1:064x}",
                "verdict": verdict if index >= 2 else "fail",
                "rationale": f"case {index} judged against the frozen rubric",
            }
            for index in range(1, 6)
        ]
    elif check_id == "NC20.manual":
        base["playbooks"] = [
            {
                "playbook_id": playbook,
                "sample": f"{playbook} sample",
                "steps": "ran the documented walkthrough",
                "outcome": "observed output and limitations",
                "blockers": [],
                "verdict": verdict,
            }
            for playbook in (
                "structured-output-mismatch",
                "tool-failure-and-recovery",
                "context-and-handoff-gap",
            )
        ]
    elif check_id == "HC13":
        base["handoff"] = {
            "operator": "owner",
            "no_chat_history": True,
            "target_identified": True,
            "facts_identified": True,
            "gaps_identified": True,
            "next_step_identified": True,
            "result": "continued from the handoff alone",
            "blockers": [],
            "verdict": verdict,
        }
    return base


def result_payload(
    check_id: str,
    target,
    *,
    status: str = "pass",
    attempt: str = "1",
    source: str = "pytest",
    case_hash: str | None = None,
    artifacts: list[dict] | None = None,
    nodes: list[dict] | None = None,
) -> dict:
    if case_hash is None:
        case_hash = _CASE_HASHES.get(check_id)
    human = source == "human_review"
    execution = {}
    if source == "pytest":
        execution = {
            "argv": ["pytest", "tests/example.py::test_example"],
            "exit_code": 0 if status == "pass" else (1 if status in {"fail", "not_run"} else 2),
            "interpreter_version": sys.version.split()[0],
            "import_root_checked": True,
            "executor_id": "test",
        }
    elif source == "run_artifact":
        execution = {
            "argv": ["agent-triage", "analyze"],
            "exit_code": 0 if status == "pass" else 1,
            "executor_id": "live-bundle",
        }
    elif human:
        execution = {"executor_id": "reviewer"}
    payload = {
        "schema_version": 1,
        "evaluation_id": "eval-20261025-000000-abcdef",
        "check_id": check_id,
        "attempt_id": attempt,
        "status": status,
        "reason_code": None if status == "pass" else "synthetic",
        "reason": None if status == "pass" else "synthetic negative fixture",
        "source_type": source,
        "case_definition_hash": case_hash,
        "target": target.model_dump(mode="json"),
        "execution": execution,
        "artifacts": artifacts or [],
        "nodes": nodes
        if nodes is not None
        else ([{"nodeid": "tests/example.py::test_example", "outcome": "passed"}] if status == "pass" and source == "pytest" else []),
    }
    if human:
        payload["human_review"] = human_review_payload(check_id)
    return payload


def write_result(evidence: Path, payload: dict) -> Path:
    path = evidence / "results" / f"{payload['check_id']}-{payload['attempt_id']}.json"
    atomic_write_json(path, payload)
    return path


def seed_all(
    evidence: Path,
    target,
    checks: list[str],
    *,
    hashes: dict[str, str] | None = None,
    skip: set[str] | None = None,
    statuses: dict[str, str] | None = None,
) -> None:
    skip = skip or set()
    statuses = statuses or {}
    hashes = hashes or {}
    write_envelope(evidence, target)
    human_ids = {"AC15.quality", "NC20.manual", "HC10", "HC13"}
    for check_id in checks:
        if check_id in skip:
            continue
        status = statuses.get(check_id, "pass")
        if check_id in human_ids:
            # Schema 2 stays readable as history, but it is not current proof.
            # Tests that need a current review build a schema 3 bundle.
            atomic_write_json(
                evidence / "human" / f"{check_id}.json",
                {
                    "schema_version": 2,
                    "record_id": f"legacy-{check_id}",
                    "evaluation_id": EVALUATION_ID,
                    "check_id": check_id,
                    "target": target.model_dump(mode="json"),
                    "artifacts": [],
                    "human_review": human_review_payload(check_id, verdict="pass"),
                },
            )
            continue
        if check_id == "HC09":
            source = "run_artifact"
        elif check_id in human_ids:
            source = "human_review"
        else:
            source = "pytest"
        payload = result_payload(
            check_id,
            target,
            status=status,
            source=source,
            case_hash=hashes.get(check_id),
        )
        if check_id == "HC09" and status == "pass":
            artifact_path = evidence / "artifacts" / "HC09-live-bundle.txt"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text("synthetic verified live bundle\n", encoding="utf-8")
            payload["related_run_ids"] = ["run-20261025-000000-abcdef"]
            payload["artifacts"] = [
                {
                    "relative_path": "artifacts/HC09-live-bundle.txt",
                    "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                    "media_type": "text/plain",
                    "size_bytes": artifact_path.stat().st_size,
                }
            ]
        write_result(evidence, payload)


def accept(tmp_path: Path, target_root: Path, evidence: Path, profile: str):
    return accept_profile(
        profile=profile,
        evidence_dir=evidence,
        project_root=target_root,
    )


def test_profile_semantics_and_exit_codes(tmp_path: Path) -> None:
    root = make_project(tmp_path / "project")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    target = build_target(root)
    offline = profile_required("offline")
    seed_all(evidence, target, offline, hashes=case_hashes(root))

    accepted = accept(tmp_path, root, evidence, "offline")
    assert accepted.overall == "pass"
    assert accepted.exit_code == 0
    assert accepted.summary.pass_count == len(offline)

    # Offline evidence alone cannot promote a connected or reviewed profile.
    connected = accept(tmp_path, root, evidence, "connected")
    assert connected.overall == "incomplete"
    assert connected.exit_code == 2
    assert any(entry.check_id == "AC15.live" and entry.execution_status.value == "not_run" for entry in connected.checks)

    # A missing required check is incomplete, not pass.
    (evidence / "results" / "AC05-1.json").unlink()
    missing = accept(tmp_path, root, evidence, "offline")
    assert missing.overall == "incomplete"
    assert missing.exit_code == 2
    assert any(entry.check_id == "AC05" and entry.effective_state == "unknown" for entry in missing.checks)

    # A current failure outranks an incomplete result.
    write_result(evidence, result_payload("AC05", target, status="fail", nodes=[{"nodeid": "tests/example.py::test_example", "outcome": "failed"}]))
    failed = accept(tmp_path, root, evidence, "offline")
    assert failed.overall == "fail"
    assert failed.exit_code == 1

    # An execution error outranks a failure.
    write_result(evidence, result_payload("AC06", target, status="error"))
    errored = accept(tmp_path, root, evidence, "offline")
    assert errored.overall == "error"
    assert errored.exit_code == 4

    # A required check cannot be waived with not_applicable.
    write_result(evidence, result_payload("AC06", target, status="not_applicable"))
    waived = accept(tmp_path, root, evidence, "offline")
    assert waived.overall != "pass"
    assert any(entry.check_id == "AC06" and entry.execution_status.value == "not_applicable" for entry in waived.checks)

    # A corrupt result file makes the evidence directory an error, not a pass.
    (evidence / "results" / "broken.json").write_text('{"check_id": "AC07"}\n', encoding="utf-8")
    corrupt = accept(tmp_path, root, evidence, "offline")
    assert corrupt.overall == "error"
    assert corrupt.evidence_issues


def test_legacy_human_records_do_not_make_reviewed_current(tmp_path: Path) -> None:
    root = make_project(tmp_path / "project")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    target = build_target(root)
    seed_all(evidence, target, profile_required("reviewed"), hashes=case_hashes(root))
    accepted = accept(tmp_path, root, evidence, "reviewed")
    assert accepted.overall == "incomplete"
    assert accepted.exit_code == 2
    human = next(entry for entry in accepted.checks if entry.check_id == "AC15.quality")
    assert human.source_type == "human_review"
    assert human.effective_state == "unknown"
    assert human.execution_status.value == "not_run"

    # A human pass without a declared target is rejected by the contract.
    bad = result_payload("HC10", target, source="human_review", status="pass")
    bad["target"] = None
    with pytest.raises(Exception):
        ExecutionResult.model_validate(bad)


def test_stale_targets_and_artifact_integrity(tmp_path: Path) -> None:
    root = make_project(tmp_path / "project")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    target = build_target(root)
    offline = profile_required("offline")
    seed_all(evidence, target, offline, hashes=case_hashes(root))
    assert accept(tmp_path, root, evidence, "offline").overall == "pass"

    # Any relevant source change invalidates the old result; history is kept.
    (root / "src" / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    old_digest = target.relevant_tree_digest
    stale = accept(tmp_path, root, evidence, "offline")
    assert stale.overall == "incomplete"
    assert old_digest != stale.evaluated_target.relevant_tree_digest
    assert all(entry.effective_state == "stale" for entry in stale.checks)
    assert (evidence / "results" / "AC01-1.json").exists(), "history must not be deleted"

    # Config and corpus changes are separate staleness dimensions.
    fresh_target = build_target(root)
    seed_all(evidence, fresh_target, offline, hashes=case_hashes(root))
    assert accept(tmp_path, root, evidence, "offline").overall == "pass"
    changed = json.loads((root / "tests" / "cases" / "AC01.json").read_text(encoding="utf-8"))
    changed["expectation"] = "changed definition after the result was produced"
    (root / "tests" / "cases" / "AC01.json").write_text(json.dumps(changed), encoding="utf-8")
    corpus_stale = accept(tmp_path, root, evidence, "offline")
    assert corpus_stale.overall == "incomplete"
    assert all(entry.effective_state == "stale" for entry in corpus_stale.checks)

    # A missing or hash-mismatched artifact is an execution error.
    bad = result_payload("AC09", build_target(root))
    bad["artifacts"] = [
        {"relative_path": "logs/missing.log", "sha256": "0" * 64, "media_type": "text/plain"}
    ]
    write_result(evidence, bad)
    broken = accept(tmp_path, root, evidence, "offline")
    assert broken.overall == "error"
    assert any("artifact" in item for item in broken.evidence_issues)

    on_disk = evidence / "logs" / "present.log"
    on_disk.parent.mkdir(parents=True, exist_ok=True)
    on_disk.write_text("real\n", encoding="utf-8")
    bad["artifacts"] = [
        {"relative_path": "logs/present.log", "sha256": "1" * 64, "media_type": "text/plain"}
    ]
    write_result(evidence, bad)
    mismatch = accept(tmp_path, root, evidence, "offline")
    assert mismatch.overall == "error"
    assert any("hash" in item for item in mismatch.evidence_issues)


def test_legacy_and_different_target_results_do_not_count(tmp_path: Path) -> None:
    root = make_project(tmp_path / "project")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    offline = profile_required("offline")

    # A v1 eval.json has no provenance: every result is unknown, not current.
    cases = [{"id": check_id, "status": "pass"} for check_id in offline]
    atomic_write_json(
        evidence / "eval.json",
        {"schema_version": 1, "overall": "pass", "summary": {}, "cases": cases},
    )
    legacy = accept(tmp_path, root, evidence, "offline")
    assert legacy.overall == "incomplete"
    assert all(entry.effective_state == "unknown" for entry in legacy.checks)

    # A failure for target A does not disappear when target B passes.
    target_a = build_target(root)
    seed_all(evidence, target_a, offline, hashes=case_hashes(root), statuses={"AC01": "fail"})
    (root / "src" / "app.py").write_text("VALUE = 3\n", encoding="utf-8")
    target_b = build_target(root)
    seed_all(evidence, target_b, offline, hashes=case_hashes(root))
    assert target_a.relevant_tree_digest != target_b.relevant_tree_digest
    assert accept(tmp_path, root, evidence, "offline").overall == "pass"

    # A current failure is not covered by an older pass for a different target.
    seed_all(evidence, target_b, offline, hashes=case_hashes(root), statuses={"AC01": "fail"})
    (root / "src" / "app.py").write_text("VALUE = 4\n", encoding="utf-8")
    target_c = build_target(root)
    seed_all(evidence, target_c, offline, hashes=case_hashes(root))
    write_result(
        evidence,
        result_payload("AC01", target_c, status="fail", attempt="2", nodes=[{"nodeid": "tests/example.py::test_example", "outcome": "failed"}]),
    )
    assert accept(tmp_path, root, evidence, "offline").overall == "fail"


def test_structured_pytest_results_and_runner_errors(tmp_path: Path, monkeypatch) -> None:
    root = make_project(tmp_path / "project")
    tests = root / "tests"
    tests.mkdir(exist_ok=True)
    (tests / "test_probe.py").write_text(
        "import pytest\n\n"
        "def test_stdout_mentions_skipped():\n"
        "    print('skipped: this is only stdout text')\n"
        "    assert True\n\n"
        "def test_real_skip():\n"
        "    pytest.skip('missing optional thing')\n\n"
        "def test_fails():\n"
        "    assert False\n\n"
        "def test_errors():\n"
        "    raise RuntimeError('boom')\n\n"
        "@pytest.fixture\n"
        "def broken_fixture():\n"
        "    raise RuntimeError('setup boom')\n\n"
        "def test_error_outcome(broken_fixture):\n"
        "    assert True\n",
        encoding="utf-8",
    )
    (tests / "test_broken.py").write_text("def test_never_closes(:\n", encoding="utf-8")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    target = build_target(root)

    def run(name: str, nodes: list[str]) -> dict:
        case = EvalCase(
            id=name,
            title=name,
            level="offline",
            source="pytest",
            pytest=nodes,
            expectation="test",
        )
        return _run_pytest_case(
            case,
            evaluation_id="eval-20261025-000000-abcdef",
            project_root=root,
            evidence_dir=evidence,
            target=target,
            include_live=False,
        )

    passed = run("PASS01", ["tests/test_probe.py::test_stdout_mentions_skipped"])
    assert passed["status"] == "pass", passed
    skipped = run("SKIP01", ["tests/test_probe.py::test_real_skip"])
    assert skipped["status"] == "not_run"
    failed = run("FAIL01", ["tests/test_probe.py::test_fails"])
    assert failed["status"] == "fail"
    errored = run("ERROR01", ["tests/test_probe.py::test_error_outcome"])
    assert errored["status"] == "error"
    missing = run("MISS01", ["tests/test_probe.py::test_does_not_exist"])
    assert missing["status"] == "error"
    collection = run("COLL01", ["tests/test_broken.py::test_never_closes"])
    assert collection["status"] == "error"
    assert collection["reason_code"] in {"not_collected", "missing_report", "test_error"}

    # A timeout still produces an error result with a log.
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=1)

    monkeypatch.setattr("agent_triage.eval.subprocess.run", fake_run)
    timed_out = run("TIME01", ["tests/test_probe.py::test_stdout_mentions_skipped"])
    assert timed_out["status"] == "error"
    assert timed_out["reason_code"] == "timeout"
    assert (evidence / "logs" / "TIME01.log").exists()
    monkeypatch.undo()

    # Every attempt is preserved, including a later pass for the same check.
    first = result_payload("AC01", target, status="fail", attempt="1", nodes=[{"nodeid": "x", "outcome": "failed"}])
    second = result_payload("AC01", target, status="pass", attempt="2")
    write_result(evidence, first)
    write_result(evidence, second)
    write_envelope(evidence, target)
    results, issues = load_execution_results(evidence)
    ac01 = [item for item in results if item.check_id == "AC01"]
    assert {item.attempt_id for item in ac01} == {"1", "2"}
    assert not issues


def test_offline_degradation_is_not_connected_success(tmp_path: Path) -> None:
    from agent_triage.demo import run_demo

    result = run_demo(tmp_path / "demo")
    run_dir = tmp_path / "demo" / "runs" / result.run_id
    checks = {item["check_id"]: item for item in json.loads((run_dir / "checks.json").read_text(encoding="utf-8"))["checks"]}
    assert result.analysis.analysis_status.value == "partial"
    assert result.remote_status == "not_configured"
    assert checks["github-history"]["status"] == "not_run"
    assert checks["claim-support"]["status"] == "not_run"

    root = make_project(tmp_path / "project")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    target = build_target(root)
    seed_all(evidence, target, profile_required("offline"), hashes=case_hashes(root))
    offline = accept(tmp_path, root, evidence, "offline")
    assert offline.overall == "pass"
    assert any("真实" in item for item in offline.not_established)
    connected = accept(tmp_path, root, evidence, "connected")
    assert connected.overall == "incomplete"
    assert connected.not_established

    # The CLI preserves the existing eval exit semantics (partial still exits 0).
    cases = tmp_path / "cases"
    cases.mkdir()
    atomic_write_json(
        cases / "PASS.json",
        {
            "schema_version": 2,
            "id": "PASS",
            "title": "pass",
            "level": "offline",
            "source": "pytest",
            "pytest": ["tests/test_contracts.py::test_event_accepts_documented_sample"],
            "expectation": "test",
        },
    )
    atomic_write_json(
        cases / "HUMAN.json",
        {
            "schema_version": 2,
            "id": "HUMAN",
            "title": "human",
            "level": "human",
            "source": "human_review",
            "expectation": "test",
        },
    )
    assert cli.main(["eval", "--cases", str(cases), "--output", str(tmp_path / "eval")]) == 0


def test_accept_cli_writes_result_and_exit_code(tmp_path: Path, capsys) -> None:
    root = make_project(tmp_path / "project")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    target = build_target(root)
    seed_all(evidence, target, profile_required("offline"), hashes=case_hashes(root))
    code = cli.main(
        [
            "accept",
            "--profile",
            "offline",
            "--evidence-dir",
            str(evidence),
            "--project-root",
            str(root),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["overall"] == "pass"
    assert (evidence / "acceptance-offline.json").exists()
    assert (evidence / "acceptance.md").exists()
    assert PROFILE_ORDER == ("offline", "connected", "reviewed")
    assert tree_digest(root) == target.relevant_tree_digest