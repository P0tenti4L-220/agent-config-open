from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agent_triage.contracts import Analysis, CheckResult, Outcome
from agent_triage.errors import OutcomeRejected
from agent_triage.feedback import record_outcome
from agent_triage.runner import validate_run_artifacts
from agent_triage.store import EvidenceStore, LessonStore, RegistryStore, RunStore
from helpers import base_analysis, final_step, make_scenario, run_replay, tool_step


def _outcome_payload(run_id: str, target_hash: str, **overrides) -> dict:
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "commands": [
            {
                "command": "python -m pytest -q",
                "exit_code": 0,
                "target_path": "src/check_reader.py",
                "target_sha256": target_hash,
                "output_ref": "pytest.log",
            }
        ],
        "executed_by": "owner",
        "executed_at": "2026-10-24T20:00:00+00:00",
        "owner_confirmed": True,
    }
    payload.update(overrides)
    return payload


def _run_for_outcome(tmp_path: Path):
    result, project, output = run_replay(tmp_path)
    store = RunStore.load(output, result.run_id)
    evidence = EvidenceStore(store)
    lessons = LessonStore(output)
    return result, project, output, store, evidence, lessons


def test_validate_accepts_valid_refs_and_fails_random_ids(tmp_path: Path) -> None:
    result, project, output, store, evidence, _ = _run_for_outcome(tmp_path)
    checks = validate_run_artifacts(store, evidence)
    by_id = {check.check_id: check for check in checks}
    for check_id in ("analysis-schema", "evidence-refs", "claim-display", "handoff-contract"):
        assert by_id[check_id].status.value == "pass"
    assert by_id["claim-support"].status.value == "not_run"

    payload = json.loads((store.dir / "analysis.json").read_text(encoding="utf-8"))
    payload["facts"][0]["evidence_ids"] = ["ev-aaaaaaaaaaaa"]
    (store.dir / "analysis.json").write_text(json.dumps(payload), encoding="utf-8")
    checks = validate_run_artifacts(store, evidence)
    by_id = {check.check_id: check for check in checks}
    assert by_id["evidence-refs"].status.value == "fail"
    assert "unknown evidence" in by_id["evidence-refs"].scope


def test_check_status_and_old_hash_do_not_promote(tmp_path: Path) -> None:
    result, project, output, store, evidence, lessons = _run_for_outcome(tmp_path)
    current = hashlib.sha256((project / "src/check_reader.py").read_bytes()).hexdigest()

    incomplete = Outcome.model_validate(_outcome_payload(result.run_id, current, owner_confirmed=False))
    with pytest.raises(OutcomeRejected, match="remains a candidate"):
        record_outcome(store=store, evidence=evidence, lesson_store=lessons, outcome=incomplete)
    assert store.meta.status == "awaiting_verification"
    assert all(lesson.status.value == "candidate" for lesson in lessons.load())

    old_hash = Outcome.model_validate(_outcome_payload(result.run_id, "0" * 64))
    with pytest.raises(OutcomeRejected, match="remains a candidate"):
        record_outcome(store=store, evidence=evidence, lesson_store=lessons, outcome=old_hash)
    checks = json.loads((store.dir / "checks.json").read_text(encoding="utf-8"))
    outcome_check = next(item for item in checks["checks"] if item["check_id"] == "outcome-verification")
    assert outcome_check["status"] == "error"
    assert "hash" in outcome_check["scope"]

    failed = Outcome.model_validate(
        _outcome_payload(
            result.run_id,
            current,
            commands=[
                {
                    "command": "python -m pytest -q",
                    "exit_code": 1,
                    "target_path": "src/check_reader.py",
                    "target_sha256": current,
                    "output_ref": "pytest.log",
                }
            ],
        )
    )
    with pytest.raises(OutcomeRejected):
        record_outcome(store=store, evidence=evidence, lesson_store=lessons, outcome=failed)
    checks = json.loads((store.dir / "checks.json").read_text(encoding="utf-8"))
    outcome_check = next(item for item in checks["checks"] if item["check_id"] == "outcome-verification")
    assert outcome_check["status"] == "fail"
    assert outcome_check["exit_code"] == 1


def test_owner_confirmed_outcome_promotes_lesson_and_resolves_run(tmp_path: Path) -> None:
    result, project, output, store, evidence, lessons = _run_for_outcome(tmp_path)
    current = hashlib.sha256((project / "src/check_reader.py").read_bytes()).hexdigest()
    (project / "pytest.log").write_text("1 passed in 0.01s\n", encoding="utf-8")
    outcome = Outcome.model_validate(
        {
            "schema_version": 2,
            "run_id": result.run_id,
            "commands": [
                {
                    "command": "python -m pytest -q",
                    "exit_code": 0,
                    "target_path": "src/check_reader.py",
                    "target_sha256": current,
                    "output_ref": "pytest.log",
                    "check_id": "adapter-check",
                    "hypothesis_id": "claim-hyp-001",
                }
            ],
            "executed_by": "owner",
            "executed_at": "2026-10-24T20:00:00+00:00",
            "owner_confirmed": True,
        }
    )
    payload = record_outcome(store=store, evidence=evidence, lesson_store=lessons, outcome=outcome)
    assert payload["promoted"] is True
    assert payload["status"] == "resolved"
    store = RunStore.load(output, result.run_id)
    assert store.meta.status == "resolved"
    stored_lessons = lessons.load()
    assert len(stored_lessons) == 1
    assert stored_lessons[0].status.value == "verified"
    assert stored_lessons[0].verification is not None
    assert stored_lessons[0].verification.owner_confirmed is True
    registry = RegistryStore(output).load()
    assert next(run for run in registry.runs if run.run_id == result.run_id).status == "resolved"
    report = (store.dir / "report.md").read_text(encoding="utf-8")
    assert "外部验证记录" in report
    assert "不是本 Agent 亲自运行" in report
    handoff = (store.dir / "handoff.md").read_text(encoding="utf-8")
    assert "状态：`resolved`" in handoff
    assert "`outcome-verification`：`pass`" in handoff


def test_same_event_id_is_idempotent_and_new_event_counts(tmp_path: Path) -> None:
    first, project, output = run_replay(tmp_path)
    second, _, _ = run_replay(tmp_path)
    assert first.duplicate_event is False and first.occurrence == 1
    assert second.duplicate_event is True and second.occurrence == 1
    registry = RegistryStore(output).load()
    assert len(registry.events) == 1
    event = registry.events["test-event-001"]
    assert len(event.run_ids) == 2

    from helpers import make_event
    from agent_triage.config import Config
    from agent_triage.pipeline import run_new_analysis

    new_event = make_event(event_id="test-event-002")
    scenario = make_scenario(
        [
            tool_step(("read_file", {"path": "logs/failure.log", "start_line": 1, "max_lines": 20})),
            final_step(base_analysis()),
        ]
    )
    third = run_new_analysis(
        event=new_event,
        project_root=project,
        output_root=tmp_path / "runs",
        config=Config(),
        replay_scenario=scenario,
    )
    assert third.duplicate_event is False
    assert third.occurrence == 2
    registry = RegistryStore(tmp_path / "runs").load()
    assert len(registry.events) == 2
    group = registry.fingerprints[next(iter(registry.fingerprints))]
    assert group.occurrence == 2
    candidates = RegistryStore(tmp_path / "runs").duplicate_candidates(
        event_id="test-event-002", fingerprint=group_value(registry)
    )
    assert any(candidate["reference"] == "test-event-001" for candidate in candidates)
    assert all(candidate["state"] == "unknown" for candidate in candidates)


def group_value(registry) -> str:
    return next(iter(registry.fingerprints))


def test_business_defect_gets_no_central_issue_draft_and_no_absolute_paths(tmp_path: Path) -> None:
    scenario = make_scenario(
        [
            tool_step(("read_file", {"path": "logs/failure.log", "start_line": 1, "max_lines": 20})),
            final_step(
                base_analysis(
                    facts=[
                        {
                            "text": "日志显示读取 payload['items'] 时抛出 KeyError",
                            "evidence_ids": ["@path:logs/failure.log"],
                        }
                    ],
                    attribution="business_defect",
                    hypotheses=[],
                    suggested_checks=[],
                    analysis_status="partial",
                    stop_reason="insufficient_evidence",
                    missing_evidence=["business source owner must confirm the data contract"],
                )
            ),
        ]
    )
    result, project, output = run_replay(tmp_path, scenario=scenario)
    run_dir = output / result.run_id
    assert not (run_dir / "issue-draft.md").exists()
    assert not (run_dir / "lesson-candidate.json").exists()
    for name in ("report.md", "context.json", "handoff.md"):
        text = (run_dir / name).read_text(encoding="utf-8")
        assert str(project) not in text
        assert "/private/" not in text


def test_public_workflow_issue_draft_is_local_and_sanitized(tmp_path: Path) -> None:
    result, project, output = run_replay(tmp_path)
    run_dir = output / result.run_id
    draft = (run_dir / "issue-draft.md").read_text(encoding="utf-8")
    assert "尚未发布" in draft
    assert str(project) not in draft
    assert "http" not in draft or "example.invalid" in draft
    assert "FAKE_SECRET_CANARY" not in draft
    payload = json.loads((run_dir / "lesson-candidate.json").read_text(encoding="utf-8"))
    assert payload["status"] == "candidate"
    assert payload["verification"] is None


def test_artifacts_are_atomic_and_parse(tmp_path: Path) -> None:
    result, project, output = run_replay(tmp_path)
    run_dir = output / result.run_id
    temp_files = list(run_dir.rglob("*.tmp-*"))
    assert temp_files == []
    for name in ("analysis.json", "context.json", "checks.json", "lesson-candidate.json"):
        json.loads((run_dir / name).read_text(encoding="utf-8"))
    for name in ("report.md", "handoff.md", "issue-draft.md", "trace.jsonl"):
        assert (run_dir / name).read_text(encoding="utf-8").strip()

def test_public_rendering_redacts_secrets_and_local_paths(tmp_path: Path) -> None:
    scenario = make_scenario(
        [
            tool_step(("read_file", {"path": "logs/failure.log", "start_line": 1, "max_lines": 20})),
            final_step(
                base_analysis(
                    summary="leaked sk-abcdefghijklmnopqrstuvwxyz at /Users/example/private/project",
                    hypotheses=[],
                    facts=[
                        {
                            "text": "token ghp_abcdefghijklmnopqrstuvwxyz appears in /Users/example/private/project/log",
                            "evidence_ids": ["@path:logs/failure.log"],
                        }
                    ],
                )
            ),
        ]
    )
    result, project, output = run_replay(tmp_path, scenario=scenario)
    run_dir = output / result.run_id
    for name in ("report.md", "handoff.md", "issue-draft.md"):
        text = (run_dir / name).read_text(encoding="utf-8")
        assert "ghp_abcdefghijklmnopqrstuvwxyz" not in text
        assert "/Users/example" not in text
        assert "[REDACTED]" in text or "<LOCAL_PATH>" in text
