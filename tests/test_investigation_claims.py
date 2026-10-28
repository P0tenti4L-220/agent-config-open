from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent_triage.claims import handoff_problems, support_review_label, validate_claims
from agent_triage.contracts import (
    Analysis,
    CheckStatus,
    ContextPacket,
    DisplaySource,
    DisplayedEvidence,
    Fact,
    InvestigationState,
    Lesson,
    LessonStatus,
    SupportReview,
    VerificationRecord,
)
from agent_triage.context import ContextBuilder
from agent_triage.contracts import ContextLimits
from agent_triage.playbooks import (
    PLAYBOOK_CATALOG_HASH,
    catalog_manifest,
    get_playbook,
    render_playbook,
    route_playbook,
)
from agent_triage.store import LessonStore, RunStore, hash_file
from helpers import base_analysis, make_event, make_scenario, make_tool_executor, run_replay, tool_step, final_step


def _display(evidence_id: str, turn: int = 0) -> DisplayedEvidence:
    return DisplayedEvidence(
        evidence_id=evidence_id,
        turn=turn,
        source=DisplaySource.TOOL_RESULT,
        mode="full",
        visible_chars=12,
        visible_sha256=hashlib.sha256(b"visible body").hexdigest(),
    )


def _analysis_with(fact_evidence: str, *, display: bool = True) -> Analysis:
    payload = base_analysis()
    payload["facts"] = [{"text": "观察到的日志", "evidence_ids": [fact_evidence]}]
    payload["hypotheses"] = []
    payload["suggested_checks"] = []
    payload["analysis_status"] = "partial"
    payload["stop_reason"] = "insufficient_evidence"
    payload["missing_evidence"] = ["需要复现"]
    return Analysis.model_validate(payload)


def test_claim_display_is_separate_from_reference(tmp_path: Path) -> None:
    executor, store, evidence, _, _, _ = make_tool_executor(tmp_path)
    result = executor.execute(
        "read_file", {"path": "logs/failure.log", "start_line": 1, "max_lines": 10}, call_id="c1", turn=0
    )
    evidence_id = result.evidence_refs[0]
    analysis = _analysis_with(evidence_id)
    not_displayed = validate_claims(analysis, evidence.all(), [])
    assert not_displayed.display_checked
    assert any("never displayed" in item for item in not_displayed.problems)
    displayed = validate_claims(analysis, evidence.all(), [_display(evidence_id)])
    assert displayed.ok
    assert displayed.display_ok


def test_stale_evidence_fails_reference_and_validate(tmp_path: Path) -> None:
    executor, store, evidence, _, _, output = make_tool_executor(tmp_path)
    result = executor.execute(
        "read_file", {"path": "logs/failure.log", "start_line": 1, "max_lines": 10}, call_id="c1", turn=0
    )
    evidence_id = result.evidence_refs[0]
    evidence.mark_stale(evidence_id, "source changed")
    analysis = _analysis_with(evidence_id)
    validation = validate_claims(analysis, evidence.all(), [_display(evidence_id)])
    assert any("stale" in item for item in validation.problems)

    from agent_triage.runner import validate_run_artifacts

    run_store = RunStore.load(output, store.run_id)
    run_store.write_json("analysis.json", analysis.model_dump(mode="json"))
    checks = {item.check_id: item for item in validate_run_artifacts(run_store, evidence)}
    assert checks["evidence-refs"].status.value == "fail"


def test_semantic_support_failure_is_not_rendered_as_pass(tmp_path: Path) -> None:
    review = SupportReview(
        status=CheckStatus.FAIL,
        reviewer="owner",
        method="manual-read",
        scope="read cited snippet against claim",
        reviewed_at=datetime.now(timezone.utc),
    )
    payload = base_analysis()
    payload["facts"][0]["evidence_ids"] = ["ev-0123456789ab"]
    payload["facts"][0]["support_review"] = review.model_dump(mode="json")
    payload["hypotheses"] = []
    analysis = Analysis.model_validate(payload)
    assert support_review_label(analysis) == "fail"

    from agent_triage.render import render_report

    event = make_event()
    result, _, output = run_replay(tmp_path)
    store = RunStore.load(output, result.run_id)
    report = render_report(
        event=event,
        analysis=analysis,
        evidence=[],
        checks=[],
        meta=store.meta,
        budget=__import__("agent_triage.budget", fromlist=["Budget"]).Budget(
            __import__("agent_triage.contracts", fromlist=["BudgetLimits"]).BudgetLimits()
        ).snapshot(),
        remote_status="not_configured",
    )
    assert "支持性复核：`fail`" in report
    assert "语义已验证" not in report


def test_handoff_contract_requires_next_step_and_counter_gap(tmp_path: Path) -> None:
    payload = base_analysis()
    payload["facts"][0]["evidence_ids"] = ["ev-0123456789ab"]
    payload["suggested_checks"] = []
    payload["missing_evidence"] = []
    payload["hypotheses"] = [
        {
            "text": "没有反证也没有待确认问题的假设",
            "supporting_evidence_ids": [],
            "contradicting_evidence_ids": [],
            "open_questions": [],
        }
    ]
    analysis = Analysis.model_validate(payload)
    problems = handoff_problems(analysis)
    assert any("minimal next step" in item for item in problems)
    assert any("neither counterevidence nor an open question" in item for item in problems)

    result, _, output = run_replay(tmp_path)
    store = RunStore.load(output, result.run_id)
    (store.dir / "handoff.md").write_text("", encoding="utf-8")
    from agent_triage.runner import validate_run_artifacts
    from agent_triage.store import EvidenceStore

    checks = {item.check_id: item for item in validate_run_artifacts(store, EvidenceStore(store))}
    assert checks["handoff-contract"].status.value == "fail"


def test_lesson_scope_is_explicit_for_version_and_legacy(tmp_path: Path) -> None:
    store = LessonStore(tmp_path / "runs")
    now = datetime.now(timezone.utc)
    verification = VerificationRecord(
        command="python -m pytest -q",
        exit_code=0,
        target_sha256="a" * 64,
        evidence_refs=[],
        executed_by="owner",
        executed_at=now,
        owner_confirmed=True,
    )
    legacy = Lesson(
        schema_version=1,
        lesson_id="lesson-aaaaaaaaaaaa",
        status=LessonStatus.VERIFIED,
        symptom_pattern="KeyError items",
        applicability="KeyError items",
        guidance="check adapter",
        verification=verification,
        applicable_versions=[],
    )
    current = Lesson(
        schema_version=2,
        lesson_id="lesson-bbbbbbbbbbbb",
        status=LessonStatus.VERIFIED,
        symptom_pattern="KeyError items",
        applicability="KeyError items",
        guidance="check adapter now",
        verification=VerificationRecord(
            command="python -m pytest -q",
            exit_code=0,
            target_sha256="a" * 64,
            evidence_refs=[],
            executed_by="owner",
            executed_at=now,
            owner_confirmed=True,
            check_ids=["adapter-check"],
            contract="v2",
        ),
        applicable_versions=["v0.1.0"],
        source_check_ids=["adapter-check"],
    )
    other = current.model_copy(
        update={"lesson_id": "lesson-cccccccccccc", "applicable_versions": ["v9.9.9"]}
    )
    store.save([legacy, current, other])
    summaries = {item.lesson_id: item for item in store.search("KeyError items", 10, rule_version="v0.1.0")}
    assert summaries[legacy.lesson_id].scope_status == "legacy"
    assert summaries[current.lesson_id].scope_status == "current"
    assert summaries[other.lesson_id].scope_status == "out_of_scope"
    assert summaries[legacy.lesson_id].verification_contract == "v1"


def test_counterevidence_is_kept_first_and_omissions_carry_impact(tmp_path: Path) -> None:
    executor, store, evidence, budget, project, _ = make_tool_executor(tmp_path)
    big = project / "logs" / "huge.log"
    big.write_text(("irrelevant line\n" * 4000) + "decisive counterexample\n", encoding="utf-8")
    first = executor.execute(
        "read_file", {"path": "logs/huge.log", "start_line": 1, "max_lines": 20}, call_id="c1", turn=0
    )
    auxiliary = executor.execute(
        "read_file", {"path": "logs/failure.log", "start_line": 1, "max_lines": 20}, call_id="c2", turn=0
    )
    decisive_id = first.evidence_refs[0]
    auxiliary_id = auxiliary.evidence_refs[0]
    investigation = InvestigationState(
        active_question="KeyError 是否由格式版本差异导致？",
        counterevidence_ids=[decisive_id],
        required_evidence_ids=[auxiliary_id],
        needed_materials=["logs/huge.log"],
        next_action_reason="先看反证，再看最小必要材料",
    )
    builder = ContextBuilder(ContextLimits(current_evidence_tokens=120), versions={"policy_hash": "a" * 64})
    context = builder.build(
        run_id=store.run_id,
        event=make_event(),
        evidence=evidence.all(),
        lessons=[],
        trace=[],
        conversation=[],
        budget_state=budget.snapshot(),
        turn=1,
        is_final_turn=False,
        tool_specs=[],
        investigation=investigation,
    )
    assert decisive_id in [item.evidence_id for item in context.packet.evidence_index]
    assert context.packet.investigation.counterevidence_ids == [decisive_id]
    if context.packet.omissions:
        assert all(item.affected for item in context.packet.omissions)


def test_playbooks_route_by_error_and_fall_back_to_generic() -> None:
    keyerror = make_event(symptom="KeyError: 'items'")
    timeout = make_event(symptom="command timed out after 30s", goal="恢复工具调用")
    handoff = make_event(symptom="new session lost the handoff context", goal="继续上一次调查")
    ambiguous = make_event(symptom="something odd happened", goal="看看")
    assert route_playbook(keyerror).playbook_id == "structured-output-mismatch"
    assert route_playbook(timeout).playbook_id == "tool-failure-and-recovery"
    assert route_playbook(handoff).playbook_id == "context-and-handoff-gap"
    assert route_playbook(ambiguous).playbook_id == "generic-triage"
    assert len(PLAYBOOK_CATALOG_HASH) == 64
    assert catalog_manifest()["playbook_catalog_hash"] == PLAYBOOK_CATALOG_HASH
    rendered = render_playbook(get_playbook("structured-output-mismatch"))
    assert "first question" in rendered.lower() or "第一" in rendered


def test_playbook_does_not_change_tool_permissions(tmp_path: Path) -> None:
    from agent_triage.lifecycle import LifecycleEngine, LifecycleEvent, PolicyCheck, allow, deny

    executor, _, _, _, _, _ = make_tool_executor(tmp_path)
    engine = LifecycleEngine(
        [
            PolicyCheck("scope", LifecycleEvent.BEFORE_TOOL, "R01", lambda inp: allow("R01")),
            PolicyCheck("deny", LifecycleEvent.BEFORE_TOOL, "R09", lambda inp: deny("R09", "tool_not_allowed", "no")),
        ]
    )
    executor.policy = engine
    result = executor.execute("read_file", {"path": "logs/failure.log"}, call_id="c1", turn=0)
    assert result.status.value == "error"
    assert result.error_code == "policy_denied:tool_not_allowed"


def test_failed_attempt_ledger_survives_interruption(tmp_path: Path) -> None:
    scenario = make_scenario(
        [
            tool_step(("read_file", {"path": "logs/missing.log", "start_line": 1})),
            tool_step(("read_file", {"path": "logs/failure.log", "start_line": 1, "max_lines": 20})),
            tool_step(("read_file", {"path": "src/check_reader.py", "start_line": 1, "max_lines": 20})),
            tool_step(("read_file", {"path": "data/checks.json", "start_line": 1, "max_lines": 20})),
            final_step(base_analysis()),
        ]
    )
    from agent_triage.errors import RunInterrupted
    from agent_triage.pipeline import resume_analysis
    from agent_triage.config import Config

    with pytest.raises(RunInterrupted):
        run_replay(tmp_path, scenario=scenario, interrupt_after_rounds=1)
    output = tmp_path / "runs"
    run_dir = next(path for path in output.iterdir() if path.name.startswith("run-"))
    investigation = json.loads((run_dir / "investigation.json").read_text(encoding="utf-8"))
    assert investigation["attempt_ledger"], "a failed tool call must enter the attempt ledger"
    assert any(entry["status"].startswith("error") for entry in investigation["attempt_ledger"])
    store = RunStore.load(output, run_dir.name)
    resumed = resume_analysis(store=store, config=Config(), replay_scenario=scenario)
    assert resumed.run_id == store.run_id
    context = json.loads((run_dir / "context.json").read_text(encoding="utf-8"))
    assert context["investigation"]["attempt_ledger"]

def test_mixed_claim_support_is_not_rounded_up_to_pass() -> None:
    review = SupportReview(
        status=CheckStatus.PASS,
        reviewer="owner",
        method="manual-read",
        scope="one claim only",
        reviewed_at=datetime.now(timezone.utc),
    )
    payload = base_analysis()
    payload["facts"] = [
        {
            "text": "已复核的主张",
            "evidence_ids": ["ev-0123456789ab"],
            "support_review": review.model_dump(mode="json"),
        },
        {
            "text": "未复核的主张",
            "evidence_ids": ["ev-0123456789ac"],
        },
    ]
    payload["hypotheses"] = []
    analysis = Analysis.model_validate(payload)
    assert support_review_label(analysis) == "not_run"
