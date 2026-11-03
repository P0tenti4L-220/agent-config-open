"""Information-loss regressions against actual submitted request snapshots."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_triage.context import ContextBuilder, estimate_tokens
from agent_triage.contracts import ContextLimits, InvestigationState
from agent_triage.errors import ValidationFailure
from agent_triage.request_views import RequestViewStore, verify_display_records
from agent_triage.tools import tool_specs
from helpers import make_event, make_tool_executor


def _context(store, evidence, budget, event, *, limit=700, investigation=None, conversation=None, working=1500):
    return ContextBuilder(ContextLimits(current_evidence_tokens=limit, working_state_tokens=working),
                          versions={"prompt_version": "prompt-v3"}).build(
        run_id=store.run_id, event=event, evidence=evidence.all(), lessons=[],
        trace=store.read_trace(), conversation=conversation or [], budget_state=budget.snapshot(),
        turn=2, is_final_turn=False, tool_specs=tool_specs(), investigation=investigation,
    )


def test_middle_exception_and_conditions_enter_the_actual_request(tmp_path: Path) -> None:
    event = make_event(symptom="RevisionConflict", error_files=["logs/window.log"], related_files=[])
    executor, store, evidence, budget, project, _ = make_tool_executor(tmp_path, event=event)
    lines = [f"startup-{n}: " + "x" * 100 for n in range(50)]
    lines[24:27] = ["only revision v2 may fail; this is not an unconditional rule",
                    "RevisionConflict: expected revision 42", "except when force=true; keep this exception"]
    original = "\n".join(lines) + "\n"
    (project / "logs/window.log").write_text(original)
    result = executor.execute("read_file", {"path": "logs/window.log", "max_lines": 120}, call_id="read-1", turn=0)
    item = evidence.require(result.evidence_refs[0])
    investigation = InvestigationState(active_question="Why RevisionConflict for revision 42?",
                                       required_evidence_ids=[item.evidence_id], next_action_reason="inspect exception")
    context = _context(store, evidence, budget, event, investigation=investigation)
    windows = context.packet.evidence_windows
    assert windows and any("RevisionConflict" in w.text for w in windows)
    assert any("only revision v2" in w.text and "except when force=true" in w.text for w in windows)
    assert all(w.text.encode() == item.snippet.encode()[w.start_byte:w.end_byte] for w in windows)
    assert any(w.start_byte > 1000 for w in windows)
    assert any(o.evidence_id == item.evidence_id and o.state == "partially_displayed" for o in context.omissions)
    views = RequestViewStore(store.dir)
    view = views.begin(run_id=store.run_id, turn=2, provider="replay", system=context.system,
                       messages=context.messages, evidence=evidence.all(), windows=windows)
    assert views.confirmed_displayed(view) == []
    confirmed = views.mark_response_received(view.request_id)
    displays = views.confirmed_displayed(confirmed)
    assert displays and all(d.schema_version == 3 for d in displays)
    assert verify_display_records(displays, [confirmed], evidence.all()) == ([], "request_view")
    assert context.input_tokens == (estimate_tokens(context.system)
                                    + estimate_tokens(json.dumps(context.messages, ensure_ascii=False))
                                    + estimate_tokens(json.dumps([s.model_dump(mode="json") for s in tool_specs()],
                                                                 ensure_ascii=False, sort_keys=True, separators=(",", ":"))))


def test_two_distant_windows_never_prove_the_gap_or_forged_offsets(tmp_path: Path) -> None:
    event = make_event(symptom="AlphaFailure BetaLimit", related_files=[])
    executor, store, evidence, budget, project, _ = make_tool_executor(tmp_path, event=event)
    lines = [f"filler-{n}: " + "q" * 120 for n in range(45)]
    lines[8] = "AlphaFailure: version 2 only"
    lines[22] = "UNSHOWN_MIDDLE_CANARY"
    lines[36] = "BetaLimit: not permitted unless explicitly configured"
    (project / "logs/failure.log").write_text("\n".join(lines))
    result = executor.execute("read_file", {"path": "logs/failure.log", "max_lines": 120}, call_id="read-1", turn=0)
    item = evidence.require(result.evidence_refs[0])
    context = _context(store, evidence, budget, event, limit=850)
    windows = context.packet.evidence_windows
    assert len(windows) == 2
    assert windows[0].end_byte < windows[1].start_byte
    assert "UNSHOWN_MIDDLE_CANARY" not in context.system
    store_views = RequestViewStore(store.dir)
    view = store_views.begin(run_id=store.run_id, turn=2, provider="replay", system=context.system,
                             messages=context.messages, evidence=[item], windows=windows)
    confirmed = store_views.mark_response_received(view.request_id)
    records = store_views.confirmed_displayed(confirmed)
    assert len(records) == 2
    assert all("UNSHOWN_MIDDLE_CANARY" not in r.window.text for r in records)
    forged_window = windows[0].model_copy(update={"start_byte": windows[0].start_byte + 1,
                                                "end_byte": windows[0].end_byte + 1})
    forged = records[0].model_copy(update={"window": forged_window})
    assert verify_display_records([forged], [confirmed], [item])[0]
    # Registering a window whose body is absent cannot create display proof.
    absent = store_views.begin(run_id=store.run_id, turn=3, provider="replay", system=item.evidence_id,
                              messages=[], evidence=[item], windows=windows)
    assert absent.evidence_views == []


def test_retained_tool_body_is_not_duplicated_in_evidence_layer(tmp_path: Path) -> None:
    executor, store, evidence, budget, project, _ = make_tool_executor(tmp_path)
    (project / "logs/failure.log").write_text("UNIQUE_SOURCE_BODY: KeyError on version 2\n")
    result = executor.execute("read_file", {"path": "logs/failure.log"}, call_id="call-1", turn=0)
    conversation = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "call-1", "name": "read_file", "input": {"path": "logs/failure.log"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": result.model_dump_json()}]},
    ]
    context = _context(store, evidence, budget, make_event(), conversation=conversation)
    assert "UNIQUE_SOURCE_BODY" in json.dumps(context.messages)
    assert "UNIQUE_SOURCE_BODY" not in context.system
    assert not context.packet.evidence_windows
    assert any("already present" in s.reason for s in context.packet.selections)


def test_unresolved_early_failure_survives_many_successes_and_large_protocol(tmp_path: Path) -> None:
    from agent_triage.contracts import AttemptLedgerEntry, BudgetLimits

    executor, store, evidence, budget, _, _ = make_tool_executor(tmp_path, limits=BudgetLimits(max_tool_attempts=30))
    executor.execute("read_file", {"path": "missing-required.log"}, call_id="old-failure", turn=0)
    for n in range(14):
        executor.execute("search_workspace", {"query": "no-hit", "paths": ["src"]}, call_id=f"later-{n}", turn=n + 1)
    investigation = InvestigationState(
        active_question="Where is missing-required.log?", needed_materials=["missing-required.log"],
        next_action_reason="provide the missing source; no repair was executed",
        attempt_ledger=[AttemptLedgerEntry(schema_version=2, turn=n + 1, action="search_workspace:no-hit",
                                           status="ok", new_information=False, call_id=f"later-{n}",
                                           source_ref=f"trace.jsonl:{n + 2}") for n in range(14)],
    )
    conversation = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "large", "name": "read_file", "input": {"path": "x"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "large", "content": "x" * 10000}]},
    ]
    context = _context(store, evidence, budget, make_event(), investigation=investigation,
                       conversation=conversation, working=500)
    assert any(a.call_id == "old-failure" and a.reason == "missing" for a in context.packet.failed_attempts)
    assert "old-failure" in context.system and "not_run" in context.system
    assert "failed=1 no_progress=14" in context.system
    assert context.packet.state_projection["needed_materials"] == ["missing-required.log"]
    assert len(context.packet.state_projection["no_progress_attempts"]) == 14
    assert len(context.messages) == 1
    assert any(s.layer == "working_state" and s.decision.value == "omitted" for s in context.packet.selections)


def test_too_small_critical_state_budget_fails_explicitly(tmp_path: Path) -> None:
    _, store, evidence, budget, _, _ = make_tool_executor(tmp_path)
    with pytest.raises(ValidationFailure, match="critical working state"):
        _context(store, evidence, budget, make_event(), working=0)


def test_counterevidence_and_required_sources_are_visible_or_explicitly_omitted(tmp_path: Path) -> None:
    event = make_event(symptom="CompatFailure", error_files=[], related_files=[])
    executor, store, evidence, budget, project, _ = make_tool_executor(tmp_path, event=event)
    for name, text in [("support.txt", "CompatFailure happens only on revision 2.\n"),
                       ("counter.txt", "CompatFailure does not happen when the legacy flag is set.\n")]:
        (project / name).write_text(text)
        executor.execute("read_file", {"path": name}, call_id=name, turn=0)
    support, counter = evidence.all()
    investigation = InvestigationState(active_question="Which condition causes CompatFailure?",
                                       required_evidence_ids=[support.evidence_id],
                                       counterevidence_ids=[counter.evidence_id], next_action_reason="compare both versions")
    context = _context(store, evidence, budget, event, limit=180, investigation=investigation)
    shown = {w.evidence_id for w in context.packet.evidence_windows}
    omitted = {o.evidence_id for o in context.omissions}
    assert {support.evidence_id, counter.evidence_id} <= shown | omitted
    assert counter.evidence_id in shown
    assert context.packet.state_projection["counterevidence_ids"] == [counter.evidence_id]
    assert context.packet.state_projection["omissions"]


def test_many_pages_from_one_source_do_not_starve_the_earlier_relevant_page(tmp_path: Path) -> None:
    from agent_triage.contracts import BudgetLimits

    event = make_event(symptom="CriticalConflict", error_files=["logs/pages.log"], related_files=[])
    executor, store, evidence, budget, project, _ = make_tool_executor(
        tmp_path, event=event, limits=BudgetLimits(max_tool_attempts=40))
    lines = ["CriticalConflict: preserve this earlier observation"] + [f"later harmless line {n}" for n in range(29)]
    (project / "logs/pages.log").write_text("\n".join(lines) + "\n")
    for line in range(1, 31):
        executor.execute("read_file", {"path": "logs/pages.log", "start_line": line, "max_lines": 1},
                         call_id=f"page-{line}", turn=line)
    important = evidence.all()[0]
    investigation = InvestigationState(active_question="Why CriticalConflict?",
                                       required_evidence_ids=evidence.ids(), next_action_reason="inspect the conflict")
    context = _context(store, evidence, budget, event, limit=400, investigation=investigation)
    assert any(w.evidence_id == important.evidence_id and "CriticalConflict" in w.text
               for w in context.packet.evidence_windows)
    assert context.packet.budget.total_used_tokens <= 12000
