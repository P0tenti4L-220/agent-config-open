from __future__ import annotations

import json
from pathlib import Path

from agent_triage.context import ContextBuilder, estimate_tokens
from agent_triage.contracts import ContextLimits, Lesson, LessonStatus
from agent_triage.models import RemainingBudget
from agent_triage.store import BudgetState, LessonStore, RunStore
from agent_triage.tools import tool_specs
from helpers import (
    base_analysis,
    final_step,
    make_event,
    make_scenario,
    make_tool_executor,
    run_replay,
    tool_step,
)


def _build(tmp_path: Path, *, limits: ContextLimits | None = None, conversation=None):
    executor, store, evidence, budget, project, output = make_tool_executor(tmp_path)
    executor.execute(
        "read_file",
        {"path": "logs/failure.log", "start_line": 1, "max_lines": 20},
        call_id="call-1",
        turn=0,
    )
    executor.execute(
        "read_file",
        {"path": "src/check_reader.py", "start_line": 1, "max_lines": 20},
        call_id="call-2",
        turn=0,
    )
    lessons = LessonStore(output).search("KeyError items checks", 3)
    builder = ContextBuilder(
        limits or ContextLimits(),
        versions={"package": "0.1.0", "context": "v1", "rules": "v0.1.0"},
    )
    context = builder.build(
        run_id=store.run_id,
        event=make_event(),
        evidence=evidence.all(),
        lessons=lessons,
        trace=store.read_trace(),
        conversation=conversation or [],
        budget_state=budget.snapshot(),
        turn=1,
        is_final_turn=False,
        tool_specs=tool_specs(),
    )
    return context, builder, evidence, store, project


def test_context_budget_counts_tools_and_protocol(tmp_path: Path) -> None:
    context, builder, evidence, store, _ = _build(tmp_path)
    packet = context.packet
    assert packet.budget.total_used_tokens <= packet.budget.total_limit_tokens == 12_000
    layers = {layer.layer: layer for layer in packet.budget.layers}
    assert layers["tool_definitions"].used_tokens > 0
    assert layers["current_evidence"].used_tokens > 0
    assert any(record.layer == "protocol" for record in packet.selections)
    assert packet.versions["package"] == "0.1.0"
    assert all(record.count_mode.value == "estimated" for record in packet.selections)
    assert "payload" in context.system
    assert packet.evidence_index
    assert all(item.evidence_id.startswith("ev-") for item in packet.evidence_index)


def test_long_evidence_is_truncated_with_visible_reason(tmp_path: Path) -> None:
    executor, store, evidence, budget, project, output = make_tool_executor(tmp_path)
    long_text = "\n".join(f"line {index} " + "x" * 200 for index in range(1, 400))
    (project / "logs" / "huge.log").write_text(long_text, encoding="utf-8")
    executor.execute(
        "read_file",
        {"path": "logs/huge.log", "start_line": 1, "max_lines": 120},
        call_id="call-1",
        turn=0,
    )
    limits = ContextLimits(current_evidence_tokens=200)
    builder = ContextBuilder(limits, versions={"package": "0.1.0"})
    context = builder.build(
        run_id=store.run_id,
        event=make_event(),
        evidence=evidence.all(),
        lessons=[],
        trace=store.read_trace(),
        conversation=[],
        budget_state=budget.snapshot(),
        turn=1,
        is_final_turn=False,
        tool_specs=tool_specs(),
    )
    decisions = {record.decision.value for record in context.packet.selections if record.layer == "current_evidence"}
    assert "truncated" in decisions or "omitted" in decisions
    assert context.packet.budget.total_used_tokens <= context.packet.budget.total_limit_tokens
    assert "snippet truncated for context budget" in context.system


def test_stale_evidence_is_excluded_from_current_context(tmp_path: Path) -> None:
    context, builder, evidence, store, project = _build(tmp_path)
    assert "payload" in context.system
    project.joinpath("src/check_reader.py").write_text("# changed after capture\n", encoding="utf-8")
    stale = evidence.refresh_staleness(project)
    assert stale
    context2 = builder.build(
        run_id=store.run_id,
        event=make_event(),
        evidence=evidence.all(),
        lessons=[],
        trace=store.read_trace(),
        conversation=[],
        budget_state=BudgetState(),
        turn=2,
        is_final_turn=False,
        tool_specs=tool_specs(),
    )
    stale_records = [
        record
        for record in context2.packet.selections
        if record.layer == "current_evidence" and record.decision.value == "omitted" and "stale" in record.reason
    ]
    assert stale_records
    assert "changed after capture" not in context2.system


def test_candidate_lessons_are_labelled_reference_not_fact(tmp_path: Path) -> None:
    executor, store, evidence, budget, project, output = make_tool_executor(tmp_path)
    lesson = Lesson(
        lesson_id="lesson-0123456789ab",
        status=LessonStatus.CANDIDATE,
        symptom_pattern="KeyError items when reading checks",
        applicability="python json readers",
        guidance="add an explicit adapter",
        event_ids=["old-event"],
    )
    LessonStore(output).upsert(lesson)
    executor.execute(
        "read_file",
        {"path": "src/check_reader.py", "start_line": 1, "max_lines": 20},
        call_id="call-1",
        turn=0,
    )
    builder = ContextBuilder(ContextLimits(), versions={"package": "0.1.0"})
    context = builder.build(
        run_id=store.run_id,
        event=make_event(),
        evidence=evidence.all(),
        lessons=LessonStore(output).search("KeyError items checks", 3),
        trace=store.read_trace(),
        conversation=[],
        budget_state=budget.snapshot(),
        turn=1,
        is_final_turn=False,
        tool_specs=tool_specs(),
    )
    assert context.packet.lessons[0].status == LessonStatus.CANDIDATE
    assert context.packet.lessons[0].verified is False
    assert "candidate lesson; not a fact" in {
        record.reason for record in context.packet.selections if record.layer == "history"
    }
    assert "status=candidate" in context.system
    assert "verified=False" in context.system


def test_missing_evidence_stays_partial_and_is_corrected(tmp_path: Path) -> None:
    scenario = make_scenario(
        [
            tool_step(("read_file", {"path": "logs/failure.log", "start_line": 1, "max_lines": 20})),
            final_step(
                base_analysis(
                    facts=[
                        {
                            "text": "data/checks.json contains checks",
                            "evidence_ids": ["@path:data/checks.json"],
                        }
                    ]
                )
            ),
            final_step(
                base_analysis(
                    facts=[
                        {
                            "text": "日志只显示读取 payload['items'] 时抛出 KeyError",
                            "evidence_ids": ["@path:logs/failure.log"],
                        }
                    ],
                    hypotheses=[],
                    missing_evidence=["data/checks.json was not read; its content is unknown"],
                    analysis_status="partial",
                    stop_reason="insufficient_evidence",
                )
            ),
        ]
    )
    result, project, output = run_replay(tmp_path, scenario=scenario)
    assert result.analysis.analysis_status.value == "partial"
    assert result.analysis.stop_reason.value in {"insufficient_evidence", "tool_unavailable"}
    assert any("data/checks.json" in item for item in result.analysis.missing_evidence)
    assert all("checks" not in fact.text.lower() or "payload" in fact.text.lower() for fact in result.analysis.facts)
    trace = (output / result.run_id / "trace.jsonl").read_text(encoding="utf-8")
    assert "invalid_output" in trace
    assert result.budget_used["corrections_used"] == 1
    checkpoint = RunStore.load(output, result.run_id).load_checkpoint()
    assert checkpoint is not None
    roles = [message["role"] for message in checkpoint.conversation]
    assert roles == ["assistant", "user", "assistant", "user"]
    assert all(message["content"] for message in checkpoint.conversation)


def test_injection_text_and_illegal_tool_requests_are_not_trusted(tmp_path: Path) -> None:
    scenario = make_scenario(
        [
            tool_step(("read_file", {"path": "logs/failure.log", "start_line": 1, "max_lines": 20})),
            tool_step(("read_file", {"path": "notes.md", "start_line": 1, "max_lines": 20})),
            tool_step(
                ("shell", {"command": "cat .env"}),
                ("read_file", {"path": ".env"}),
                ("read_file", {"path": "../../etc/passwd"}),
            ),
            final_step(
                base_analysis(
                    facts=[
                        {
                            "text": "日志显示 payload['items'] 触发 KeyError",
                            "evidence_ids": ["@path:logs/failure.log"],
                        }
                    ],
                    hypotheses=[],
                    missing_evidence=["no test was executed"],
                    analysis_status="partial",
                    stop_reason="insufficient_evidence",
                )
            ),
        ]
    )
    result, project, output = run_replay(tmp_path, scenario=scenario)
    run_dir = output / result.run_id
    trace = [json.loads(line) for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    tool_trace = [record for record in trace if record["kind"] == "tool"]
    assert any(record["error_code"] == "unknown_tool" for record in tool_trace)
    assert any(record["error_code"] == "sensitive_path" for record in tool_trace)
    assert any(record["error_code"] == "invalid_arguments" for record in tool_trace)
    evidence_index = json.loads((run_dir / "evidence" / "index.json").read_text(encoding="utf-8"))
    paths = {item["path"] for item in evidence_index["evidence"]}
    assert ".env" not in paths
    assert "notes.md" in paths
    assert result.analysis.analysis_status.value == "partial"


def test_protocol_compaction_keeps_complete_pairs(tmp_path: Path) -> None:
    conversation = [{"role": "user", "content": [{"type": "text", "text": "start"}]}]
    for index in range(10):
        conversation.append(
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": f"call-{index}", "name": "read_file", "input": {"path": "x"}}],
            }
        )
        conversation.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": f"call-{index}", "content": "{}"}],
            }
        )
    context, _, _, _, _ = _build(
        tmp_path,
        limits=ContextLimits(
            constraints_tokens=300,
            tool_definitions_tokens=800,
            task_facts_tokens=300,
            current_evidence_tokens=300,
            history_tokens=100,
            working_state_tokens=200,
            total_input_tokens=2000,
        ),
        conversation=conversation,
    )
    messages = context.messages
    assert messages[0]["role"] == "user"
    assert messages[-1]["role"] == "user"
    for index, message in enumerate(messages):
        if message["role"] == "assistant":
            tool_id = message["content"][0]["id"]
            assert messages[index + 1]["content"][0]["tool_use_id"] == tool_id
    assert len(messages) < len(conversation)
    assert context.packet.budget.total_used_tokens <= context.packet.budget.total_limit_tokens