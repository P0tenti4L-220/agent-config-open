from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agent_triage.budget import Budget
from agent_triage.config import Config
from agent_triage.contracts import BudgetLimits, ToolStatus
from agent_triage.errors import RunInterrupted, ValidationFailure
from agent_triage.models import FinalResponse, ToolRequests
from agent_triage.pipeline import resume_analysis
from agent_triage.store import RunStore
from agent_triage.tools import IssueSearchOutcome
from helpers import (
    base_analysis,
    final_step,
    make_event,
    make_scenario,
    make_tool_executor,
    run_replay,
    tool_step,
)


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class FlakyGithub:
    available = True
    unavailable_reason = None

    def __init__(self) -> None:
        self.calls = 0

    def search_issues(self, query: str, limit: int) -> IssueSearchOutcome:
        self.calls += 1
        if self.calls == 1:
            from agent_triage.errors import ToolFailure

            raise ToolFailure("mcp_timeout", "transient timeout", retryable=True)
        return IssueSearchOutcome(
            issues=[
                {
                    "issue_ref": "example/repo#1",
                    "number": 1,
                    "title": "synthetic issue",
                    "state": "open",
                    "html_url": "https://example.invalid/issues/1",
                    "updated_at": "2026-10-24T20:00:00Z",
                    "body": None,
                }
            ],
            total_count=1,
            complete=True,
            query_scope="repo:example/repo",
        )

    def read_issue(self, issue_ref: str):  # pragma: no cover - not used
        raise NotImplementedError


class AdvanceAfterFirstModel:
    provider = "replay"

    def __init__(self, inner, clock: FakeClock) -> None:
        self.inner = inner
        self.clock = clock
        self.position = 0

    def next(self, context, tool_specs, remaining_budget):
        response = self.inner.next(context, tool_specs, remaining_budget)
        self.position = getattr(self.inner, "position", 0)
        if isinstance(response, ToolRequests):
            self.clock.now = 10_000
        return response


def test_duplicate_call_without_new_evidence_stops_path(tmp_path: Path) -> None:
    scenario = make_scenario(
        [
            tool_step(("read_file", {"path": "logs/failure.log", "start_line": 1, "max_lines": 20})),
            tool_step(("read_file", {"path": "logs/failure.log", "start_line": 1, "max_lines": 20})),
            final_step(base_analysis()),
        ]
    )
    result, _, output = run_replay(tmp_path, scenario=scenario)
    assert result.analysis.stop_reason.value == "insufficient_evidence"
    assert result.analysis.analysis_status.value in {"failed", "partial"}
    assert result.budget_used["tool_attempts_used"] == 2
    trace = (output / result.run_id / "trace.jsonl").read_text(encoding="utf-8")
    assert "read_file" in trace


def test_read_only_tool_retries_once_and_counts_both_attempts(tmp_path: Path) -> None:
    github = FlakyGithub()
    executor, store, evidence, budget, _, _ = make_tool_executor(tmp_path, github=github)
    result = executor.execute(
        "github_search_issues",
        {"query": "KeyError items", "limit": 3},
        call_id="call-1",
        turn=0,
    )
    assert result.status == ToolStatus.OK
    assert github.calls == 2
    assert budget.state.tool_attempts_used == 2
    trace = store.read_trace()
    assert len(trace) == 2
    assert [record.attempt for record in trace] == [1, 2]
    assert result.evidence_refs


def test_tool_budget_exhaustion_stops_before_more_work(tmp_path: Path) -> None:
    limits = BudgetLimits(max_tool_attempts=1)
    scenario = make_scenario(
        [
            tool_step(("read_file", {"path": "logs/failure.log", "start_line": 1, "max_lines": 20})),
            tool_step(("read_file", {"path": "src/check_reader.py", "start_line": 1, "max_lines": 20})),
            final_step(base_analysis()),
        ]
    )
    result, _, output = run_replay(tmp_path, scenario=scenario, config=Config(budgets=limits))
    assert result.analysis.stop_reason.value == "budget_exhausted"
    assert result.budget_used["tool_attempts_used"] == 1
    assert result.budget_used["model_requests_used"] == 2
    trace = [
        json.loads(line)
        for line in (output / result.run_id / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(record.get("error_code") == "tool_attempt_limit" for record in trace)


def test_budget_refuses_request_after_deadline() -> None:
    clock = FakeClock(0)
    budget = Budget(BudgetLimits(total_seconds=10), clock=clock)
    assert budget.can_start_model_request() == (True, None)
    clock.now = 11
    assert budget.can_start_model_request() == (False, "total_time_exhausted")
    assert budget.remaining_seconds() == 0


def test_runner_does_not_request_model_after_deadline(tmp_path: Path) -> None:
    from agent_triage.context import ContextBuilder
    from agent_triage.runner import Runner

    clock = FakeClock(0)
    scenario = make_scenario(
        [
            tool_step(("read_file", {"path": "logs/failure.log", "start_line": 1, "max_lines": 20})),
            final_step(base_analysis()),
        ]
    )
    result, project, output = run_replay(tmp_path, scenario=scenario)
    # Build the same run again with a clock-advancing wrapper to check the loop guard.
    from agent_triage.config import config_fingerprint
    from agent_triage.models import ReplayModel
    from agent_triage.store import EvidenceStore, LessonStore, RegistryStore, RunStore
    from agent_triage.tools import Workspace

    event = make_event(event_id="deadline-event")
    store = RunStore.create(
        tmp_path / "deadline-runs",
        event_payload=event.model_dump(mode="json"),
        event_id=event.event_id,
        project_id=event.project_id,
        project_root=project,
        provider="replay",
        model="replay",
        config_fingerprint=config_fingerprint(Config()),
    )
    RegistryStore(tmp_path / "deadline-runs").register(
        event_id=event.event_id,
        project_id=event.project_id,
        fingerprint="f" * 64,
        fingerprint_algorithm="fp1",
        symptom=event.symptom,
        run_id=store.run_id,
        provider="replay",
    )
    model = AdvanceAfterFirstModel(ReplayModel(scenario), clock)
    runner = Runner(
        store=store,
        event=event,
        config=Config(),
        model=model,
        context_builder=ContextBuilder(Config().context, versions={"package": "test"}),
        evidence_store=EvidenceStore(store),
        lesson_store=LessonStore(tmp_path / "deadline-runs"),
        workspace=Workspace(project),
        clock=clock,
    )
    result = runner.run()
    assert result.stop_reason.value == "budget_exhausted"
    assert result.budget_used["model_requests_used"] == 1


def test_missing_final_output_becomes_failed_after_one_correction(tmp_path: Path) -> None:
    scenario = make_scenario(
        [tool_step(("read_file", {"path": "logs/failure.log", "start_line": 1, "max_lines": 20}))]
    )
    result, _, output = run_replay(tmp_path, scenario=scenario)
    assert result.analysis.analysis_status.value == "failed"
    assert result.analysis.stop_reason.value == "invalid_output"
    assert result.budget_used["model_requests_used"] == 3
    assert result.budget_used["corrections_used"] == 1
    trace = (output / result.run_id / "trace.jsonl").read_text(encoding="utf-8")
    assert "invalid_output" in trace


def test_resume_preserves_budget_and_does_not_repeat_completed_tool_calls(tmp_path: Path) -> None:
    scenario = make_scenario(
        [
            tool_step(("read_file", {"path": "logs/failure.log", "start_line": 1, "max_lines": 20})),
            tool_step(("read_file", {"path": "src/check_reader.py", "start_line": 1, "max_lines": 20})),
            tool_step(("read_file", {"path": "data/checks.json", "start_line": 1, "max_lines": 20})),
            final_step(base_analysis()),
        ]
    )
    with pytest.raises(RunInterrupted):
        run_replay(tmp_path, scenario=scenario, interrupt_after_rounds=1)
    output = tmp_path / "runs"
    run_dir = next(path for path in output.iterdir() if path.name.startswith("run-"))
    store = RunStore.load(output, run_dir.name)
    checkpoint = store.load_checkpoint()
    assert checkpoint is not None
    assert checkpoint.budget.model_requests_used == 1
    assert checkpoint.budget.tool_attempts_used == 1
    assert store.meta.interruption == "checkpointed_after_complete_tool_round"

    result = resume_analysis(store=store, config=Config(), replay_scenario=scenario)
    assert result.analysis.analysis_status.value == "partial"
    assert result.budget_used["model_requests_used"] == 4
    assert result.budget_used["tool_attempts_used"] == 3
    assert result.budget_used["elapsed_ms"] >= checkpoint.budget.elapsed_ms
    evidence = json.loads((run_dir / "evidence" / "index.json").read_text(encoding="utf-8"))
    assert len(evidence["evidence"]) == 3
    assert len(store.read_trace()) >= 5


def test_resume_rejects_changed_sources_unless_explicitly_accepted(tmp_path: Path) -> None:
    scenario = make_scenario(
        [
            tool_step(("read_file", {"path": "logs/failure.log", "start_line": 1, "max_lines": 20})),
            tool_step(("read_file", {"path": "src/check_reader.py", "start_line": 1, "max_lines": 20})),
            tool_step(("read_file", {"path": "data/checks.json", "start_line": 1, "max_lines": 20})),
            final_step(
                base_analysis(
                    facts=[
                        {
                            "text": "读取器访问 items，而输入数据顶层是 checks",
                            "evidence_ids": ["@path:src/check_reader.py", "@path:data/checks.json"],
                        }
                    ]
                )
            ),
        ]
    )
    with pytest.raises(RunInterrupted):
        run_replay(tmp_path, scenario=scenario, interrupt_after_rounds=1)
    output = tmp_path / "runs"
    run_dir = next(path for path in output.iterdir() if path.name.startswith("run-"))
    store = RunStore.load(output, run_dir.name)
    project = store.project_root()
    project.joinpath("logs/failure.log").write_text("changed after capture\n", encoding="utf-8")
    with pytest.raises(ValidationFailure, match="source files changed"):
        resume_analysis(store=store, config=Config(), replay_scenario=scenario)

    result = resume_analysis(
        store=store,
        config=Config(),
        replay_scenario=scenario,
        accept_changed_sources=True,
    )
    assert result.analysis.analysis_status.value in {"partial", "complete"}
    evidence = json.loads((run_dir / "evidence" / "index.json").read_text(encoding="utf-8"))
    assert any(item["stale"] for item in evidence["evidence"])

def test_resume_rejects_changed_config_fingerprint(tmp_path: Path) -> None:
    scenario = make_scenario(
        [
            tool_step(("read_file", {"path": "logs/failure.log", "start_line": 1, "max_lines": 20})),
            final_step(base_analysis()),
        ]
    )
    with pytest.raises(RunInterrupted):
        run_replay(tmp_path, scenario=scenario, interrupt_after_rounds=1)
    output = tmp_path / "runs"
    run_dir = next(path for path in output.iterdir() if path.name.startswith("run-"))
    store = RunStore.load(output, run_dir.name)
    changed = Config(budgets=BudgetLimits(max_tool_attempts=5))
    with pytest.raises(ValidationFailure, match="configuration differs"):
        resume_analysis(store=store, config=changed, replay_scenario=scenario)
