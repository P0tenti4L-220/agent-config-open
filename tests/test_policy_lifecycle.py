from __future__ import annotations

import time
from pathlib import Path

import pytest

from agent_triage.context import PROMPT_HASH
from agent_triage.lifecycle import (
    Decision,
    HookResult,
    LifecycleEngine,
    LifecycleEvent,
    PolicyCheck,
    allow,
    deny,
)
from agent_triage.policy import POLICY_HASH, RULES, policy_prompt_text, rule_ids
from agent_triage.store import RunStore
from helpers import make_event, make_scenario, make_tool_executor, run_replay, tool_step, final_step, base_analysis


class CountingGithub:
    available = True
    unavailable_reason = None

    def __init__(self) -> None:
        self.calls = 0

    def search_issues(self, query: str, limit: int):
        self.calls += 1
        raise AssertionError("a denied GitHub query must never reach the backend")

    def read_issue(self, issue_ref: str):  # pragma: no cover - not used
        raise AssertionError("a denied GitHub read must never reach the backend")


def test_policy_ids_hashes_and_prompt_are_stable() -> None:
    assert rule_ids() == [f"R0{index}" for index in range(1, 10)]
    assert len(POLICY_HASH) == 64
    assert len(PROMPT_HASH) == 64
    prompt = policy_prompt_text()
    for rule in RULES:
        assert rule.rule_id in prompt
    assert POLICY_HASH == __import__("agent_triage.policy", fromlist=["policy_hash"]).POLICY_HASH


def test_versions_are_frozen_into_context_and_checkpoint(tmp_path: Path) -> None:
    result, _, output = run_replay(tmp_path)
    run_dir = output / result.run_id
    versions = __import__("json").loads((run_dir / "versions.json").read_text(encoding="utf-8"))
    assert versions["policy_version"] == "policy-v3"
    assert len(versions["policy_hash"]) == 64
    assert versions["prompt_version"] == "prompt-v4"
    assert versions["tool_schema_version"] == "tools-v4"
    assert versions["playbook_id"]
    assert versions["playbook_hash"]
    checkpoint = __import__("json").loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["policy_versions"]["policy_hash"] == versions["policy_hash"]
    context = __import__("json").loads((run_dir / "context.json").read_text(encoding="utf-8"))
    assert context["versions"]["policy_hash"] == versions["policy_hash"]


def test_resume_rejects_changed_policy_version(tmp_path: Path) -> None:
    scenario = make_scenario(
        [
            tool_step(("read_file", {"path": "logs/failure.log", "start_line": 1, "max_lines": 20})),
            final_step(base_analysis()),
        ]
    )
    from agent_triage.errors import RunInterrupted, ValidationFailure
    from agent_triage.pipeline import resume_analysis

    with pytest.raises(RunInterrupted):
        run_replay(tmp_path, scenario=scenario, interrupt_after_rounds=1)
    output = tmp_path / "runs"
    run_dir = next(path for path in output.iterdir() if path.name.startswith("run-"))
    store = RunStore.load(output, run_dir.name)
    payload = store.read_json("checkpoint.json")
    payload["policy_versions"]["policy_hash"] = "0" * 64
    store.write_json("checkpoint.json", payload)
    with pytest.raises(ValidationFailure, match="versions changed"):
        resume_analysis(store=store, config=__import__("agent_triage.config", fromlist=["Config"]).Config(), replay_scenario=scenario)


def test_before_tool_exception_blocks_execution(tmp_path: Path) -> None:
    executor, _, _, budget, _, _ = make_tool_executor(tmp_path)
    calls = {"dispatch": 0}

    def broken(inp):
        raise RuntimeError("synthetic hook failure")

    engine = LifecycleEngine(
        [PolicyCheck("broken", LifecycleEvent.BEFORE_TOOL, "R01", broken)],
    )
    executor.policy = engine
    original = executor._dispatch

    def counting(*args, **kwargs):
        calls["dispatch"] += 1
        return original(*args, **kwargs)

    executor._dispatch = counting
    result = executor.execute(
        "read_file", {"path": "logs/failure.log", "start_line": 1}, call_id="call-1", turn=0
    )
    assert calls["dispatch"] == 0
    assert result.status.value == "error"
    assert result.error_code == "policy_denied:check_exception"
    assert result.data["rule_id"] == "R01"
    assert budget.state.tool_attempts_used == 1


def test_before_tool_timeout_blocks_execution(tmp_path: Path) -> None:
    executor, _, _, _, _, _ = make_tool_executor(tmp_path)
    calls = {"dispatch": 0}

    def slow(inp):
        time.sleep(0.05)
        return allow("R01")

    engine = LifecycleEngine(
        [PolicyCheck("slow", LifecycleEvent.BEFORE_TOOL, "R01", slow, timeout_ms=5)],
    )
    executor.policy = engine
    original = executor._dispatch

    def counting(*args, **kwargs):
        calls["dispatch"] += 1
        return original(*args, **kwargs)

    executor._dispatch = counting
    result = executor.execute(
        "read_file", {"path": "logs/failure.log", "start_line": 1}, call_id="call-1", turn=0
    )
    assert calls["dispatch"] == 0
    assert result.error_code == "policy_denied:check_timeout"
    assert result.data["rule_id"] == "R01"


def test_deny_wins_and_both_checks_see_the_same_immutable_input(tmp_path: Path) -> None:
    executor, _, _, _, _, _ = make_tool_executor(tmp_path)
    seen: list[dict] = []
    calls = {"dispatch": 0}

    def allowing(inp):
        seen.append(dict(inp.data))
        return allow("R01")

    def denying(inp):
        seen.append(dict(inp.data))
        return deny("R09", "synthetic_deny", "not allowed")

    engine = LifecycleEngine(
        [
            PolicyCheck("allow", LifecycleEvent.BEFORE_TOOL, "R01", allowing),
            PolicyCheck("deny", LifecycleEvent.BEFORE_TOOL, "R09", denying),
        ],
        run_id="run-test",
    )
    executor.policy = engine
    original = executor._dispatch

    def counting(*args, **kwargs):
        calls["dispatch"] += 1
        return original(*args, **kwargs)

    executor._dispatch = counting
    result = executor.execute("read_file", {"path": "logs/failure.log"}, call_id="call-1", turn=0)
    assert calls["dispatch"] == 0
    assert result.error_code == "policy_denied:synthetic_deny"
    assert len(seen) == 2
    assert seen[0] == seen[1]
    assert [record["decision"] for record in engine.records] == ["allow", "deny"]


def test_github_query_with_token_is_not_sent(tmp_path: Path) -> None:
    github = CountingGithub()
    executor, store, _, budget, _, _ = make_tool_executor(tmp_path, github=github)
    engine = LifecycleEngine(
        [
            PolicyCheck("scope", LifecycleEvent.BEFORE_TOOL, "R01", lambda inp: allow("R01")),
            PolicyCheck("budget", LifecycleEvent.BEFORE_TOOL, "R07", lambda inp: allow("R07")),
            PolicyCheck(
                "github",
                LifecycleEvent.BEFORE_TOOL,
                "R09",
                __import__("agent_triage.lifecycle", fromlist=["_github_query"])._github_query,
            ),
        ]
    )
    executor.policy = engine
    result = executor.execute(
        "github_search_issues",
        {"query": "failure ghp_abcdefghijklmnopqrstuvwxyz", "limit": 3},
        call_id="call-1",
        turn=0,
    )
    assert github.calls == 0
    assert result.error_code == "policy_denied:query_contains_secret"
    assert result.data["rule_id"] == "R09"
    assert budget.state.tool_attempts_used == 1
    trace = store.read_trace()
    assert trace and trace[0].status == "rejected"


def test_unknown_decision_and_missing_fields_are_denied(tmp_path: Path) -> None:
    executor, _, _, _, _, _ = make_tool_executor(tmp_path)
    calls = {"dispatch": 0}

    def invalid(inp):
        return {"decision": "maybe", "reason_code": "x"}

    engine = LifecycleEngine(
        [PolicyCheck("invalid", LifecycleEvent.BEFORE_TOOL, "R01", invalid, critical=True)]
    )
    executor.policy = engine
    original = executor._dispatch

    def counting(*args, **kwargs):
        calls["dispatch"] += 1
        return original(*args, **kwargs)

    executor._dispatch = counting
    result = executor.execute("read_file", {"path": "logs/failure.log"}, call_id="call-1", turn=0)
    assert calls["dispatch"] == 0
    assert result.error_code == "policy_denied:invalid_result"
    assert result.data["rule_id"] == "R01"
