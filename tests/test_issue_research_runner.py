"""End-to-end layered research through the real Runner (offline backend)."""

from __future__ import annotations

from pathlib import Path

from agent_triage.config import Config, config_fingerprint
from agent_triage.context import ContextBuilder
from agent_triage.contracts import (
    BudgetLimits,
    DependencyMapping,
    IssueResearchConfig,
    MCPConfig,
)
from agent_triage.models import ReplayModel
from agent_triage.runner import Runner
from agent_triage.store import EvidenceStore, LessonStore, RunStore
from agent_triage.tools import IssueReadOutcome, IssueSearchOutcome, Workspace
from helpers import base_analysis, final_step, make_event, make_scenario, tool_step, write_sample_project


class LayeredBackend:
    available = True
    unavailable_reason = None

    def __init__(self) -> None:
        self.searches: list[tuple[str, str]] = []
        self.reads: list[str] = []

    def search_issues(self, query, limit, *, scope=None):
        self.searches.append((query, scope.label))
        if scope.layer.value == "current":
            items = [
                {
                    "issue_ref": "example/application#3",
                    "owner": "example",
                    "repo": "application",
                    "number": 3,
                    "title": "KeyError items adapter",
                    "state": "open",
                    "html_url": "https://github.com/example/application/issues/3",
                }
            ]
        else:
            items = [
                {
                    "issue_ref": "upstream/client#7",
                    "owner": "upstream",
                    "repo": "client",
                    "number": 7,
                    "title": "adapter crash on 1.4",
                    "state": "closed",
                    "html_url": "https://github.com/upstream/client/issues/7",
                }
            ]
        return IssueSearchOutcome(
            issues=items,
            total_count=len(items),
            complete=True,
            query_scope=scope.label,
            query=query,
            layer=scope.layer.value,
        )

    def read_issue(self, issue_ref, *, allow_any_repo=False):
        self.reads.append(issue_ref)
        owner, rest = issue_ref.split("/", 1)
        repo, number = rest.split("#", 1)
        return IssueReadOutcome(
            issue={
                "issue_ref": issue_ref,
                "owner": owner,
                "repo": repo,
                "number": int(number),
                "title": "adapter crash",
                "state": "open",
                "html_url": f"https://github.com/{owner}/{repo}/issues/{number}",
                "body": "ClientLib adapter raises KeyError for items payloads.",
            },
            comments=[],
            url=f"https://github.com/{owner}/{repo}/issues/{number}",
            updated_at=None,
        )


def _config() -> Config:
    return Config(
        schema_version=2,
        budgets=BudgetLimits(max_model_requests=6, max_tool_attempts=12),
        mcp=MCPConfig(
            enabled=True,
            command="python3",
            args=["-c", "pass"],
            repo="example/application",
        ),
        issue_research=IssueResearchConfig(
            mode="layered",
            max_scope="dependencies",
            dependencies=[
                DependencyMapping(package="client-lib", import_names=["client_lib"], repo="upstream/client")
            ],
        ),
    )


def test_layered_run_records_layers_receipts_and_related_issues(tmp_path: Path) -> None:
    project = write_sample_project(tmp_path / "project")
    (project / "pyproject.toml").write_text(
        '[project]\nname = "app"\ndependencies = ["client-lib>=1.0"]\n', encoding="utf-8"
    )
    event = make_event(symptom="KeyError: 'items' in client_lib adapter")
    scenario = make_scenario(
        [
            tool_step(("read_file", {"path": "logs/failure.log", "start_line": 1, "max_lines": 20})),
            final_step(base_analysis()),
        ]
    )
    config = _config()
    store = RunStore.create(
        tmp_path / "runs",
        event_payload=event.model_dump(mode="json"),
        event_id=event.event_id,
        project_id=event.project_id,
        project_root=project,
        provider="replay",
        model="replay",
        config_fingerprint=config_fingerprint(config),
    )
    backend = LayeredBackend()
    runner = Runner(
        store=store,
        event=event,
        config=config,
        model=ReplayModel(scenario),
        context_builder=ContextBuilder(config.context, versions={"package": "test"}),
        evidence_store=EvidenceStore(store),
        lesson_store=LessonStore(tmp_path / "runs"),
        workspace=Workspace(project),
        github=backend,
    )
    result = runner.run()

    assert backend.searches[0][1] == "repo:example/application"
    assert backend.reads == ["example/application#3", "upstream/client#7"]
    assert result.remote_status == "available"
    assert (store.dir / "issue-research.json").is_file()
    payload = store.read_json("issue-research.json")
    assert payload["layer"] == "dependencies"
    assert len(payload["receipts"]) == 2
    assert {candidate["issue_ref"] for candidate in payload["candidates"]} == {
        "example/application#3",
        "upstream/client#7",
    }
    assert result.analysis.related_issues
    assert {item.issue_ref for item in result.analysis.related_issues} == {
        "example/application#3",
        "upstream/client#7",
    }
    report = (store.dir / "report.md").read_text(encoding="utf-8")
    assert "分层 Issue 检索" in report
    # A located lead is not a confirmed root cause.
    assert "适用线索不等于根因已确认" in report

def test_interrupted_layered_run_resumes_with_receipts(tmp_path: Path) -> None:
    from agent_triage.errors import RunInterrupted

    project = write_sample_project(tmp_path / "project")
    (project / "pyproject.toml").write_text(
        '[project]\nname = "app"\ndependencies = ["client-lib>=1.0"]\n', encoding="utf-8"
    )
    event = make_event(symptom="KeyError: 'items' in client_lib adapter")
    config = _config()
    store = RunStore.create(
        tmp_path / "runs",
        event_payload=event.model_dump(mode="json"),
        event_id=event.event_id,
        project_id=event.project_id,
        project_root=project,
        provider="replay",
        model="replay",
        config_fingerprint=config_fingerprint(config),
    )
    backend = LayeredBackend()
    scenario = make_scenario(
        [
            tool_step(("read_file", {"path": "logs/failure.log", "start_line": 1, "max_lines": 20})),
            final_step(base_analysis()),
        ]
    )

    def build(checkpoint=None):
        return Runner(
            store=store,
            event=event,
            config=config,
            model=ReplayModel(scenario),
            context_builder=ContextBuilder(config.context, versions={"package": "test"}),
            evidence_store=EvidenceStore(store),
            lesson_store=LessonStore(tmp_path / "runs"),
            workspace=Workspace(project),
            github=backend,
            checkpoint=checkpoint,
            interrupt_after_rounds=1 if checkpoint is None else None,
        )

    try:
        build().run()
    except RunInterrupted:
        pass
    else:  # pragma: no cover - the interrupt contract changed
        raise AssertionError("the run should have been interrupted")

    checkpoint = store.load_checkpoint()
    assert checkpoint is not None
    assert checkpoint.issue_research is not None
    receipts_before = [receipt.receipt_id for receipt in checkpoint.issue_research.receipts]
    assert receipts_before

    result = build(checkpoint).run()
    payload = store.read_json("issue-research.json")
    assert [receipt["receipt_id"] for receipt in payload["receipts"]][: len(receipts_before)] == receipts_before
    assert result.analysis is not None
