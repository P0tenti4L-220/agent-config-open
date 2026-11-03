"""Executor-level tests: receipts, cross-repository reads and assessments."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent_triage.budget import Budget
from agent_triage.config import config_fingerprint
from agent_triage.contracts import (
    BudgetLimits,
    Config,
    Event,
    IssueResearchConfig,
    ProfileDependency,
    ProjectProfile,
)
from agent_triage.errors import ToolFailure
from agent_triage.issue_research import IssueResearchController
from agent_triage.store import EvidenceStore, RunStore
from agent_triage.tools import (
    IssueReadOutcome,
    IssueSearchOutcome,
    ToolExecutor,
    Workspace,
)
from tests.helpers import make_event


class FakeScopedBackend:
    available = True
    unavailable_reason = None

    def __init__(self) -> None:
        self.searches = []
        self.reads = []

    def search_issues(self, query, limit, *, scope=None):
        self.searches.append((query, scope.label))
        if scope.layer.value == "current":
            items = [
                {
                    "issue_ref": "example/application#12",
                    "owner": "example",
                    "repo": "application",
                    "number": 12,
                    "title": "KeyError items in adapter",
                    "state": "open",
                    "html_url": "https://github.com/example/application/issues/12",
                    "updated_at": "2026-11-01T00:00:00Z",
                }
            ]
        elif scope.layer.value == "dependencies":
            items = [
                {
                    "issue_ref": "upstream/client#4",
                    "owner": "upstream",
                    "repo": "client",
                    "number": 4,
                    "title": "adapter KeyError fixed in 1.5",
                    "state": "closed",
                    "html_url": "https://github.com/upstream/client/issues/4",
                    "updated_at": "2026-10-01T00:00:00Z",
                }
            ]
        else:
            items = []
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
                "title": "adapter KeyError",
                "state": "open",
                "html_url": f"https://github.com/{owner}/{repo}/issues/{number}",
                "body": "ClientLib adapter raises KeyError for items payloads on 1.4.x.",
            },
            comments=[],
            url=f"https://github.com/{owner}/{repo}/issues/{number}",
            updated_at="2026-11-01T00:00:00Z",
        )


def _profile() -> ProjectProfile:
    return ProjectProfile(
        project_repo="example/application",
        repo_source="config",
        language="python",
        runtime_version="3.11.9",
        dependencies=[
            ProfileDependency(
                ecosystem="pypi",
                package="client-lib",
                import_names=["client_lib"],
                upstream_repo="upstream/client",
                locked="1.4.2",
            )
        ],
    )


def _lookup(evidence):
    def exists(evidence_id: str) -> bool:
        try:
            evidence.require(evidence_id)
        except Exception:  # noqa: BLE001 - absence is the answer
            return False
        return True

    return exists


def _executor(tmp_path: Path, event: Event, *, policy=None):
    project = tmp_path / "project"
    project.mkdir()
    (project / "app.py").write_text("print('hello')\n", encoding="utf-8")
    store = RunStore.create(
        tmp_path / "runs",
        event_payload=event.model_dump(mode="json"),
        event_id=event.event_id,
        project_id=event.project_id,
        project_root=project,
        provider="replay",
        model="replay",
        config_fingerprint=config_fingerprint(Config()),
    )
    evidence = EvidenceStore(store)
    controller = IssueResearchController(
        config=IssueResearchConfig(mode="layered", max_scope="public"),
        profile=_profile(),
        event=event,
        run_id=store.run_id,
    )
    executor = ToolExecutor(
        workspace=Workspace(project),
        project_id=event.project_id,
        budget=Budget(BudgetLimits()),
        evidence_sink=evidence.add,
        github=FakeScopedBackend(),
        trace_sink=store.append_trace,
        run_id=store.run_id,
        research=controller,
        evidence_lookup=_lookup(evidence),
        evidence_store=evidence,
        displayed_lookup=lambda _evidence_id: True,
        policy=policy,
    )
    return executor, controller, evidence


def test_search_and_read_bind_real_identity_and_receipt(tmp_path: Path) -> None:
    event = make_event(symptom="KeyError: 'items' in client_lib")
    executor, controller, _evidence = _executor(tmp_path, event)

    search = executor.execute(
        "github_search_issues", {"query": "KeyError items client_lib", "limit": 3}, call_id="call-1", turn=0
    )
    assert search.status.value == "ok", search.data
    assert search.data["scope"] == "repo:example/application"
    assert search.data["receipt_id"].startswith("rcpt-")
    assert controller.state.searches_used == 1
    assert controller.state.candidates[0].issue_ref == "example/application#12"

    read = executor.execute(
        "github_read_issue", {"issue_ref": "example/application#12"}, call_id="call-2", turn=0
    )
    assert read.status.value == "ok", read.data
    assert read.data["issue"]["issue_ref"] == "example/application#12"
    candidate = controller.candidate("example/application#12")
    assert candidate is not None and candidate.read
    read_evidence = read.data["evidence_id"]

    assessment_args = {
        "issue_ref": "example/application#12",
        "component": "match",
        "symptom": "match",
        "version": "unknown",
        "environment": "unknown",
        "trigger": "unknown",
        "fix": "unknown",
    }
    summary_only = executor.execute(
        "record_issue_assessment",
        {**assessment_args, "evidence_refs": [candidate.search_evidence_id]},
        call_id="call-3",
        turn=0,
    )
    assert summary_only.status.value == "error"
    assert summary_only.error_code == "assessment_missing_issue_body_evidence"
    assert candidate.assessment is None
    assert not controller.finished

    assessment = executor.execute(
        "record_issue_assessment",
        {**assessment_args, "evidence_refs": [candidate.search_evidence_id, read_evidence]},
        call_id="call-4",
        turn=0,
    )
    assert assessment.status.value == "ok", assessment.data
    assert assessment.data["applicable"] is True

    from agent_triage.contracts import IssueAssessment

    controller.record_assessment(IssueAssessment.model_validate(assessment.data["assessment"]))
    assert controller.finished
    assert controller.state.stop_reason == "applicable_candidate"


def test_read_without_receipt_is_rejected(tmp_path: Path) -> None:
    event = make_event(symptom="KeyError: 'items' in client_lib")
    executor, _controller, _evidence = _executor(tmp_path, event)
    result = executor.execute(
        "github_read_issue", {"issue_ref": "other/repo#1"}, call_id="call-1", turn=0
    )
    assert result.status.value == "error"
    assert result.error_code == "out_of_scope_issue"


def test_search_query_cannot_override_repository(tmp_path: Path) -> None:
    event = make_event(symptom="KeyError: 'items' in client_lib")
    executor, _controller, _evidence = _executor(tmp_path, event)
    result = executor.execute(
        "github_search_issues",
        {"query": "KeyError repo:other/project", "limit": 2},
        call_id="call-1",
        turn=0,
    )
    assert result.status.value == "error"
    assert result.error_code == "invalid_query_scope"


def test_assessment_requires_read_body(tmp_path: Path) -> None:
    event = make_event(symptom="KeyError: 'items' in client_lib")
    executor, controller, _evidence = _executor(tmp_path, event)
    executor.execute(
        "github_search_issues", {"query": "KeyError items", "limit": 2}, call_id="call-1", turn=0
    )
    candidate = controller.state.candidates[0]
    result = executor.execute(
        "record_issue_assessment",
        {
            "issue_ref": candidate.issue_ref,
            "component": "match",
            "symptom": "match",
            "version": "unknown",
            "environment": "unknown",
            "trigger": "unknown",
            "fix": "unknown",
            "evidence_refs": [candidate.search_evidence_id],
        },
        call_id="call-2",
        turn=0,
    )
    assert result.status.value == "error"
    assert result.error_code == "issue_not_read"

def test_assessment_tool_passes_lifecycle_allowlist(tmp_path: Path) -> None:
    from agent_triage.lifecycle import build_default_engine

    event = make_event(symptom="KeyError: 'items' in client_lib")
    executor, _controller, _evidence = _executor(
        tmp_path, event, policy=build_default_engine(run_id="run-20261103-120000-abcdef")
    )
    executor.execute(
        "github_search_issues", {"query": "KeyError items", "limit": 2}, call_id="call-1", turn=0
    )
    read = executor.execute(
        "github_read_issue", {"issue_ref": "example/application#12"}, call_id="call-2", turn=0
    )
    assert read.status.value == "ok", read.data
    result = executor.execute(
        "record_issue_assessment",
        {
            "issue_ref": "example/application#12",
            "component": "match",
            "symptom": "match",
            "version": "unknown",
            "environment": "unknown",
            "trigger": "unknown",
            "fix": "unknown",
            "evidence_refs": [read.data["evidence_id"]],
        },
        call_id="call-3",
        turn=0,
    )
    assert result.status.value == "ok", result.data
