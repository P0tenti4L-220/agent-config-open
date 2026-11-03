"""Offline tests for layered issue research: scope, receipts and stopping."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest

from agent_triage.contracts import (
    Config,
    DependencyMapping,
    IssueAssessment,
    IssueCandidate,
    IssueResearchConfig,
    ProfileDependency,
    ProjectProfile,
    ReceiptIssue,
    ResearchLayer,
    SearchReceipt,
)
from agent_triage.github_scope import (
    ScopeViolation,
    SearchScope,
    compile_query,
    parse_issue_identity,
    scope_for,
    split_issue_ref,
)
from agent_triage.issue_research import IssueResearchController

RUN_ID = "run-20261103-120000-abcdef"
NOW = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _profile(*, dependency_repo: str | None = "upstream/client") -> ProjectProfile:
    dependencies = [
        ProfileDependency(
            ecosystem="pypi",
            package="client-lib",
            import_names=["client_lib"],
            upstream_repo=dependency_repo,
            declared=">=1.0,<2",
            locked="1.4.2",
        )
    ]
    return ProjectProfile(
        project_repo="example/application",
        repo_source="config",
        language="python",
        runtime_version="3.11.9",
        dependencies=dependencies,
    )


def _controller(config: IssueResearchConfig | None = None) -> IssueResearchController:
    from tests.helpers import make_event

    return IssueResearchController(
        config=config
        or IssueResearchConfig(mode="layered", max_scope="public", max_searches=6, max_issue_reads=4),
        profile=_profile(),
        event=make_event(symptom="KeyError: 'items' in client_lib adapter"),
        run_id=RUN_ID,
    )


def _receipt(layer: ResearchLayer, scope: SearchScope, issues: list[ReceiptIssue], *, suffix: str = "a") -> SearchReceipt:
    return SearchReceipt(
        receipt_id=f"rcpt-{suffix * 12}"[:17],
        run_id=RUN_ID,
        layer=layer,
        scope=scope.label,
        query_sha256=_sha("query"),
        call_id=f"research-{suffix}",
        evidence_id="ev-" + suffix * 12,
        snapshot_sha256=_sha("snapshot" + suffix),
        issues=issues,
        captured_at=NOW,
    )


def _candidate(layer: ResearchLayer, receipt: SearchReceipt, *, suffix: str = "a") -> IssueCandidate:
    issue = receipt.issues[0]
    owner, repo, _number = split_issue_ref(issue.issue_ref)
    return IssueCandidate(
        issue_ref=issue.issue_ref,
        layer=layer,
        owner=owner,
        repo=repo,
        number=issue.number,
        title=issue.title,
        state=issue.state,
        url=issue.url,
        receipt_id=receipt.receipt_id,
        search_evidence_id=issue.evidence_id,
    )


def test_config_rejects_layered_mode_without_v2() -> None:
    with pytest.raises(ValueError):
        Config.model_validate(
            {
                "schema_version": 1,
                "mcp": {"repo": "example/application"},
                "issue_research": {"mode": "layered", "max_scope": "public"},
            }
        )


def test_legacy_default_preserves_single_repo_behaviour() -> None:
    config = Config()
    assert config.schema_version == 1
    assert config.issue_research.mode == "legacy"
    assert config.issue_research.max_scope == "current"


def test_query_cannot_widen_scope() -> None:
    scope = scope_for(ResearchLayer.CURRENT, "example/application")
    for forbidden in (
        "repo:other/project",
        "org:other",
        "is:public",
        "KeyError OR repo:other",
        "-repo:other",
        'KeyError "quoted"',
    ):
        with pytest.raises(ScopeViolation):
            compile_query([forbidden], scope)


def test_public_scope_appends_public_qualifier() -> None:
    query = compile_query(["KeyError", "client_lib"], scope_for(ResearchLayer.PUBLIC, None))
    assert query.endswith("is:public is:issue")
    assert "repo:" not in query


def test_identity_comes_from_response_not_configuration() -> None:
    item = {
        "number": 7,
        "title": "Adapter crash",
        "html_url": "https://github.com/upstream/client/issues/7",
        "repository": {"full_name": "upstream/client"},
    }
    owner, repo, number, url = parse_issue_identity(item)
    assert (owner, repo, number) == ("upstream", "client", 7)
    assert url.endswith("/upstream/client/issues/7")
    with pytest.raises(Exception) as info:
        parse_issue_identity(item, expected_repo="example/application")
    assert getattr(info.value, "code", "") == "out_of_scope_issue"


def test_identity_conflict_is_rejected() -> None:
    item = {
        "number": 7,
        "title": "Adapter crash",
        "html_url": "https://github.com/other/repo/issues/8",
    }
    with pytest.raises(Exception):
        parse_issue_identity(item, expected_repo="other/repo")


def test_layers_advance_search_read_wait_then_widen() -> None:
    controller = _controller()
    action = controller.next_action()
    assert action is not None and action.kind == "search"
    assert action.scope.layer == ResearchLayer.CURRENT
    assert "repo:example/application" in (action.query or "")

    scope = action.scope
    receipt = _receipt(
        ResearchLayer.CURRENT,
        scope,
        [
            ReceiptIssue(
                issue_ref="example/application#12",
                number=12,
                title="KeyError items adapter",
                state="open",
                url="https://github.com/example/application/issues/12",
                evidence_id="ev-" + "1" * 12,
            )
        ],
    )
    controller.accept_search(
        action,
        receipt=receipt,
        candidates=[_candidate(ResearchLayer.CURRENT, receipt)],
        complete=True,
    )
    read = controller.next_action()
    assert read is not None and read.kind == "read"
    assert read.issue_ref == "example/application#12"
    controller.accept_read(read, evidence_id="ev-" + "2" * 12)
    assert controller.next_action() is None
    assert controller.state.status == "waiting_assessment"

    controller.note_model_round()
    widened = controller.next_action()
    assert widened is not None and widened.kind == "search"
    assert widened.scope.layer == ResearchLayer.DEPENDENCIES
    assert "repo:upstream/client" in (widened.query or "")


def test_applicable_assessment_stops_widening() -> None:
    controller = _controller()
    action = controller.next_action()
    assert action is not None
    scope = action.scope
    receipt = _receipt(
        ResearchLayer.CURRENT,
        scope,
        [
            ReceiptIssue(
                issue_ref="example/application#5",
                number=5,
                title="KeyError items",
                state="open",
                url="https://github.com/example/application/issues/5",
                evidence_id="ev-" + "3" * 12,
            )
        ],
    )
    controller.accept_search(
        action,
        receipt=receipt,
        candidates=[_candidate(ResearchLayer.CURRENT, receipt)],
        complete=True,
    )
    read = controller.next_action()
    assert read is not None
    controller.accept_read(read, evidence_id="ev-" + "4" * 12)
    controller.record_assessment(
        IssueAssessment(
            issue_ref="example/application#5",
            component="match",
            symptom="match",
            version="unknown",
            environment="unknown",
            trigger="unknown",
            fix="unknown",
            evidence_refs=["ev-" + "4" * 12],
        )
    )
    assert controller.finished
    assert controller.state.stop_reason == "applicable_candidate"
    assert controller.next_action() is None


def test_version_mismatch_keeps_searching() -> None:
    controller = _controller()
    action = controller.next_action()
    assert action is not None
    scope = action.scope
    receipt = _receipt(
        ResearchLayer.CURRENT,
        scope,
        [
            ReceiptIssue(
                issue_ref="example/application#9",
                number=9,
                title="KeyError items on 2.x",
                state="closed",
                url="https://github.com/example/application/issues/9",
                evidence_id="ev-" + "5" * 12,
            )
        ],
    )
    controller.accept_search(
        action, receipt=receipt, candidates=[_candidate(ResearchLayer.CURRENT, receipt)], complete=True
    )
    read = controller.next_action()
    assert read is not None
    controller.accept_read(read, evidence_id="ev-" + "6" * 12)
    controller.record_assessment(
        IssueAssessment(
            issue_ref="example/application#9",
            component="match",
            symptom="match",
            version="mismatch",
            environment="unknown",
            trigger="unknown",
            fix="unknown",
            evidence_refs=["ev-" + "6" * 12],
        )
    )
    assert not controller.finished
    controller.note_model_round()
    widened = controller.next_action()
    assert widened is not None and widened.scope.layer == ResearchLayer.DEPENDENCIES


def test_search_budget_stops_with_gap() -> None:
    controller = _controller(
        IssueResearchConfig(mode="layered", max_scope="current", max_searches=1, max_issue_reads=1)
    )
    action = controller.next_action()
    assert action is not None
    controller.accept_search_error(action, "mcp_timeout", "timed out")
    controller.note_model_round()
    assert controller.next_action() is None
    assert controller.finished
    assert any("search_failed" in gap for gap in controller.state.gaps)


def test_dependency_mapping_is_required_for_dependency_layer() -> None:
    controller = _controller()
    controller._dependencies = []  # no mapped dependency survived profile filtering
    controller.config.max_scope = "dependencies"
    action = controller.next_action()
    assert action is not None
    scope = action.scope
    receipt = _receipt(ResearchLayer.CURRENT, scope, [])
    controller.accept_search(action, receipt=receipt, candidates=[], complete=True)
    controller.note_model_round()
    assert controller.finished
    assert "no_applicable_candidate_in_authorized_scope" in controller.state.gaps


def test_state_round_trip_keeps_receipts_and_assessment() -> None:
    controller = _controller()
    action = controller.next_action()
    assert action is not None
    receipt = _receipt(
        ResearchLayer.CURRENT,
        action.scope,
        [
            ReceiptIssue(
                issue_ref="example/application#3",
                number=3,
                title="KeyError items",
                state="open",
                url="https://github.com/example/application/issues/3",
                evidence_id="ev-" + "7" * 12,
            )
        ],
    )
    controller.accept_search(
        action, receipt=receipt, candidates=[_candidate(ResearchLayer.CURRENT, receipt)], complete=True
    )
    read = controller.next_action()
    assert read is not None
    controller.accept_read(read, evidence_id="ev-" + "8" * 12)
    state = controller.snapshot()
    restored = IssueResearchController(
        config=controller.config,
        profile=controller.profile,
        event=controller.event,
        run_id=RUN_ID,
        state=state,
    )
    assert restored.state.receipts == state.receipts
    assert restored.state.candidates[0].read
    assert restored.next_action() is None

def test_public_layer_uses_public_qualifier_and_real_identity() -> None:
    controller = _controller(
        IssueResearchConfig(mode="layered", max_scope="public", max_searches=3, max_issue_reads=3)
    )
    controller._dependencies = []  # no mapped dependency: current -> public
    first = controller.next_action()
    assert first is not None
    controller.accept_search(
        first,
        receipt=_receipt(ResearchLayer.CURRENT, first.scope, []),
        candidates=[],
        complete=True,
    )
    controller.note_model_round()
    public = controller.next_action()
    assert public is not None and public.scope.public
    assert (public.query or "").endswith("is:public is:issue")
    assert "repo:" not in (public.query or "")


def test_same_number_in_two_repositories_stays_distinct() -> None:
    controller = _controller()
    action = controller.next_action()
    assert action is not None
    current_receipt = _receipt(
        ResearchLayer.CURRENT,
        action.scope,
        [
            ReceiptIssue(
                issue_ref="example/application#1",
                number=1,
                title="current repo issue",
                state="open",
                url="https://github.com/example/application/issues/1",
                evidence_id="ev-" + "a" * 12,
            )
        ],
    )
    controller.accept_search(
        action,
        receipt=current_receipt,
        candidates=[_candidate(ResearchLayer.CURRENT, current_receipt)],
        complete=True,
    )
    read = controller.next_action()
    assert read is not None
    controller.accept_read(read, evidence_id="ev-" + "b" * 12)
    controller.note_model_round()
    dep_action = controller.next_action()
    assert dep_action is not None
    dep_receipt = _receipt(
        ResearchLayer.DEPENDENCIES,
        dep_action.scope,
        [
            ReceiptIssue(
                issue_ref="upstream/client#1",
                number=1,
                title="upstream issue with the same number",
                state="open",
                url="https://github.com/upstream/client/issues/1",
                evidence_id="ev-" + "c" * 12,
            )
        ],
        suffix="d",
    )
    controller.accept_search(
        dep_action,
        receipt=dep_receipt,
        candidates=[_candidate(ResearchLayer.DEPENDENCIES, dep_receipt, suffix="d")],
        complete=True,
    )
    refs = {candidate.issue_ref for candidate in controller.state.candidates}
    assert refs == {"example/application#1", "upstream/client#1"}
    assert controller.can_read("upstream/client#1")
    assert not controller.can_read("other/repo#1")
