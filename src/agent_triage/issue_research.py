"""Deterministic layered issue research controller.

The controller decides which layer may be searched, which bounded queries are
allowed, which candidate may be read, and when widening stops.  The model may
judge applicability through ``record_issue_assessment``; it cannot choose a
repository, a search scope or the next layer.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from .contracts import (
    Event,
    IssueAssessment,
    IssueCandidate,
    IssueResearchConfig,
    IssueResearchState,
    LayerRecord,
    ProfileDependency,
    ProjectProfile,
    ReceiptIssue,
    ResearchLayer,
    SearchReceipt,
)
from .github_scope import SearchScope, compile_query, derive_terms, scope_for
from .project_profile import related_dependencies


@dataclass(frozen=True)
class ResearchAction:
    kind: Literal["search", "read"]
    call_id: str
    layer: ResearchLayer
    scope: SearchScope
    query: str | None = None
    issue_ref: str | None = None
    limit: int = 3

    def arguments(self) -> dict[str, Any]:
        if self.kind == "search":
            return {"query": self.query or "", "limit": self.limit}
        return {"issue_ref": self.issue_ref or ""}


class IssueResearchController:
    def __init__(
        self,
        *,
        config: IssueResearchConfig,
        profile: ProjectProfile,
        event: Event,
        run_id: str,
        state: IssueResearchState | None = None,
        now: datetime | None = None,
    ) -> None:
        self.config = config
        self.profile = profile
        self.event = event
        self.run_id = run_id
        self._now = now
        self.state = state.model_copy(deep=True) if state is not None else IssueResearchState(
            scope_limit=config.max_scope,
            started_at=self._timestamp(),
            updated_at=self._timestamp(),
        )
        if state is None:
            for gap in profile.gaps:
                if gap not in self.state.gaps and len(self.state.gaps) < 20:
                    self.state.gaps.append(gap)
        self._terms = derive_terms(f"{event.symptom} {event.goal}", limit=4)
        self._dependencies = related_dependencies(
            profile,
            event.symptom,
            limit=config.max_dependency_repos,
        )
        if config.max_scope == "current":
            self._dependencies = []
        self._action_counter = len(self.state.receipts) + self.state.searches_used + self.state.reads_used

    # -- persistence --------------------------------------------------------
    def _timestamp(self) -> datetime:
        return self._now or datetime.now().astimezone()

    def snapshot(self) -> IssueResearchState:
        self.state.updated_at = self._timestamp()
        return self.state.model_copy(deep=True)

    # -- public inspection --------------------------------------------------
    @property
    def enabled(self) -> bool:
        return True

    @property
    def finished(self) -> bool:
        return self.state.status == "done"

    def current_scope(self) -> SearchScope:
        return self._scope_for_layer(self.state.layer)

    def model_scope(self) -> SearchScope:
        """The scope a model-initiated remote call must use right now."""

        if self.state.layer == ResearchLayer.DEPENDENCIES:
            record = self._record(ResearchLayer.DEPENDENCIES)
            index = min(record.searches, max(0, len(self._dependencies) - 1))
            repo = self._dependencies[index].upstream_repo if self._dependencies else None
            return self._scope_for_layer(self.state.layer, repo)
        return self._scope_for_layer(self.state.layer)

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.state.schema_version,
            "mode": self.state.mode,
            "layer": self.state.layer.value,
            "scope_limit": self.state.scope_limit,
            "status": self.state.status,
            "searches_used": self.state.searches_used,
            "searches_limit": self.config.max_searches,
            "reads_used": self.state.reads_used,
            "reads_limit": self.config.max_issue_reads,
            "remote_requests_used": self.state.remote_requests_used,
            "remote_requests_limit": self.config.max_remote_requests,
            "stop_reason": self.state.stop_reason,
            "gaps": list(self.state.gaps),
            "candidates": [
                {
                    "issue_ref": candidate.issue_ref,
                    "layer": candidate.layer.value,
                    "title": candidate.title,
                    "read": candidate.read,
                    "applicable": bool(candidate.assessment and candidate.assessment.applicable),
                    "rejected": bool(candidate.assessment and candidate.assessment.rejected),
                }
                for candidate in self.state.candidates
            ],
        }

    def next_reason(self) -> str:
        if self.finished:
            return self.state.stop_reason or "done"
        if self.state.status == "waiting_assessment":
            return "waiting for candidate assessment before widening the search"
        if self.state.layer == ResearchLayer.CURRENT:
            return "search the current project first"
        if self.state.layer == ResearchLayer.DEPENDENCIES:
            return "search mapped upstream dependency repositories"
        return "search the public issue space as a bounded fallback"

    # -- action planning ----------------------------------------------------
    def next_action(self) -> ResearchAction | None:
        if self.state.status == "done":
            return None
        if self.state.status == "waiting_assessment":
            return None
        if self._applicable_candidate() is not None:
            self._finish("applicable_candidate")
            return None
        for _ in range(4):
            layer = self.state.layer
            record = self._record(layer)
            if layer == ResearchLayer.CURRENT:
                if record.searches == 0:
                    return self._search_action(ResearchLayer.CURRENT, self._current_terms())
                if record.reads == 0:
                    action = self._read_action(layer)
                    if action is not None:
                        return action
                if self._advance_layer():
                    continue
                return None
            if layer == ResearchLayer.DEPENDENCIES:
                if not self._dependencies:
                    if self._advance_layer():
                        continue
                    return None
                index = record.searches
                if index < len(self._dependencies):
                    dependency = self._dependencies[index]
                    return self._search_action(
                        ResearchLayer.DEPENDENCIES,
                        self._dependency_terms(dependency),
                        repo=dependency.upstream_repo,
                    )
                if record.reads == 0:
                    action = self._read_action(layer)
                    if action is not None:
                        return action
                if self._advance_layer():
                    continue
                return None
            if layer == ResearchLayer.PUBLIC:
                if record.searches == 0:
                    return self._search_action(ResearchLayer.PUBLIC, self._public_terms())
                if record.reads == 0:
                    action = self._read_action(layer)
                    if action is not None:
                        return action
                self._finish("scopes_exhausted")
                return None
        return None

    def _search_action(
        self,
        layer: ResearchLayer,
        terms: list[str],
        *,
        repo: str | None = None,
    ) -> ResearchAction | None:
        if self.state.searches_used >= self.config.max_searches:
            self._finish("search_budget_exhausted")
            return None
        if self.state.remote_requests_used >= self.config.max_remote_requests:
            self._finish("remote_budget_exhausted")
            return None
        target_repo = repo or (self.profile.project_repo if layer == ResearchLayer.CURRENT else None)
        if layer == ResearchLayer.CURRENT and not target_repo:
            self._gap("project_repo_unresolved")
            self._finish("project_repo_unresolved")
            return None
        scope = scope_for(layer, target_repo)
        try:
            query = compile_query(terms, scope)
        except ValueError as exc:
            self._gap(f"query_rejected:{str(exc)[:200]}")
            record = self._record(layer)
            record.errors.append(str(exc)[:500])
            self.state.searches_used += 1
            if not self._advance_layer():
                self._finish("query_rejected")
            return None
        self._action_counter += 1
        return ResearchAction(
            kind="search",
            call_id=f"research-{self._action_counter:03d}",
            layer=layer,
            scope=scope,
            query=query,
            limit=self.config.candidates_per_query,
        )

    def _read_action(self, layer: ResearchLayer) -> ResearchAction | None:
        if self.state.reads_used >= self.config.max_issue_reads:
            self._finish("read_budget_exhausted")
            return None
        if self.state.remote_requests_used >= self.config.max_remote_requests:
            self._finish("remote_budget_exhausted")
            return None
        candidate = self._next_unread(layer)
        if candidate is None:
            return None
        self._action_counter += 1
        self.state.pending_issue_ref = candidate.issue_ref
        return ResearchAction(
            kind="read",
            call_id=f"research-{self._action_counter:03d}",
            layer=layer,
            scope=self._scope_for_layer(layer, f"{candidate.owner}/{candidate.repo}"),
            issue_ref=candidate.issue_ref,
        )

    def _current_terms(self) -> list[str]:
        terms = list(self._terms)
        code = _error_code(self.event.symptom)
        if code and code not in terms:
            terms.insert(0, code)
        return terms[:3] or [self.event.project_id]

    def _dependency_terms(self, dependency: ProfileDependency) -> list[str]:
        primary = dependency.import_names[0] if dependency.import_names else dependency.package
        terms = [primary, *self._terms[:2]]
        return terms[:3]

    def _public_terms(self) -> list[str]:
        packages = [dependency.package for dependency in self.profile.dependencies if dependency.upstream_repo]
        terms = [*packages[:1], *self._terms[:2]]
        return terms[:3]

    # -- observation --------------------------------------------------------
    def note_remote_request(self) -> bool:
        if self.state.remote_requests_used >= self.config.max_remote_requests:
            return False
        self.state.remote_requests_used += 1
        return True

    def accept_search(
        self,
        action: ResearchAction,
        *,
        receipt: SearchReceipt,
        candidates: list[IssueCandidate],
        complete: bool,
        coverage: str | None = None,
    ) -> None:
        if action.kind != "search":
            raise ValueError("accept_search requires a search action")
        if receipt.run_id != self.run_id:
            raise ValueError("receipt belongs to another run")
        if receipt.layer != action.layer:
            raise ValueError("receipt layer does not match the executed action")
        if any(existing.receipt_id == receipt.receipt_id for existing in self.state.receipts):
            raise ValueError("receipt was already registered")
        self.state.receipts.append(receipt)
        known = {candidate.issue_ref for candidate in self.state.candidates}
        for candidate in candidates:
            if candidate.receipt_id != receipt.receipt_id:
                raise ValueError("candidate receipt does not match the registered receipt")
            if candidate.issue_ref not in known:
                self.state.candidates.append(candidate)
                known.add(candidate.issue_ref)
        record = self._record(action.layer)
        record.searches += 1
        record.queries.append(action.query or "")
        self.state.searches_used += 1
        if not complete:
            record.coverage.append("search result was incomplete or truncated")
        if coverage:
            record.coverage.append(coverage[:500])
        self.state.updated_at = self._timestamp()

    def accept_search_error(self, action: ResearchAction, code: str, message: str) -> None:
        record = self._record(action.layer)
        record.searches += 1
        record.errors.append(f"{code}: {message}"[:500])
        self.state.searches_used += 1
        self._gap(f"{action.layer.value}_search_failed:{code}")
        self.state.updated_at = self._timestamp()

    def accept_read(
        self,
        action: ResearchAction,
        *,
        evidence_id: str,
    ) -> None:
        if self.state.status == "done":
            raise ValueError("issue research already finished; no further reads are accepted")
        if self.state.reads_used >= self.config.max_issue_reads:
            raise ValueError("the configured issue read budget is exhausted")
        candidate = self._candidate(action.issue_ref or "")
        if candidate is None:
            raise ValueError("read action references an unknown candidate")
        candidate.read_evidence_id = evidence_id
        self.state.pending_issue_ref = None
        self.state.reads_used += 1
        self._record(action.layer).reads += 1
        self.state.status = "waiting_assessment"
        self.state.updated_at = self._timestamp()

    def accept_read_error(self, action: ResearchAction, code: str, message: str) -> None:
        candidate = self._candidate(action.issue_ref or "")
        if candidate is not None:
            candidate.read_error = f"{code}: {message}"[:500]
        self.state.pending_issue_ref = None
        self.state.status = "pending"
        self._record(action.layer).errors.append(f"read_failed:{code}: {message}"[:500])
        self._gap(f"{action.layer.value}_read_failed:{code}")
        self.state.updated_at = self._timestamp()

    def finalize(self, reason: str = "model_finalized_before_research_complete") -> None:
        self._finish(reason)

    def action_failed(self, action: ResearchAction, code: str, message: str) -> None:
        if action.kind == "search":
            self.accept_search_error(action, code, message)
        else:
            self.accept_read_error(action, code, message)

    def can_read(self, issue_ref: str) -> bool:
        return self._candidate(issue_ref) is not None

    def candidate(self, issue_ref: str) -> IssueCandidate | None:
        return self._candidate(issue_ref)

    def record_assessment(self, assessment: IssueAssessment) -> None:
        candidate = self._candidate(assessment.issue_ref)
        if candidate is None:
            raise ValueError("assessment references an issue that this run did not locate")
        if not candidate.read:
            raise ValueError("assessment references an issue whose body was not read")
        if candidate.receipt_id not in {receipt.receipt_id for receipt in self.state.receipts}:
            raise ValueError("assessment references a candidate without a persisted search receipt")
        candidate.assessment = assessment
        unknowns = [
            name
            for name in ("component", "symptom", "version", "environment", "trigger", "fix")
            if getattr(assessment, name) == "unknown"
        ]
        if unknowns:
            self._gap(f"applicability_unknowns:{assessment.issue_ref}:{','.join(unknowns)}")
        if assessment.applicable:
            self._finish("applicable_candidate")

    def note_model_round(self) -> None:
        """The model had one request to assess the current layer's candidate."""

        if self.state.status == "done":
            return
        if self.state.status == "waiting_assessment":
            self.state.status = "pending"
        if self._applicable_candidate() is not None:
            self._finish("applicable_candidate")
            return
        record = self._record(self.state.layer)
        layer_candidates = [item for item in self.state.candidates if item.layer == self.state.layer]
        if not layer_candidates:
            layer_complete = record.searches > 0
        else:
            layer_complete = record.searches > 0 and (
                record.reads >= 1 or all(item.read_error is not None for item in layer_candidates)
            )
        if layer_complete:
            self._advance_layer()

    # -- internals ----------------------------------------------------------
    def _advance_layer(self) -> bool:
        order = [ResearchLayer.CURRENT, ResearchLayer.DEPENDENCIES, ResearchLayer.PUBLIC]
        if self.state.layer == ResearchLayer.CURRENT:
            if self.config.max_scope in {"dependencies", "public"} and self._dependencies:
                self.state.layer = ResearchLayer.DEPENDENCIES
                self.state.status = "pending"
                return True
            if self.config.max_scope == "public":
                self.state.layer = ResearchLayer.PUBLIC
                self.state.status = "pending"
                return True
            self._gap("no_applicable_candidate_in_authorized_scope")
            self._finish("scopes_exhausted")
            return False
        if self.state.layer == ResearchLayer.DEPENDENCIES:
            if self.config.max_scope == "public":
                self.state.layer = ResearchLayer.PUBLIC
                self.state.status = "pending"
                return True
            self._gap("no_applicable_candidate_in_authorized_scope")
            self._finish("scopes_exhausted")
            return False
        index = order.index(self.state.layer)
        if index == len(order) - 1:
            self._finish("scopes_exhausted")
            return False
        self.state.layer = order[index + 1]
        return True

    def _finish(self, reason: str) -> None:
        if self.state.status == "done":
            return
        self.state.status = "done"
        self.state.stop_reason = reason
        self.state.pending_issue_ref = None
        self.state.updated_at = self._timestamp()

    def _gap(self, text: str) -> None:
        text = text[:500]
        if text and text not in self.state.gaps:
            self.state.gaps.append(text)
            del self.state.gaps[:-20]

    def _scope_for_layer(self, layer: ResearchLayer, repo: str | None = None) -> SearchScope:
        if layer == ResearchLayer.PUBLIC:
            return scope_for(layer, None)
        if repo is None:
            repo = self.profile.project_repo if layer == ResearchLayer.CURRENT else None
        return scope_for(layer, repo)

    def _record(self, layer: ResearchLayer) -> LayerRecord:
        for record in self.state.layers:
            if record.layer == layer:
                return record
        record = LayerRecord(layer=layer)
        self.state.layers.append(record)
        return record

    def _candidate(self, issue_ref: str) -> IssueCandidate | None:
        for candidate in self.state.candidates:
            if candidate.issue_ref == issue_ref:
                return candidate
        return None

    def _next_unread(self, layer: ResearchLayer) -> IssueCandidate | None:
        for candidate in self.state.candidates:
            if candidate.layer == layer and not candidate.read and not candidate.assessment and candidate.read_error is None:
                return candidate
        return None

    def _applicable_candidate(self) -> IssueCandidate | None:
        for candidate in self.state.candidates:
            if candidate.assessment and candidate.assessment.applicable:
                return candidate
        return None

    # -- conversion ---------------------------------------------------------
    def related_issues(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for candidate in self.state.candidates:
            assessment = candidate.assessment
            if assessment is None:
                applicability = "unassessed"
            elif assessment.applicable:
                applicability = "applicable"
            elif assessment.rejected:
                applicability = "rejected"
            else:
                applicability = "reference"
            items.append(
                {
                    "issue_ref": candidate.issue_ref,
                    "layer": candidate.layer.value,
                    "title": candidate.title,
                    "state": candidate.state,
                    "url": candidate.url,
                    "applicability": applicability,
                    "assessment": assessment.model_dump(mode="json") if assessment else None,
                }
            )
        return items


def issue_receipt(
    *,
    run_id: str,
    layer: ResearchLayer,
    scope: SearchScope,
    query: str,
    call_id: str,
    evidence_id: str,
    snapshot_sha256: str,
    issues: list[ReceiptIssue],
    captured_at: datetime,
) -> SearchReceipt:
    digest = hashlib.sha256(
        f"{run_id}|{layer.value}|{scope.label}|{query}|{call_id}|{evidence_id}".encode()
    ).hexdigest()[:12]
    receipt_id = f"rcpt-{digest}"
    return SearchReceipt(
        receipt_id=receipt_id,
        run_id=run_id,
        layer=layer,
        scope=scope.label,
        query_sha256=hashlib.sha256(query.encode()).hexdigest(),
        call_id=call_id,
        evidence_id=evidence_id,
        snapshot_sha256=snapshot_sha256,
        issues=issues,
        captured_at=captured_at,
    )


def _error_code(symptom: str) -> str | None:
    match = re.search(r"\b(?:[A-Z][A-Za-z]*(?:Error|Exception)|E[0-9]{4}|[A-Z]{2,}[0-9]{2,})\b", symptom or "")
    return match.group(0) if match else None