"""Claim-level reference, display and handoff checks.

Three questions are kept separate:

* structure: does the analysis satisfy the declared schema?
* reference/display: do cited evidence ids exist, remain current, and did the
  model actually see a bounded view of them?
* semantic support: does the quoted text support the claim?  This remains
  ``not_run`` until a human (or an explicitly evaluated reviewer) says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .contracts import (
    Analysis,
    AnalysisStatus,
    CheckStatus,
    ContextPacket,
    DisplayedEvidence,
    Evidence,
)


@dataclass
class ClaimValidation:
    problems: list[str] = field(default_factory=list)
    claim_ids: list[str] = field(default_factory=list)
    display_checked: bool = False
    display_ok: bool = True
    display_mode: str = "none"  # none | legacy | request_view
    semantic_status: CheckStatus = CheckStatus.NOT_RUN

    @property
    def ok(self) -> bool:
        return not self.problems


def validate_claims(
    analysis: Analysis,
    evidence: list[Evidence],
    displayed: list[DisplayedEvidence] | None,
) -> ClaimValidation:
    known = {item.evidence_id: item for item in evidence}
    result = ClaimValidation()
    records = list(displayed or [])
    request_view_records = [item for item in records if item.schema_version >= 2]
    if displayed is not None:
        result.display_checked = True
    if request_view_records:
        # Once a run has request-view records, only those prove that a body
        # entered a provider request.  Legacy v1 views cannot be mixed in to
        # rescue a missing v2 record.
        result.display_mode = "request_view"
        display_ids = {item.evidence_id for item in request_view_records}
    elif records:
        result.display_mode = "legacy"
        display_ids = {item.evidence_id for item in records}
    else:
        display_ids = set()
    for claim in [*analysis.facts, *analysis.hypotheses]:
        if claim.claim_id:
            result.claim_ids.append(claim.claim_id)
    for fact in analysis.facts:
        for evidence_id in fact.evidence_ids:
            item = known.get(evidence_id)
            if item is None:
                result.problems.append(f"fact {fact.claim_id or '?'} cites unknown evidence {evidence_id}")
                continue
            if item.stale:
                result.problems.append(f"fact {fact.claim_id or '?'} cites stale evidence {evidence_id}")
                continue
            if displayed is not None and evidence_id not in display_ids:
                result.problems.append(
                    f"fact {fact.claim_id or '?'} cites evidence {evidence_id} "
                    "that was read but never displayed to the model; read it again in a narrower window"
                )
    for hypothesis in analysis.hypotheses:
        for evidence_id in [*hypothesis.supporting_evidence_ids, *hypothesis.contradicting_evidence_ids]:
            item = known.get(evidence_id)
            if item is None:
                result.problems.append(
                    f"hypothesis {hypothesis.claim_id or '?'} cites unknown evidence {evidence_id}"
                )
                continue
            if item.stale:
                result.problems.append(
                    f"hypothesis {hypothesis.claim_id or '?'} cites stale evidence {evidence_id}"
                )
                continue
            if displayed is not None and evidence_id not in display_ids:
                result.problems.append(
                    f"hypothesis {hypothesis.claim_id or '?'} cites evidence {evidence_id} "
                    "that was read but never displayed to the model"
                )
    if analysis.analysis_status == AnalysisStatus.COMPLETE and not analysis.facts:
        result.problems.append("complete analysis requires at least one cited fact")
    if result.problems:
        result.display_ok = not any("never displayed" in item for item in result.problems)
    return result


def handoff_problems(
    analysis: Analysis,
    *,
    packet: ContextPacket | None = None,
    evidence_count: int | None = None,
) -> list[str]:
    """Program-checkable minimum for a handoff: goal, next step, falsifiability."""
    if analysis.analysis_status == AnalysisStatus.FAILED:
        return []
    problems: list[str] = []
    if analysis.analysis_status != AnalysisStatus.FAILED and not (
        analysis.suggested_checks or analysis.missing_evidence
    ):
        problems.append("handoff has no minimal next step: add suggested_checks or missing_evidence")
    for hypothesis in analysis.hypotheses:
        if not hypothesis.contradicting_evidence_ids and not hypothesis.open_questions:
            problems.append(
                f"hypothesis {hypothesis.claim_id or '?'} has neither counterevidence nor an open question"
            )
    if packet is not None:
        if not packet.goal.strip():
            problems.append("handoff is missing the user goal")
        if analysis.facts and not packet.evidence_index:
            problems.append("handoff cites facts but has an empty evidence index")
        indexed = {item.evidence_id for item in packet.evidence_index}
        omitted = {item.evidence_id for item in packet.omissions if item.evidence_id}
        cited = {
            evidence_id
            for claim in [*analysis.facts, *analysis.hypotheses]
            for evidence_id in (
                *getattr(claim, "evidence_ids", []),
                *getattr(claim, "supporting_evidence_ids", []),
                *getattr(claim, "contradicting_evidence_ids", []),
            )
        }
        missing_from_handoff = sorted(cited - indexed - omitted)
        if missing_from_handoff:
            problems.append(
                "handoff evidence index omits cited evidence: " + ", ".join(missing_from_handoff[:5])
            )
        if not packet.not_executed:
            problems.append("handoff does not state which checks remain unexecuted")
        if not packet.versions.get("policy_hash"):
            problems.append("handoff does not record the policy version/hash")
        if packet.investigation is not None and not packet.investigation.active_question.strip():
            problems.append("handoff has no active investigation question")
    if evidence_count == 0 and analysis.facts:
        problems.append("handoff cites facts but no evidence was stored")
    return problems


def support_review_label(analysis: Analysis) -> str:
    statuses = [claim.support_review.status for claim in [*analysis.facts, *analysis.hypotheses]]
    if not statuses:
        return CheckStatus.NOT_RUN.value
    if any(status == CheckStatus.FAIL for status in statuses):
        return CheckStatus.FAIL.value
    if any(status in (CheckStatus.ERROR, CheckStatus.NOT_APPLICABLE) for status in statuses):
        return CheckStatus.ERROR.value
    if all(status == CheckStatus.PASS for status in statuses):
        return CheckStatus.PASS.value
    # A mixed pass/not_run set is not fully reviewed; do not round it up to pass.
    return CheckStatus.NOT_RUN.value