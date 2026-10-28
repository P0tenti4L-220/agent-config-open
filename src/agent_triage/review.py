"""External semantic review import and derived claim-support status.

The model may write a ``support_review`` object in its analysis JSON.  That
object is *not* evidence of an external review: a model can type
``reviewer="owner", status="pass"`` and nothing prevents it.  This module makes
that boundary explicit:

* :func:`sanitize_model_review` downgrades every model-supplied review to
  ``not_run`` with a fixed note, before the analysis is stored;
* only :func:`import_external_review` can create a trusted review record;
* the record is bound to the canonical analysis content and to the displayed
  evidence snapshot, and becomes stale when either changes;
* a hypothesis ``pass`` means "the reasoning is sound and uncertainty is
  preserved", never "the root cause is proven".
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError, field_validator, model_validator

from .contracts import (
    Analysis,
    CheckResult,
    CheckStatus,
    ClaimId,
    DisplayedEvidence,
    Evidence,
    EvidenceId,
    NonEmptyStr,
    ShortStr,
    StrictModel,
)
from .errors import InputError, ValidationFailure
from .store import RunStore, atomic_write_json, atomic_write_text, hash_file, read_json, utcnow

MODEL_REVIEW_NOTE = (
    "model-supplied support_review was ignored at the model boundary; "
    "only an imported record-review can establish claim support"
)
REVIEW_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,63}$")
MAX_REVIEW_BYTES = 1024 * 1024


class ReviewEvidenceRef(StrictModel):
    evidence_id: EvidenceId
    view_ref: str | None = None


class ReviewClaim(StrictModel):
    claim_id: ClaimId
    verdict: CheckStatus
    rationale: NonEmptyStr
    evidence_refs: list[ReviewEvidenceRef] = Field(default_factory=list, max_length=20)


class NextActionReview(StrictModel):
    verdict: CheckStatus
    rationale: NonEmptyStr


class ExternalReview(StrictModel):
    schema_version: Literal[1] = 1
    review_id: str
    run_id: str
    analysis_content_hash: str
    evidence_snapshot_hash: str
    reviewer: NonEmptyStr
    method: Literal["human"]
    reviewed_at: datetime
    rubric_version: NonEmptyStr
    claims: list[ReviewClaim] = Field(min_length=1, max_length=60)
    next_action: NextActionReview
    unresolved_questions: list[ShortStr] = Field(default_factory=list, max_length=20)

    @field_validator("review_id")
    @classmethod
    def _review_id_shape(cls, value: str) -> str:
        if not REVIEW_ID_RE.fullmatch(value):
            raise ValueError("review_id contains unsupported characters")
        return value

    @field_validator("analysis_content_hash", "evidence_snapshot_hash")
    @classmethod
    def _hash_shape(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("expected a lowercase sha256 digest")
        return value

    @model_validator(mode="after")
    def _no_duplicate_claims(self) -> "ExternalReview":
        ids = [claim.claim_id for claim in self.claims]
        if len(ids) != len(set(ids)):
            raise ValueError("a review must not repeat a claim_id")
        return self


def canonical_analysis(analysis: Analysis) -> str:
    """Canonical JSON of the diagnostic content, excluding review metadata.

    Importing a review must not change the hash it was bound to, so the mutable
    ``support_review`` block is removed before hashing.
    """
    payload = analysis.model_dump(mode="json")

    def strip(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: strip(item) for key, item in value.items() if key != "support_review"}
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value

    return _canonical(strip(payload))


def analysis_content_hash(analysis: Analysis) -> str:
    return _sha256_text(canonical_analysis(analysis))


def view_ref(displayed: DisplayedEvidence) -> str:
    request = f"{displayed.request_id}:" if displayed.schema_version >= 2 else ""
    return (
        f"{displayed.evidence_id}@{displayed.turn}:{request}{displayed.source.value}:"
        f"{displayed.mode}:{displayed.visible_sha256[:12]}"
    )


def evidence_snapshot_hash(evidence: list[Evidence], displayed: list[DisplayedEvidence] | None) -> str:
    views = sorted(view_ref(item) for item in (displayed or []))
    payload = {
        "evidence": [
            {
                "evidence_id": item.evidence_id,
                "kind": item.kind.value,
                "path": item.path,
                "url": item.url,
                "sha256": item.sha256,
                "snippet_sha256": _sha256_text(item.snippet),
                "stale": item.stale,
                "stale_reason": item.stale_reason,
            }
            for item in sorted(evidence, key=lambda item: item.evidence_id)
        ],
        "views": views,
    }
    return _sha256_text(_canonical(payload))


def _canonical(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sanitize_model_review(analysis: Analysis) -> tuple[Analysis, int]:
    """Downgrade model-authored review metadata to ``not_run``.

    Returns the rewritten analysis and how many claims were changed.  The
    original diagnostic content is preserved; only the review block changes.
    """
    from .contracts import SupportReview

    changed = 0
    facts = []
    for fact in analysis.facts:
        review = fact.support_review
        if (
            review.status != CheckStatus.NOT_RUN
            or review.reviewer
            or review.method
            or review.scope
            or review.reviewed_at
        ):
            fact = fact.model_copy(
                update={"support_review": SupportReview(status=CheckStatus.NOT_RUN, note=MODEL_REVIEW_NOTE)}
            )
            changed += 1
        facts.append(fact)
    hypotheses = []
    for hypothesis in analysis.hypotheses:
        review = hypothesis.support_review
        if (
            review.status != CheckStatus.NOT_RUN
            or review.reviewer
            or review.method
            or review.scope
            or review.reviewed_at
        ):
            hypothesis = hypothesis.model_copy(
                update={"support_review": SupportReview(status=CheckStatus.NOT_RUN, note=MODEL_REVIEW_NOTE)}
            )
            changed += 1
        hypotheses.append(hypothesis)
    if not changed:
        return analysis, 0
    return analysis.model_copy(update={"facts": facts, "hypotheses": hypotheses}), changed


def load_external_review(path: Path) -> ExternalReview:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise InputError(f"cannot read review file: {path.name}", detail=str(exc)) from exc
    if len(raw) > MAX_REVIEW_BYTES:
        raise InputError(f"review file is larger than {MAX_REVIEW_BYTES} bytes")
    try:
        return ExternalReview.model_validate_json(raw)
    except ValidationError as exc:
        raise ValidationFailure("review file does not match the ExternalReview contract", detail=str(exc)) from exc


def _claim_ids(analysis: Analysis) -> set[str]:
    return {claim.claim_id for claim in [*analysis.facts, *analysis.hypotheses] if claim.claim_id}


def import_external_review(
    *,
    store: RunStore,
    analysis: Analysis,
    evidence: list[Evidence],
    displayed: list[DisplayedEvidence] | None,
    review: ExternalReview,
    source_path: Path,
) -> dict[str, Any]:
    if review.run_id != store.run_id:
        raise ValidationFailure(
            "review belongs to a different run",
            detail=f"review run_id={review.run_id}; current run_id={store.run_id}",
        )
    current_analysis = analysis_content_hash(analysis)
    if review.analysis_content_hash != current_analysis:
        raise ValidationFailure(
            "review was bound to different analysis content",
            detail="re-import is required after analysis.json changes",
        )
    current_evidence = evidence_snapshot_hash(evidence, displayed)
    if review.evidence_snapshot_hash != current_evidence:
        raise ValidationFailure("review was bound to a different evidence snapshot")

    known_claims = _claim_ids(analysis)
    references = {item.evidence_id for item in evidence}
    views = {item.evidence_id: set() for item in evidence}
    for item in displayed or []:
        views.setdefault(item.evidence_id, set()).add(view_ref(item))
    for claim in review.claims:
        if claim.claim_id not in known_claims:
            raise ValidationFailure(f"review references unknown claim {claim.claim_id}")
        for ref in claim.evidence_refs:
            if ref.evidence_id not in references:
                raise ValidationFailure(f"review references unknown evidence {ref.evidence_id}")
            if ref.view_ref is not None and ref.view_ref not in views.get(ref.evidence_id, set()):
                raise ValidationFailure(
                    f"review references a view that was not displayed: {ref.evidence_id}"
                )

    source_sha = hash_file(source_path)
    target = store.dir / "evidence" / "reviews" / f"{review.review_id}.json"
    if target.exists():
        existing = read_json(target)
        if existing.get("source_sha256") != source_sha:
            raise ValidationFailure(
                "a review with this id already exists with different content; use a new review_id"
            )
        record = {**existing, "imported_at": existing.get("imported_at") or utcnow().isoformat()}
    else:
        record = {
            "schema_version": 1,
            "imported_at": utcnow().isoformat(),
            "source_sha256": source_sha,
            "review": review.model_dump(mode="json"),
        }
        atomic_write_json(target, record)

    view = build_review_view(store=store, analysis=analysis, evidence=evidence, displayed=displayed)
    atomic_write_json(store.dir / "review-status.json", view)
    return {
        "review_id": review.review_id,
        "run_id": store.run_id,
        "imported": True,
        "record": target.relative_to(store.dir).as_posix(),
        "overall": view["overall"],
        "claims": view["claims"],
    }


def _load_reviews(store: RunStore) -> tuple[list[ExternalReview], list[str]]:
    reviews: list[ExternalReview] = []
    problems: list[str] = []
    directory = store.dir / "evidence" / "reviews"
    if not directory.is_dir():
        return reviews, problems
    for path in sorted(directory.glob("*.json")):
        try:
            payload = read_json(path)
            reviews.append(ExternalReview.model_validate(payload.get("review", payload)))
        except (InputError, ValidationError) as exc:
            problems.append(f"{path.name}: {exc}")
    return reviews, problems


def build_review_view(
    *,
    store: RunStore,
    analysis: Analysis,
    evidence: list[Evidence],
    displayed: list[DisplayedEvidence] | None,
) -> dict[str, Any]:
    reviews, problems = _load_reviews(store)
    current_hash = analysis_content_hash(analysis)
    current_evidence = evidence_snapshot_hash(evidence, displayed)
    valid = [
        review
        for review in reviews
        if review.analysis_content_hash == current_hash and review.evidence_snapshot_hash == current_evidence
    ]
    stale = [review for review in reviews if review not in valid]

    claims: dict[str, dict[str, Any]] = {}
    for claim in [*analysis.facts, *analysis.hypotheses]:
        if not claim.claim_id:
            continue
        matching = [
            (review, item)
            for review in valid
            for item in review.claims
            if item.claim_id == claim.claim_id
        ]
        if not matching:
            claims[claim.claim_id] = {
                "status": "not_run",
                "rationale": "no current external review covers this claim",
                "review_id": None,
                "reviewer": None,
                "reviewed_at": None,
            }
            continue
        if any(item.verdict == CheckStatus.FAIL for _, item in matching):
            review, item = next((review, item) for review, item in matching if item.verdict == CheckStatus.FAIL)
            status = "fail"
        elif all(item.verdict == CheckStatus.PASS for _, item in matching):
            review, item = matching[-1]
            status = "pass"
        else:
            review, item = matching[-1]
            status = "not_run"
        claims[claim.claim_id] = {
            "status": status,
            "rationale": item.rationale,
            "review_id": review.review_id,
            "reviewer": review.reviewer,
            "reviewed_at": review.reviewed_at.isoformat(),
            "hypothesis": claim.claim_id.startswith("claim-hyp-"),
        }

    statuses = [item["status"] for item in claims.values()]
    if any(status == "fail" for status in statuses):
        overall = "fail"
    elif statuses and all(status == "pass" for status in statuses):
        overall = "pass"
    else:
        overall = "not_run"

    next_action = None
    if valid:
        latest = sorted(valid, key=lambda review: review.reviewed_at)[-1]
        next_action = {
            "verdict": latest.next_action.verdict.value,
            "rationale": latest.next_action.rationale,
            "review_id": latest.review_id,
        }
    return {
        "schema_version": 1,
        "run_id": store.run_id,
        "analysis_content_hash": current_hash,
        "evidence_snapshot_hash": current_evidence,
        "overall": overall,
        "claims": claims,
        "next_action": next_action,
        "reviews": [
            {
                "review_id": review.review_id,
                "reviewer": review.reviewer,
                "reviewed_at": review.reviewed_at.isoformat(),
                "current": review in valid,
            }
            for review in reviews
        ],
        "stale_review_ids": [review.review_id for review in stale],
        "problems": problems,
    }


def claim_support_check(
    *,
    store: RunStore,
    analysis: Analysis,
    evidence: list[Evidence],
    displayed: list[DisplayedEvidence] | None,
    analysis_sha: str,
) -> tuple[CheckResult, dict[str, Any]]:
    view = build_review_view(store=store, analysis=analysis, evidence=evidence, displayed=displayed)
    overall = view["overall"]
    now = utcnow()
    if overall in {"pass", "fail"}:
        reviewed = [item for item in view["claims"].values() if item["status"] == overall]
        reviewer = next((item["reviewer"] for item in reviewed if item["reviewer"]), "unknown")
        reviewed_at = next((item["reviewed_at"] for item in reviewed if item["reviewed_at"]), None)
        return (
            CheckResult(
                check_id="claim-support",
                status=CheckStatus(overall),
                description="semantic support was reviewed by an imported external record",
                command="review:external-v1",
                exit_code=0 if overall == "pass" else 1,
                output_ref="review-status.json",
                target_sha256=analysis_sha,
                scope=(
                    "imported review bound to this analysis and evidence snapshot; a hypothesis pass "
                    "means the reasoning and uncertainty are reasonable, not that the root cause is proven"
                ),
                executed_by=reviewer,
                executed_at=datetime.fromisoformat(reviewed_at) if reviewed_at else now,
            ),
            view,
        )
    reasons = ["no current external review covers every claim"]
    if view["stale_review_ids"]:
        reasons.append("stale reviews ignored: " + ", ".join(view["stale_review_ids"][:5]))
    if view["problems"]:
        reasons.append("; ".join(view["problems"][:5]))
    return (
        CheckResult(
            check_id="claim-support",
            status=CheckStatus.NOT_RUN,
            description="whether each citation actually supports its claim",
            command=None,
            exit_code=None,
            output_ref=None,
            target_sha256=None,
            scope=(
                "semantic support is not implied by schema, reference or display checks; "
                + "; ".join(reasons)
            ),
            executed_by=None,
            executed_at=None,
        ),
        view,
    )


def write_review_status(store: RunStore, view: dict[str, Any]) -> Path:
    return atomic_write_json(store.dir / "review-status.json", view)


def render_review_section(view: dict[str, Any] | None) -> list[str]:
    lines = ["", "## 外部复核（与结构/引用检查分开）", ""]
    if not view or not view.get("claims"):
        lines.append("- 没有导入任何外部复核记录；`claim-support` 保持 `not_run`。")
        return lines
    lines.append(f"- 总体：`{view['overall']}`；复核记录：{len(view.get('reviews', []))} 份。")
    for claim_id, item in view["claims"].items():
        suffix = "（假设：通过仅表示推理与不确定性表达合理，不表示根因证实）" if item.get("hypothesis") else ""
        lines.append(
            f"- `{claim_id}`：`{item['status']}`；复核人 `{item.get('reviewer') or 'unknown'}`；"
            f"理由：{_safe_review_text(item.get('rationale'))}{suffix}"
        )
    if view.get("stale_review_ids"):
        lines.append("- 已过期复核：" + ", ".join(f"`{item}`" for item in view["stale_review_ids"]))
    return lines


def _safe_review_text(value: str | None) -> str:
    text = (value or "").replace("\n", " ").strip()
    return text[:500] or "-"


def atomic_write_review_text(path: Path, text: str) -> Path:
    return atomic_write_text(path, text)