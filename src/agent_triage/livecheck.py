"""End-to-end assertions for the live GitHub issue chain.

The live smoke test must prove more than "a tool name appeared in a log".  The
same assertions are exercised offline with synthetic positive and negative
fixtures, so a broken chain is rejected before anyone spends credentials on it.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .contracts import Analysis, DisplayedEvidence, Evidence
from .errors import ValidationFailure


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationFailure(f"cannot read {path.name}", detail=str(exc)) from exc


def _issue_number(issue_ref: str) -> tuple[str, int]:
    match = re.fullmatch(r"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)#([1-9][0-9]*)", issue_ref)
    if not match:
        raise ValidationFailure("expected issue_ref in owner/repo#number form")
    return match.group(1), int(match.group(2))


def verify_issue_chain(run_dir: Path, *, issue_ref: str, marker: str) -> dict[str, Any]:
    """Validate the model -> tool -> evidence -> request view -> display -> claim chain.

    Checks are recomputed from the artifacts; an old ``checks.json`` pass is not
    trusted.  Raises :class:`ValidationFailure` with the first concrete missing
    link.
    """
    from .claims import validate_claims
    from .request_views import load_request_views, verify_display_records

    repo, number = _issue_number(issue_ref)
    expected_url = f"https://github.com/{repo}/issues/{number}"
    trace_path = run_dir / "trace.jsonl"
    if not trace_path.is_file():
        raise ValidationFailure("run has no trace.jsonl")
    trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tool_trace = [item for item in trace if item.get("kind") == "tool"]
    search = [
        item
        for item in tool_trace
        if item.get("tool_name") == "github_search_issues" and item.get("status") == "ok"
    ]
    read = [
        item
        for item in tool_trace
        if item.get("tool_name") == "github_read_issue" and item.get("status") == "ok"
    ]
    if not search:
        raise ValidationFailure("the model did not complete a successful github_search_issues call")
    if not read:
        raise ValidationFailure("the model only searched; it did not successfully read the issue")
    if len(read) > 1:
        raise ValidationFailure("the trace contains more than one successful issue read; expected one designated read")

    evidence_payload = _read_json(run_dir / "evidence" / "index.json")
    try:
        evidence = [Evidence.model_validate(item) for item in evidence_payload.get("evidence", [])]
    except ValidationError as exc:
        raise ValidationFailure("evidence index is invalid", detail=str(exc)) from exc
    by_id = {item.evidence_id: item for item in evidence}
    read_call = read[0]
    call_id = read_call.get("call_id")
    if not call_id:
        raise ValidationFailure("the issue read has no tool call id")
    arguments = read_call.get("arguments") or {}
    if arguments.get("issue_ref") != issue_ref:
        raise ValidationFailure("the successful read was for a different issue_ref")
    read_refs = [by_id[item] for item in (read_call.get("evidence_refs") or []) if item in by_id]
    read_issue = [item for item in read_refs if item.kind.value == "issue"]
    if not read_issue:
        raise ValidationFailure("the issue read result does not reference the stored issue evidence")
    if len(read_issue) != 1:
        raise ValidationFailure("the issue read must reference exactly one issue evidence item")
    issue = read_issue[0]
    if issue.url != expected_url:
        raise ValidationFailure(f"the designated issue {issue_ref} was not stored as issue evidence")
    if marker not in issue.snippet:
        raise ValidationFailure("the designated issue content does not contain the frozen marker")
    if issue.stale:
        raise ValidationFailure("the issue evidence is stale")
    if issue.sha256 != hashlib.sha256(issue.snippet.encode("utf-8")).hexdigest():
        raise ValidationFailure("the issue evidence source hash does not match its snippet")

    search_hits_target = False
    for call in search:
        for evidence_id in call.get("evidence_refs") or []:
            item = by_id.get(evidence_id)
            if item is not None and item.kind.value == "issue" and item.url == expected_url:
                search_hits_target = True
    if not search_hits_target:
        raise ValidationFailure("github_search_issues did not return the designated issue")

    views = load_request_views(run_dir)
    if not views:
        raise ValidationFailure("run has no provider request views")
    run_id = run_dir.name
    if (run_dir / "run.json").is_file():
        run_id = str(_read_json(run_dir / "run.json").get("run_id") or run_id)
    if any(view.run_id != run_id for view in views):
        raise ValidationFailure("request views belong to a different run")
    called = [view for view in views if call_id in view.tool_call_ids]
    returned = [view for view in views if call_id in view.tool_result_ids]
    if not called:
        raise ValidationFailure("the issue read call never appeared in a model request")
    if not returned:
        raise ValidationFailure("the issue tool result never appeared in a later model request")
    carrying = [
        view
        for view in returned
        if view.status == "response_received"
        and any(item.evidence_id == issue.evidence_id for item in view.evidence_views)
    ]
    if not carrying:
        raise ValidationFailure("the issue evidence body never appeared in a request that received a response")

    display_path = run_dir / "display.json"
    if not display_path.is_file():
        raise ValidationFailure("run has no display.json")
    try:
        displayed = [
            DisplayedEvidence.model_validate(item)
            for item in _read_json(display_path).get("displayed", [])
        ]
    except ValidationError as exc:
        raise ValidationFailure("display.json is invalid", detail=str(exc)) from exc
    if not any(item.evidence_id == issue.evidence_id for item in displayed):
        raise ValidationFailure("the issue evidence was stored but never displayed to the model")
    display_problems, display_mode = verify_display_records(displayed, views, evidence)
    if display_mode != "request_view":
        raise ValidationFailure("display.json is legacy v1; it cannot prove the issue body entered a request")
    if display_problems:
        raise ValidationFailure("display proof does not match the request views", detail="; ".join(display_problems[:3]))

    try:
        analysis = Analysis.model_validate(_read_json(run_dir / "analysis.json"))
    except ValidationError as exc:
        raise ValidationFailure("analysis.json is invalid", detail=str(exc)) from exc
    cited = {
        evidence_id
        for claim in [*analysis.facts, *analysis.hypotheses]
        for evidence_id in (
            *getattr(claim, "evidence_ids", []),
            *getattr(claim, "supporting_evidence_ids", []),
            *getattr(claim, "contradicting_evidence_ids", []),
        )
    }
    if issue.evidence_id not in cited:
        raise ValidationFailure("no claim cites the designated issue evidence")
    validation = validate_claims(analysis, evidence, displayed)
    if validation.problems:
        raise ValidationFailure(
            "claims do not satisfy reference/display checks",
            detail="; ".join(validation.problems[:3]),
        )

    report = (run_dir / "report.md").read_text(encoding="utf-8") if (run_dir / "report.md").is_file() else ""
    if issue.evidence_id not in report:
        raise ValidationFailure("the report does not reference the designated issue evidence")

    carrying_view = carrying[0]
    return {
        "run_id": run_dir.name,
        "issue_ref": issue_ref,
        "issue_url": expected_url,
        "evidence_id": issue.evidence_id,
        "issue_sha256": issue.sha256,
        "search_call_ids": [item.get("call_id") for item in search],
        "read_call_id": call_id,
        "request_id": carrying_view.request_id,
        "request_payload_sha256": carrying_view.payload_sha256,
        "displayed_views": len(
            [item for item in displayed if item.evidence_id == issue.evidence_id]
        ),
        "cited_claims": sorted(
            claim.claim_id or "?"
            for claim in [*analysis.facts, *analysis.hypotheses]
            if issue.evidence_id
            in (
                *getattr(claim, "evidence_ids", []),
                *getattr(claim, "supporting_evidence_ids", []),
                *getattr(claim, "contradicting_evidence_ids", []),
            )
        ),
    }
