"""Sourced state views, regenerated from existing records without a model call."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

from .contracts import FailedAttempt, InvestigationState, TraceRecord
from .errors import ValidationFailure
from .windows import query_terms, relevance

NOT_EXECUTED = [
    "no test, reproduction or fix has been executed by this agent",
    "suggested checks remain not_run until a human records an outcome",
]


def arguments_hash(arguments: dict[str, Any] | None) -> str:
    return hashlib.sha256(json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def failed_attempts(trace: list[TraceRecord], question: str = "") -> list[FailedAttempt]:
    terms = query_terms(question)
    ranked: list[tuple[int, int, FailedAttempt]] = []
    for index, record in enumerate(trace):
        if record.kind.value != "tool" or record.status == "ok":
            continue
        retry = {
            "missing": "correct the path or provide the missing file",
            "line_out_of_range": "provide a valid line range",
            "sensitive_path": "owner configuration must authorize the source; do not retry unchanged",
            "path_escape": "provide a path inside the verified project root",
            "tool_timeout": "time budget or tool availability must change",
        }.get(record.error_code or "", "unknown")
        attempt = FailedAttempt(
            schema_version=2,
            action=f"{record.tool_name} ({record.call_id})"[:500],
            status=record.status, reason=record.error_code or "unknown",
            call_id=record.call_id, arguments_sha256=arguments_hash(record.arguments),
            source_ref=f"trace.jsonl:{index + 1}", retry_condition=retry,
        )
        score = relevance(json.dumps(record.arguments or {}, ensure_ascii=False) + " " +
                          (record.error_code or ""), terms)
        ranked.append((-score, index, attempt))
    # No explicit resolution event exists in v1. A later successful call is
    # not permission to erase an earlier failure. Keep all identities in JSON.
    return [item for _score, _index, item in sorted(ranked, key=lambda row: row[:2])]


def project_state(*, investigation: InvestigationState | None,
                  trace: list[TraceRecord], budget_state: Any, is_final_turn: bool,
                  goal: str, versions: dict[str, str]) -> dict[str, Any]:
    failures = failed_attempts(trace, investigation.active_question if investigation else "")
    no_progress = [
        {"call_id": entry.call_id, "action": entry.action, "status": entry.status,
         "source_ref": entry.source_ref or f"investigation.json:attempt_ledger[{index}]",
         "observation": "successful tool return without new evidence; resolution unknown"}
        for index, entry in enumerate(investigation.attempt_ledger if investigation else [])
        if entry.status == "ok" and not entry.new_information
    ]
    return {
        "schema_version": 1,
        "goal": goal,
        "versions": dict(versions),
        "active_question": investigation.active_question if investigation else None,
        "needed_materials": list(investigation.needed_materials) if investigation else [],
        "counterevidence_ids": list(investigation.counterevidence_ids) if investigation else [],
        "required_evidence_ids": list(investigation.required_evidence_ids) if investigation else [],
        "omissions": [item.model_dump(mode="json") for item in investigation.omissions] if investigation else [],
        "prior_omission_count": len(investigation.omissions) if investigation else 0,
        "failed_attempts": [item.model_dump(mode="json") for item in failures],
        "no_progress_attempts": no_progress,
        "not_executed": list(NOT_EXECUTED),
        "next_action_reason": investigation.next_action_reason if investigation else None,
        "budget_used": budget_state.model_dump(mode="json"),
        "final_turn": is_final_turn,
        "sources": ["event.json", "versions.json", "investigation.json", "trace.jsonl", "checks.json"],
    }


def render_state(projection: dict[str, Any], limit: int,
                 estimate: Callable[[str], int]) -> tuple[str, list[tuple[str, bool, int]]]:
    failures = projection["failed_attempts"]
    no_progress = projection["no_progress_attempts"]
    minimal = (
        "WORKING STATE (deterministic; unresolved)\n"
        "goal: event.json:goal; versions: versions.json\n"
        "checks/tests/fix=not_run; suggested actions are not executions\n"
        f"failed={len(failures)} no_progress={len(no_progress)} "
        f"needed={len(projection['needed_materials'])} counter={len(projection['counterevidence_ids'])} "
        f"prior_omissions={projection['prior_omission_count']}\n"
        "all details: investigation.json, trace.jsonl, checks.json; "
        "unexpanded entries remain unresolved; this-turn omissions: context.json:omissions\n"
    )
    if projection["active_question"]:
        # The full question remains in investigation.json; a reference is an
        # explicit fallback, never an empty 'resolved' question.
        question = "active_question: " + projection["active_question"]
        if estimate(minimal + question) <= limit:
            minimal += question + "\n"
        else:
            minimal += "active_question: see investigation.json:active_question (unexpanded)\n"
    if estimate(minimal) > limit:
        raise ValidationFailure("critical working state does not fit the context budget",
                                detail=f"need {estimate(minimal)} estimated tokens; available {limit}")
    parts = [minimal.rstrip()]
    decisions: list[tuple[str, bool, int]] = []
    rows: list[tuple[str, str]] = []
    for key in ("needed_materials", "counterevidence_ids", "required_evidence_ids"):
        for index, value in enumerate(projection[key]):
            rows.append((f"investigation.json:{key}[{index}]", f"{key}: {value}"))
    for attempt in failures:
        rows.append((attempt["source_ref"],
                     f"failed call={attempt['call_id']} status={attempt['status']} "
                     f"error={attempt['reason']} source={attempt['source_ref']} "
                     f"retry_condition={attempt['retry_condition']}"))
    for attempt in no_progress:
        rows.append((attempt["source_ref"],
                     f"no_progress call={attempt['call_id']} action={attempt['action']} "
                     f"status={attempt['status']} source={attempt['source_ref']}"))
    for index, omission in enumerate(projection["omissions"]):
        rows.append((f"investigation.json:omissions[{index}]",
                     f"omission: {omission['request']} {omission['reason']}"))
    if projection["next_action_reason"]:
        rows.append(("investigation.json:next_action_reason", "suggested next: " + projection["next_action_reason"]))
    rows.append(("budget", "budget_used: " + json.dumps(projection["budget_used"], sort_keys=True)))
    for source, row in rows:
        candidate = "\n".join([*parts, row])
        include = estimate(candidate) <= limit
        decisions.append((source, include, estimate(row) if include else 0))
        if include:
            parts.append(row)
    return "\n".join(parts), decisions
