"""Three small, versioned investigation playbooks plus a generic fallback.

A playbook is selected from the event's error type and symptom keywords.  At
most one is loaded; when routing is ambiguous the generic playbook is used.
Playbooks teach an investigation order, they do not add tools, permissions or
facts.  The loaded id/version/hash is recorded for every run.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .ingest import error_type_of

PLAYBOOK_CATALOG_VERSION = "playbooks-v1"


class Playbook(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    playbook_id: str
    version: str
    title: str
    summary: str
    first_question: str
    trigger_error_types: tuple[str, ...] = ()
    trigger_keywords: tuple[str, ...] = ()
    excluded_when: tuple[str, ...] = ()
    required_evidence: tuple[str, ...] = ()
    steps: tuple[str, ...] = ()
    counterevidence_questions: tuple[str, ...] = ()
    stop_conditions: tuple[str, ...] = ()
    output_requirements: tuple[str, ...] = ()
    good_example: str = ""
    bad_example: str = ""
    eval_ids: tuple[str, ...] = ()


STRUCTURED = Playbook(
    playbook_id="structured-output-mismatch",
    version="v1",
    title="Structured output mismatch",
    summary="A reader/consumer and a producer disagree about fields, types or envelope shape.",
    first_question="Which concrete input did the failing access actually receive, and which reader assumption does it contradict?",
    trigger_error_types=("KeyError", "IndexError", "TypeError", "AttributeError", "JSONDecodeError", "ValidationError"),
    trigger_keywords=("schema", "field", "items", "keyerror", "type", "api", "mcp", "adapter", "payload", "json"),
    excluded_when=("the failure is a timeout, permission error or network outage",),
    required_evidence=(
        "the exact error line and the access expression",
        "the producer contract or version marker",
        "the actual input shape at the failing boundary (bounded snippet)",
    ),
    steps=(
        "Read the traceback location and the single access that failed.",
        "Read the producer/adapter that defines the expected shape; note its version.",
        "Read a bounded window of the actual input near the failing key or element.",
        "State the observed mismatch (e.g. reader indexes items, input exposes checks) as one fact per claim.",
        "Keep adapter absence or version drift as a hypothesis; list what would falsify it.",
    ),
    counterevidence_questions=(
        "Could an earlier transformation rename or wrap the field before this access?",
        "Does another input version satisfy the current reader, which would weaken the adapter hypothesis?",
    ),
    stop_conditions=(
        "the actual input shape and the reader assumption are both evidenced, or",
        "the actual input cannot be read inside the allowed root: report the gap instead of guessing.",
    ),
    output_requirements=(
        "A minimal check that exercises the same input shape and reader branch.",
        "A distinct check for unknown format/version behaviour.",
    ),
    good_example="The reader indexes payload['items']; the captured input exposes only the top-level 'checks' key.",
    bad_example="The cache is corrupted, because a version log mentions a cache change.",
    eval_ids=("AC01", "AC06", "NC13", "NC14"),
)

TOOL_FAILURE = Playbook(
    playbook_id="tool-failure-and-recovery",
    version="v1",
    title="Tool failure and recovery",
    summary="A timeout, permission error, pagination limit or retry loop hides whether data was read.",
    first_question="Did the action not run, fail, or partially run, and what safe recovery point follows from the recorded attempts?",
    trigger_error_types=("TimeoutError", "PermissionError", "ConnectionError", "HTTPError"),
    trigger_keywords=("timeout", "timed out", "permission", "403", "401", "rate limit", "pagination", "retry", "connection", "network"),
    excluded_when=("the failure is a deterministic field/type mismatch",),
    required_evidence=(
        "the failing tool call, its parameters and attempt count",
        "the error class or status code",
        "whether any result or evidence was persisted before the failure",
    ),
    steps=(
        "Classify the failure: not executed, failed, partially read, or succeeded with an error text.",
        "Check the attempt ledger and whether a retry produced new information.",
        "Read the smallest continuation point rather than repeating the whole request.",
        "State partial coverage explicitly; do not turn an error into an empty result.",
        "Recommend a recovery step with its precondition and expected result.",
    ),
    counterevidence_questions=(
        "Would a second identical attempt return new bytes, or is the failure deterministic?",
        "Does the truncated page contain the decisive item, making a broader retry unnecessary?",
    ),
    stop_conditions=(
        "the failure class and the safe continuation point are known, or",
        "the budget is exhausted: return the partial state and the missing material.",
    ),
    output_requirements=(
        "Separate not_executed, failed and partial_read in the handoff.",
        "Name one recovery action and what a failure of that action would mean.",
    ),
    good_example="MCP search timed out after one retry; local evidence is complete, remote history is unchecked and marked partial.",
    bad_example="GitHub returned no issues, so this problem has never been reported elsewhere.",
    eval_ids=("AC08", "AC09", "AC13", "NC05"),
)

CONTEXT_HANDOFF = Playbook(
    playbook_id="context-and-handoff-gap",
    version="v1",
    title="Context and handoff gap",
    summary="A new session, moved root, stale evidence or repeated request blocks continuation.",
    first_question="What does the next executor need to locate the object and choose one next check without the old chat?",
    trigger_error_types=("FileNotFoundError", "OSError"),
    trigger_keywords=("handoff", "context", "stale", "resume", "checkpoint", "session", "migrat", "path", "missing file", "workspace"),
    excluded_when=("the failure is a freshly introduced schema mismatch with complete local input",),
    required_evidence=(
        "current project root identity and version",
        "evidence index with capture time and staleness",
        "attempt ledger and unexecuted checks",
        "the exact missing material and where it should come from",
    ),
    steps=(
        "Confirm the current root and the artifact set; never read the old root by habit.",
        "List what was attempted, what failed and what has not run.",
        "Check each cited evidence item for staleness before using it.",
        "State the smallest material that unblocks the next executor.",
        "Write a handoff that contains goal, version, claims, counterevidence gap, next check and evidence index.",
    ),
    counterevidence_questions=(
        "Is the cited evidence still current, or does it describe a previous root/version?",
        "Can the next executor reproduce the next step from the handoff without the original conversation?",
    ),
    stop_conditions=(
        "the handoff is self-contained and the missing material is named, or",
        "the current root/version cannot be confirmed: stop and report the mismatch.",
    ),
    output_requirements=(
        "A next check with precondition, expected result and failure meaning.",
        "A material gap list that is specific enough to act on.",
    ),
    good_example="The run was resumed after a root move; old evidence is stale, the new root is named, and one check remains not_run.",
    bad_example="Please provide more context.",
    eval_ids=("AC12", "AC05", "NC12", "NC20"),
)

GENERIC = Playbook(
    playbook_id="generic-triage",
    version="v1",
    title="Generic triage",
    summary="No specialised route matched; use the smallest evidence that separates the candidate explanations.",
    first_question="What is the observed failure, and which single piece of evidence would most change the leading explanation?",
    trigger_keywords=(),
    required_evidence=(
        "the exact error and its location",
        "the relevant input or environment fact",
        "whether a check or reproduction has actually run",
    ),
    steps=(
        "Separate the observed failure from any causal explanation.",
        "Read the smallest relevant evidence window; avoid scanning the whole project.",
        "Name at most three hypotheses and the observation that would weaken each.",
        "Record attempts and missing material, then propose one minimal next check.",
    ),
    counterevidence_questions=("What observation would make the leading hypothesis wrong?",),
    stop_conditions=(
        "the leading hypothesis has support and a counter-check, or",
        "further reading needs material outside the allowed root: report the gap.",
    ),
    output_requirements=("A minimal next check with status not_run.",),
    good_example="One log line, one source location and one missing input are separated; the next check is specific.",
    bad_example="The root cause is probably an environment issue.",
    eval_ids=("AC06", "NC14"),
)

CATALOG: tuple[Playbook, ...] = (STRUCTURED, TOOL_FAILURE, CONTEXT_HANDOFF, GENERIC)


def _canonical(playbook: Playbook) -> str:
    return json.dumps(playbook.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def playbook_hash(playbook: Playbook) -> str:
    return hashlib.sha256(_canonical(playbook).encode("utf-8")).hexdigest()


def catalog_hash() -> str:
    payload = [_canonical(playbook) for playbook in CATALOG]
    return hashlib.sha256("\n".join(payload).encode("utf-8")).hexdigest()


PLAYBOOK_CATALOG_HASH = catalog_hash()


def catalog_manifest() -> dict[str, str]:
    return {
        "playbook_catalog_version": PLAYBOOK_CATALOG_VERSION,
        "playbook_catalog_hash": PLAYBOOK_CATALOG_HASH,
    }


def get_playbook(playbook_id: str) -> Playbook:
    for playbook in CATALOG:
        if playbook.playbook_id == playbook_id:
            return playbook
    return GENERIC


def route_playbook(event: Any) -> Playbook:
    """Pick at most one playbook; ties and weak matches use the generic one."""
    text = f"{event.symptom} {event.goal}".casefold()
    error_type = error_type_of(event.symptom)
    scores: dict[str, int] = {}
    for playbook in CATALOG:
        if playbook.playbook_id == GENERIC.playbook_id:
            continue
        score = 0
        if error_type in playbook.trigger_error_types:
            score += 3
        score += sum(1 for keyword in playbook.trigger_keywords if keyword.casefold() in text)
        scores[playbook.playbook_id] = score
    if not scores:
        return GENERIC
    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    best_id, best_score = ranked[0]
    if best_score <= 0:
        return GENERIC
    if len(ranked) > 1 and ranked[1][1] == best_score:
        return GENERIC
    return get_playbook(best_id)


def render_playbook(playbook: Playbook, *, max_chars: int = 4000) -> str:
    lines = [
        f"SELECTED PLAYBOOK [{playbook.playbook_id} {playbook.version}] {playbook.title}",
        f"summary: {playbook.summary}",
        f"first question: {playbook.first_question}",
        "required evidence:",
        *[f"- {item}" for item in playbook.required_evidence],
        "steps:",
        *[f"{index}. {item}" for index, item in enumerate(playbook.steps, start=1)],
        "counterevidence:",
        *[f"- {item}" for item in playbook.counterevidence_questions],
        "stop conditions:",
        *[f"- {item}" for item in playbook.stop_conditions],
        "output requirements:",
        *[f"- {item}" for item in playbook.output_requirements],
    ]
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n<playbook truncated for context budget>"