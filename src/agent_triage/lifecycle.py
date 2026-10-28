"""Internal lifecycle checks: the program-side half of the behaviour policy.

This is deliberately not a general plugin bus.  There is a fixed event list, a
typed decision, and one execution path per event.  Checks receive an immutable
snapshot and return ``allow``/``deny``/``degrade``; a deny on a critical check
blocks the action before it happens.  Every decision is recorded with a stable
rule id and reason code in ``lifecycle.json``.
"""

from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .contracts import RULE_ID_RE
from .policy import POLICY_VERSION


class LifecycleEvent(StrEnum):
    RUN_START = "run_start"
    BEFORE_MODEL = "before_model"
    BEFORE_TOOL = "before_tool"
    AFTER_TOOL = "after_tool"
    BEFORE_FINALIZE = "before_finalize"
    RESUME = "resume"
    BEFORE_PROMOTE_LESSON = "before_promote_lesson"


class Decision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    DEGRADE = "degrade"


INTERNAL_RULE = "R00"


@dataclass(frozen=True)
class HookInput:
    event: LifecycleEvent
    run_id: str
    turn: int
    call_id: str | None
    policy_version: str
    data: Mapping[str, Any]
    budget: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HookResult:
    decision: Decision
    rule_id: str
    reason_code: str
    message: str
    affected: tuple[str, ...] = ()
    duration_ms: int = 0


@dataclass(frozen=True)
class PolicyCheck:
    name: str
    event: LifecycleEvent
    rule_id: str
    func: Callable[[HookInput], HookResult]
    critical: bool = True
    timeout_ms: int = 1000


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def allow(rule_id: str, reason_code: str = "ok", message: str = "") -> HookResult:
    return HookResult(Decision.ALLOW, rule_id, reason_code, message)


def deny(rule_id: str, reason_code: str, message: str, *, affected: tuple[str, ...] = ()) -> HookResult:
    return HookResult(Decision.DENY, rule_id, reason_code, message, affected)


def degrade(rule_id: str, reason_code: str, message: str, *, affected: tuple[str, ...] = ()) -> HookResult:
    return HookResult(Decision.DEGRADE, rule_id, reason_code, message, affected)


class LifecycleEngine:
    """Ordered checks, deny-wins aggregation, bounded records."""

    def __init__(
        self,
        checks: list[PolicyCheck],
        *,
        run_id: str = "run-unknown",
        policy_version: str = POLICY_VERSION,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.checks = list(checks)
        self.run_id = run_id
        self.policy_version = policy_version
        self.clock = clock
        self.records: list[dict[str, Any]] = []

    def checks_for(self, event: LifecycleEvent) -> list[PolicyCheck]:
        return [check for check in self.checks if check.event == event]

    def evaluate(
        self,
        event: LifecycleEvent,
        *,
        data: Mapping[str, Any] | None = None,
        turn: int = 0,
        call_id: str | None = None,
        budget: Mapping[str, Any] | None = None,
    ) -> HookResult:
        checks = self.checks_for(event)
        snapshot = HookInput(
            event=event,
            run_id=self.run_id,
            turn=turn,
            call_id=call_id,
            policy_version=self.policy_version,
            data=_freeze(data or {}),
            budget=_freeze(budget or {}),
        )
        results: list[HookResult] = []
        for check in checks:
            result = self._invoke(check, snapshot)
            results.append(result)
            self._record(event, check, result, snapshot)
        return self._aggregate(event, checks, results)

    # -- internals ----------------------------------------------------------
    def _invoke(self, check: PolicyCheck, snapshot: HookInput) -> HookResult:
        """Run one check in a bounded worker; a late result is discarded."""
        outcome: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

        def target() -> None:
            try:
                outcome.put((True, check.func(snapshot)))
            except BaseException as exc:  # noqa: BLE001 - a failing check is a decision
                outcome.put((False, exc))

        started = self.clock()
        worker = threading.Thread(target=target, name=f"lifecycle-{check.name}", daemon=True)
        worker.start()
        try:
            ok, value = outcome.get(timeout=max(check.timeout_ms, 1) / 1000)
        except queue.Empty:
            elapsed = int((self.clock() - started) * 1000)
            return HookResult(
                Decision.DENY if check.critical else Decision.DEGRADE,
                check.rule_id,
                "check_timeout",
                f"{check.name} exceeded its {check.timeout_ms}ms budget",
                duration_ms=elapsed,
            )
        elapsed = int((self.clock() - started) * 1000)
        if not ok:
            exc = value
            return HookResult(
                Decision.DENY if check.critical else Decision.DEGRADE,
                check.rule_id,
                "check_exception",
                f"{check.name} raised {type(exc).__name__}: {str(exc)[:200]}",
                duration_ms=elapsed,
            )
        result = self._normalize(value, check, elapsed)
        if result.duration_ms == 0:
            result = HookResult(
                result.decision,
                result.rule_id,
                result.reason_code,
                result.message,
                result.affected,
                elapsed,
            )
        return result

    def _normalize(self, raw: Any, check: PolicyCheck, elapsed: int) -> HookResult:
        if isinstance(raw, HookResult):
            result = raw
        elif isinstance(raw, Mapping):
            try:
                result = HookResult(
                    decision=Decision(str(raw.get("decision", ""))),
                    rule_id=str(raw.get("rule_id") or ""),
                    reason_code=str(raw.get("reason_code") or ""),
                    message=str(raw.get("message") or ""),
                    affected=tuple(str(item) for item in raw.get("affected") or ()),
                )
            except (ValueError, TypeError):
                return HookResult(
                    Decision.DENY if check.critical else Decision.DEGRADE,
                    check.rule_id,
                    "invalid_result",
                    f"{check.name} returned an unknown decision or missing fields",
                    duration_ms=elapsed,
                )
        else:
            return HookResult(
                Decision.DENY if check.critical else Decision.DEGRADE,
                check.rule_id,
                "invalid_result",
                f"{check.name} returned no typed decision",
                duration_ms=elapsed,
            )
        if not RULE_ID_RE.fullmatch(result.rule_id):
            return HookResult(
                Decision.DENY if check.critical else Decision.DEGRADE,
                check.rule_id,
                "invalid_rule_id",
                f"{check.name} returned an invalid rule id",
                duration_ms=elapsed,
            )
        if not result.reason_code:
            return HookResult(
                Decision.DENY if check.critical else Decision.DEGRADE,
                check.rule_id,
                "invalid_result",
                f"{check.name} returned no reason code",
                duration_ms=elapsed,
            )
        return result

    def _aggregate(
        self, event: LifecycleEvent, checks: list[PolicyCheck], results: list[HookResult]
    ) -> HookResult:
        if not checks:
            return allow(INTERNAL_RULE, "no_check")
        if not results:
            return allow(INTERNAL_RULE, "no_result")
        denials = [result for result in results if result.decision == Decision.DENY]
        if denials:
            first = denials[0]
            affected = tuple(item for result in denials for item in result.affected)
            return HookResult(
                Decision.DENY,
                first.rule_id,
                first.reason_code,
                first.message,
                affected,
                sum(result.duration_ms for result in results),
            )
        degradations = [result for result in results if result.decision == Decision.DEGRADE]
        if degradations:
            first = degradations[0]
            return HookResult(
                Decision.DEGRADE,
                first.rule_id,
                first.reason_code,
                first.message,
                tuple(item for result in degradations for item in result.affected),
                sum(result.duration_ms for result in results),
            )
        return HookResult(
            Decision.ALLOW,
            checks[0].rule_id,
            "allow",
            "",
            (),
            sum(result.duration_ms for result in results),
        )

    def _record(self, event: LifecycleEvent, check: PolicyCheck, result: HookResult, snapshot: HookInput) -> None:
        self.records.append(
            {
                "schema_version": 1,
                "event": event.value,
                "check": check.name,
                "rule_id": result.rule_id,
                "decision": result.decision.value,
                "reason_code": result.reason_code,
                "message": result.message[:500],
                "affected": list(result.affected)[:20],
                "duration_ms": result.duration_ms,
                "run_id": snapshot.run_id,
                "turn": snapshot.turn,
                "call_id": snapshot.call_id,
                "policy_version": snapshot.policy_version,
                "critical": check.critical,
            }
        )
        if len(self.records) > 1000:
            del self.records[:200]

    def dump(self) -> dict[str, Any]:
        summary = {decision.value: 0 for decision in Decision}
        for record in self.records:
            summary[record["decision"]] = summary.get(record["decision"], 0) + 1
        return {
            "schema_version": 1,
            "policy_version": self.policy_version,
            "summary": summary,
            "records": self.records,
        }


# ---------------------------------------------------------------------------
# Default checks
# ---------------------------------------------------------------------------

SECRET_LIKE_RE = re.compile(
    r"(sk-[A-Za-z0-9_-]{10,}|gh[pousr]_[A-Za-z0-9]{10,}|github_pat_[A-Za-z0-9_]{10,}"
    r"|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|Bearer\s+[A-Za-z0-9._-]{10,})"
)
ALLOWED_TOOL_NAMES = (
    "read_file",
    "search_workspace",
    "get_git_context",
    "search_lessons",
    "github_search_issues",
    "github_read_issue",
)


def _tool_scope(inp: HookInput) -> HookResult:
    tool = str(inp.data.get("tool") or "")
    if tool not in ALLOWED_TOOL_NAMES:
        return deny("R08", "unknown_tool", f"tool {tool!r} is not in the allowlist", affected=(tool,))
    workspace = inp.data.get("workspace")
    if workspace is None:
        return deny("R01", "workspace_missing", "no verified workspace was provided")
    paths: list[str] = []
    if isinstance(inp.data.get("path"), str):
        paths.append(str(inp.data["path"]))
    raw_paths = inp.data.get("paths")
    if isinstance(raw_paths, (list, tuple)):
        paths.extend(str(item) for item in raw_paths)
    for relative in paths:
        try:
            workspace.resolve(relative, must_exist=False)
        except Exception as exc:  # noqa: BLE001 - converted to a bounded denial
            code = getattr(exc, "code", "path_rejected")
            return deny("R01", code, f"path {relative!r} is outside the allowed scope")
    return allow("R01", "scope_ok")


def _tool_budget(inp: HookInput) -> HookResult:
    budget = inp.data.get("budget")
    if budget is None:
        return deny("R07", "budget_missing", "no budget object was provided")
    can, reason = budget.can_attempt_tool()
    if not can:
        return deny("R07", reason or "tool_attempt_limit", "tool attempt budget is exhausted")
    return allow("R07", "budget_ok")


def _github_query(inp: HookInput) -> HookResult:
    tool = str(inp.data.get("tool") or "")
    if not tool.startswith("github_"):
        return allow("R09", "not_remote")
    query = inp.data.get("query")
    if tool == "github_read_issue":
        issue_ref = str(inp.data.get("issue_ref") or "")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+#[1-9][0-9]*", issue_ref):
            return deny("R09", "invalid_issue_ref", "issue_ref is not a bounded repository reference")
        return allow("R09", "issue_ref_ok")
    if not isinstance(query, str):
        return deny("R09", "query_missing", "GitHub search requires a text query")
    if len(query) > 500:
        return deny("R09", "query_too_long", "GitHub query exceeds 500 characters")
    if "\n" in query or "\r" in query:
        return deny("R09", "query_multiline", "GitHub query must be a single line")
    if "://" in query:
        return deny("R09", "query_contains_url", "GitHub query must not contain a URL")
    if SECRET_LIKE_RE.search(query):
        return deny("R09", "query_contains_secret", "GitHub query looks like it contains a token")
    if "```" in query:
        return deny("R09", "query_contains_code", "GitHub query must not carry a code block")
    return allow("R09", "query_ok")


def _after_tool_evidence(inp: HookInput) -> HookResult:
    refs = inp.data.get("evidence_refs") or ()
    persisted = inp.data.get("evidence_persisted")
    if refs and persisted is not True:
        return degrade(
            "R03",
            "evidence_not_persisted",
            "tool returned evidence refs that were not confirmed in the evidence store",
            affected=tuple(str(item) for item in refs),
        )
    return allow("R03", "evidence_ok")


def _before_model_budget(inp: HookInput) -> HookResult:
    budget = inp.data.get("budget")
    if budget is None:
        return deny("R07", "budget_missing", "no budget object was provided")
    can, reason = budget.can_start_model_request()
    if not can:
        return deny("R07", reason or "model_request_limit", "model request budget is exhausted")
    if bool(inp.data.get("final_turn_has_tool_request")):
        return deny("R07", "final_turn_tools", "tools must not run on the reserved final turn")
    return allow("R07", "budget_ok")


def _before_model_context(inp: HookInput) -> HookResult:
    used = inp.data.get("context_tokens")
    limit = inp.data.get("context_limit")
    if isinstance(used, int) and isinstance(limit, int) and used > limit:
        return deny("R04", "context_over_budget", f"context uses {used} tokens over {limit}")
    return allow("R04", "context_ok")


def _before_finalize_claims(inp: HookInput) -> HookResult:
    problems = [str(item) for item in (inp.data.get("problems") or ())]
    if problems:
        return deny("R02", "claim_check_failed", "; ".join(problems[:8]), affected=tuple(problems[:8]))
    return allow("R02", "claims_ok")


def _run_start_versions(inp: HookInput) -> HookResult:
    versions = inp.data.get("versions")
    if not isinstance(versions, Mapping) or not versions.get("policy_hash"):
        return deny("R01", "versions_missing", "policy/version manifest was not fixed at run start")
    if inp.data.get("workspace_ok") is not True:
        return deny("R01", "workspace_invalid", "workspace was not verified")
    return allow("R01", "start_ok")


def _resume_versions(inp: HookInput) -> HookResult:
    recorded = inp.data.get("recorded_versions") or {}
    current = inp.data.get("current_versions") or {}
    changed = [key for key in ("policy_hash", "prompt_hash", "playbook_hash") if recorded.get(key) != current.get(key)]
    if changed:
        return deny("R05", "version_drift", f"checkpoint versions changed: {', '.join(changed)}")
    return allow("R05", "versions_match")


def _promote_guard(inp: HookInput) -> HookResult:
    reasons = [str(item) for item in (inp.data.get("reasons") or ())]
    if reasons:
        return deny("R03", "promotion_rejected", "; ".join(reasons[:8]), affected=tuple(reasons[:8]))
    return allow("R03", "promotion_ok")


def default_checks() -> list[PolicyCheck]:
    return [
        PolicyCheck("run_start_versions", LifecycleEvent.RUN_START, "R01", _run_start_versions),
        PolicyCheck("before_model_budget", LifecycleEvent.BEFORE_MODEL, "R07", _before_model_budget),
        PolicyCheck("before_model_context", LifecycleEvent.BEFORE_MODEL, "R04", _before_model_context),
        PolicyCheck("tool_scope", LifecycleEvent.BEFORE_TOOL, "R01", _tool_scope),
        PolicyCheck("tool_budget", LifecycleEvent.BEFORE_TOOL, "R07", _tool_budget),
        PolicyCheck("github_query", LifecycleEvent.BEFORE_TOOL, "R09", _github_query),
        PolicyCheck("after_tool_evidence", LifecycleEvent.AFTER_TOOL, "R03", _after_tool_evidence, critical=False),
        PolicyCheck("before_finalize_claims", LifecycleEvent.BEFORE_FINALIZE, "R02", _before_finalize_claims),
        PolicyCheck("resume_versions", LifecycleEvent.RESUME, "R05", _resume_versions),
        PolicyCheck("promote_guard", LifecycleEvent.BEFORE_PROMOTE_LESSON, "R03", _promote_guard),
    ]


def build_default_engine(*, run_id: str) -> LifecycleEngine:
    return LifecycleEngine(default_checks(), run_id=run_id)


def denied_by_policy(result: Any) -> bool:
    return getattr(result, "decision", None) == Decision.DENY