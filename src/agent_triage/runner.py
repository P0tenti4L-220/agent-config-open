"""Bounded model loop, checkpoint/resume, validation and artifact assembly."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from .budget import Budget
from .claims import ClaimValidation, handoff_problems, validate_claims
from .review import claim_support_check, write_review_status
from .config import Config
from .context import PROMPT_MANIFEST, ContextBuilder
from .contracts import (
    Analysis,
    AnalysisStatus,
    AttemptLedgerEntry,
    Attribution,
    CheckResult,
    CheckStatus,
    ContextPacket,
    DisplayedEvidence,
    Event,
    Evidence,
    EvidenceKind,
    Fact,
    InvestigationPhase,
    InvestigationState,
    OmissionImpact,
    StopReason,
    ToolStatus,
    TraceKind,
    TraceRecord,
)
from .lifecycle import Decision, LifecycleEngine, LifecycleEvent, build_default_engine
from .playbooks import Playbook, catalog_manifest, get_playbook, playbook_hash, route_playbook
from .policy import POLICY_VERSION, version_manifest
from .errors import InputError, ModelError, ModelOutputError, RunInterrupted, StoreError, ValidationFailure
from .ingest import fingerprint
from .models import FinalResponse, ModelClient, ModelContext, RemainingBudget, ToolRequests
from .request_views import RequestViewStore, load_request_views, verify_display_records
from .state_projection import arguments_hash
from .render import (
    checks_to_json,
    render_handoff,
    render_issue_draft,
    render_lesson_candidate,
    render_report,
)
from .store import (
    Checkpoint,
    EvidenceStore,
    LessonStore,
    RegistryStore,
    RunMetadata,
    RunStore,
    hash_file,
    utcnow,
)
from .tools import GithubBackend, ToolExecutor, Workspace, redact_arguments, tool_specs


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    analysis: Analysis
    checks: list[CheckResult]
    stop_reason: StopReason
    artifacts: list[str] = field(default_factory=list)
    remote_status: str = "not_configured"
    duplicate_event: bool = False
    occurrence: int = 1
    budget_used: dict[str, int] = field(default_factory=dict)


def initial_investigation(event: Event, playbook: Playbook) -> InvestigationState:
    needed = list(dict.fromkeys([*event.error_files, *event.related_files]))
    if not event.attempts:
        needed.append("actual input or environment value at the failing boundary, if not captured")
    return InvestigationState(
        active_question=playbook.first_question,
        playbook_id=playbook.playbook_id,
        playbook_version=playbook.version,
        playbook_hash=playbook_hash(playbook),
        needed_materials=needed[:10],
        next_action_reason="read the smallest necessary evidence before proposing a cause",
    )


class Runner:
    def __init__(
        self,
        *,
        store: RunStore,
        event: Event,
        config: Config,
        model: ModelClient,
        context_builder: ContextBuilder,
        evidence_store: EvidenceStore,
        lesson_store: LessonStore,
        workspace: Workspace,
        github: GithubBackend | None = None,
        checkpoint: Checkpoint | None = None,
        clock: Callable[[], float] = time.monotonic,
        interrupt_after_rounds: int | None = None,
        accept_changed_sources: bool = False,
        duplicate_event: bool = False,
        occurrence: int = 1,
    ) -> None:
        self.store = store
        self.event = event
        self.config = config
        self.model = model
        self.builder = context_builder
        self.evidence = evidence_store
        self.lessons = lesson_store
        self.workspace = workspace
        self.github = github
        self.clock = clock
        self.interrupt_after_rounds = interrupt_after_rounds
        self.accept_changed_sources = accept_changed_sources
        self.duplicate_event = duplicate_event
        self.occurrence = occurrence
        self.budget = Budget(config.budgets, state=checkpoint.budget if checkpoint else None, clock=clock)
        self.trace: list[TraceRecord] = self.store.read_trace()
        self.conversation: list[dict[str, Any]] = list(checkpoint.conversation) if checkpoint else []
        self.turn = checkpoint.turn if checkpoint else 0
        self.completed_rounds = 0
        self.remote_status = self._initial_remote_status()
        self.duplicate_stop = False
        self.notes: list[str] = []
        self._last_signature: str | None = None
        self.versions = {
            **version_manifest(),
            **PROMPT_MANIFEST,
            **catalog_manifest(),
            **dict(context_builder.versions),
        }
        self.playbook = (
            get_playbook(checkpoint.playbook_id)
            if checkpoint is not None and checkpoint.playbook_id
            else route_playbook(event)
        )
        self.versions["playbook_id"] = self.playbook.playbook_id
        self.versions["playbook_version"] = self.playbook.version
        self.versions["playbook_hash"] = playbook_hash(self.playbook)
        self.builder.versions = self.versions
        self.investigation = (
            checkpoint.investigation
            if checkpoint is not None and checkpoint.investigation is not None
            else initial_investigation(event, self.playbook)
        )
        self.displayed: list[DisplayedEvidence] = (
            list(checkpoint.displayed) if checkpoint is not None else []
        )
        self.request_views = RequestViewStore(store.dir)
        self.hooks = build_default_engine(run_id=store.run_id)
        self.executor = ToolExecutor(
            workspace=workspace,
            project_id=event.project_id,
            budget=self.budget,
            evidence_sink=self.evidence.add,
            lesson_store=lesson_store,
            github=github,
            trace_sink=self._append_trace,
            run_id=store.run_id,
            policy=self.hooks,
            evidence_lookup=self._evidence_exists,
            evidence_store=self.evidence,
        )
        if checkpoint is not None:
            self._verify_resume(checkpoint)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self) -> RunResult:
        lock = self.store.acquire_lock()
        try:
            if self.store.exists("analysis.json"):
                raise InputError("run already has a final analysis; nothing to resume")
            gate = self.hooks.evaluate(
                LifecycleEvent.RUN_START,
                data={"versions": self.versions, "workspace_ok": self.workspace.root.is_dir()},
            )
            if gate.decision == Decision.DENY:
                raise ValidationFailure(
                    "run start policy check failed",
                    detail=f"{gate.rule_id}/{gate.reason_code}: {gate.message}",
                )
            analysis = self._loop()
            checks = self._write_outputs(analysis)
            self.store.set_status("needs_review")
            return RunResult(
                run_id=self.store.run_id,
                run_dir=self.store.dir,
                analysis=analysis,
                checks=checks,
                stop_reason=analysis.stop_reason,
                artifacts=self._artifact_names(),
                remote_status=self.remote_status,
                duplicate_event=self.duplicate_event,
                occurrence=self.occurrence,
                budget_used=self.budget.snapshot().model_dump(mode="json"),
            )
        finally:
            lock.release()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _loop(self) -> Analysis:
        specs = tool_specs()
        stop_reason: StopReason | None = None
        analysis: Analysis | None = None
        invalid_detail: str | None = None
        corrections = 0

        while True:
            if self.turn > 200:
                stop_reason = StopReason.BUDGET_EXHAUSTED
                break
            can, reason = self.budget.can_start_model_request()
            if not can:
                stop_reason = StopReason.BUDGET_EXHAUSTED
                self.notes.append(f"stopped before a model request: {reason}")
                break
            is_final_turn = self.budget.model_requests_remaining() == 1
            lessons = self.lessons.search(
                f"{self.event.symptom} {self.event.goal}", 3, rule_version=self.event.rule_version
            )
            context = self.builder.build(
                run_id=self.store.run_id,
                event=self.event,
                evidence=self.evidence.all(),
                lessons=lessons,
                trace=self.trace,
                conversation=self.conversation,
                budget_state=self.budget.snapshot(),
                turn=self.turn,
                is_final_turn=is_final_turn,
                tool_specs=specs,
                investigation=self.investigation,
                playbook=self.playbook,
            )
            self.store.write_json("context.json", context.packet.model_dump(mode="json"))
            self._record_context(context)
            gate = self.hooks.evaluate(
                LifecycleEvent.BEFORE_MODEL,
                data={
                    "budget": self.budget,
                    "context_tokens": context.packet.budget.total_used_tokens,
                    "context_limit": context.packet.budget.total_limit_tokens,
                },
                turn=self.turn,
            )
            if gate.decision == Decision.DENY:
                self.notes.append(f"model request blocked by {gate.rule_id}/{gate.reason_code}: {gate.message}")
                stop_reason = (
                    StopReason.BUDGET_EXHAUSTED
                    if gate.reason_code in {"total_time_exhausted", "model_request_limit"}
                    else StopReason.INVALID_OUTPUT
                )
                break
            self.budget.consume_model_request()
            request_view = self.request_views.begin(
                run_id=self.store.run_id,
                turn=self.turn,
                provider=getattr(self.model, "provider", "model"),
                system=context.system,
                messages=context.messages,
                evidence=self.evidence.all(),
                windows=context.packet.evidence_windows,
            )
            started = utcnow()
            started_monotonic = self.clock()
            try:
                response = self.model.next(
                    context,
                    specs,
                    RemainingBudget(
                        model_requests=self.budget.model_requests_remaining(),
                        tool_attempts=self.budget.tool_attempts_remaining(),
                        seconds=self.budget.per_model_timeout(),
                    ),
                )
            except ModelOutputError as exc:
                # The provider returned a response, but it was not a valid
                # final object.  The model still saw this request payload.
                self._confirm_request_view(request_view.request_id)
                duration = int((self.clock() - started_monotonic) * 1000)
                self._append_trace(self._model_trace(started, duration, "error", f"invalid_output:{exc.message}"))
                invalid_detail = f"{exc.message}; {exc.detail or ''}".strip()
                if self.budget.corrections_remaining() <= 0 or self.budget.model_requests_remaining() <= 0:
                    stop_reason = StopReason.INVALID_OUTPUT
                    break
                self.budget.consume_correction()
                corrections += 1
                self.conversation.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": (getattr(exc, "raw_text", None) or "<invalid final output>")[:20_000],
                            }
                        ],
                    }
                )
                self.conversation.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Your final response did not satisfy the contract. "
                                    f"Fix it and return only the JSON object. Details: {invalid_detail[:1500]}"
                                ),
                            }
                        ],
                    }
                )
                self.turn += 1
                self._save_checkpoint()
                continue
            except ModelError as exc:
                self.request_views.mark_failed(request_view.request_id)
                duration = int((self.clock() - started_monotonic) * 1000)
                self._append_trace(self._model_trace(started, duration, "error", f"model_error:{exc.message}"))
                self.notes.append(f"model provider failed: {exc.message}")
                stop_reason = StopReason.TOOL_UNAVAILABLE
                break
            except Exception as exc:  # noqa: BLE001 - no unbounded retry
                self.request_views.mark_failed(request_view.request_id)
                duration = int((self.clock() - started_monotonic) * 1000)
                self._append_trace(self._model_trace(started, duration, "error", f"provider_exception:{type(exc).__name__}"))
                self.notes.append(f"provider raised {type(exc).__name__}")
                stop_reason = StopReason.TOOL_UNAVAILABLE
                break

            self._confirm_request_view(request_view.request_id)
            duration = int((self.clock() - started_monotonic) * 1000)

            if isinstance(response, ToolRequests):
                self._append_trace(
                    self._model_trace(started, duration, "ok", self._usage_note(response))
                )
                if is_final_turn:
                    stop_reason = StopReason.INVALID_OUTPUT
                    invalid_detail = "model requested tools on the reserved final turn"
                    self.notes.append(invalid_detail)
                    break
                if not response.calls:
                    invalid_detail = "model returned an empty tool request list"
                    if self.budget.corrections_remaining() <= 0 or self.budget.model_requests_remaining() <= 0:
                        stop_reason = StopReason.INVALID_OUTPUT
                        break
                    self.budget.consume_correction()
                    self.conversation.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "(empty tool request)"}],
                        }
                    )
                    self.conversation.append(
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "The tool request was empty. Either call a tool or return final JSON."}],
                        }
                    )
                    self.turn += 1
                    self._save_checkpoint()
                    continue
                seen_ids: set[str] = set()
                duplicate_id = False
                for call in response.calls:
                    if call.call_id in seen_ids:
                        duplicate_id = True
                    seen_ids.add(call.call_id)
                if duplicate_id:
                    invalid_detail = "duplicate call_id within one tool request"
                    stop_reason = StopReason.INVALID_OUTPUT
                    break
                assistant_content = response.assistant_content or [
                    {
                        "type": "tool_use",
                        "id": call.call_id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                    for call in response.calls
                ]
                if not assistant_content:
                    assistant_content = [{"type": "text", "text": "(empty tool request)"}]
                self.conversation.append({"role": "assistant", "content": assistant_content})
                tool_results: list[dict[str, Any]] = []
                budget_exhausted_result = False
                for call in response.calls:
                    before = set(self.evidence.ids())
                    result = self.executor.execute(
                        call.name,
                        call.arguments,
                        call_id=call.call_id,
                        turn=self.turn,
                    )
                    after = set(self.evidence.ids())
                    new_evidence = after - before
                    self._record_tool_attempt(call, result, new_evidence)
                    if isinstance(result.data, dict) and "policy_warning" in result.data:
                        warning = result.data["policy_warning"]
                        self.notes.append(
                            f"after_tool policy {warning.get('rule_id')}/{warning.get('reason_code')}: "
                            f"{warning.get('message')}"
                        )
                    signature = call.name + "|" + json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)
                    if signature == self._last_signature and not new_evidence:
                        self.duplicate_stop = True
                    self._last_signature = signature
                    if result.error_code in {"tool_attempt_limit", "total_time_exhausted", "model_request_limit"}:
                        budget_exhausted_result = True
                    elif (
                        self.budget.tool_attempts_remaining() == 0
                        and result.status == ToolStatus.ERROR
                    ):
                        budget_exhausted_result = True
                    if self.budget.remaining_seconds() <= 0:
                        budget_exhausted_result = True
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": call.call_id,
                            "content": self._tool_result_content(result),
                            "is_error": result.status == ToolStatus.ERROR,
                        }
                    )
                self.conversation.append({"role": "user", "content": tool_results})
                self.turn += 1
                self.completed_rounds += 1
                self._save_checkpoint(context)
                if self.interrupt_after_rounds is not None and self.completed_rounds >= self.interrupt_after_rounds:
                    self.store.set_status("needs_review", interruption="checkpointed_after_complete_tool_round")
                    raise RunInterrupted(
                        "run interrupted after a complete tool round; state was checkpointed",
                        detail=f"resume with run id {self.store.run_id}",
                    )
                if self.duplicate_stop:
                    self.notes.append("stopped a path after an identical call produced no new evidence")
                    stop_reason = StopReason.INSUFFICIENT_EVIDENCE
                    break
                if budget_exhausted_result:
                    self.notes.append("tool budget was exhausted during the last round")
                    stop_reason = StopReason.BUDGET_EXHAUSTED
                    break
                continue

            if isinstance(response, FinalResponse):
                analysis = self._apply_remote_constraint(response.analysis)
                problems = self._candidate_problems(analysis, context.packet)
                gate = self.hooks.evaluate(
                    LifecycleEvent.BEFORE_FINALIZE,
                    data={"problems": problems, "claim_support": "not_run"},
                    turn=self.turn,
                )
                if gate.decision == Decision.DENY:
                    if not problems:
                        problems = [f"{gate.rule_id}/{gate.reason_code}: {gate.message}"]
                    self.notes.append(
                        f"final analysis blocked by {gate.rule_id}/{gate.reason_code}: {gate.message}"
                    )
                if problems:
                    invalid_detail = "; ".join(problems)
                    self._append_trace(
                        self._model_trace(
                            started,
                            duration,
                            "error",
                            f"invalid_output:{invalid_detail}",
                        )
                    )
                    if self.budget.corrections_remaining() <= 0 or self.budget.model_requests_remaining() <= 0:
                        stop_reason = StopReason.INVALID_OUTPUT
                        break
                    self.budget.consume_correction()
                    self.conversation.append(
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        response.text
                                        or json.dumps(
                                            analysis.model_dump(mode="json"),
                                            ensure_ascii=False,
                                        )
                                    )[:20_000],
                                }
                            ],
                        }
                    )
                    self.conversation.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Your final analysis failed validation: "
                                        f"{invalid_detail[:1500]}. Fix evidence ids/claims/next step and return only JSON."
                                    ),
                                }
                            ],
                        }
                    )
                    self.turn += 1
                    self._save_checkpoint()
                    continue
                self._append_trace(
                    self._model_trace(started, duration, "ok", self._usage_note(response))
                )
                self.investigation = self._finish_investigation(analysis)
                analysis = self._apply_omission_impacts(analysis)
                stop_reason = analysis.stop_reason
                self._save_checkpoint(context)
                break
            self._append_trace(
                self._model_trace(started, duration, "error", "invalid_output:unknown response type")
            )
            stop_reason = StopReason.INVALID_OUTPUT
            break

        if analysis is None:
            analysis = self._fallback_analysis(stop_reason or StopReason.INVALID_OUTPUT)
        analysis = self._apply_remote_constraint(analysis)
        analysis = self._enrich_duplicate_candidates(analysis)
        if invalid_detail:
            self.notes.append(f"invalid output: {invalid_detail}")
        return analysis

    # ------------------------------------------------------------------
    # Validation and constraints
    # ------------------------------------------------------------------
    def _candidate_problems(self, analysis: Analysis, packet: ContextPacket) -> list[str]:
        validation = validate_claims(analysis, self.evidence.all(), list(self.displayed))
        problems = list(validation.problems)
        problems.extend(
            handoff_problems(
                analysis,
                packet=packet,
                evidence_count=len(self.evidence.all()),
            )
        )
        return problems

    def _evidence_exists(self, evidence_id: str) -> bool:
        try:
            self.evidence.require(evidence_id)
        except Exception:  # noqa: BLE001 - absence is the answer
            return False
        return True

    def _record_context(self, context: ModelContext) -> None:
        # Display proof is derived from the actual provider request view, not
        # from the context builder's candidate list.  See _confirm_request_view.
        if context.omissions:
            merged = list(self.investigation.omissions)
            known = {(item.evidence_id, item.reason) for item in merged}
            for omission in context.omissions:
                key = (omission.evidence_id, omission.reason)
                if key not in known:
                    merged.append(omission)
                    known.add(key)
            self._update_investigation(omissions=merged[-50:])
        self.store.write_json("investigation.json", self.investigation.model_dump(mode="json"))
        self.store.write_json("versions.json", self.versions)

    def _confirm_request_view(self, request_id: str) -> None:
        view = self.request_views.mark_response_received(request_id)
        self.displayed.extend(self.request_views.confirmed_displayed(view))
        self.displayed = self.displayed[-800:]
        self._write_display()

    def _write_display(self) -> None:
        views = self.request_views.all()
        self.store.write_json(
            "display.json",
            {
                "schema_version": 3,
                "displayed": [item.model_dump(mode="json") for item in self.displayed],
                "request_views": [
                    {
                        "request_id": item.request_id,
                        "turn": item.turn,
                        "status": item.status,
                        "payload_sha256": item.payload_sha256,
                        "payload_chars": item.payload_chars,
                        "evidence_views": len(item.evidence_views),
                    }
                    for item in views
                ],
            },
        )

    def _record_tool_attempt(self, call: Any, result: Any, new_evidence: set[str]) -> None:
        # A successful tool call makes evidence available; it does not prove the
        # model saw the body.  The next request view decides that.
        action = call.name
        if isinstance(call.arguments.get("path"), str):
            action += f":{call.arguments['path']}"
        elif isinstance(call.arguments.get("query"), str):
            action += f":{call.arguments['query'][:120]}"
        entry = AttemptLedgerEntry(
            schema_version=2,
            turn=self.turn,
            action=action[:500],
            status=result.status.value + (f":{result.error_code}" if result.error_code else ""),
            new_information=bool(new_evidence),
            detail=(result.error_code or "")[:500] or None,
            call_id=call.call_id,
            arguments_sha256=arguments_hash(redact_arguments(call.arguments)),
            source_ref=f"trace.jsonl:{len(self.trace)}",
        )
        ledger = [*self.investigation.attempt_ledger, entry][-100:]
        required = list(self.investigation.required_evidence_ids)
        for evidence_id in sorted(new_evidence):
            if evidence_id not in required:
                required.append(evidence_id)
        needed = list(self.investigation.needed_materials)
        path = call.arguments.get("path")
        if result.status.value == "ok" and isinstance(path, str) and path in needed:
            needed.remove(path)
        if result.error_code:
            next_reason = f"recover from {result.error_code} without repeating an identical call"
        elif new_evidence:
            next_reason = "new evidence was added; test whether the active question is now answerable"
        else:
            next_reason = "no new information; avoid repeating the same call"
        phase = InvestigationPhase.COMPARE if len(required) >= 2 else self.investigation.phase
        if len(required) >= 2 and phase == InvestigationPhase.LOCATE:
            phase = InvestigationPhase.COMPARE
        self._update_investigation(
            attempt_ledger=ledger,
            required_evidence_ids=required[:50],
            needed_materials=needed[:20],
            next_action_reason=next_reason,
            phase=phase,
        )

    def _update_investigation(self, **changes: Any) -> None:
        payload = {**self.investigation.model_dump(mode="json"), "schema_version": 2, **changes}
        self.investigation = InvestigationState.model_validate(payload)

    def _finish_investigation(self, analysis: Analysis) -> InvestigationState:
        claims = [*analysis.facts, *analysis.hypotheses]
        resolved: list[OmissionImpact] = []
        for omission in self.investigation.omissions:
            affected = sorted(
                {
                    claim.claim_id
                    for claim in claims
                    if claim.claim_id
                    and omission.evidence_id is not None
                    and omission.evidence_id
                    in {
                        *getattr(claim, "evidence_ids", []),
                        *getattr(claim, "supporting_evidence_ids", []),
                        *getattr(claim, "contradicting_evidence_ids", []),
                    }
                }
            )
            resolved.append(
                omission.model_copy(update={"affected_claim_ids": affected})
            )
        payload = {
            **self.investigation.model_dump(mode="json"),
            "phase": InvestigationPhase.HANDOFF.value,
            "omissions": [item.model_dump(mode="json") for item in resolved],
            "next_action_reason": "handoff is ready; the next executor runs the listed checks",
        }
        return InvestigationState.model_validate(payload)

    def _apply_omission_impacts(self, analysis: Analysis) -> Analysis:
        if analysis.analysis_status == AnalysisStatus.FAILED or not self.investigation.omissions:
            return analysis
        missing = list(analysis.missing_evidence)
        additions: list[str] = []
        for omission in self.investigation.omissions:
            if omission.affected_claim_ids:
                text = (
                    f"Context omitted {omission.request} ({omission.reason}); "
                    f"re-read it before relying on {', '.join(omission.affected_claim_ids)}"
                )
            else:
                text = f"Context omitted {omission.request} ({omission.reason})"
            text = text[:500]
            if text not in missing and text not in additions:
                additions.append(text)
        if not additions:
            return analysis
        payload = analysis.model_dump(mode="json")
        payload["missing_evidence"] = (missing + additions)[:20]
        try:
            return Analysis.model_validate(payload)
        except ValidationError:
            return analysis

    def _apply_remote_constraint(self, analysis: Analysis) -> Analysis:
        missing = list(analysis.missing_evidence)
        status = analysis.analysis_status
        stop = analysis.stop_reason
        if self.remote_status != "available":
            text = {
                "not_configured": "GitHub history was not checked: MCP is not configured",
                "available": "",
            }.get(self.remote_status, f"GitHub history was not checked: {self.remote_status}")
            if text and text not in missing:
                missing.append(text)
            if status == AnalysisStatus.COMPLETE:
                status = AnalysisStatus.PARTIAL
            if stop in (StopReason.COMPLETED,):
                stop = StopReason.TOOL_UNAVAILABLE
        elif not self._remote_checked():
            text = "GitHub history was not queried by the agent during this run"
            if text not in missing:
                missing.append(text)
            if status == AnalysisStatus.COMPLETE:
                status = AnalysisStatus.PARTIAL
            if stop == StopReason.COMPLETED:
                stop = StopReason.INSUFFICIENT_EVIDENCE
        if status != analysis.analysis_status or stop != analysis.stop_reason or missing != analysis.missing_evidence:
            payload = analysis.model_dump(mode="json")
            payload["analysis_status"] = status.value
            payload["stop_reason"] = stop.value
            payload["missing_evidence"] = missing[:20]
            try:
                analysis = Analysis.model_validate(payload)
            except ValidationError as exc:
                raise StoreError("program could not apply remote constraint to analysis", detail=str(exc)) from exc
        return analysis

    def _enrich_duplicate_candidates(self, analysis: Analysis) -> Analysis:
        if analysis.analysis_status == AnalysisStatus.FAILED:
            return analysis
        registry = RegistryStore(self.store.output_root)
        try:
            extras = registry.duplicate_candidates(
                event_id=self.event.event_id,
                fingerprint=self._fingerprint_value(),
            )
        except Exception:  # noqa: BLE001 - registry is advisory, analysis remains usable
            return analysis
        existing = {candidate.reference for candidate in analysis.duplicate_candidates}
        payload = analysis.model_dump(mode="json")
        for extra in extras:
            if extra["reference"] in existing:
                continue
            payload["duplicate_candidates"].append(extra)
        if len(payload["duplicate_candidates"]) > 20:
            payload["duplicate_candidates"] = payload["duplicate_candidates"][:20]
        try:
            return Analysis.model_validate(payload)
        except ValidationError:
            return analysis

    def _fingerprint_value(self) -> str:
        return fingerprint(self.event)["value"]

    def _remote_checked(self) -> bool:
        return any(
            record.kind == TraceKind.TOOL
            and (record.tool_name or "").startswith("github_")
            and record.status == "ok"
            for record in self.trace
        )

    # ------------------------------------------------------------------
    # Fallbacks
    # ------------------------------------------------------------------
    def _fallback_analysis(self, stop: StopReason) -> Analysis:
        if stop == StopReason.COMPLETED:
            # A completed stop reason without a validated final object is a
            # contract failure, never a successful analysis.
            stop = StopReason.INVALID_OUTPUT
        facts: list[Fact] = []
        for item in self.evidence.all():
            if item.stale or item.kind not in (EvidenceKind.LOCAL_FILE, EvidenceKind.GIT):
                continue
            location = item.path or "workspace"
            if item.line_start is not None:
                location += f":{item.line_start}-{item.line_end}"
            facts.append(Fact(text=f"Read {location} (sha256 {item.sha256[:12]})", evidence_ids=[item.evidence_id]))
        partial = bool(facts) and stop in (StopReason.BUDGET_EXHAUSTED, StopReason.TOOL_UNAVAILABLE)
        status = AnalysisStatus.PARTIAL if partial else AnalysisStatus.FAILED
        summary = {
            StopReason.INVALID_OUTPUT: "The model did not produce a contract-valid final analysis.",
            StopReason.BUDGET_EXHAUSTED: "The run stopped on its budget before a final diagnosis.",
            StopReason.TOOL_UNAVAILABLE: "A provider or tool needed for the final diagnosis was unavailable.",
            StopReason.INTERRUPTED: "The run was interrupted before a final diagnosis.",
            StopReason.INSUFFICIENT_EVIDENCE: "The run stopped after repeated calls produced no new evidence.",
            StopReason.COMPLETED: "The run ended without a final analysis.",
        }[stop]
        return Analysis(
            schema_version=1,
            summary=summary,
            attribution=Attribution.UNKNOWN,
            facts=facts[:20],
            hypotheses=[],
            missing_evidence=[summary[:500]],
            suggested_checks=[],
            duplicate_candidates=[],
            analysis_status=status,
            stop_reason=stop,
        )

    # ------------------------------------------------------------------
    # Output assembly
    # ------------------------------------------------------------------
    def _write_outputs(self, analysis: Analysis) -> list[CheckResult]:
        self.store.write_json("versions.json", self.versions)
        self.store.write_json("lifecycle.json", self.hooks.dump())
        self._write_display()
        self.store.write_json("investigation.json", self.investigation.model_dump(mode="json"))
        analysis_path = self.store.write_json("analysis.json", analysis.model_dump(mode="json"))
        analysis_sha = hash_file(analysis_path)
        packet = self._final_packet(analysis)
        self.store.write_json("context.json", packet.model_dump(mode="json"))
        review_check, review_view = claim_support_check(
            store=self.store,
            analysis=analysis,
            evidence=self.evidence.all(),
            displayed=list(self.displayed),
            analysis_sha=analysis_sha,
        )
        write_review_status(self.store, review_view)
        handoff = render_handoff(
            packet=packet,
            analysis=analysis,
            checks=[],
            meta=self.store.meta,
            versions=self.versions,
            investigation=self.investigation,
        )
        checks = self._program_checks(
            analysis,
            analysis_sha,
            packet=packet,
            handoff_text=handoff,
            review_check=review_check,
        )
        self.store.write_json("checks.json", checks_to_json(checks))
        report = render_report(
            event=self.event,
            analysis=analysis,
            evidence=self.evidence.all(),
            checks=checks,
            meta=self.store.meta,
            budget=self.budget.snapshot(),
            remote_status=self.remote_status,
            versions=self.versions,
            investigation=self.investigation,
            playbook=self.playbook,
            review_view=review_view,
        )
        self.store.write_text("report.md", report)
        handoff = render_handoff(
            packet=packet,
            analysis=analysis,
            checks=checks,
            meta=self.store.meta,
            versions=self.versions,
            investigation=self.investigation,
        )
        self.store.write_text("handoff.md", handoff)
        final_handoff_sha = hashlib.sha256(handoff.encode("utf-8")).hexdigest()
        checks = [
            check.model_copy(update={"target_sha256": final_handoff_sha})
            if check.check_id == "handoff-contract"
            else check
            for check in checks
        ]
        self.store.write_json("checks.json", checks_to_json(checks))
        if (
            analysis.attribution == Attribution.PUBLIC_WORKFLOW
            and analysis.analysis_status != AnalysisStatus.FAILED
            and analysis.facts
        ):
            draft = render_issue_draft(event=self.event, analysis=analysis, evidence=self.evidence.all())
            self.store.write_text("issue-draft.md", draft)
        lesson = render_lesson_candidate(event=self.event, analysis=analysis, run_id=self.store.run_id)
        if lesson is not None:
            self.store.write_json("lesson-candidate.json", lesson.model_dump(mode="json"))
            try:
                self.lessons.upsert(lesson)
            except StoreError:
                self.notes.append("lesson candidate was written but the local lesson store could not be updated")
        self._save_checkpoint()
        self.store.save_meta()
        return checks

    def _program_checks(
        self,
        analysis: Analysis,
        analysis_sha: str,
        *,
        packet: ContextPacket | None = None,
        handoff_text: str | None = None,
        review_check: CheckResult | None = None,
    ) -> list[CheckResult]:
        trace_path = self.store.dir / "trace.jsonl"
        trace_sha = hash_file(trace_path) if trace_path.exists() else hashlib.sha256(b"").hexdigest()
        evidence_path = self.store.dir / "evidence" / "index.json"
        evidence_sha = hash_file(evidence_path) if evidence_path.exists() else hashlib.sha256(b"").hexdigest()
        now = utcnow()
        checks: list[CheckResult] = [
            CheckResult(
                check_id="analysis-schema",
                status=CheckStatus.PASS,
                description="analysis.json validates against the Analysis contract",
                command="schema:analysis-v1",
                exit_code=0,
                output_ref="analysis.json",
                target_sha256=analysis_sha,
                scope="final analysis structure only",
                executed_by="agent-triage",
                executed_at=now,
            ),
            CheckResult(
                check_id="evidence-refs",
                status=CheckStatus.PASS,
                description="every fact and hypothesis evidence id was read in this run and is not stale",
                command="evidence:refs-v1",
                exit_code=0,
                output_ref="evidence/index.json",
                target_sha256=evidence_sha,
                scope="source existence and snippet identity, not truth of the conclusion",
                executed_by="agent-triage",
                executed_at=now,
            ),
            CheckResult(
                check_id="project-tests",
                status=CheckStatus.NOT_RUN,
                description="the agent did not execute the project's tests",
                command=None,
                exit_code=None,
                output_ref=None,
                target_sha256=None,
                scope="no test command was run by this agent; report real runs with record-outcome",
                executed_by=None,
                executed_at=None,
            ),
            CheckResult(
                check_id="reproduction",
                status=CheckStatus.NOT_RUN,
                description="the root cause was not reproduced or disproved",
                command=None,
                exit_code=None,
                output_ref=None,
                target_sha256=None,
                scope="hypotheses remain hypotheses until a human or authorized executor verifies them",
                executed_by=None,
                executed_at=None,
            ),
        ]
        if self._remote_checked():
            checks.append(
                CheckResult(
                    check_id="github-history",
                    status=CheckStatus.PASS,
                    description="at least one GitHub MCP tool returned a valid result in this run",
                    command="mcp:github-read-v1",
                    exit_code=0,
                    output_ref="trace.jsonl",
                    target_sha256=trace_sha,
                    scope="protocol and adapter contract; completeness is limited by the wrapper page limit",
                    executed_by="agent-triage",
                    executed_at=now,
                )
            )
        elif self.remote_status == "available":
            checks.append(
                CheckResult(
                    check_id="github-history",
                    status=CheckStatus.NOT_RUN,
                    description="GitHub MCP was available but the agent did not query it",
                    command=None,
                    exit_code=None,
                    output_ref=None,
                    target_sha256=None,
                    scope="remote history remains unchecked; no claim of no duplicates",
                    executed_by=None,
                    executed_at=None,
                )
            )
        elif self.remote_status == "not_configured":
            checks.append(
                CheckResult(
                    check_id="github-history",
                    status=CheckStatus.NOT_RUN,
                    description="GitHub MCP is not configured",
                    command=None,
                    exit_code=None,
                    output_ref=None,
                    target_sha256=None,
                    scope="local analysis only; remote history not checked and not equivalent to MCP live success",
                    executed_by=None,
                    executed_at=None,
                )
            )
        else:
            checks.append(
                CheckResult(
                    check_id="github-history",
                    status=CheckStatus.ERROR,
                    description="GitHub MCP failed to start or was unavailable",
                    command=None,
                    exit_code=None,
                    output_ref="trace.jsonl",
                    target_sha256=trace_sha,
                    scope=f"remote history not checked: {self.remote_status}",
                    executed_by="agent-triage",
                    executed_at=now,
                )
            )

        validation = validate_claims(analysis, self.evidence.all(), list(self.displayed))
        display_problems = [item for item in validation.problems if "never displayed" in item]
        request_views = self.request_views.all()
        verified_problems, display_mode = verify_display_records(
            list(self.displayed), request_views, self.evidence.all()
        )
        display_problems.extend(verified_problems)
        if not validation.display_checked:
            display_status = CheckStatus.NOT_RUN
            display_scope = "no display log was recorded; run-level display provenance is unknown"
        elif display_mode == "legacy":
            display_status = CheckStatus.NOT_RUN
            display_scope = (
                "display.json is legacy v1; it records a candidate view but not the provider "
                "request that carried the body"
            )
        elif display_problems:
            display_status = CheckStatus.FAIL
            display_scope = "; ".join(display_problems[:5])
        else:
            display_status = CheckStatus.PASS
            display_scope = (
                "every cited evidence body or exact fragment appears in a provider request that "
                "received a response"
            )
        checks.append(
            CheckResult(
                check_id="claim-display",
                status=display_status,
                description="cited evidence was actually displayed to the model, not merely stored",
                command="claims:display-v2" if display_status != CheckStatus.NOT_RUN else None,
                exit_code=(
                    (1 if display_problems else 0)
                    if display_status != CheckStatus.NOT_RUN
                    else None
                ),
                output_ref="display.json" if display_status != CheckStatus.NOT_RUN else None,
                target_sha256=(
                    hash_file(self.store.dir / "display.json")
                    if display_status != CheckStatus.NOT_RUN and (self.store.dir / "display.json").exists()
                    else None
                ),
                scope=display_scope,
                executed_by="agent-triage" if display_status != CheckStatus.NOT_RUN else None,
                executed_at=now if display_status != CheckStatus.NOT_RUN else None,
            )
        )

        if review_check is None:
            review_check, _ = claim_support_check(
                store=self.store,
                analysis=analysis,
                evidence=self.evidence.all(),
                displayed=list(self.displayed),
                analysis_sha=analysis_sha,
            )
        checks.append(review_check)

        handoff_problems_list = (
            handoff_problems(
                analysis,
                packet=packet,
                evidence_count=len(self.evidence.all()),
            )
            if packet is not None
            else []
        )
        if handoff_text is not None and not handoff_text.strip():
            handoff_problems_list.append("handoff.md is empty")
        handoff_sha = hashlib.sha256((handoff_text or "").encode("utf-8")).hexdigest()
        checks.append(
            CheckResult(
                check_id="handoff-contract",
                status=CheckStatus.PASS if not handoff_problems_list else CheckStatus.FAIL,
                description="handoff contains goal, status, next step, counterevidence gap and evidence index",
                command="handoff:contract-v1",
                exit_code=0 if not handoff_problems_list else 1,
                output_ref="handoff.md",
                target_sha256=handoff_sha,
                scope=(
                    "contract fields only; does not prove the next executor can continue"
                    if not handoff_problems_list
                    else "; ".join(handoff_problems_list[:5])
                ),
                executed_by="agent-triage",
                executed_at=now,
            )
        )
        return checks

    def _final_packet(self, analysis: Analysis) -> ContextPacket:
        context = self.builder.build(
            run_id=self.store.run_id,
            event=self.event,
            evidence=self.evidence.all(),
            lessons=self.lessons.search(
                f"{self.event.symptom} {self.event.goal}", 3, rule_version=self.event.rule_version
            ),
            trace=self.trace,
            conversation=self.conversation,
            budget_state=self.budget.snapshot(),
            turn=self.turn,
            is_final_turn=False,
            tool_specs=tool_specs(),
            investigation=self.investigation,
            playbook=self.playbook,
        )
        payload = context.packet.model_dump(mode="json")
        payload["facts"] = [fact.model_dump(mode="json") for fact in analysis.facts]
        payload["hypotheses"] = [hypothesis.model_dump(mode="json") for hypothesis in analysis.hypotheses]
        payload["not_executed"] = [
            "no test, reproduction or fix has been executed by this agent",
            *[check.description[:500] for check in analysis.suggested_checks],
        ][:20]
        payload["resume_from"]["next_model_turn"] = self.turn
        payload["displayed_count"] = len(self.displayed)
        payload["state_projection"]["not_executed"] = payload["not_executed"]
        payload["state_projection"]["open_questions"] = [question for hypothesis in analysis.hypotheses
                                                         for question in hypothesis.open_questions]
        payload["state_projection"]["suggested_checks"] = [check.model_dump(mode="json") for check in analysis.suggested_checks]
        try:
            return ContextPacket.model_validate(payload)
        except ValidationError as exc:
            raise StoreError("could not build final context packet", detail=str(exc)) from exc

    # ------------------------------------------------------------------
    # Checkpoint and resume
    # ------------------------------------------------------------------
    def _save_checkpoint(self, context: ModelContext | None = None) -> None:
        replay_position = int(getattr(self.model, "position", 0))
        source_hashes = self.evidence.source_hashes()
        checkpoint = Checkpoint(
            run_id=self.store.run_id,
            turn=self.turn,
            replay_position=replay_position,
            budget=self.budget.snapshot(),
            conversation=self.conversation,
            evidence_ids=self.evidence.ids(),
            completed_call_ids=[record.call_id for record in self.trace if record.call_id][-200:],
            source_hashes=source_hashes,
            config_fingerprint=self.store.meta.config_fingerprint,
            investigation=self.investigation,
            displayed=self.displayed[-800:],
            policy_versions=self.versions,
            playbook_id=self.playbook.playbook_id,
            playbook_version=self.playbook.version,
            playbook_hash=playbook_hash(self.playbook),
            updated_at=utcnow(),
        )
        self.store.save_checkpoint(checkpoint)
        self.store.write_json("investigation.json", self.investigation.model_dump(mode="json"))

    def _verify_resume(self, checkpoint: Checkpoint) -> None:
        if checkpoint.config_fingerprint != self.store.meta.config_fingerprint:
            raise ValidationFailure(
                "configuration changed since the checkpoint; start a new run or restore the old config"
            )
        gate = self.hooks.evaluate(
            LifecycleEvent.RESUME,
            data={
                "recorded_versions": checkpoint.policy_versions,
                "current_versions": self.versions,
            },
        )
        if gate.decision == Decision.DENY:
            raise ValidationFailure(
                "policy or playbook versions changed since the checkpoint",
                detail=f"{gate.rule_id}/{gate.reason_code}: {gate.message}",
            )
        if hasattr(self.model, "position"):
            self.model.position = checkpoint.replay_position  # type: ignore[attr-defined]
        changed: list[str] = []
        for path, recorded in checkpoint.source_hashes.items():
            target = self.workspace.root / path
            try:
                current = hash_file(target)
            except OSError:
                changed.append(f"{path} (missing)")
                continue
            if current != recorded:
                changed.append(f"{path} (changed)")
        if changed:
            if not self.accept_changed_sources:
                raise ValidationFailure(
                    "source files changed since the checkpoint; start a new run or pass --accept-changes",
                    detail=", ".join(changed[:10]),
                )
            for path in checkpoint.source_hashes:
                for item in self.evidence.all():
                    if item.path == path and not item.stale:
                        self.evidence.mark_stale(item.evidence_id, f"changed_since_checkpoint:{path}")
            self.notes.append("resume accepted changed source files; affected evidence is stale")
        stale = self.evidence.refresh_staleness(self.workspace.root)
        if stale:
            self.notes.append(f"marked {len(stale)} evidence item(s) stale before resume")

    # ------------------------------------------------------------------
    # Trace helpers and presentation
    # ------------------------------------------------------------------
    def _append_trace(self, record: TraceRecord) -> None:
        self.trace.append(record)
        self.store.append_trace(record)

    def _model_trace(self, started: datetime, duration_ms: int, status: str, note: str) -> TraceRecord:
        finished = utcnow()
        return TraceRecord(
            kind=TraceKind.MODEL,
            run_id=self.store.run_id,
            turn=self.turn,
            tool_name=getattr(self.model, "provider", "model"),
            status=status,
            duration_ms=max(0, min(duration_ms, 600_000)),
            started_at=started,
            finished_at=finished,
            note=note[:2000],
        )

    def _usage_note(self, response: ToolRequests | FinalResponse) -> str:
        note = {
            "provider": getattr(self.model, "provider", "unknown"),
            "kind": "tool_requests" if isinstance(response, ToolRequests) else "final",
        }
        if response.usage:
            note["usage"] = response.usage
            note["count_mode"] = "measured"
        else:
            note["count_mode"] = "estimated"
        return json.dumps(note, ensure_ascii=False)

    def _tool_result_content(self, result: Any) -> str:
        # The executor already bounds the complete JSON result. A second
        # character cut would break exact windows, cursors and JSON structure.
        return json.dumps(result.model_dump(mode="json"), ensure_ascii=False)

    def _initial_remote_status(self) -> str:
        if not self.config.mcp.enabled:
            return "not_configured"
        if self.github is not None and getattr(self.github, "available", False):
            return "available"
        reason = getattr(self.github, "unavailable_reason", None) if self.github is not None else None
        return reason or "mcp_unavailable"

    def _artifact_names(self) -> list[str]:
        names = [
            "event.json",
            "analysis.json",
            "report.md",
            "context.json",
            "handoff.md",
            "trace.jsonl",
            "checks.json",
            "run.json",
            "lifecycle.json",
            "display.json",
            "investigation.json",
            "versions.json",
            "review-status.json",
        ]
        if self.request_views.directory.is_dir():
            names.append("request-views")
        for optional in ("issue-draft.md", "lesson-candidate.json", "checkpoint.json"):
            if self.store.exists(optional):
                names.append(optional)
        return names


def validate_run_artifacts(store: RunStore, evidence: EvidenceStore) -> list[CheckResult]:
    """Re-run structural, reference, display and handoff checks without touching status."""
    if not store.exists("analysis.json"):
        raise InputError("run has no analysis.json to validate")
    payload = store.read_json("analysis.json")
    try:
        analysis = Analysis.model_validate(payload)
    except ValidationError as exc:
        raise ValidationFailure("analysis.json does not match the contract", detail=str(exc)) from exc

    all_evidence = evidence.all()
    displayed: list[DisplayedEvidence] | None = None
    if store.exists("display.json"):
        raw = store.read_json("display.json")
        try:
            displayed = [DisplayedEvidence.model_validate(item) for item in raw.get("displayed", [])]
        except ValidationError as exc:
            raise ValidationFailure("display.json does not match the contract", detail=str(exc)) from exc
    validation = validate_claims(analysis, all_evidence, displayed)
    ref_problems = [item for item in validation.problems if "never displayed" not in item]
    display_problems = [item for item in validation.problems if "never displayed" in item]
    display_mode = "none"
    if displayed is not None:
        try:
            request_views = load_request_views(store.dir)
        except ValidationFailure as exc:
            request_views = []
            display_problems.append(str(exc))
        verified_problems, display_mode = verify_display_records(displayed, request_views, all_evidence)
        display_problems.extend(verified_problems)

    packet: ContextPacket | None = None
    if store.exists("context.json"):
        try:
            packet = ContextPacket.model_validate(store.read_json("context.json"))
        except ValidationError:
            packet = None
    handoff_text = ""
    if store.exists("handoff.md"):
        handoff_text = (store.dir / "handoff.md").read_text(encoding="utf-8")
    handoff_issues = handoff_problems(analysis, packet=packet, evidence_count=len(all_evidence))
    if not handoff_text.strip():
        handoff_issues.append("handoff.md is missing or empty")

    analysis_sha = hash_file(store.dir / "analysis.json")
    now = utcnow()
    checks = [
        CheckResult(
            check_id="analysis-schema",
            status=CheckStatus.PASS,
            description="analysis.json validates against the Analysis contract",
            command="schema:analysis-v2",
            exit_code=0,
            output_ref="analysis.json",
            target_sha256=analysis_sha,
            scope="structure only; not a truth or reproduction check",
            executed_by="agent-triage validate",
            executed_at=now,
        ),
        CheckResult(
            check_id="evidence-refs",
            status=CheckStatus.PASS if not ref_problems else CheckStatus.FAIL,
            description="every cited evidence id exists and is not stale",
            command="evidence:refs-v2",
            exit_code=0 if not ref_problems else 1,
            output_ref="analysis.json",
            target_sha256=analysis_sha,
            scope=(
                "; ".join(ref_problems[:5])
                if ref_problems
                else "source existence and snippet identity only"
            ),
            executed_by="agent-triage validate",
            executed_at=now,
        ),
    ]
    if displayed is None:
        checks.append(
            CheckResult(
                check_id="claim-display",
                status=CheckStatus.NOT_RUN,
                description="cited evidence was displayed to the model",
                command=None,
                exit_code=None,
                output_ref=None,
                target_sha256=None,
                scope="no display.json; provenance of display is unknown for this legacy run",
                executed_by=None,
                executed_at=None,
            )
        )
    elif display_mode == "legacy":
        checks.append(
            CheckResult(
                check_id="claim-display",
                status=CheckStatus.NOT_RUN,
                description="cited evidence was displayed to the model",
                command=None,
                exit_code=None,
                output_ref="display.json",
                target_sha256=hash_file(store.dir / "display.json"),
                scope=(
                    "display.json is legacy v1; it cannot prove the body entered a provider request"
                ),
                executed_by=None,
                executed_at=None,
            )
        )
    else:
        display_sha = hash_file(store.dir / "display.json")
        checks.append(
            CheckResult(
                check_id="claim-display",
                status=CheckStatus.PASS if not display_problems else CheckStatus.FAIL,
                description="cited evidence fragments were present in a provider request",
                command="claims:display-v2",
                exit_code=0 if not display_problems else 1,
                output_ref="display.json",
                target_sha256=display_sha,
                scope=(
                    "; ".join(display_problems[:5])
                    if display_problems
                    else "request-view proof only; semantic support is a separate check"
                ),
                executed_by="agent-triage validate",
                executed_at=now,
            )
        )
    review_check, review_view = claim_support_check(
        store=store,
        analysis=analysis,
        evidence=all_evidence,
        displayed=displayed,
        analysis_sha=analysis_sha,
    )
    write_review_status(store, review_view)
    checks.append(review_check)
    checks.append(
        CheckResult(
            check_id="handoff-contract",
            status=CheckStatus.PASS if not handoff_issues else CheckStatus.FAIL,
            description="handoff has goal, next step, counterevidence gap, versions and evidence index",
            command="handoff:contract-v1",
            exit_code=0 if not handoff_issues else 1,
            output_ref="handoff.md",
            target_sha256=hashlib.sha256(handoff_text.encode("utf-8")).hexdigest(),
            scope=(
                "contract fields only; blind continuation is a separate manual test"
                if not handoff_issues
                else "; ".join(handoff_issues[:5])
            ),
            executed_by="agent-triage validate",
            executed_at=now,
        )
    )
    return checks
