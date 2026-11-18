"""Explicit verification feedback: candidate -> verified, with owner confirmation.

Version 2 requires a planned check id, imported output material and every
verified target hash.  A successful but unrelated command cannot promote a
lesson.  Version 1 records remain readable but are treated as legacy and do not
promote under the new contract.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .contracts import (
    Analysis,
    CheckResult,
    CheckStatus,
    ContextPacket,
    Event,
    Lesson,
    LessonStatus,
    Outcome,
    OutcomeCommand,
    VerificationRecord,
)
from .errors import InputError, OutcomeRejected, StoreError
from .lifecycle import Decision, LifecycleEvent, build_default_engine
from .render import checks_to_json, render_handoff, render_lesson_candidate
from .store import (
    EvidenceStore,
    LessonStore,
    RegistryStore,
    RunStatus,
    RunStore,
    exclusive_file_lock,
    hash_file,
    read_json,
    utcnow,
)
from .tools import is_sensitive

MAX_OUTPUT_BYTES = 2 * 1024 * 1024
AUDIT_DIR = "outcome-audits"
PENDING_AUDIT = f"{AUDIT_DIR}/.pending.json"


def load_outcome(path: Path) -> Outcome:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise InputError("cannot read outcome file", detail=str(exc)) from exc
    try:
        return Outcome.model_validate_json(raw)
    except ValidationError as exc:
        raise InputError("outcome does not match the expected shape", detail=str(exc)) from exc


def record_outcome(
    *,
    store: RunStore,
    evidence: EvidenceStore,
    lesson_store: LessonStore,
    outcome: Outcome,
) -> dict[str, Any]:
    if outcome.run_id != store.run_id:
        raise InputError("outcome.run_id does not match the target run")
    payload = outcome.model_dump(mode="json")
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    outcome_id = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    audit_path = f"{AUDIT_DIR}/{outcome_id}.json"
    # Serialize feedback callers and coordinate with the analysis writer.
    with exclusive_file_lock(store.dir / AUDIT_DIR / ".lock"), store.acquire_lock():
        store.meta = RunStore.load(store.output_root, store.run_id).meta
        if store.exists(PENDING_AUDIT):
            _commit_outcome(store, lesson_store, store.read_json(PENDING_AUDIT))
        if store.exists(audit_path):
            audit = store.read_json(audit_path)
        else:
            if not store.exists("analysis.json"):
                raise InputError("run has no analysis.json; feedback requires a finished analysis")
            audit = _prepare_outcome(store, outcome, outcome_id)
            store.write_json(PENDING_AUDIT, audit)
            audit = _commit_outcome(store, lesson_store, audit)
    if audit["receipt"] is None:
        raise OutcomeRejected(audit["rejection"], detail="; ".join(audit["record"]["reasons"][:8]))
    return audit["receipt"]


def _prepare_outcome(store: RunStore, outcome: Outcome, outcome_id: str) -> dict[str, Any]:
    event = Event.model_validate(store.read_json("event.json"))
    analysis = Analysis.model_validate(store.read_json("analysis.json"))
    project_root = store.project_root()

    reasons: list[str] = []
    if not outcome.owner_confirmed:
        reasons.append("owner_confirmed is false")
    planned_checks = {check.check_id for check in analysis.suggested_checks}
    hypothesis_ids = {item.claim_id for item in analysis.hypotheses if item.claim_id}
    target_hashes: dict[str, str] = {}
    imports: list[dict[str, Any]] = []
    check_ids: list[str] = []

    for command in outcome.commands:
        _verify_targets(project_root, command, reasons, target_hashes)
        if command.exit_code != 0:
            reasons.append(f"{command.command}: exit_code {command.exit_code} is not zero")
        if not command.output_ref:
            reasons.append(f"{command.command}: missing output reference")
        if outcome.schema_version != 2:
            continue
        if not command.check_id:
            reasons.append(f"{command.command}: v2 outcome requires a planned check_id")
        elif command.check_id not in planned_checks:
            reasons.append(
                f"{command.command}: check_id {command.check_id!r} is not one of the analysis suggested checks"
            )
        else:
            check_ids.append(command.check_id)
        if command.hypothesis_id and command.hypothesis_id not in hypothesis_ids:
            reasons.append(
                f"{command.command}: hypothesis_id {command.hypothesis_id!r} is not a hypothesis in this analysis"
            )
        if command.output_ref:
            imported = _import_output(store, project_root, command)
            if isinstance(imported, str):
                reasons.append(f"{command.command}: {imported}")
            else:
                imports.append(imported)

    if not target_hashes:
        reasons.append("no verifiable target file was supplied")
    if outcome.schema_version != 2:
        reasons.append(
            "legacy schema_version=1 outcome cannot promote; resubmit with check_id, "
            "an imported output_ref and target hashes"
        )
    elif not imports:
        reasons.append("no output material could be imported for the v2 outcome")
    elif not check_ids:
        reasons.append("no planned check was bound to the v2 outcome")

    lesson = _load_or_build_lesson(store, event, analysis)
    if not reasons and lesson is None:
        reasons.append("analysis has no reusable lesson content to verify")
    engine = build_default_engine(run_id=store.run_id)
    gate = engine.evaluate(
        LifecycleEvent.BEFORE_PROMOTE_LESSON,
        data={"reasons": reasons},
    )
    if gate.decision == Decision.DENY and not reasons:
        reasons.append(f"{gate.rule_id}/{gate.reason_code}: {gate.message}")

    recorded_at = utcnow().isoformat()
    promotion = {
        "schema_version": 1,
        "outcome_id": outcome_id,
        "schema_outcome": outcome.schema_version,
        "reasons": reasons,
        "decision": gate.decision.value,
        "rule_id": gate.rule_id,
        "reason_code": gate.reason_code,
        "check_ids": check_ids,
        "outputs": imports,
        "lifecycle": engine.dump(),
        "recorded_at": recorded_at,
    }
    outcome_record = {
        "schema_version": 2,
        "outcome_id": outcome_id,
        "outcome": outcome.model_dump(mode="json"),
        "current_target_hashes": target_hashes,
        "imported_outputs": imports,
        "reasons": reasons,
        "promoted": False,
        "recorded_at": recorded_at,
    }
    if not reasons and lesson is not None:
        lesson = _verified_lesson(lesson, event, outcome, target_hashes, check_ids, imports)
    return {
        "schema_version": 1,
        "outcome_id": outcome_id,
        "record": outcome_record,
        "gate": promotion,
        "lesson": lesson.model_dump(mode="json") if lesson is not None else None,
    }


def _verified_lesson(
    lesson: Lesson,
    event: Event,
    outcome: Outcome,
    target_hashes: dict[str, str],
    check_ids: list[str],
    imports: list[dict[str, Any]],
) -> Lesson:
    primary = outcome.commands[0]
    verification = VerificationRecord(
        command=" && ".join(command.command for command in outcome.commands)[:2000],
        exit_code=0,
        target_sha256=primary.target_sha256,
        evidence_refs=[],
        executed_by=outcome.executed_by,
        executed_at=outcome.executed_at,
        owner_confirmed=True,
        note=(
            "External executor supplied this v2 verification; the triage agent did not run it. "
            f"Targets: {', '.join(f'{path}={digest[:12]}' for path, digest in target_hashes.items())}"
        ),
        check_ids=sorted(set(check_ids)),
        hypothesis_id=next((command.hypothesis_id for command in outcome.commands if command.hypothesis_id), None),
        target_hashes={path: digest for path, digest in target_hashes.items()},
        output_refs=[item["snapshot"] for item in imports],
        contract="v2",
    )
    return Lesson.model_validate(
        {
            **lesson.model_dump(mode="json"),
            "status": LessonStatus.VERIFIED.value,
            "verification": verification.model_dump(mode="json"),
            "applicable_versions": [event.rule_version],
            "source_run_id": outcome.run_id,
            "source_check_ids": sorted(set(check_ids)),
            "promotion_reason": (
                "owner-confirmed external verification of planned checks "
                f"{', '.join(sorted(set(check_ids)))} on {len(target_hashes)} target(s)"
            ),
        }
    )


def _commit_outcome(store: RunStore, lesson_store: LessonStore, audit: dict[str, Any]) -> dict[str, Any]:
    audit_path = f"{AUDIT_DIR}/{audit['outcome_id']}.json"
    if store.exists(audit_path):
        committed = store.read_json(audit_path)
    else:
        outcome = Outcome.model_validate(audit["record"]["outcome"])
        checks = _load_checks(store)
        lesson = Lesson.model_validate(audit["lesson"]) if audit["lesson"] is not None else None
        reasons = audit["record"]["reasons"]
        committed = {**audit, "receipt": None}
        if reasons:
            protected = store.meta.status == "resolved" or any(
                check.check_id == "outcome-verification" and check.status == CheckStatus.PASS
                for check in checks
            )
            committed["rejection"] = "verification did not satisfy the promotion contract"
            if not protected:
                if lesson is not None and lesson.status == LessonStatus.CANDIDATE:
                    lesson_store.upsert(lesson)
                _record_rejected_check(store, outcome, audit, checks)
                committed["rejection"] += "; lesson remains a candidate"
        else:
            if lesson is None or lesson.verification is None:
                raise StoreError("prepared feedback has no verified lesson")
            for imported in audit["record"]["imported_outputs"]:
                snapshot = imported["snapshot"]
                if not store.exists(snapshot) or hash_file(store.dir / snapshot) != imported["sha256"]:
                    raise StoreError("prepared feedback output snapshot is missing or changed")
            checks = _promote_outcome(store, lesson_store, outcome, audit, lesson, checks)
            committed["record"] = {**audit["record"], "promoted": True}
            store.write_json("outcome.json", committed["record"])
            committed["receipt"] = {
                "run_id": store.run_id,
                "promoted": True,
                "lesson_id": lesson.lesson_id,
                "outcome_id": audit["outcome_id"],
                "status": "resolved",
                "checks": [check.model_dump(mode="json") for check in checks],
            }
        store.write_json(audit_path, committed)
    (store.dir / PENDING_AUDIT).unlink()
    return committed


def _record_rejected_check(
    store: RunStore, outcome: Outcome, audit: dict[str, Any], checks: list[CheckResult]
) -> None:
    store.write_json("promotion-check.json", audit["gate"])
    store.write_json("outcome.json", audit["record"])
    failed = next((command for command in outcome.commands if command.exit_code != 0), None)
    if failed is not None and all(command.output_ref for command in outcome.commands):
        check = CheckResult(
            check_id="outcome-verification",
            status=CheckStatus.FAIL,
            description="external verification command reported a non-zero exit code",
            command=failed.command,
            exit_code=failed.exit_code,
            output_ref=failed.output_ref,
            target_sha256=failed.target_sha256,
            scope="external executor supplied this result; the agent did not run it",
            executed_by=outcome.executed_by,
            executed_at=outcome.executed_at,
        )
    else:
        check = CheckResult(
            check_id="outcome-verification",
            status=CheckStatus.ERROR,
            description="verification is incomplete or does not prove the planned check",
            command=" && ".join(command.command for command in outcome.commands)[:500],
            exit_code=0,
            output_ref="outcome.json",
            target_sha256=hash_file(store.dir / "outcome.json"),
            scope="; ".join(audit["record"]["reasons"][:5]),
            executed_by=outcome.executed_by,
            executed_at=outcome.executed_at,
        )
    checks = [existing for existing in checks if existing.check_id != check.check_id]
    store.write_json("checks.json", checks_to_json([*checks, check]))
    _update_run_status(store, "awaiting_verification")


def _promote_outcome(
    store: RunStore,
    lesson_store: LessonStore,
    outcome: Outcome,
    audit: dict[str, Any],
    lesson: Lesson,
    checks: list[CheckResult],
) -> list[CheckResult]:
    verification = lesson.verification
    assert verification is not None
    lesson_store.upsert(lesson)
    store.write_json("promotion-check.json", audit["gate"])
    checks = [check for check in checks if check.check_id != "outcome-verification"]
    checks.append(
        CheckResult(
            check_id="outcome-verification",
            status=CheckStatus.PASS,
            description="owner-confirmed v2 verification bound to planned checks, targets and imported output",
            command=verification.command,
            exit_code=0,
            output_ref="promotion-check.json",
            target_sha256=verification.target_sha256,
            scope=(
                "external executor result; not a reproduction performed by this agent; "
                f"checks={','.join(verification.check_ids)}; targets={len(verification.target_hashes)}"
            ),
            executed_by=outcome.executed_by,
            executed_at=outcome.executed_at,
        )
    )
    store.write_json("checks.json", checks_to_json(checks))
    _update_run_status(store, "resolved")
    _append_report_verification(store, outcome, verification, audit["outcome_id"])
    _refresh_handoff(store)
    return checks


def _update_run_status(store: RunStore, status: RunStatus) -> None:
    if store.meta.status != status:
        store.set_status(status)
    RegistryStore(store.output_root).set_run_status(store.run_id, status, missing_ok=True)


def _verify_targets(
    project_root: Path,
    command: OutcomeCommand,
    reasons: list[str],
    target_hashes: dict[str, str],
) -> None:
    targets = [(command.target_path, command.target_sha256)]
    targets.extend((item.path, item.sha256) for item in command.additional_targets)
    for relative, expected in targets:
        target = (project_root / relative).resolve()
        try:
            target.relative_to(project_root.resolve())
        except ValueError:
            reasons.append(f"{relative}: target escapes the project root")
            continue
        if not target.is_file():
            reasons.append(f"{relative}: target file is missing")
            continue
        current = hash_file(target)
        target_hashes[relative] = current
        if current != expected:
            reasons.append(
                f"{relative}: supplied hash does not match the current file "
                f"(old hash does not verify the current object)"
            )


def _import_output(store: RunStore, project_root: Path, command: OutcomeCommand) -> dict[str, Any] | str:
    raw_ref = command.output_ref or ""
    if not raw_ref or raw_ref.startswith("/") or ".." in Path(raw_ref).parts:
        return "output_ref must be a project-relative file path without '..'"
    source = (project_root / raw_ref).resolve()
    try:
        source.relative_to(project_root.resolve())
    except ValueError:
        return "output_ref escapes the project root"
    if not source.is_file():
        return f"output_ref {raw_ref!r} does not exist"
    if is_sensitive(raw_ref):
        return "output_ref points at a sensitive file type"
    try:
        size = source.stat().st_size
        if size <= 0:
            return "output_ref is empty; an empty output cannot prove a check"
        if size > MAX_OUTPUT_BYTES:
            return "output_ref is larger than 2 MiB; select a bounded excerpt"
        with source.open("rb") as handle:
            raw = handle.read(MAX_OUTPUT_BYTES + 1)
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return "output_ref is not UTF-8 text"
    except OSError as exc:
        return f"output_ref cannot be read: {exc}"
    if not raw or len(raw) > MAX_OUTPUT_BYTES:
        return "output_ref size changed during import"
    digest = hashlib.sha256(raw).hexdigest()
    if command.output_sha256 is not None and digest != command.output_sha256:
        return "output_sha256 does not match the imported output"
    suffix = source.suffix.lower() if source.suffix.lower() in {".log", ".txt", ".md", ".json", ".xml", ".out"} else ".txt"
    snapshot = f"evidence/outcomes/{digest}{suffix}"
    if store.exists(snapshot):
        if hash_file(store.dir / snapshot) != digest:
            raise StoreError("stored verification output hash does not match its snapshot name")
    else:
        store.write_text(snapshot, text)
    return {
        "check_id": command.check_id,
        "hypothesis_id": command.hypothesis_id,
        "source_ref": raw_ref,
        "snapshot": snapshot,
        "sha256": digest,
        "bytes": len(raw),
    }


def _load_or_build_lesson(store: RunStore, event: Event, analysis: Analysis) -> Lesson | None:
    if store.exists("lesson-candidate.json"):
        return Lesson.model_validate(store.read_json("lesson-candidate.json"))
    return render_lesson_candidate(event=event, analysis=analysis, run_id=store.run_id)


def _load_checks(store: RunStore) -> list[CheckResult]:
    if not store.exists("checks.json"):
        return []
    payload = store.read_json("checks.json")
    checks: list[CheckResult] = []
    for item in payload.get("checks", []):
        try:
            checks.append(CheckResult.model_validate(item))
        except ValidationError as exc:
            raise StoreError("existing checks.json contains an invalid record", detail=str(exc)) from exc
    return checks


def _refresh_handoff(store: RunStore) -> None:
    if not (store.exists("context.json") and store.exists("analysis.json") and store.exists("checks.json")):
        return
    try:
        packet = ContextPacket.model_validate(store.read_json("context.json"))
        analysis = Analysis.model_validate(store.read_json("analysis.json"))
        checks = _load_checks(store)
        versions = store.read_json("versions.json") if store.exists("versions.json") else None
        investigation = None
        if store.exists("investigation.json"):
            from .contracts import InvestigationState

            investigation = InvestigationState.model_validate(store.read_json("investigation.json"))
        text = render_handoff(
            packet=packet,
            analysis=analysis,
            checks=checks,
            meta=store.meta,
            versions=versions,
            investigation=investigation,
        )
    except (ValidationError, StoreError):
        return
    store.write_text("handoff.md", text)


def _append_report_verification(
    store: RunStore,
    outcome: Outcome,
    verification: VerificationRecord,
    outcome_id: str,
) -> None:
    path = store.dir / "report.md"
    if not path.exists():
        return
    lines = [
        "",
        "## 外部验证记录",
        "",
        f"- 执行者：`{outcome.executed_by}`；确认人字段：`owner_confirmed={outcome.owner_confirmed}`",
        f"- 时间：`{outcome.executed_at.isoformat()}`",
        f"- 命令：`{verification.command}`",
        f"- 绑定检查：{', '.join(verification.check_ids) or '（无）'}",
        f"- 目标 hash：" + "；".join(f"`{path}={digest[:12]}`" for path, digest in verification.target_hashes.items()),
        f"- 导入输出：{', '.join(verification.output_refs) or '（无）'}",
        "- 注意：这是外部执行者提供的验证记录，不是本 Agent 亲自运行的结果。",
        "",
    ]
    existing = path.read_text(encoding="utf-8")
    marker = f"<!-- outcome:{outcome_id} -->"
    if marker in existing.splitlines():
        return
    # Older completed outcomes wrote the same section without an identifier.
    if store.exists("outcome.json"):
        previous = store.read_json("outcome.json")
        if "outcome_id" not in previous and previous.get("outcome") == outcome.model_dump(mode="json"):
            if "\n".join(lines).strip() in existing:
                return
    existing = re.sub(
        r"工作流状态：`[^`]+`",
        f"工作流状态：`{store.meta.status}`",
        existing,
        count=1,
    )
    from .store import atomic_write_text

    atomic_write_text(path, existing.rstrip() + "\n" + "\n".join([*lines, marker, ""]))
