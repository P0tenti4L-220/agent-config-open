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
from datetime import datetime, timezone
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
from .errors import InputError, OutcomeRejected, StoreError, ValidationFailure
from .lifecycle import Decision, LifecycleEvent, build_default_engine
from .render import checks_to_json, render_handoff, render_lesson_candidate
from .store import (
    EvidenceStore,
    LessonStore,
    RegistryStore,
    RunStore,
    hash_file,
    read_json,
    utcnow,
)
from .tools import is_sensitive

MAX_OUTPUT_BYTES = 2 * 1024 * 1024


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
    if not store.exists("analysis.json"):
        raise InputError("run has no analysis.json; feedback requires a finished analysis")
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

    engine = build_default_engine(run_id=store.run_id)
    gate = engine.evaluate(
        LifecycleEvent.BEFORE_PROMOTE_LESSON,
        data={"reasons": reasons},
    )
    if gate.decision == Decision.DENY and not reasons:
        reasons.append(f"{gate.rule_id}/{gate.reason_code}: {gate.message}")
    store.write_json(
        "promotion-check.json",
        {
            "schema_version": 1,
            "schema_outcome": outcome.schema_version,
            "reasons": reasons,
            "decision": gate.decision.value,
            "rule_id": gate.rule_id,
            "reason_code": gate.reason_code,
            "check_ids": check_ids,
            "outputs": imports,
            "lifecycle": engine.dump(),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    outcome_record = {
        "schema_version": 2,
        "outcome": outcome.model_dump(mode="json"),
        "current_target_hashes": target_hashes,
        "imported_outputs": imports,
        "reasons": reasons,
        "promoted": not reasons,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    store.write_json("outcome.json", outcome_record)

    lesson = _load_or_build_lesson(store, event, analysis)
    checks = _load_checks(store)
    if reasons:
        if lesson is not None and lesson.status == LessonStatus.CANDIDATE:
            lesson_store.upsert(lesson)
        store.set_status("awaiting_verification")
        try:
            RegistryStore(store.output_root).set_run_status(store.run_id, "awaiting_verification")
        except StoreError:
            pass
        has_failed_exit = any(command.exit_code != 0 for command in outcome.commands)
        checks = [check for check in checks if check.check_id != "outcome-verification"]
        if has_failed_exit and all(command.output_ref for command in outcome.commands):
            failed = next(command for command in outcome.commands if command.exit_code != 0)
            checks.append(
                CheckResult(
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
            )
        else:
            outcome_sha = hash_file(store.dir / "outcome.json")
            checks.append(
                CheckResult(
                    check_id="outcome-verification",
                    status=CheckStatus.ERROR,
                    description="verification is incomplete or does not prove the planned check",
                    command=" && ".join(command.command for command in outcome.commands)[:500],
                    exit_code=0,
                    output_ref="outcome.json",
                    target_sha256=outcome_sha,
                    scope="; ".join(reasons[:5]),
                    executed_by=outcome.executed_by,
                    executed_at=outcome.executed_at,
                )
            )
        store.write_json("checks.json", checks_to_json(checks))
        raise OutcomeRejected(
            "verification did not satisfy the promotion contract; lesson remains a candidate",
            detail="; ".join(reasons[:8]),
        )

    if lesson is None:
        raise ValidationFailure("analysis has no reusable lesson content to verify")
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
    verified = Lesson.model_validate(
        {
            **lesson.model_dump(mode="json"),
            "status": LessonStatus.VERIFIED.value,
            "verification": verification.model_dump(mode="json"),
            "applicable_versions": [event.rule_version],
            "source_run_id": store.run_id,
            "source_check_ids": sorted(set(check_ids)),
            "promotion_reason": (
                "owner-confirmed external verification of planned checks "
                f"{', '.join(sorted(set(check_ids)))} on {len(target_hashes)} target(s)"
            ),
        }
    )
    lesson_store.upsert(verified)
    checks = [check for check in checks if check.check_id != "outcome-verification"]
    checks.append(
        CheckResult(
            check_id="outcome-verification",
            status=CheckStatus.PASS,
            description="owner-confirmed v2 verification bound to planned checks, targets and imported output",
            command=verification.command,
            exit_code=0,
            output_ref="promotion-check.json",
            target_sha256=primary.target_sha256,
            scope=(
                "external executor result; not a reproduction performed by this agent; "
                f"checks={','.join(verification.check_ids)}; targets={len(target_hashes)}"
            ),
            executed_by=outcome.executed_by,
            executed_at=outcome.executed_at,
        )
    )
    store.write_json("checks.json", checks_to_json(checks))
    store.set_status("resolved")
    try:
        RegistryStore(store.output_root).set_run_status(store.run_id, "resolved")
    except StoreError:
        pass
    _append_report_verification(store, outcome, verification, imports)
    _refresh_handoff(store)
    return {
        "run_id": store.run_id,
        "promoted": True,
        "lesson_id": verified.lesson_id,
        "status": "resolved",
        "checks": [check.model_dump(mode="json") for check in checks],
    }


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
        raw = source.read_bytes()
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return "output_ref is not UTF-8 text"
    except OSError as exc:
        return f"output_ref cannot be read: {exc}"
    digest = hashlib.sha256(raw).hexdigest()
    if command.output_sha256 is not None and digest != command.output_sha256:
        return "output_sha256 does not match the imported output"
    suffix = source.suffix.lower() if source.suffix.lower() in {".log", ".txt", ".md", ".json", ".xml", ".out"} else ".txt"
    snapshot = f"evidence/outcomes/{digest}{suffix}"
    store.write_text(snapshot, text)
    return {
        "check_id": command.check_id,
        "hypothesis_id": command.hypothesis_id,
        "source_ref": raw_ref,
        "snapshot": snapshot,
        "sha256": digest,
        "bytes": size,
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
    imports: list[dict[str, Any]],
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
    existing = re.sub(
        r"工作流状态：`[^`]+`",
        f"工作流状态：`{store.meta.status}`",
        existing,
        count=1,
    )
    from .store import atomic_write_text

    atomic_write_text(path, existing.rstrip() + "\n" + "\n".join(lines))