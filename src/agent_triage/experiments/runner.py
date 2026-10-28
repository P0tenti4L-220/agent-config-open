"""Sequential local experiment runner built on the existing ``run_new_analysis``.

Every trial gets its own output root, registry, lesson store and workspace.
The runner never edits the product tools or the model loop; it only schedules
the same pipeline on frozen inputs and records what happened.
"""

from __future__ import annotations

import hashlib
import platform
import secrets
import shutil
import time
from pathlib import Path

from ..acceptance import Artifact, build_target
from ..config import Config, config_fingerprint
from ..contracts import AnalysisStatus, CheckStatus, Event
from ..errors import InputError, ValidationFailure
from ..models import ReplayScenario, load_replay
from ..pipeline import run_new_analysis
from ..quality import DatasetCase, QualityDataset, load_dataset
from ..store import atomic_write_json, read_json, utcnow
from .contracts import (
    ExperimentLimits,
    ExperimentPlan,
    ExperimentSpec,
    TrialRecord,
    TrialSlot,
)

EXPERIMENT_SCHEMA = 1
EVALUATOR_MANIFEST = {
    "schema_version": 1,
    "evaluators": [
        {"id": "code.execution", "version": "1", "producer": "code"},
        {"id": "code.analysis_available", "version": "1", "producer": "code"},
        {"id": "code.claims_fact_count", "version": "1", "producer": "code"},
        {"id": "code.claims_with_evidence_ratio", "version": "1", "producer": "code"},
        {"id": "code.tool_calls", "version": "1", "producer": "code"},
        {"id": "code.tool_errors", "version": "1", "producer": "code"},
    ],
}


def _experiment_id() -> str:
    return f"exp-{utcnow().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"


def _copy_dataset(dataset: QualityDataset, destination: Path) -> None:
    destination.mkdir(parents=True)
    atomic_write_json(destination / "manifest.json", dataset.manifest.model_dump(mode="json"))
    for relative, source in dataset.files.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    copied = load_dataset(destination)
    if copied.digest != dataset.digest:
        raise ValidationFailure("copied experiment dataset does not reproduce its digest")


def _trial_root(experiment_dir: Path, case_id: str, repeat_index: int, attempt: int) -> Path:
    return experiment_dir / "trials" / case_id / str(repeat_index) / f"attempt-{attempt}"


def _latest_attempt(trial_dir: Path) -> int:
    if not trial_dir.is_dir():
        return 1
    attempts = [
        int(path.name.split("-", 1)[1])
        for path in trial_dir.iterdir()
        if path.is_dir() and path.name.startswith("attempt-") and path.name.split("-", 1)[1].isdigit()
    ]
    return max(attempts, default=0)


def _record_path(experiment_dir: Path, slot: TrialSlot) -> Path | None:
    trial_dir = experiment_dir / "trials" / slot.case_id / str(slot.repeat_index)
    if not trial_dir.is_dir():
        return None
    latest = _latest_attempt(trial_dir)
    if latest == 0:
        return None
    return trial_dir / f"attempt-{latest}" / "trial.json"


def _workspace(dataset: QualityDataset, case: DatasetCase, trial_dir: Path) -> Path:
    workspace = trial_dir / "workspace"
    workspace.mkdir(parents=True)
    prefix = f"fixtures/{case.case_id}/"
    for ref in case.fixture_refs:
        source = dataset.file_for(ref)
        relative = ref.relative_path
        if relative.startswith(prefix):
            relative = relative[len(prefix):]
        target = (workspace / relative).resolve()
        if workspace.resolve() not in target.parents and target != workspace.resolve():
            raise ValidationFailure(f"fixture escapes the trial workspace: {ref.relative_path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return workspace


def _scenario_for(dataset: QualityDataset, case: DatasetCase, replay: Path | None) -> ReplayScenario | None:
    if case.replay_ref is not None:
        return load_replay(dataset.file_for(case.replay_ref))
    if replay is not None:
        return load_replay(replay)
    return None


def _event_for(dataset: QualityDataset, case: DatasetCase) -> Event:
    try:
        return Event.model_validate(read_json(dataset.file_for(case.input_ref)))
    except ValidationFailure:
        raise
    except Exception as exc:  # noqa: BLE001 - convert parser detail into a typed failure
        raise ValidationFailure(f"case {case.case_id} input is not a valid event", detail=str(exc)) from exc


def _artifacts(run_dir: Path) -> list[Artifact]:
    artifacts: list[Artifact] = []
    for name in ("run.json", "analysis.json", "display.json", "event.json", "checks.json"):
        path = run_dir / name
        if not path.is_file():
            continue
        artifacts.append(
            Artifact(
                relative_path=f"runs/{run_dir.name}/{name}",
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                size_bytes=path.stat().st_size,
                media_type="application/json",
            )
        )
    return artifacts


def plan_experiment(
    *,
    dataset_root: Path,
    output_root: Path,
    variant_id: str,
    repetitions: int,
    provider_mode: str,
    project_root: Path,
    config: Config,
    limits: ExperimentLimits | None = None,
) -> tuple[ExperimentSpec, ExperimentPlan, Path]:
    dataset = load_dataset(dataset_root)
    if provider_mode not in {"replay", "live"}:
        raise InputError("provider mode must be replay or live")
    if repetitions < 1 or repetitions > 10:
        raise InputError("repetitions must be between 1 and 10")
    limits = limits or ExperimentLimits()
    if len(dataset.manifest.cases) * repetitions > limits.max_trials:
        raise ValidationFailure("the plan exceeds max_trials")
    experiment_id = _experiment_id()
    experiment_dir = Path(output_root).resolve() / experiment_id
    experiment_dir.mkdir(parents=True)
    _copy_dataset(dataset, experiment_dir / "dataset")
    target = build_target(
        project_root,
        config=config,
        live_config=config if provider_mode == "live" else None,
    )
    spec = ExperimentSpec(
        experiment_id=experiment_id,
        dataset_id=dataset.dataset_id,
        dataset_digest=dataset.digest,
        variant_id=variant_id,
        target=target,
        provider_mode=provider_mode,
        repetitions=repetitions,
        limits=limits,
        config_hash=config_fingerprint(config),
        python_version=platform.python_version(),
        created_at=utcnow(),
    )
    slots = []
    for case in sorted(dataset.manifest.cases, key=lambda item: item.case_id):
        for repeat in range(repetitions):
            slots.append(
                TrialSlot(
                    trial_id=f"{case.case_id}-r{repeat}",
                    case_id=case.case_id,
                    repeat_index=repeat,
                    status=CheckStatus.NOT_RUN,
                    reason="registered; not executed",
                )
            )
    plan = ExperimentPlan(experiment_id=experiment_id, slots=slots, created_at=utcnow())
    atomic_write_json(experiment_dir / "spec.json", spec.model_dump(mode="json"))
    atomic_write_json(experiment_dir / "plan.json", plan.model_dump(mode="json"))
    atomic_write_json(experiment_dir / "evaluator-manifest.json", EVALUATOR_MANIFEST)
    for slot in slots:
        record = TrialRecord(
            trial_id=slot.trial_id,
            experiment_id=experiment_id,
            case_id=slot.case_id,
            repeat_index=slot.repeat_index,
            attempt_id=slot.attempt_id,
            execution_status=CheckStatus.NOT_RUN,
            reason_code="registered",
            reason="registered; not executed",
        )
        atomic_write_json(_trial_root(experiment_dir, slot.case_id, slot.repeat_index, 1) / "trial.json", record.model_dump(mode="json"))
    return spec, plan, experiment_dir


def load_experiment(experiment_dir: Path) -> tuple[ExperimentSpec, ExperimentPlan]:
    spec = ExperimentSpec.model_validate(read_json(experiment_dir / "spec.json"))
    plan = ExperimentPlan.model_validate(read_json(experiment_dir / "plan.json"))
    if plan.experiment_id != spec.experiment_id:
        raise ValidationFailure("experiment plan and spec disagree on the experiment id")
    return spec, plan


def run_experiment(
    *,
    experiment_dir: Path,
    mode: str,
    project_root: Path,
    config: Config,
    replay: Path | None = None,
) -> dict:
    experiment_dir = Path(experiment_dir).resolve()
    spec, plan = load_experiment(experiment_dir)
    if mode != spec.provider_mode:
        raise ValidationFailure(
            f"experiment was planned for {spec.provider_mode}, not {mode}; refusing an implicit mode change"
        )
    if config_fingerprint(config) != spec.config_hash:
        raise ValidationFailure("configuration differs from the configuration frozen in the plan")
    dataset = load_dataset(experiment_dir / "dataset")
    if dataset.digest != spec.dataset_digest:
        raise ValidationFailure("experiment dataset no longer matches its frozen digest")
    case_map = dataset.cases
    outcomes: list[dict] = []
    deadline = time.monotonic() + spec.limits.max_duration_seconds
    model_requests = 0
    stopped_early = False
    for slot in plan.slots:
        if time.monotonic() > deadline or model_requests >= spec.limits.max_model_requests:
            stopped_early = True
            break
        case = case_map.get(slot.case_id)
        if case is None:
            raise ValidationFailure(f"plan references unknown case {slot.case_id}")
        trial_dir = experiment_dir / "trials" / slot.case_id / str(slot.repeat_index)
        latest = _latest_attempt(trial_dir)
        existing = _record_path(experiment_dir, slot)
        if existing is not None:
            record = TrialRecord.model_validate(read_json(existing))
            if record.execution_status != CheckStatus.NOT_RUN or record.reason_code != "registered":
                # Any real attempt (including a failure) is preserved; a rerun
                # adds a new attempt instead of erasing the evidence.
                latest += 1
        attempt = max(latest, 1)
        attempt_dir = _trial_root(experiment_dir, slot.case_id, slot.repeat_index, attempt)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        result = _run_one(
            dataset=dataset,
            case=case,
            attempt_dir=attempt_dir,
            spec=spec,
            slot=slot,
            config=config,
            project_root=project_root,
            replay=replay,
            mode=mode,
        )
        atomic_write_json(attempt_dir / "trial.json", result.model_dump(mode="json"))
        outcomes.append(result.model_dump(mode="json"))
        model_requests += _model_requests(attempt_dir)
    trials = latest_trials(experiment_dir)
    statuses = [trial.execution_status for trial in trials]
    return {
        "experiment_id": spec.experiment_id,
        "planned": len(plan.slots),
        "completed": sum(1 for item in statuses if item in {CheckStatus.PASS, CheckStatus.FAIL}),
        "pass": statuses.count(CheckStatus.PASS),
        "fail": statuses.count(CheckStatus.FAIL),
        "error": statuses.count(CheckStatus.ERROR),
        "not_run": statuses.count(CheckStatus.NOT_RUN),
        "stopped_early": stopped_early,
        "outcomes": outcomes,
    }


def _model_requests(attempt_dir: Path) -> int:
    trace = attempt_dir / "runs"
    count = 0
    if not trace.is_dir():
        return count
    for path in trace.glob("*/trace.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if '"kind":"model"' in line or '"kind": "model"' in line:
                count += 1
    return count


def _run_one(
    *,
    dataset: QualityDataset,
    case: DatasetCase,
    attempt_dir: Path,
    spec: ExperimentSpec,
    slot: TrialSlot,
    config: Config,
    project_root: Path,
    replay: Path | None,
    mode: str,
) -> TrialRecord:
    started = utcnow()
    clock = time.monotonic()
    try:
        workspace = _workspace(dataset, case, attempt_dir)
        event = _event_for(dataset, case)
        scenario = _scenario_for(dataset, case, replay)
        if mode == "replay" and scenario is None:
            raise ValidationFailure(f"case {case.case_id} has no replay scenario")
        result = run_new_analysis(
            event=event,
            project_root=workspace,
            output_root=attempt_dir / "runs",
            config=config,
            replay_scenario=scenario,
        )
    except Exception as exc:  # noqa: BLE001 - a failed trial is data, not a crash
        ended = utcnow()
        return TrialRecord(
            trial_id=slot.trial_id,
            experiment_id=spec.experiment_id,
            case_id=case.case_id,
            repeat_index=slot.repeat_index,
            attempt_id=attempt_dir.name,
            execution_status=CheckStatus.ERROR,
            reason_code="trial_failed",
            reason=f"{type(exc).__name__}: {exc}",
            duration_ms=int((time.monotonic() - clock) * 1000),
            started_at=started,
            ended_at=ended,
        )
    run_dir = attempt_dir / "runs" / result.run_id
    duration = int((time.monotonic() - clock) * 1000)
    available = (run_dir / "analysis.json").is_file()
    failed = result.analysis.analysis_status == AnalysisStatus.FAILED
    return TrialRecord(
        trial_id=slot.trial_id,
        experiment_id=spec.experiment_id,
        case_id=case.case_id,
        repeat_index=slot.repeat_index,
        attempt_id=attempt_dir.name,
        run_id=result.run_id,
        execution_status=(CheckStatus.ERROR if not available else CheckStatus.FAIL if failed else CheckStatus.PASS),
        reason_code="analysis_missing" if not available else "analysis_failed" if failed else None,
        reason=(
            "analysis.json was not produced" if not available
            else f"analysis failed: {result.analysis.stop_reason.value}" if failed else None
        ),
        result_available=available,
        duration_ms=duration,
        artifacts=_artifacts(run_dir),
        started_at=started,
        ended_at=utcnow(),
    )


def latest_trials(experiment_dir: Path) -> list[TrialRecord]:
    experiment_dir = Path(experiment_dir).resolve()
    records: list[TrialRecord] = []
    trials = experiment_dir / "trials"
    if not trials.is_dir():
        return records
    for case_dir in sorted(path for path in trials.iterdir() if path.is_dir()):
        for repeat_dir in sorted(path for path in case_dir.iterdir() if path.is_dir()):
            latest = _latest_attempt(repeat_dir)
            if latest == 0:
                continue
            records.append(TrialRecord.model_validate(read_json(repeat_dir / f"attempt-{latest}" / "trial.json")))
    return records
