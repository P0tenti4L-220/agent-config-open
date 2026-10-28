"""Deterministic trial scoring and a local, inspectable report.

Only registered code evaluators are used here.  Human and judge scores are
separate producers and never satisfy the acceptance profile.  A metric with no
usable observation is ``not_run`` with a null value, not zero.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..contracts import Analysis, CheckStatus, TraceKind, TraceRecord
from ..errors import InputError, ValidationFailure
from ..store import atomic_write_json, atomic_write_text, hash_file, read_json, utcnow
from .contracts import (
    ExperimentScore,
    MetricResult,
    TrialRecord,
    TrialScore,
)
from .runner import latest_trials, load_experiment


def _metric(
    metric_id: str,
    *,
    trial: TrialRecord,
    status: CheckStatus,
    value: Any | None,
    rationale: str | None = None,
    direction: str = "neutral",
    evidence_refs: list[str] | None = None,
    input_digest: str | None = None,
) -> MetricResult:
    return MetricResult(
        metric_id=metric_id,
        scope="trial",
        subject_id=trial.trial_id,
        producer="code",
        execution_status=status,
        value=value,
        direction=direction,
        rationale=rationale,
        evidence_refs=evidence_refs or [],
        input_digest=input_digest,
        created_at=utcnow(),
    )


def _missing(trial: TrialRecord, metric_id: str, reason: str) -> MetricResult:
    status = CheckStatus.ERROR if trial.execution_status == CheckStatus.ERROR else CheckStatus.NOT_RUN
    return _metric(metric_id, trial=trial, status=status, value=None, rationale=reason)


def _trace(run_dir: Path) -> list[TraceRecord]:
    path = run_dir / "trace.jsonl"
    if not path.is_file():
        return []
    records: list[TraceRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(TraceRecord.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValidationFailure("trace.jsonl contains an invalid record", detail=str(exc)) from exc
    return records


def score_trial(experiment_dir: Path, trial: TrialRecord) -> TrialScore:
    metrics: list[MetricResult] = []
    if trial.execution_status == CheckStatus.PASS:
        metrics.append(_metric("code.execution", trial=trial, status=CheckStatus.PASS, value=1, direction="higher"))
    elif trial.execution_status == CheckStatus.FAIL:
        metrics.append(_metric("code.execution", trial=trial, status=CheckStatus.PASS, value=0, direction="higher"))
    else:
        metrics.append(_missing(trial, "code.execution", trial.reason or "trial did not execute"))

    run_dir = None
    if trial.run_id:
        candidate = experiment_dir / "trials" / trial.case_id / str(trial.repeat_index) / trial.attempt_id / "runs" / trial.run_id
        if candidate.is_dir():
            run_dir = candidate
    if run_dir is None:
        for metric_id in (
            "code.analysis_available",
            "code.claims_fact_count",
            "code.claims_with_evidence_ratio",
            "code.tool_calls",
            "code.tool_errors",
        ):
            metrics.append(_missing(trial, metric_id, "no run artifact"))
        return TrialScore(
            trial_id=trial.trial_id,
            experiment_id=trial.experiment_id,
            case_id=trial.case_id,
            repeat_index=trial.repeat_index,
            metrics=metrics,
        )

    analysis_path = run_dir / "analysis.json"
    if not analysis_path.is_file():
        metrics.append(_missing(trial, "code.analysis_available", "analysis.json is missing"))
        metrics.extend(
            _missing(trial, metric_id, "analysis.json is missing")
            for metric_id in (
                "code.claims_fact_count",
                "code.claims_with_evidence_ratio",
            )
        )
        analysis = None
        analysis_digest = None
    else:
        analysis_digest = hash_file(analysis_path)
        metrics.append(
            _metric(
                "code.analysis_available",
                trial=trial,
                status=CheckStatus.PASS,
                value=1,
                direction="higher",
                input_digest=analysis_digest,
            )
        )
        try:
            analysis = Analysis.model_validate(read_json(analysis_path))
        except Exception as exc:  # noqa: BLE001
            raise ValidationFailure("analysis.json is not a valid Analysis", detail=str(exc)) from exc
        fact_count = len(analysis.facts)
        metrics.append(
            _metric(
                "code.claims_fact_count",
                trial=trial,
                status=CheckStatus.PASS,
                value=fact_count,
                direction="higher",
                input_digest=analysis_digest,
            )
        )
        facts_with_evidence = sum(1 for fact in analysis.facts if fact.evidence_ids)
        if fact_count == 0:
            metrics.append(
                _metric(
                    "code.claims_with_evidence_ratio",
                    trial=trial,
                    status=CheckStatus.NOT_RUN,
                    value=None,
                    rationale="n=0: no fact claims; a ratio would be undefined",
                    input_digest=analysis_digest,
                )
            )
        else:
            metrics.append(
                _metric(
                    "code.claims_with_evidence_ratio",
                    trial=trial,
                    status=CheckStatus.PASS,
                    value=facts_with_evidence / fact_count,
                    direction="higher",
                    rationale=f"{facts_with_evidence}/{fact_count} facts cite at least one evidence id",
                    input_digest=analysis_digest,
                )
            )

    trace = _trace(run_dir)
    tool_records = [record for record in trace if record.kind == TraceKind.TOOL]
    metrics.append(
        _metric(
            "code.tool_calls",
            trial=trial,
            status=CheckStatus.PASS,
            value=len(tool_records),
            direction="neutral",
        )
    )
    errors = [record for record in tool_records if record.status not in {"ok", "success", "pass"}]
    metrics.append(
        _metric(
            "code.tool_errors",
            trial=trial,
            status=CheckStatus.PASS,
            value=len(errors),
            direction="lower",
            evidence_refs=[record.call_id for record in errors if record.call_id][:20],
        )
    )
    return TrialScore(
        trial_id=trial.trial_id,
        experiment_id=trial.experiment_id,
        case_id=trial.case_id,
        repeat_index=trial.repeat_index,
        metrics=metrics,
    )


def _aggregate(scores: list[TrialScore]) -> dict[str, Any]:
    metric_values: dict[str, list[float]] = {}
    metric_missing: dict[str, int] = {}
    for score in scores:
        for metric in score.metrics:
            if metric.value is None:
                metric_missing[metric.metric_id] = metric_missing.get(metric.metric_id, 0) + 1
            elif isinstance(metric.value, (int, float)) and not isinstance(metric.value, bool):
                metric_values.setdefault(metric.metric_id, []).append(float(metric.value))
    metrics = {}
    for metric_id, values in metric_values.items():
        metrics[metric_id] = {
            "n": len(values),
            "missing": metric_missing.get(metric_id, 0),
            "mean": statistics.fmean(values),
            "min": min(values),
            "max": max(values),
        }
    for metric_id, missing in metric_missing.items():
        metrics.setdefault(metric_id, {"n": 0, "missing": missing, "mean": None, "min": None, "max": None})
    return {"metrics": metrics}


def _median(values: list[int]) -> float | None:
    return statistics.median(values) if values else None


def score_experiment(experiment_dir: Path) -> dict[str, Any]:
    experiment_dir = Path(experiment_dir).resolve()
    spec, _plan = load_experiment(experiment_dir)
    trials = latest_trials(experiment_dir)
    if not trials:
        raise InputError("experiment has no trial records")
    scores = [score_trial(experiment_dir, trial) for trial in trials]
    for score in scores:
        atomic_write_json(experiment_dir / "scores" / f"{score.trial_id}.json", score.model_dump(mode="json"))
    accepted = ExperimentScore(
        experiment_id=spec.experiment_id,
        dataset_digest=spec.dataset_digest,
        variant_id=spec.variant_id,
        trials=scores,
        created_at=utcnow(),
    )
    atomic_write_json(experiment_dir / "score.json", accepted.model_dump(mode="json"))
    counts = {
        "planned": len(_plan.slots),
        "scored": len(scores),
        "pass": sum(1 for trial in trials if trial.execution_status == CheckStatus.PASS),
        "fail": sum(1 for trial in trials if trial.execution_status == CheckStatus.FAIL),
        "error": sum(1 for trial in trials if trial.execution_status == CheckStatus.ERROR),
        "not_run": sum(1 for trial in trials if trial.execution_status == CheckStatus.NOT_RUN),
    }
    aggregate = _aggregate(scores)
    aggregate.update(
        {
            "experiment_id": spec.experiment_id,
            "variant_id": spec.variant_id,
            "dataset_digest": spec.dataset_digest,
            "counts": counts,
            "median_duration_ms": _median([trial.duration_ms for trial in trials if trial.duration_ms is not None]),
            "trials": [
                {
                    "trial_id": trial.trial_id,
                    "case_id": trial.case_id,
                    "repeat_index": trial.repeat_index,
                    "attempt_id": trial.attempt_id,
                    "status": trial.execution_status.value,
                    "run_id": trial.run_id,
                    "duration_ms": trial.duration_ms,
                }
                for trial in trials
            ],
        }
    )
    atomic_write_json(experiment_dir / "report.json", aggregate)
    atomic_write_text(experiment_dir / "report.md", _markdown(aggregate))
    return aggregate


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# 实验 {report['experiment_id']}",
        "",
        f"- variant: `{report['variant_id']}`",
        f"- dataset digest: `{report['dataset_digest']}`",
        f"- 计划/已评分: {report['counts']['planned']}/{report['counts']['scored']}",
        f"- pass/fail/error/not_run: {report['counts']['pass']}/{report['counts']['fail']}/"
        f"{report['counts']['error']}/{report['counts']['not_run']}",
        f"- trial 耗时中位数: {report['median_duration_ms']} ms" if report["median_duration_ms"] is not None else "- trial 耗时中位数: null",
        "",
        "## 指标",
        "",
        "| 指标 | n | 缺失 | 均值 | 最小 | 最大 |",
        "|---|---|---|---|---|---|",
    ]
    for metric_id, item in sorted(report["metrics"].items()):
        lines.append(
            f"| {metric_id} | {item['n']} | {item['missing']} | {item['mean']} | {item['min']} | {item['max']} |"
        )
    lines.extend(
        [
            "",
            "## 试验",
            "",
            "| trial | case | repeat | status | run | 耗时 ms |",
            "|---|---|---|---|---|---|",
        ]
    )
    for trial in report["trials"]:
        lines.append(
            f"| {trial['trial_id']} | {trial['case_id']} | {trial['repeat_index']} | "
            f"`{trial['status']}` | {trial['run_id'] or '-'} | {trial['duration_ms'] or '-'} |"
        )
    lines.extend(
        [
            "",
            "> 这是本地确定性评分，不是人工质量验收；`not_run`/`error` 保留在分母中，",
            "> 均值只覆盖有值的 trial。样本量小，不构成统计结论。",
            "",
        ]
    )
    return "\n".join(lines)
