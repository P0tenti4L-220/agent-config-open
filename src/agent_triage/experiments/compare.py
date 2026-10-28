"""Compare two completed local experiments without rerunning the Agent.

The comparison is deliberately report-only: it does not promote a variant or
change an acceptance profile.  Incompatible datasets, repetitions or evaluator
manifests are reported as ``not_comparable`` instead of being silently paired.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..errors import InputError
from ..store import atomic_write_json, atomic_write_text, read_json, utcnow
from .runner import load_experiment


def _evaluator_digest(experiment_dir: Path) -> str:
    path = experiment_dir / "evaluator-manifest.json"
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _score_values(experiment_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    score_path = experiment_dir / "score.json"
    if not score_path.is_file():
        raise InputError("experiment has not been scored; run 'experiment score' first")
    payload = read_json(score_path)
    values: dict[tuple[str, int], dict[str, Any]] = {}
    for trial in payload.get("trials", []):
        key = (trial["case_id"], int(trial["repeat_index"]))
        values[key] = {metric["metric_id"]: metric for metric in trial.get("metrics", [])}
    return values


def compare_experiments(baseline_dir: Path, candidate_dir: Path) -> tuple[dict[str, Any], int]:
    baseline_dir = Path(baseline_dir).resolve()
    candidate_dir = Path(candidate_dir).resolve()
    base_spec, base_plan = load_experiment(baseline_dir)
    cand_spec, cand_plan = load_experiment(candidate_dir)
    differences: list[str] = []
    if base_spec.dataset_digest != cand_spec.dataset_digest:
        differences.append("dataset_digest differs")
    if base_spec.repetitions != cand_spec.repetitions:
        differences.append("repetitions differ")
    if base_spec.provider_mode != cand_spec.provider_mode:
        differences.append("provider_mode differs")
    if {slot.case_id for slot in base_plan.slots} != {slot.case_id for slot in cand_plan.slots}:
        differences.append("case set differs")
    if _evaluator_digest(baseline_dir) != _evaluator_digest(candidate_dir):
        differences.append("evaluator manifest differs")
    comparable = not differences
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": utcnow().isoformat(),
        "baseline": {
            "experiment_id": base_spec.experiment_id,
            "variant_id": base_spec.variant_id,
            "target_commit": base_spec.target.code_commit,
        },
        "candidate": {
            "experiment_id": cand_spec.experiment_id,
            "variant_id": cand_spec.variant_id,
            "target_commit": cand_spec.target.code_commit,
        },
        "comparable": comparable,
        "differences": differences,
        "policy": "report_only",
        "release_decision": "not_evaluated",
    }
    if comparable:
        base_scores = _score_values(baseline_dir)
        cand_scores = _score_values(candidate_dir)
        keys = sorted(set(base_scores) | set(cand_scores))
        rows = []
        wins = losses = ties = missing = 0
        for case_id, repeat in keys:
            base = base_scores.get((case_id, repeat), {})
            cand = cand_scores.get((case_id, repeat), {})
            base_exec = base.get("code.execution", {}).get("value")
            cand_exec = cand.get("code.execution", {}).get("value")
            if base_exec is None or cand_exec is None:
                missing += 1
                outcome = "missing"
            elif cand_exec > base_exec:
                wins += 1
                outcome = "win"
            elif cand_exec < base_exec:
                losses += 1
                outcome = "loss"
            else:
                ties += 1
                outcome = "tie"
            rows.append(
                {
                    "case_id": case_id,
                    "repeat_index": repeat,
                    "baseline_execution": base_exec,
                    "candidate_execution": cand_exec,
                    "outcome": outcome,
                }
            )
        report.update(
            {
                "paired": {"total": len(keys), "wins": wins, "losses": losses, "ties": ties, "missing": missing},
                "cases": rows,
                "note": "execution is the deterministic code metric; quality/cost conclusions need human scoring",
            }
        )
    output = candidate_dir / "comparisons" / f"{base_spec.experiment_id}.json"
    atomic_write_json(output, report)
    atomic_write_text(output.with_suffix(".md"), _markdown(report))
    return report, (0 if comparable else 2)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# 比较 {report['candidate']['experiment_id']} vs {report['baseline']['experiment_id']}",
        "",
        f"- baseline: `{report['baseline']['variant_id']}` @ `{report['baseline']['target_commit']}`",
        f"- candidate: `{report['candidate']['variant_id']}` @ `{report['candidate']['target_commit']}`",
        f"- comparable: `{report['comparable']}`",
        f"- release decision: `{report['release_decision']}`（本命令只生成报告）",
        "",
    ]
    if report["differences"]:
        lines.extend(["## 不可比原因", ""])
        lines.extend(f"- {item}" for item in report["differences"])
        return "\n".join(lines) + "\n"
    paired = report["paired"]
    lines.extend(
        [
            f"配对: {paired['total']}；win/loss/tie/missing = "
            f"{paired['wins']}/{paired['losses']}/{paired['ties']}/{paired['missing']}",
            "",
            "| case | repeat | baseline | candidate | outcome |",
            "|---|---|---|---|---|",
        ]
    )
    for row in report["cases"]:
        lines.append(
            f"| {row['case_id']} | {row['repeat_index']} | {row['baseline_execution']} | "
            f"{row['candidate_execution']} | {row['outcome']} |"
        )
    lines.extend(
        [
            "",
            "> 只比较了确定性执行指标；质量、成本和稳定性需要人工/judge 评分与更大样本，",
            "> 这里不据此宣称改进或回归。",
            "",
        ]
    )
    return "\n".join(lines)
