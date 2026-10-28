"""Local experiment planning, isolation, scoring and comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_triage.config import Config
from agent_triage.errors import ValidationFailure
from agent_triage.experiments.compare import compare_experiments
from agent_triage.experiments.runner import (
    latest_trials,
    load_experiment,
    plan_experiment,
    run_experiment,
)
from agent_triage.experiments.scoring import score_experiment
from agent_triage.quality import write_dataset_manifest
from agent_triage.store import atomic_write_json, hash_file, read_json

from helpers import base_analysis, final_step, make_event, make_scenario, tool_step, write_sample_project
from test_acceptance import make_project

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES = ("EX01", "EX02")


def _artifact(path: Path, root: Path) -> dict:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": hash_file(path),
        "size_bytes": path.stat().st_size,
        "media_type": "application/json" if path.suffix == ".json" else "text/plain",
    }


def _dataset(tmp_path: Path, *, marker: str = "v1", scenario=None) -> Path:
    project = make_project(tmp_path / "project")
    write_sample_project(project)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    scenario = scenario or make_scenario(
        [
            tool_step(("read_file", {"path": "logs/failure.log", "start_line": 1, "max_lines": 20})),
            tool_step(("read_file", {"path": "src/check_reader.py", "start_line": 1, "max_lines": 20})),
            tool_step(("read_file", {"path": "data/checks.json", "start_line": 1, "max_lines": 20})),
            final_step(base_analysis()),
        ]
    )
    cases = []
    for index, case_id in enumerate(CASES):
        event = make_event(event_id=f"{case_id.lower()}-{marker}", symptom=f"{marker} failure {case_id}")
        input_path = dataset / "inputs" / case_id / "event.json"
        input_path.parent.mkdir(parents=True)
        atomic_write_json(input_path, event.model_dump(mode="json"))
        reference = dataset / "references" / f"{case_id}.md"
        reference.parent.mkdir(parents=True, exist_ok=True)
        reference.write_text(f"# reference {case_id}\n", encoding="utf-8")
        rubric = dataset / "rubrics" / f"{case_id}.json"
        rubric.parent.mkdir(parents=True, exist_ok=True)
        rubric.write_text('{"criteria": ["evidence"]}\n', encoding="utf-8")
        replay = dataset / "replay" / f"{case_id}.json"
        replay.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(replay, scenario.model_dump(mode="json"))
        fixtures = []
        for relative in ("logs/failure.log", "src/check_reader.py", "data/checks.json"):
            source = project / relative
            target = dataset / "fixtures" / case_id / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            fixtures.append(_artifact(target, dataset))
        cases.append(
            {
                "case_id": case_id,
                "group_id": f"group-{index}",
                "split": "dev",
                "category": "structured",
                "playbook_id": "structured-output-mismatch",
                "input_ref": _artifact(input_path, dataset),
                "fixture_refs": fixtures,
                "reference_ref": _artifact(reference, dataset),
                "rubric_ref": _artifact(rubric, dataset),
                "replay_ref": _artifact(replay, dataset),
                "origin": "synthetic",
                "source_run_ref": None,
            }
        )
    write_dataset_manifest(
        dataset,
        {"schema_version": 1, "dataset_id": f"experiment-{marker}", "revision": "1", "cases": cases},
    )
    return dataset


def _make_plan(tmp_path: Path, dataset: Path, *, variant: str = "baseline", repetitions: int = 2):
    return plan_experiment(
        dataset_root=dataset,
        output_root=tmp_path / "experiments",
        variant_id=variant,
        repetitions=repetitions,
        provider_mode="replay",
        project_root=REPO_ROOT,
        config=Config(),
    )


def test_plan_registers_every_slot(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    spec, plan, directory = _make_plan(tmp_path, dataset)
    assert spec.dataset_digest
    assert len(plan.slots) == 4
    for slot in plan.slots:
        record = read_json(directory / "trials" / slot.case_id / str(slot.repeat_index) / "attempt-1" / "trial.json")
        assert record["execution_status"] == "not_run"
        assert record["reason_code"] == "registered"
    assert (directory / "dataset" / "manifest.json").is_file()
    assert (directory / "evaluator-manifest.json").is_file()


def test_replay_run_is_isolated_and_attempts_are_preserved(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    _spec, _plan, directory = _make_plan(tmp_path, dataset)
    first = run_experiment(
        experiment_dir=directory,
        mode="replay",
        project_root=REPO_ROOT,
        config=Config(),
    )
    assert first["pass"] == 4
    assert first["error"] == 0
    assert len(latest_trials(directory)) == 4
    run_dirs = []
    for trial in latest_trials(directory):
        attempt = directory / "trials" / trial.case_id / str(trial.repeat_index) / trial.attempt_id
        run_dir = attempt / "runs" / trial.run_id
        run_dirs.append(run_dir)
        assert (run_dir / "analysis.json").is_file()
        assert (attempt / "trial.json").is_file()
    assert len({str(path) for path in run_dirs}) == 4

    second = run_experiment(
        experiment_dir=directory,
        mode="replay",
        project_root=REPO_ROOT,
        config=Config(),
    )
    assert second["pass"] == 4
    for trial in latest_trials(directory):
        trial_root = directory / "trials" / trial.case_id / str(trial.repeat_index)
        assert (trial_root / "attempt-1" / "trial.json").is_file()
        assert trial.attempt_id == "attempt-2"


def test_score_keeps_missing_trials_in_the_denominator(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    _spec, _plan, directory = _make_plan(tmp_path, dataset)
    report = score_experiment(directory)
    assert report["counts"]["not_run"] == 4
    execution = report["metrics"]["code.execution"]
    assert execution["n"] == 0
    assert execution["missing"] == 4
    score = read_json(directory / "scores" / "EX01-r0.json")
    metric = next(item for item in score["metrics"] if item["metric_id"] == "code.execution")
    assert metric["execution_status"] == "not_run"
    assert metric["value"] is None


@pytest.mark.parametrize("failure_kind", ["reported_failure", "unread_evidence"])
def test_failed_analysis_is_a_failed_trial_and_cli_exit(tmp_path: Path, capsys, failure_kind: str) -> None:
    from agent_triage.cli import main

    scenario = make_scenario([final_step(base_analysis(
        facts=[], hypotheses=[], suggested_checks=[],
        analysis_status="failed", stop_reason="tool_unavailable",
        missing_evidence=["Synthetic provider failure"],
    ))])
    if failure_kind == "unread_evidence":
        # Preserve the old invalid fixture as a negative regression: it cited
        # data/checks.json without reading it, previously hidden by trial=pass.
        scenario = make_scenario([
            tool_step(("read_file", {"path": "logs/failure.log", "start_line": 1, "max_lines": 20})),
            tool_step(("read_file", {"path": "src/check_reader.py", "start_line": 1, "max_lines": 20})),
            final_step(base_analysis()),
        ])
    dataset = _dataset(tmp_path, scenario=scenario)
    _spec, _plan, directory = _make_plan(tmp_path, dataset, repetitions=1)
    code = main(["experiment", "run", "--experiment-dir", str(directory), "--mode", "replay"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["fail"] == 2
    assert payload["pass"] == 0
    for trial in latest_trials(directory):
        assert trial.execution_status.value == "fail"
        assert trial.reason_code == "analysis_failed"
        assert trial.result_available is True
        assert trial.run_id
        attempt = directory / "trials" / trial.case_id / str(trial.repeat_index) / trial.attempt_id
        assert read_json(attempt / "runs" / trial.run_id / "analysis.json")["analysis_status"] == "failed"
    report = score_experiment(directory)
    assert report["counts"]["fail"] == 2
    assert report["metrics"]["code.execution"]["n"] == 2
    for case_id in CASES:
        metrics = read_json(directory / "scores" / f"{case_id}-r0.json")["metrics"]
        execution = next(item for item in metrics if item["metric_id"] == "code.execution")
        assert execution["value"] == 0
    assert main(["experiment", "score", "--experiment-dir", str(directory)]) == 1


def test_completed_run_scores_deterministic_metrics(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    _spec, _plan, directory = _make_plan(tmp_path, dataset)
    run_experiment(experiment_dir=directory, mode="replay", project_root=REPO_ROOT, config=Config())
    report = score_experiment(directory)
    assert report["counts"]["pass"] == 4
    assert report["metrics"]["code.execution"]["n"] == 4
    assert report["metrics"]["code.execution"]["missing"] == 0
    assert report["metrics"]["code.tool_calls"]["n"] == 4
    assert (directory / "report.md").is_file()


def test_compare_rejects_a_different_dataset(tmp_path: Path) -> None:
    first = _dataset(tmp_path / "one", marker="a")
    second = _dataset(tmp_path / "two", marker="b")
    _s1, _p1, base_dir = _make_plan(tmp_path / "one", first)
    _s2, _p2, cand_dir = _make_plan(tmp_path / "two", second, variant="candidate")
    run_experiment(experiment_dir=base_dir, mode="replay", project_root=REPO_ROOT, config=Config())
    run_experiment(experiment_dir=cand_dir, mode="replay", project_root=REPO_ROOT, config=Config())
    score_experiment(base_dir)
    score_experiment(cand_dir)
    report, code = compare_experiments(base_dir, cand_dir)
    assert code == 2
    assert report["comparable"] is False
    assert any("dataset" in item for item in report["differences"])


def test_comparable_experiments_report_paired_results(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    _s1, _p1, base_dir = _make_plan(tmp_path, dataset, variant="baseline")
    _s2, _p2, cand_dir = _make_plan(tmp_path, dataset, variant="candidate")
    run_experiment(experiment_dir=base_dir, mode="replay", project_root=REPO_ROOT, config=Config())
    run_experiment(experiment_dir=cand_dir, mode="replay", project_root=REPO_ROOT, config=Config())
    score_experiment(base_dir)
    score_experiment(cand_dir)
    report, code = compare_experiments(base_dir, cand_dir)
    assert code == 0
    assert report["comparable"] is True
    assert report["paired"]["total"] == 4
    assert report["paired"]["ties"] == 4
    assert report["release_decision"] == "not_evaluated"


def test_cli_plan_run_score_and_compare(tmp_path: Path, capsys) -> None:
    from agent_triage.cli import main

    dataset = _dataset(tmp_path)
    output = tmp_path / "experiments"
    assert main(["experiment", "plan", "--dataset", str(dataset), "--output", str(output), "--variant", "cli", "--repetitions", "1", "--mode", "replay"]) == 0
    planned = json.loads(capsys.readouterr().out)
    experiment_dir = planned["experiment_dir"]
    assert main(["experiment", "run", "--experiment-dir", experiment_dir, "--mode", "replay"]) == 0
    capsys.readouterr()
    assert main(["experiment", "score", "--experiment-dir", experiment_dir]) == 0
    capsys.readouterr()
    assert main(["experiment", "compare", "--baseline", experiment_dir, "--candidate", experiment_dir]) == 0
    compared = json.loads(capsys.readouterr().out)
    assert compared["comparable"] is True


def test_mode_change_and_live_optin_are_rejected(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    _spec, _plan, directory = _make_plan(tmp_path, dataset)
    with pytest.raises(ValidationFailure, match="planned for replay"):
        run_experiment(
            experiment_dir=directory,
            mode="live",
            project_root=REPO_ROOT,
            config=Config(),
        )
    spec = load_experiment(directory)[0]
    assert spec.provider_mode == "replay"
