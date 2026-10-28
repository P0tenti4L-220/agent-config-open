"""Command-line entry point for the error and context triage agent."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .acceptance import PROFILE_ORDER, accept_profile, write_acceptance
from .config import config_fingerprint, load_config
from .contracts import Analysis, Config, DisplayedEvidence
from .demo import run_demo
from .errors import ConfigError, InputError, TriageError
from .eval import run_eval
from .feedback import load_outcome, record_outcome
from .ingest import event_from_symptom_file, load_event
from .models import load_replay
from .pipeline import resume_analysis, run_new_analysis
from .review import import_external_review, load_external_review
from .render import checks_to_json
from .runner import RunResult, validate_run_artifacts
from .store import EvidenceStore, LessonStore, RegistryStore, RunStore


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except TriageError as exc:
        print(json.dumps({"status": "error", **exc.as_dict()}, ensure_ascii=False), file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print(json.dumps({"status": "error", "error": "interrupted by user"}), file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - last-resort visible failure, no traceback by default
        if getattr(args, "debug", False):
            raise
        print(
            json.dumps(
                {"status": "error", "error": "unexpected failure", "detail": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 4


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-triage",
        description="Turn one failure event into an evidence-backed diagnosis and handoff context.",
    )
    parser.add_argument("--version", action="version", version=f"agent-triage {__version__}")
    parser.add_argument("--debug", action="store_true", help="show unexpected tracebacks")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run the synthetic offline demo; no credentials or network")
    demo.add_argument("--output", type=Path, default=Path("local/demo"))
    demo.add_argument("--quiet", action="store_true")
    demo.set_defaults(func=_cmd_demo)

    analyze = subparsers.add_parser("analyze", help="analyze one event with replay or a real model")
    analyze.add_argument("--event", type=Path, help="event JSON")
    analyze.add_argument("--symptom-file", type=Path, help="alternative: .log/.txt/.md inside the project")
    analyze.add_argument("--goal", help="goal when using --symptom-file")
    analyze.add_argument("--project-id", help="project alias when using --symptom-file")
    analyze.add_argument("--project-root", type=Path, default=Path("."))
    analyze.add_argument("--config", type=Path)
    analyze.add_argument("--replay", type=Path, help="replay scenario JSON; forces provider=replay")
    analyze.add_argument("--output", type=Path, default=Path("local/runs"))
    analyze.set_defaults(func=_cmd_analyze)

    resume = subparsers.add_parser("resume", help="continue an interrupted run without resetting its budget")
    resume.add_argument("--run-id", required=True)
    resume.add_argument("--output", type=Path, default=Path("local/runs"))
    resume.add_argument("--config", type=Path)
    resume.add_argument("--accept-changes", action="store_true", help="mark changed evidence stale and continue")
    resume.set_defaults(func=_cmd_resume)

    listing = subparsers.add_parser("list", help="list local runs")
    listing.add_argument(
        "--status",
        choices=["all", "needs_review", "awaiting_verification", "resolved"],
        default="all",
    )
    listing.add_argument("--limit", type=_positive_int, default=20)
    listing.add_argument("--output", type=Path, default=Path("local/runs"))
    listing.set_defaults(func=_cmd_list)

    outcome = subparsers.add_parser("record-outcome", help="import an externally executed verification")
    outcome.add_argument("--run-id", required=True)
    outcome.add_argument("--outcome", type=Path, required=True)
    outcome.add_argument("--output", type=Path, default=Path("local/runs"))
    outcome.set_defaults(func=_cmd_record_outcome)

    validate = subparsers.add_parser("validate", help="re-run structural and evidence-reference checks")
    validate.add_argument("--run-id", required=True)
    validate.add_argument("--output", type=Path, default=Path("local/runs"))
    validate.set_defaults(func=_cmd_validate)

    policy = subparsers.add_parser("policy", help="print the fixed policy, rule ids and version hashes")
    policy.set_defaults(func=_cmd_policy)

    evaluation = subparsers.add_parser("eval", help="run the offline acceptance cases")
    evaluation.add_argument("--cases", type=Path, default=Path("tests/cases"))
    evaluation.add_argument("--output", type=Path, default=Path("local/eval"))
    evaluation.add_argument("--include-live", action="store_true", help="also run live cases (requires --config)")
    evaluation.add_argument("--config", type=Path, help="verified live configuration for connected evidence")
    evaluation.add_argument("--project-root", type=Path, help="defaults to this checkout")
    evaluation.set_defaults(func=_cmd_eval)

    live_import = subparsers.add_parser(
        "import-live",
        help="verify a live bundle and import it as HC09 run-artifact evidence",
    )
    live_import.add_argument("--bundle", type=Path, required=True)
    live_import.add_argument("--evidence-dir", type=Path, required=True)
    live_import.add_argument("--project-root", type=Path, help="defaults to this checkout")
    live_import.add_argument("--attempt", default="1")
    live_import.set_defaults(func=_cmd_import_live)

    accept = subparsers.add_parser("accept", help="judge an existing evidence directory against a profile")
    accept.add_argument("--profile", choices=list(PROFILE_ORDER), required=True)
    accept.add_argument("--evidence-dir", type=Path, required=True)
    accept.add_argument("--config", type=Path, help="configuration the evaluation was run with")
    accept.add_argument("--project-root", type=Path, help="defaults to this checkout")
    accept.set_defaults(func=_cmd_accept)

    review = subparsers.add_parser("record-review", help="import an external human review for one run")
    review.add_argument("--run-id", required=True)
    review.add_argument("--review", type=Path, required=True)
    review.add_argument("--output", type=Path, default=Path("local/runs"))
    review.set_defaults(func=_cmd_record_review)

    doctor = subparsers.add_parser("doctor", help="read-only environment readiness check")
    doctor.add_argument("--profile", choices=["offline", "connected"], default="offline")
    doctor.add_argument("--output", type=Path, default=Path("local/doctor"))
    doctor.add_argument("--config", type=Path)
    doctor.add_argument("--project-root", type=Path)
    doctor.set_defaults(func=_cmd_doctor)

    handoff = subparsers.add_parser("handoff-check", help="verify a development handoff against the worktree")
    handoff.add_argument("--file", type=Path, required=True)
    handoff.add_argument("--project-root", type=Path)
    handoff.set_defaults(func=_cmd_handoff_check)

    review_acceptance = subparsers.add_parser(
        "acceptance-review",
        help="prepare or import an acceptance-level human review bound to run evidence",
    )
    review_sub = review_acceptance.add_subparsers(dest="review_command", required=True)
    prepare = review_sub.add_parser("prepare", help="build a review template from frozen cases and run bundles")
    prepare.add_argument("--evidence-dir", type=Path, required=True)
    prepare.add_argument("--check", required=True, help="acceptance check id, e.g. AC15.quality")
    prepare.add_argument("--dataset", type=Path, help="frozen quality dataset directory")
    prepare.add_argument("--subjects", type=Path, help="directory of <case_id>/<run-dir> evidence")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--project-root", type=Path)
    prepare.set_defaults(func=_cmd_review_prepare)

    review_import = review_sub.add_parser("import", help="verify and import a completed schema 3 review")
    review_import.add_argument("--evidence-dir", type=Path, required=True)
    review_import.add_argument("--review", type=Path, required=True)
    review_import.add_argument("--project-root", type=Path)
    review_import.set_defaults(func=_cmd_review_import)

    experiment = subparsers.add_parser(
        "experiment",
        help="plan, run, score and compare frozen quality experiments",
    )
    experiment_sub = experiment.add_subparsers(dest="experiment_command", required=True)
    exp_plan = experiment_sub.add_parser("plan", help="freeze a dataset and register every trial slot")
    exp_plan.add_argument("--dataset", type=Path, required=True)
    exp_plan.add_argument("--output", type=Path, default=Path("local/experiments"))
    exp_plan.add_argument("--variant", required=True)
    exp_plan.add_argument("--repetitions", type=_positive_int, default=1)
    exp_plan.add_argument("--mode", choices=["replay", "live"], default="replay")
    exp_plan.add_argument("--config", type=Path)
    exp_plan.add_argument("--project-root", type=Path)
    exp_plan.set_defaults(func=_cmd_experiment_plan)

    exp_run = experiment_sub.add_parser("run", help="execute every planned trial sequentially")
    exp_run.add_argument("--experiment-dir", type=Path, required=True)
    exp_run.add_argument("--mode", choices=["replay", "live"], required=True)
    exp_run.add_argument("--config", type=Path)
    exp_run.add_argument("--project-root", type=Path)
    exp_run.add_argument("--replay", type=Path, help="fallback replay scenario when a case has none")
    exp_run.add_argument("--run-live", action="store_true", help="explicitly allow real provider calls")
    exp_run.set_defaults(func=_cmd_experiment_run)

    exp_score = experiment_sub.add_parser("score", help="compute deterministic metrics and a local report")
    exp_score.add_argument("--experiment-dir", type=Path, required=True)
    exp_score.set_defaults(func=_cmd_experiment_score)

    exp_compare = experiment_sub.add_parser("compare", help="compare two completed experiments")
    exp_compare.add_argument("--baseline", type=Path, required=True)
    exp_compare.add_argument("--candidate", type=Path, required=True)
    exp_compare.set_defaults(func=_cmd_experiment_compare)
    return parser


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _cmd_demo(args: argparse.Namespace) -> int:
    result = run_demo(args.output)
    payload = _summary(result)
    payload["note"] = (
        "Synthetic replay demo. Tool execution and validation use production code; "
        "no real model, GitHub MCP or live credentials were used."
    )
    _print(payload, quiet=args.quiet)
    return 0 if result.analysis.analysis_status.value != "failed" else 1


def _cmd_analyze(args: argparse.Namespace) -> int:
    if args.event and args.symptom_file:
        raise InputError("use either --event or --symptom-file, not both")
    if not args.event and not args.symptom_file:
        raise InputError("analyze requires --event or --symptom-file")
    project_root = args.project_root.resolve()
    if not project_root.is_dir():
        raise InputError(f"project root is not a directory: {args.project_root}")
    if args.symptom_file:
        if not args.goal or not args.project_id:
            raise InputError("--symptom-file requires --goal and --project-id")
        event = event_from_symptom_file(
            args.symptom_file,
            project_root,
            goal=args.goal,
            project_id=args.project_id,
            rule_version="v0.1.0",
        )
    else:
        event = load_event(args.event)
    config = load_config(args.config)
    if args.replay:
        # The scenario itself is stored as replay.json; do not copy the user's
        # absolute local path into config.json or the config fingerprint.
        config = Config.model_validate(
            {
                **config.model_dump(mode="json"),
                "provider": "replay",
                "model": "replay",
                "replay_script": None,
            }
        )
    if config.provider in {"anthropic", "deepseek"}:
        if config.model == "replay":
            raise ConfigError(f"provider={config.provider} requires an explicit model name")
        if not os.environ.get(config.credential_env):
            raise ConfigError(
                f"missing model credential in {config.credential_env}; live analysis remains not_run"
            )
    scenario = None
    if config.provider == "replay":
        scenario_path = args.replay or (Path(config.replay_script) if config.replay_script else None)
        if scenario_path is None:
            raise ConfigError("replay provider requires --replay or config.replay_script")
        scenario = load_replay(scenario_path)
    result = run_new_analysis(
        event=event,
        project_root=project_root,
        output_root=args.output,
        config=config,
        replay_scenario=scenario,
    )
    payload = _summary(result)
    payload["analysis_status"] = result.analysis.analysis_status.value
    payload["attribution"] = result.analysis.attribution.value
    payload["stop_reason"] = result.analysis.stop_reason.value
    payload["duplicate_event"] = result.duplicate_event
    payload["occurrence"] = result.occurrence
    _print(payload)
    return 0 if result.analysis.analysis_status.value != "failed" else 1


def _cmd_resume(args: argparse.Namespace) -> int:
    store = RunStore.load(args.output, args.run_id)
    config = _config_for_run(store, args.config)
    scenario = None
    if config.provider == "replay":
        replay_path = store.dir / "replay.json"
        if replay_path.exists():
            scenario = load_replay(replay_path)
        elif config.replay_script:
            scenario = load_replay(Path(config.replay_script))
        else:
            raise ConfigError("run uses replay but no replay.json or replay_script is available")
    result = resume_analysis(
        store=store,
        config=config,
        replay_scenario=scenario,
        accept_changed_sources=args.accept_changes,
    )
    payload = _summary(result)
    payload["resumed"] = True
    _print(payload)
    return 0 if result.analysis.analysis_status.value != "failed" else 1


def _cmd_list(args: argparse.Namespace) -> int:
    runs = RegistryStore(args.output).list_runs(args.status, args.limit)
    _print(
        {
            "output": str(args.output),
            "status": args.status,
            "runs": [run.model_dump(mode="json") for run in runs],
        }
    )
    return 0


def _cmd_record_outcome(args: argparse.Namespace) -> int:
    store = RunStore.load(args.output, args.run_id)
    outcome = load_outcome(args.outcome)
    lesson_store = LessonStore(args.output)
    evidence = EvidenceStore(store)
    payload = record_outcome(
        store=store,
        evidence=evidence,
        lesson_store=lesson_store,
        outcome=outcome,
    )
    _print(payload)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    store = RunStore.load(args.output, args.run_id)
    evidence = EvidenceStore(store)
    new_checks = validate_run_artifacts(store, evidence)
    existing: dict[str, dict] = {}
    if store.exists("checks.json"):
        for item in store.read_json("checks.json").get("checks", []):
            existing[item["check_id"]] = item
    for check in new_checks:
        existing[check.check_id] = check.model_dump(mode="json")
    from .contracts import CheckResult

    checks = [CheckResult.model_validate(item) for item in existing.values()]
    store.write_json("checks.json", checks_to_json(checks))
    blocking = [check for check in new_checks if check.status.value in {"fail", "error"}]
    _print(
        {
            "run_id": store.run_id,
            "status": "pass" if not blocking else "fail",
            "checks": [check.model_dump(mode="json") for check in new_checks],
            "note": (
                "structure, reference, display and handoff contract only; semantic support, tests and "
                "reproduction remain not_run unless executed or reviewed externally"
            ),
        }
    )
    return 0 if not blocking else 1


def _cmd_policy(args: argparse.Namespace) -> int:
    from .context import PROMPT_MANIFEST
    from .playbooks import catalog_manifest
    from .policy import RULES, version_manifest

    _print(
        {
            "manifest": {**version_manifest(), **PROMPT_MANIFEST, **catalog_manifest()},
            "rules": [{"rule_id": rule.rule_id, "title": rule.title} for rule in RULES],
            "note": "manifest is code-derived; a run records the same hashes in versions.json",
        }
    )
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    if args.include_live and not args.config:
        raise InputError("eval --include-live requires --config so live evidence binds to a verified configuration")
    config = load_config(args.config) if args.config else None
    payload = run_eval(
        cases_dir=args.cases,
        output_root=args.output,
        include_live=args.include_live,
        project_root=args.project_root,
        config=config,
        config_path=args.config,
    )
    _print(
        {
            "evaluation_id": payload["evaluation_id"],
            "evidence_dir": payload["evidence_dir"],
            "scope": payload["scope"],
            "overall": payload["overall"],
            "summary": payload["summary"],
            "note": "eval only executes tests; profile acceptance is a separate 'accept' step",
        }
    )
    return 0 if payload["overall"] != "fail" else 1


def _cmd_accept(args: argparse.Namespace) -> int:
    config = load_config(args.config) if args.config else None
    result = accept_profile(
        profile=args.profile,
        evidence_dir=args.evidence_dir,
        project_root=args.project_root or Path(__file__).resolve().parents[2],
        config=config,
    )
    write_acceptance(result, args.evidence_dir.resolve())
    _print(result.model_dump(mode="json"))
    return result.exit_code


def _cmd_import_live(args: argparse.Namespace) -> int:
    from .live_bundle import import_live_bundle

    result, path = import_live_bundle(
        args.bundle,
        args.evidence_dir,
        project_root=args.project_root or Path(__file__).resolve().parents[2],
        attempt=args.attempt,
    )
    _print(
        {
            "status": result.status.value,
            "check_id": result.check_id,
            "result_file": str(path),
            "run_ids": result.related_run_ids,
            "note": "HC09 is run-artifact evidence; it does not replace AC15.live or human review",
        }
    )
    return 0 if result.status.value == "pass" else 1


def _cmd_record_review(args: argparse.Namespace) -> int:
    store = RunStore.load(args.output, args.run_id)
    if not store.exists("analysis.json"):
        raise InputError("run has no analysis.json to review")
    analysis = Analysis.model_validate(store.read_json("analysis.json"))
    displayed: list[DisplayedEvidence] | None = None
    if store.exists("display.json"):
        raw = store.read_json("display.json")
        displayed = [DisplayedEvidence.model_validate(item) for item in raw.get("displayed", [])]
    review = load_external_review(args.review)
    payload = import_external_review(
        store=store,
        analysis=analysis,
        evidence=EvidenceStore(store).all(),
        displayed=displayed,
        review=review,
        source_path=args.review,
    )
    _print(payload)
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    from .doctor import run_doctor

    report = run_doctor(
        profile=args.profile,
        output=args.output,
        config_path=args.config,
        project_root=args.project_root,
    )
    _print(report.summary())
    return {"ready": 0, "blocked": 2, "error": 4}[report.status]


def _cmd_handoff_check(args: argparse.Namespace) -> int:
    from .handoff import verify_handoff

    payload = verify_handoff(
        args.file,
        project_root=args.project_root or Path(__file__).resolve().parents[2],
    )
    _print(payload)
    return 0 if payload["state"] == "current" else 1


def _cmd_review_prepare(args: argparse.Namespace) -> int:
    from .acceptance_review import prepare_review

    payload = prepare_review(
        evidence_dir=args.evidence_dir,
        check_id=args.check,
        dataset_root=args.dataset,
        subjects_dir=args.subjects,
        output=args.output,
        project_root=args.project_root,
    )
    _print(payload)
    return 0


def _cmd_review_import(args: argparse.Namespace) -> int:
    from .acceptance_review import import_review

    payload = import_review(args.evidence_dir, args.review, project_root=args.project_root)
    _print(payload)
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _summary(result: RunResult) -> dict:
    run_dir = result.run_dir.resolve()
    status = result.analysis.analysis_status.value
    return {
        "run_id": result.run_id,
        "analysis_status": status,
        "status_explanation": {
            "complete": "analysis complete; the report is ready to read",
            "partial": "analysis exists but some listed materials are still missing; read the report and continue",
            "failed": "this analysis failed; inspect the error and any artifacts that were written",
        }.get(status, "unrecognised analysis status; inspect analysis.json before relying on it"),
        "missing_evidence": list(result.analysis.missing_evidence),
        "stop_reason": result.analysis.stop_reason.value,
        "attribution": result.analysis.attribution.value,
        "remote_status": result.remote_status,
        "artifacts": result.artifacts,
        "output_dir": str(run_dir),
        "report_path": str(run_dir / "report.md"),
        "context_path": str(run_dir / "context.json"),
        "handoff_path": str(run_dir / "handoff.md"),
        "budget_used": result.budget_used,
    }


def _config_for_run(store: RunStore, explicit: Path | None) -> Config:
    if explicit is not None:
        config = load_config(explicit)
        if config_fingerprint(config) != store.meta.config_fingerprint:
            raise ConfigError("config differs from the configuration recorded for this run")
        return config
    if store.exists("config.json"):
        return Config.model_validate(store.read_json("config.json"))
    raise ConfigError("run has no config.json; pass --config used for the original run")


def _cmd_experiment_plan(args: argparse.Namespace) -> int:
    from .experiments.runner import plan_experiment

    config = load_config(args.config) if args.config else Config()
    if args.mode == "live" and args.config is None:
        raise ConfigError("live experiment planning requires --config")
    spec, plan, directory = plan_experiment(
        dataset_root=args.dataset,
        output_root=args.output,
        variant_id=args.variant,
        repetitions=args.repetitions,
        provider_mode=args.mode,
        project_root=args.project_root or Path(__file__).resolve().parents[2],
        config=config,
    )
    _print(
        {
            "status": "planned",
            "experiment_id": spec.experiment_id,
            "experiment_dir": str(directory),
            "dataset_digest": spec.dataset_digest,
            "slots": len(plan.slots),
        }
    )
    return 0


def _cmd_experiment_run(args: argparse.Namespace) -> int:
    from .experiments.runner import run_experiment

    if args.mode == "live" and (not args.run_live or args.config is None):
        raise ConfigError("live experiment run requires --run-live and --config")
    config = load_config(args.config) if args.config else Config()
    payload = run_experiment(
        experiment_dir=args.experiment_dir,
        mode=args.mode,
        project_root=args.project_root or Path(__file__).resolve().parents[2],
        config=config,
        replay=args.replay,
    )
    _print(payload)
    if payload["error"] or payload["not_run"]:
        return 2
    return 1 if payload["fail"] else 0


def _cmd_experiment_score(args: argparse.Namespace) -> int:
    from .experiments.scoring import score_experiment

    payload = score_experiment(args.experiment_dir)
    _print(payload)
    counts = payload["counts"]
    if counts["error"] or counts["not_run"]:
        return 2
    return 1 if counts["fail"] else 0


def _cmd_experiment_compare(args: argparse.Namespace) -> int:
    from .experiments.compare import compare_experiments

    payload, code = compare_experiments(args.baseline, args.candidate)
    _print(payload)
    return code


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _print(payload: dict, *, quiet: bool = False) -> None:
    if not quiet:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
