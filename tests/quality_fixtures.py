"""Synthetic, fully bound quality evidence for contract tests.

These helpers are explicitly synthetic.  They prove that schema 3 import and
re-verification work against real replay runs and real files; they are not a
human review and do not establish AC15.quality for the repository.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_triage.acceptance_review import prepare_review
from agent_triage.config import Config
from agent_triage.pipeline import run_new_analysis
from agent_triage.quality import write_dataset_manifest
from agent_triage.store import atomic_write_json, hash_file, read_json

from helpers import base_analysis, final_step, make_event, make_scenario, tool_step, write_sample_project
from test_acceptance import case_hashes, make_project, write_envelope
from agent_triage.acceptance import build_target

QUALITY_CASES = ("QA01", "QA02", "QA03", "QA04", "QA05")
PLAYBOOKS = (
    "structured-output-mismatch",
    "structured-output-mismatch",
    "tool-failure-and-recovery",
    "tool-failure-and-recovery",
    "context-and-handoff-gap",
)


def _artifact(path: Path, root: Path) -> dict:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": hash_file(path),
        "size_bytes": path.stat().st_size,
        "media_type": "application/json" if path.suffix == ".json" else "text/markdown",
    }


def build_fixture(tmp_path: Path) -> dict:
    project = make_project(tmp_path / "project")
    write_sample_project(project)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    target = build_target(project)
    write_envelope(evidence, target)
    subjects = tmp_path / "subjects"
    subjects.mkdir()
    scenario = make_scenario(
        [
            tool_step(("read_file", {"path": "logs/failure.log", "start_line": 1, "max_lines": 20})),
            tool_step(("read_file", {"path": "src/check_reader.py", "start_line": 1, "max_lines": 20})),
            tool_step(("read_file", {"path": "data/checks.json", "start_line": 1, "max_lines": 20})),
            final_step(base_analysis()),
        ]
    )
    runs = {}
    for case_id in QUALITY_CASES:
        event = make_event(event_id=f"{case_id.lower()}-event", symptom=f"synthetic {case_id}")
        result = run_new_analysis(
            event=event,
            project_root=project,
            output_root=subjects / case_id,
            config=Config(),
            replay_scenario=scenario,
        )
        runs[case_id] = subjects / case_id / result.run_id

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    cases = []
    for index, case_id in enumerate(QUALITY_CASES):
        run_dir = runs[case_id]
        source = dataset / "inputs" / case_id / "event.json"
        source.parent.mkdir(parents=True)
        source.write_bytes((run_dir / "event.json").read_bytes())
        reference = dataset / "references" / f"{case_id}.md"
        reference.parent.mkdir(parents=True, exist_ok=True)
        reference.write_text(f"# reference {case_id}\n\nsynthetic reference\n", encoding="utf-8")
        rubric = dataset / "rubrics" / f"{case_id}.json"
        rubric.parent.mkdir(parents=True, exist_ok=True)
        rubric.write_text(json.dumps({"case_id": case_id, "criteria": ["evidence", "next step"]}) + "\n", encoding="utf-8")
        cases.append(
            {
                "case_id": case_id,
                "group_id": f"group-{index}",
                "split": "pilot",
                "category": PLAYBOOKS[index],
                "playbook_id": PLAYBOOKS[index],
                "input_ref": _artifact(source, dataset),
                "fixture_refs": [],
                "reference_ref": _artifact(reference, dataset),
                "rubric_ref": _artifact(rubric, dataset),
                "origin": "synthetic",
                "replay_ref": None,
                "source_run_ref": None,
            }
        )
    write_dataset_manifest(
        dataset,
        {"schema_version": 1, "dataset_id": "synthetic-v1", "revision": "1", "cases": cases},
    )
    return {
        "project": project,
        "evidence": evidence,
        "target": target,
        "subjects": subjects,
        "dataset": dataset,
        "hashes": case_hashes(project),
    }


def prepare_completed(
    fixture: dict,
    *,
    check_id: str = "AC15.quality",
    verdicts: dict[str, str] | None = None,
    mutate: callable | None = None,
) -> Path:
    evidence = fixture["evidence"]
    template = evidence.parent / f"{check_id}.template.json"
    prepare_review(
        evidence_dir=evidence,
        check_id=check_id,
        dataset_root=fixture["dataset"],
        subjects_dir=fixture["subjects"],
        output=template,
    )
    payload = read_json(template)
    payload["reviewer_id"] = "synthetic-reviewer"
    payload["human_review"]["reviewer"] = "synthetic-reviewer"
    payload["reviewed_at"] = "2026-10-26T00:00:00+00:00"
    payload["human_review"]["reviewed_at"] = payload["reviewed_at"]
    payload["human_review"]["rubric_version"] = "rubric-v1"
    payload["human_review"]["summary"] = "synthetic contract fixture; not a real human review"
    for case in payload["human_review"].get("cases", []):
        case["verdict"] = (verdicts or {}).get(case["case_id"], "pass")
        case["rationale"] = "synthetic contract fixture"
    if mutate is not None:
        mutate(payload)
    completed = evidence.parent / f"{check_id}.completed.json"
    atomic_write_json(completed, payload)
    return completed


def narrative_record(check_id: str, target, case_hash: str, *, blockers: list[str] | None = None) -> dict:
    common = {
        "schema_version": 3,
        "record_id": f"review-{check_id.replace('.', '-')}",
        "evaluation_id": "eval-20261025-000000-abcdef",
        "check_id": check_id,
        "case_definition_hash": case_hash,
        "target": target.model_dump(mode="json"),
        "reviewer_id": "synthetic-reviewer",
        "review_method": "human",
        "reviewed_at": "2026-10-26T00:00:00+00:00",
        "artifacts": [],
        "human_review": {
            "reviewer": "synthetic-reviewer",
            "method": "human",
            "reviewed_at": "2026-10-26T00:00:00+00:00",
            "rubric_version": "rubric-v1",
            "summary": "synthetic contract fixture",
        },
    }
    if check_id == "NC20.manual":
        common["human_review"]["playbooks"] = [
            {
                "playbook_id": playbook,
                "sample": "synthetic sample",
                "steps": "followed the documented steps",
                "outcome": "observed output",
                "blockers": blockers or [],
                "verdict": "pass",
            }
            for playbook in (
                "structured-output-mismatch",
                "tool-failure-and-recovery",
                "context-and-handoff-gap",
            )
        ]
    elif check_id == "HC13":
        common["human_review"]["handoff"] = {
            "operator": "synthetic-reviewer",
            "no_chat_history": True,
            "target_identified": True,
            "facts_identified": True,
            "gaps_identified": True,
            "next_step_identified": True,
            "result": "continued from the written handoff",
            "blockers": blockers or [],
            "verdict": "pass",
        }
    return common