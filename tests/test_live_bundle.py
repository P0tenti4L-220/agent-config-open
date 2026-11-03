"""Live bundle integrity and HC09 import (F05).

The synthetic run is the same fixture used by the strict issue-chain tests: a
request view carries the designated issue body, a display record points at it,
and a claim cites it.  The bundle test then tampers with copies to prove that
the importer rejects missing or changed files.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from agent_triage.acceptance import accept_profile, build_target, profile_required
from agent_triage.config import Config
from agent_triage.errors import ValidationFailure
from agent_triage.live_bundle import import_live_bundle, verify_live_bundle, write_live_bundle

from test_acceptance import case_hashes, make_project, seed_all
from test_livecheck import MARKER, URL, _prepare


REPO_ROOT = Path(__file__).resolve().parents[1]


def _live_config() -> Config:
    return Config(provider="anthropic", model="synthetic-model")


def _bundle(tmp_path: Path, project: Path) -> tuple[Path, Path]:
    run_dir = _prepare(tmp_path)
    target = build_target(project, live_config=_live_config())
    bundle = tmp_path / "bundle"
    write_live_bundle(run_dir, bundle, issue_ref="example/repo#7", marker=MARKER, target=target)
    return bundle, run_dir


def test_bundle_round_trip_and_hc09_import(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    bundle, run_dir = _bundle(tmp_path, project)
    manifest = verify_live_bundle(bundle, issue_ref="example/repo#7", marker=MARKER)
    assert manifest.run_id
    assert manifest.chain["read_call_id"]
    assert (bundle / "request-views" / "req-000099.json").is_file()
    assert (bundle / "evidence" / "index.json").is_file()
    index = json.loads((bundle / "evidence/index.json").read_text())
    snapshots = [item["snapshot_sha256"] for item in index["evidence"] if item.get("snapshot_sha256")]
    assert snapshots
    assert all((bundle / "evidence/objects" / f"{digest}.txt").is_file() for digest in snapshots)

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    target = build_target(project, live_config=_live_config())
    seed_all(
        evidence,
        target,
        profile_required("connected"),
        hashes=case_hashes(project),
    )
    result, path = import_live_bundle(bundle, evidence, project_root=project)
    assert result.status.value == "pass"
    assert result.check_id == "HC09"
    assert result.case_definition_hash
    assert path.is_file()
    assert result.artifacts
    for artifact in result.artifacts:
        assert (evidence / artifact.relative_path).is_file()

    accepted = accept_profile(
        profile="connected",
        evidence_dir=evidence,
        project_root=project,
        config=_live_config(),
    )
    assert accepted.overall == "pass", [
        entry for entry in accepted.checks if entry.execution_status.value != "pass"
    ]
    hc09 = next(entry for entry in accepted.checks if entry.check_id == "HC09")
    assert hc09.source_type == "run_artifact"
    assert hc09.effective_state == "current"


def test_bundle_rejects_changed_file(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    bundle, _ = _bundle(tmp_path, project)
    (bundle / "analysis.json").write_text('{"tampered": true}\n', encoding="utf-8")
    with pytest.raises(ValidationFailure, match="hash changed|size changed"):
        verify_live_bundle(bundle, issue_ref="example/repo#7", marker=MARKER)


def test_bundle_rejects_extra_file(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    bundle, _ = _bundle(tmp_path, project)
    (bundle / "unlisted.txt").write_text("extra\n", encoding="utf-8")
    with pytest.raises(ValidationFailure, match="not covered by the manifest"):
        verify_live_bundle(bundle, issue_ref="example/repo#7", marker=MARKER)


def test_bundle_rejects_wrong_marker(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    bundle, _ = _bundle(tmp_path, project)
    with pytest.raises(ValidationFailure, match="marker"):
        verify_live_bundle(bundle, issue_ref="example/repo#7", marker="OTHER-MARKER")


def test_import_rejects_non_empty_bundle_directory(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    run_dir = _prepare(tmp_path)
    target = build_target(project, live_config=_live_config())
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "keep.txt").write_text("keep\n", encoding="utf-8")
    with pytest.raises(Exception, match="not empty"):
        write_live_bundle(run_dir, bundle, issue_ref="example/repo#7", marker=MARKER, target=target)

def test_eval_imports_a_bundle_already_under_evidence_live(tmp_path: Path) -> None:
    from agent_triage.case_manifest import load_manifest
    from agent_triage.eval import _import_live_bundles

    project = make_project(tmp_path / "project")
    run_dir = _prepare(tmp_path)
    evidence = tmp_path / "evidence"
    bundle = evidence / "live" / run_dir.name
    target = build_target(project, live_config=_live_config())
    write_live_bundle(run_dir, bundle, issue_ref="example/repo#7", marker=MARKER, target=target)
    results = {
        "HC09": {
            "id": "HC09",
            "status": "not_run",
            "reason_code": "external_record_required",
            "reason": "placeholder",
            "artifacts": [],
        }
    }
    _import_live_bundles(
        evidence_dir=evidence,
        project_root=project,
        results=results,
        by_id=load_manifest(project / "tests" / "cases").cases,
    )
    assert results["HC09"]["status"] == "pass"
    assert (evidence / "results" / "HC09-1.json").is_file()
    assert (evidence / "live" / run_dir.name / "manifest.json").is_file()


def test_eval_rejects_a_tampered_bundle_as_hc09_error(tmp_path: Path) -> None:
    from agent_triage.case_manifest import load_manifest
    from agent_triage.eval import _import_live_bundles

    project = make_project(tmp_path / "project")
    run_dir = _prepare(tmp_path)
    evidence = tmp_path / "evidence"
    bundle = evidence / "live" / run_dir.name
    target = build_target(project, live_config=_live_config())
    write_live_bundle(run_dir, bundle, issue_ref="example/repo#7", marker=MARKER, target=target)
    (bundle / "analysis.json").write_text('{"tampered": true}\n', encoding="utf-8")
    results = {"HC09": {"id": "HC09", "status": "not_run"}}
    _import_live_bundles(
        evidence_dir=evidence,
        project_root=project,
        results=results,
        by_id=load_manifest(project / "tests" / "cases").cases,
    )
    assert results["HC09"]["status"] == "error"
    assert results["HC09"]["reason_code"] == "live_bundle_invalid"


def test_live_config_change_makes_connected_evidence_incomplete(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    original = _live_config()
    other = Config(provider="anthropic", model="other-model")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    seed_all(
        evidence,
        build_target(project, live_config=other),
        profile_required("connected"),
        hashes=case_hashes(project),
    )
    run_dir = _prepare(tmp_path)
    bundle = tmp_path / "bundle"
    write_live_bundle(
        run_dir,
        bundle,
        issue_ref="example/repo#7",
        marker=MARKER,
        target=build_target(project, live_config=original),
    )
    import_live_bundle(bundle, evidence, project_root=project)
    accepted = accept_profile(
        profile="connected",
        evidence_dir=evidence,
        project_root=project,
        config=original,
    )
    assert accepted.overall == "incomplete"
    live = next(entry for entry in accepted.checks if entry.check_id == "AC15.live")
    assert live.effective_state in {"stale", "unknown"}
    hc09 = next(entry for entry in accepted.checks if entry.check_id == "HC09")
    assert hc09.effective_state == "current"


def test_run_eval_rejects_live_without_config(tmp_path: Path) -> None:
    from agent_triage.errors import InputError
    from agent_triage.eval import run_eval

    with pytest.raises(InputError, match="verified configuration"):
        run_eval(
            cases_dir=REPO_ROOT / "tests" / "cases",
            output_root=tmp_path / "eval",
            include_live=True,
        )
    assert not (tmp_path / "eval").exists()
