"""Live bundle integrity and HC09 import (F05).

The synthetic run is the same fixture used by the strict issue-chain tests: a
request view carries the designated issue body, a display record points at it,
and a claim cites it.  The bundle test then tampers with copies to prove that
the importer rejects missing or changed files.
"""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from agent_triage.acceptance import accept_profile, build_target, profile_required
from agent_triage.config import Config
from agent_triage import live_bundle
from agent_triage.cli import main
from agent_triage.errors import StoreError, ValidationFailure
from agent_triage.live_bundle import import_live_bundle, verify_live_bundle, write_live_bundle
from agent_triage.store import hash_file

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
        skip={"HC09"},
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
        skip={"HC09"},
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


def _published_hashes(evidence: Path) -> dict[str, str]:
    return {
        path.relative_to(evidence).as_posix(): hash_file(path)
        for directory in ("live", "results")
        for path in (evidence / directory).rglob("*")
        if path.is_file()
    }


def test_import_retry_and_bundle_id_conflict_preserve_old_bytes(tmp_path: Path, capsys) -> None:
    project = make_project(tmp_path / "project")
    bundle, run_dir = _bundle(tmp_path, project)
    evidence = tmp_path / "evidence"
    args = [
        "import-live", "--bundle", str(bundle), "--evidence-dir", str(evidence),
        "--project-root", str(project),
    ]
    assert main(args) == 0
    original = capsys.readouterr().out
    hashes = _published_hashes(evidence)
    assert main(args) == 0
    assert capsys.readouterr().out == original

    (run_dir / "report.md").write_text(
        (run_dir / "report.md").read_text(encoding="utf-8") + "\nIndependent note.\n",
        encoding="utf-8",
    )
    replacement = tmp_path / "replacement"
    write_live_bundle(
        run_dir, replacement, issue_ref="example/repo#7", marker=MARKER,
        target=build_target(project, live_config=_live_config()),
    )
    args[2] = str(replacement)
    assert main([*args, "--attempt", "2"]) == 1
    assert "live bundle id conflict" in capsys.readouterr().err
    assert _published_hashes(evidence) == hashes


def test_different_bundle_requires_new_attempt(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    first, _ = _bundle(tmp_path / "first", project)
    second, _ = _bundle(tmp_path / "second", project)
    evidence = tmp_path / "evidence"
    _, path = import_live_bundle(first, evidence, project_root=project)
    hashes = _published_hashes(evidence)
    with pytest.raises(ValidationFailure, match="use a new --attempt"):
        import_live_bundle(second, evidence, project_root=project)
    assert _published_hashes(evidence) == hashes
    result, _ = import_live_bundle(second, evidence, project_root=project, attempt="2")
    assert result.attempt_id == "2"
    assert hash_file(path) == hashes["results/HC09-1.json"]


def test_large_bundle_result_binds_manifest(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    run_dir = _prepare(tmp_path)
    snippets = run_dir / "evidence" / "snippets"
    snippets.mkdir(exist_ok=True)
    for index in range(51):
        (snippets / f"extra-{index}.txt").write_text("synthetic snippet", encoding="utf-8")
    bundle = tmp_path / "bundle"
    write_live_bundle(
        run_dir, bundle, issue_ref="example/repo#7", marker=MARKER,
        target=build_target(project, live_config=_live_config()),
    )

    evidence = tmp_path / "evidence"
    result, path = import_live_bundle(bundle, evidence, project_root=project)
    manifest = next(item for item in result.artifacts if item.relative_path.endswith("/manifest.json"))
    assert manifest.sha256 == hash_file(bundle / "manifest.json")
    assert import_live_bundle(bundle, evidence, project_root=project) == (result, path)


@pytest.mark.parametrize("copy_failure", ["interrupted", "corrupt_copy"])
def test_import_validates_staged_copy_and_preserves_existing_evidence(
    tmp_path: Path, monkeypatch, copy_failure: str
) -> None:
    project = make_project(tmp_path / "project")
    bundle, _ = _bundle(tmp_path, project)
    evidence = tmp_path / "evidence"
    import_live_bundle(bundle, evidence, project_root=project)
    hashes = _published_hashes(evidence)
    copytree = shutil.copytree

    def faulty_copy(source, destination, *args, **kwargs):
        if source == bundle.resolve() and copy_failure == "interrupted":
            destination.mkdir()
            shutil.copyfile(source / "manifest.json", destination / "manifest.json")
            raise OSError("copy interrupted")
        result = copytree(source, destination, *args, **kwargs)
        if source == bundle.resolve():
            (destination / "analysis.json").write_text("damaged copy", encoding="utf-8")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(live_bundle.shutil, "copytree", faulty_copy)
        error = StoreError if copy_failure == "interrupted" else ValidationFailure
        with pytest.raises(error):
            import_live_bundle(bundle, evidence, project_root=project, attempt="2")
    assert _published_hashes(evidence) == hashes
    import_live_bundle(bundle, evidence, project_root=project, attempt="2")


def test_retry_completes_result_after_bundle_publish(tmp_path: Path, monkeypatch) -> None:
    project = make_project(tmp_path / "project")
    bundle, _ = _bundle(tmp_path, project)
    evidence = tmp_path / "evidence"

    def interrupted_write(path, payload):
        raise StoreError("result write interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(live_bundle, "atomic_write_json", interrupted_write)
        with pytest.raises(StoreError, match="result write interrupted"):
            import_live_bundle(bundle, evidence, project_root=project)
    hashes = _published_hashes(evidence)
    assert hashes and all(path.startswith("live/") for path in hashes)

    _, path = import_live_bundle(bundle, evidence, project_root=project)
    assert path.is_file()
    assert {name: hash_file(evidence / name) for name in hashes} == hashes


@pytest.mark.parametrize("same_bundle", [True, False])
def test_concurrent_imports_serialize_publication(tmp_path: Path, same_bundle: bool) -> None:
    project = make_project(tmp_path / "project")
    first, _ = _bundle(tmp_path / "first", project)
    second = first if same_bundle else _bundle(tmp_path / "second", project)[0]
    evidence = tmp_path / "evidence"
    start = Barrier(2)

    def submit(bundle: Path):
        start.wait(timeout=10)
        return import_live_bundle(bundle, evidence, project_root=project)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(submit, bundle) for bundle in (first, second)]
        if same_bundle:
            assert futures[0].result() == futures[1].result()
        else:
            assert sum(future.exception() is None for future in futures) == 1
            error = next(future.exception() for future in futures if future.exception())
            assert isinstance(error, ValidationFailure)
            assert "use a new --attempt" in str(error)
    hashes = _published_hashes(evidence)
    results = [future.result() for future in futures if future.exception() is None]
    winner, _ = results[0]
    assert len(list((evidence / "live").iterdir())) == 1
    assert all(hashes[item.relative_path] == item.sha256 for item in winner.artifacts)
    assert "results/HC09-1.json" in hashes
