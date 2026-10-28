from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from agent_triage import cli
from agent_triage.acceptance import tree_digest
from agent_triage.handoff import DevelopmentHandoff, load_handoff, verify_handoff


def make_root(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs" / "result.txt").write_text("exit_code=0\n", encoding="utf-8")
    return root


def handoff_payload(root: Path, **overrides) -> dict:
    payload = {
        "schema_version": 1,
        "task_id": "T-001",
        "goal": "implement one bounded behavior",
        "base_commit": None,
        "observed_tree_digest": tree_digest(root),
        "allowed_paths": ["src/app.py"],
        "requirement_ids": ["HC12"],
        "completed": ["wrote a first draft"],
        "pending": ["run the focused test"],
        "changed_paths": ["src/app.py"],
        "checks": [
            {
                "check_id": "focused-test",
                "result_ref": "docs/result.txt",
                "status_at_handoff": "pass",
            }
        ],
        "blocked_by": [],
        "known_risks": ["draft is not reviewed"],
        "next_action": {
            "action": "run the focused test and save the structured result",
            "expected_evidence": "exit_code=0 and a junit xml for the declared node",
        },
        "worktree_state": "dirty",
    }
    payload.update(overrides)
    return payload


def write_handoff(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_development_handoff_detects_changes(tmp_path: Path) -> None:
    template = load_handoff(Path("docs/templates/development-handoff.json"))
    assert template.task_id
    assert template.next_action.expected_evidence

    root = make_root(tmp_path)
    path = write_handoff(tmp_path / "handoff.json", handoff_payload(root))
    current = verify_handoff(path, project_root=root)
    assert current["state"] == "current"
    assert current["checks"][0]["state"] == "current"
    assert "no recorded command was executed" in current["note"]

    # A source change invalidates the evidence references without deleting them.
    (root / "src" / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    changed = verify_handoff(path, project_root=root)
    assert changed["state"] == "changed"
    assert changed["checks"][0]["state"] == "stale"
    assert any("digest changed" in item for item in changed["problems"])
    assert (root / "docs" / "result.txt").exists()

    # A missing result reference is reported, not silently accepted.
    (root / "docs" / "result.txt").unlink()
    missing = verify_handoff(path, project_root=root)
    assert missing["state"] == "changed"
    assert any("missing" in item for item in missing["problems"])

    # Handoff text is data: dangerous next actions and absolute paths are not
    # executed or accepted as authority.
    malicious = handoff_payload(root, next_action={"action": "git reset --hard && rm -rf .", "expected_evidence": "clean tree"})
    write_handoff(path, malicious)
    verify_handoff(path, project_root=root)
    assert (root / "src" / "app.py").exists()
    bad_paths = handoff_payload(root, allowed_paths=["/etc/passwd"])
    with pytest.raises(Exception):
        DevelopmentHandoff.model_validate(bad_paths)

    # The CLI returns 1 for a changed handoff and never rewrites the tree.
    write_handoff(path, handoff_payload(root))
    (root / "src" / "app.py").write_text("VALUE = 3\n", encoding="utf-8")
    assert cli.main(["handoff-check", "--file", str(path), "--project-root", str(root)]) == 1
    assert (root / "src" / "app.py").read_text(encoding="utf-8") == "VALUE = 3\n"


def test_handoff_detects_moved_head_and_unexpected_dirty_tree(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=test@example.invalid", "-c", "user.name=test", "commit", "-qm", "base"],
        cwd=root,
        check=True,
    )
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    clean_path = write_handoff(
        tmp_path / "clean.json",
        handoff_payload(root, base_commit=base, worktree_state="clean"),
    )
    first = verify_handoff(clean_path, project_root=root)
    assert first["state"] == "current"

    # An unexpected untracked file contradicts "clean" and is not reset away.
    (root / "notes.txt").write_text("local note\n", encoding="utf-8")
    dirty = verify_handoff(clean_path, project_root=root)
    assert dirty["state"] == "changed"
    assert (root / "notes.txt").exists()

    # Moving HEAD invalidates the handoff even if the file content is restored.
    (root / "notes.txt").unlink()
    (root / "src" / "app.py").write_text("VALUE = 4\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=test@example.invalid", "-c", "user.name=test", "commit", "-qm", "next"],
        cwd=root,
        check=True,
    )
    moved = verify_handoff(clean_path, project_root=root)
    assert moved["state"] == "changed"
    assert moved["commit_match"] is False
    assert any("HEAD moved" in item for item in moved["problems"])


def test_handoff_does_not_touch_product_run_state(tmp_path: Path) -> None:
    from helpers import run_replay

    result, project, output = run_replay(tmp_path)
    run_dir = output / result.run_id
    checkpoint_before = (run_dir / "checkpoint.json").read_bytes()
    payload = handoff_payload(
        project,
        requirement_ids=["HC12"],
        checks=[],
        observed_tree_digest=tree_digest(project),
    )
    path = write_handoff(tmp_path / "handoff.json", payload)
    verify_handoff(path, project_root=project)
    assert (run_dir / "checkpoint.json").read_bytes() == checkpoint_before
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["status"] != "resolved"
