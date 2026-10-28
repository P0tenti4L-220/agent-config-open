"""Live tests are opt-in even when every credential variable is present (F06)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from agent_triage import cli

REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_LIVE_ENV = {
    "ANTHROPIC_API_KEY": "not-a-real-key",
    "AGENT_TRIAGE_MODEL": "synthetic-model",
    "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp-not-a-real-token",
    "AGENT_TRIAGE_TEST_REPO": "example/repo",
    "AGENT_TRIAGE_TEST_ISSUE": "example/repo#7",
    "AGENT_TRIAGE_ISSUE_MARKER": "FROZEN-ISSUE-MARKER-7",
}


def test_live_test_skips_without_explicit_opt_in(tmp_path: Path) -> None:
    env = dict(os.environ)
    env.update(FAKE_LIVE_ENV)
    env.pop("AGENT_TRIAGE_EVAL_CONFIG", None)
    env.pop("AGENT_TRIAGE_LIVE_BUNDLE_DIR", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/integration/test_live_smoke.py",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "1 skipped" in completed.stdout
    assert "passed" not in completed.stdout


def test_eval_include_live_requires_a_verified_config(tmp_path: Path) -> None:
    code = cli.main(
        [
            "eval",
            "--cases",
            str(REPO_ROOT / "tests" / "cases"),
            "--output",
            str(tmp_path / "eval"),
            "--include-live",
        ]
    )
    assert code == 2
    assert not (tmp_path / "eval").exists()
