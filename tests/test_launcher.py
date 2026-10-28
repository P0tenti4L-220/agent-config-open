"""Exercise the shipped shell entry point without local credentials or network."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from agent_triage.demo import demo_scenario, write_demo_project


@pytest.fixture
def launcher(tmp_path: Path) -> Path:
    root = tmp_path / "checkout with 'quotes'"
    (root / "scripts").mkdir(parents=True)
    (root / "local").mkdir()
    (root / ".venv" / "bin").mkdir(parents=True)
    shutil.copyfile(Path(__file__).resolve().parents[1] / "scripts/run-agent.sh", root / "scripts/run-agent.sh")
    python = root / ".venv/bin/python"
    python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n', encoding="utf-8")
    python.chmod(0o755)
    return root


def _run(root: Path, *args: str, cwd: Path | None = None, extra_env: dict | None = None):
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("AGENT_TRIAGE_", "DEEPSEEK_", "ANTHROPIC_", "GITHUB_"))
    }
    env.update(extra_env or {})
    return subprocess.run(
        ["sh", str(root / "scripts/run-agent.sh"), *args],
        cwd=cwd or root, env=env, capture_output=True, text=True, timeout=20,
    )


@pytest.mark.parametrize("args", [("--help",), ("analyze", "--help"), ("--version",)])
def test_help_and_version_do_not_source_credentials(launcher: Path, args: tuple[str, ...]) -> None:
    (launcher / "local/deepseek.env").write_text("exit 91\n", encoding="utf-8")
    result = _run(launcher, *args)
    assert result.returncode == 0, result.stderr
    assert "agent-triage" in result.stdout


def test_replay_with_local_live_config_does_not_require_key(launcher: Path, tmp_path: Path) -> None:
    caller = tmp_path / "caller's project"
    write_demo_project(caller)
    original = (caller / "logs/failure.log").read_bytes()
    (caller / "replay.json").write_text(demo_scenario().model_dump_json(), encoding="utf-8")
    (launcher / "local/config.toml").write_text(
        'provider = "deepseek"\nmodel = "deepseek-flash"\n', encoding="utf-8",
    )
    # A normal invocation must never depend on the live acceptance environment.
    (launcher / "local/live-test.env").write_text("exit 92\n", encoding="utf-8")
    goal = "分析 'items'; $(touch MUST_NOT_EXIST) `touch ALSO_MUST_NOT_EXIST`"
    result = _run(
        launcher, "analyze", "--config", str(launcher / "local/config.toml"),
        "--replay=replay.json", "--symptom-file", "logs/failure.log",
        "--project-id", "sample-project", "--goal", goal, "--output=run outputs",
        cwd=caller,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert Path(payload["output_dir"]).parent == caller / "run outputs"
    for key in ("report_path", "context_path", "handoff_path"):
        assert Path(payload[key]).is_file()
    event = json.loads((Path(payload["output_dir"]) / "event.json").read_text())
    assert event["goal"] == goal
    assert event["symptom"] == "KeyError: 'items'"
    assert (caller / "logs/failure.log").read_bytes() == original
    for root in (launcher, caller):
        assert not (root / "MUST_NOT_EXIST").exists()
        assert not (root / "ALSO_MUST_NOT_EXIST").exists()


def test_shell_credentials_take_precedence_over_local_defaults(launcher: Path) -> None:
    (launcher / "local/deepseek.env").write_text("export DEEPSEEK_API_KEY=local-placeholder\n")
    (launcher / "local/github.env").write_text("export GITHUB_PERSONAL_ACCESS_TOKEN=local-placeholder\n")
    python = launcher / ".venv/bin/python"
    python.write_text(
        '#!/bin/sh\n[ "$DEEPSEEK_API_KEY" = shell-placeholder ] || exit 93\n'
        '[ "$GITHUB_PERSONAL_ACCESS_TOKEN" = shell-placeholder ] || exit 94\n',
    )
    result = _run(launcher, "policy", extra_env={
        "DEEPSEEK_API_KEY": "shell-placeholder",
        "GITHUB_PERSONAL_ACCESS_TOKEN": "shell-placeholder",
    })
    assert result.returncode == 0, result.stderr


def test_missing_explicit_config_does_not_create_run(launcher: Path) -> None:
    result = _run(launcher, "analyze", "--config=missing.toml")
    assert result.returncode == 2
    assert "配置文件不存在" in result.stderr
    assert not (launcher / "local/runs").exists()
