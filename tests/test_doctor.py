from __future__ import annotations

import json
import socket
import subprocess
from pathlib import Path

import pytest

from agent_triage import cli
from agent_triage.doctor import run_doctor


SECRET = "sk-test-should-never-appear"


def test_doctor_readiness_and_no_side_effects(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "doctor"

    # Offline readiness does not require a model credential or MCP.
    offline = run_doctor(
        profile="offline",
        output=output,
        project_root=Path.cwd(),
        env={},
    )
    assert offline.status == "ready"
    assert offline.network == "not_run"
    assert offline.model_call == "not_run"
    assert offline.probe_cleaned is True
    assert not list(output.glob(".doctor-probe-*"))
    assert (output / "doctor.json").exists()
    by_id = {check.check_id: check for check in offline.checks}
    assert by_id["interpreter"].status == "pass"
    assert by_id["package_import"].status == "pass"
    assert by_id["dependency:pydantic"].status == "pass"

    # Connected readiness with the replay provider is blocked, not passed.
    connected = run_doctor(profile="connected", output=output, project_root=Path.cwd(), env={})
    assert connected.status == "blocked"
    assert any(check.check_id == "provider_credential" and check.status == "blocked" for check in connected.checks)

    # A live config with a missing MCP executable is blocked; the secret value
    # is never included in the report.
    config = tmp_path / "config.toml"
    config.write_text(
        'provider = "anthropic"\n'
        'model = "model-id"\n'
        'credential_env = "ANTHROPIC_API_KEY"\n'
        "[mcp]\n"
        "enabled = true\n"
        'command = "definitely-not-installed-mcp"\n'
        'args = ["stdio"]\n'
        'repo = "example/repo"\n',
        encoding="utf-8",
    )
    live = run_doctor(
        profile="connected",
        output=output,
        config_path=config,
        project_root=Path.cwd(),
        env={"ANTHROPIC_API_KEY": SECRET, "GITHUB_PERSONAL_ACCESS_TOKEN": SECRET},
    )
    assert live.status == "blocked"
    by_id = {check.check_id: check for check in live.checks}
    assert by_id["provider_credential"].status == "pass"
    assert by_id["mcp_executable"].status == "blocked"
    serialized = json.dumps([check.model_dump(mode="json") for check in live.checks], ensure_ascii=False)
    assert SECRET not in serialized
    assert "present" in by_id["provider_credential"].detail

    # A live provider with MCP disabled is still blocked for connected.
    no_mcp = tmp_path / "no-mcp.toml"
    no_mcp.write_text(
        'provider = "anthropic"\n'
        'model = "model-id"\n'
        'credential_env = "ANTHROPIC_API_KEY"\n',
        encoding="utf-8",
    )
    blocked_mcp = run_doctor(
        profile="connected",
        output=output,
        config_path=no_mcp,
        project_root=Path.cwd(),
        env={"ANTHROPIC_API_KEY": "present"},
    )
    assert blocked_mcp.status == "blocked"
    assert any(check.check_id == "mcp_config" and check.status == "blocked" for check in blocked_mcp.checks)

    # Invalid config is blocked with a readable reason.
    bad = tmp_path / "bad.toml"
    bad.write_text('provider = "unknown"\n', encoding="utf-8")
    invalid = run_doctor(profile="offline", output=output, config_path=bad, project_root=Path.cwd(), env={})
    assert invalid.status == "blocked"
    assert any(check.check_id == "config" and check.status == "blocked" for check in invalid.checks)

    # An output path that is a file is blocked and the probe is not left behind.
    not_a_dir = tmp_path / "file"
    not_a_dir.write_text("x", encoding="utf-8")
    unusable = run_doctor(profile="offline", output=not_a_dir, project_root=Path.cwd(), env={})
    assert unusable.status == "blocked"
    assert any(check.check_id == "output_probe" and check.status == "blocked" for check in unusable.checks)

    # Wrong import root is reported instead of silently accepting another tree.
    fake = tmp_path / "fake"
    (fake / "src" / "agent_triage").mkdir(parents=True)
    wrong = run_doctor(profile="offline", output=output, project_root=fake, env={})
    assert wrong.status == "blocked"
    assert any(check.check_id == "import_root" and check.status == "blocked" for check in wrong.checks)


def test_doctor_never_opens_network_or_spawns_processes(tmp_path: Path, monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("doctor must not create a socket or subprocess")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    report = run_doctor(profile="connected", output=tmp_path / "doctor", project_root=Path.cwd(), env={})
    assert report.network == "not_run"
    assert report.status in {"ready", "blocked"}


def test_doctor_cli_exit_codes(tmp_path: Path, capsys) -> None:
    assert cli.main(["doctor", "--profile", "offline", "--output", str(tmp_path / "d1")]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ready"
    assert "not AC15.live" in payload["note"]

    assert cli.main(["doctor", "--profile", "connected", "--output", str(tmp_path / "d2")]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked"
