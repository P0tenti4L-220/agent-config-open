from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_triage import cli
from agent_triage.config import Config, config_fingerprint
from agent_triage.demo import run_demo, write_demo_project, demo_event, demo_scenario
from agent_triage.pipeline import run_new_analysis
from agent_triage.store import RunStore, hash_file
from helpers import make_scenario, read_json, write_sample_project


def test_demo_pipeline_facts_and_refs(tmp_path: Path) -> None:
    result = run_demo(tmp_path / "demo")
    run_dir = tmp_path / "demo" / "runs" / result.run_id
    analysis = read_json(run_dir / "analysis.json")
    evidence = read_json(run_dir / "evidence" / "index.json")["evidence"]
    known = {item["evidence_id"]: item for item in evidence}
    assert analysis["facts"]
    for fact in analysis["facts"]:
        assert fact["evidence_ids"]
        for evidence_id in fact["evidence_ids"]:
            assert evidence_id in known
            assert known[evidence_id]["stale"] is False
    checks = {item["check_id"]: item for item in read_json(run_dir / "checks.json")["checks"]}
    assert checks["project-tests"]["status"] == "not_run"
    assert checks["reproduction"]["status"] == "not_run"
    assert checks["github-history"]["status"] == "not_run"
    assert checks["claim-display"]["status"] == "pass"
    assert checks["claim-support"]["status"] == "not_run"
    assert checks["handoff-contract"]["target_sha256"] == hash_file(run_dir / "handoff.md")
    assert analysis["analysis_status"] == "partial"
    assert result.remote_status == "not_configured"
    assert any("GitHub" in item for item in analysis["missing_evidence"])
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "未执行" in report
    assert "已通过测试" not in report
    handoff = (run_dir / "handoff.md").read_text(encoding="utf-8")
    assert "resume --run-id" in handoff
    assert (run_dir / "issue-draft.md").exists()
    trace = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert sum(1 for item in trace if item["kind"] == "model") == 5
    assert sum(1 for item in trace if item["kind"] == "tool") == 4


def test_demo_outputs_do_not_contain_absolute_paths(tmp_path: Path) -> None:
    result = run_demo(tmp_path / "demo")
    run_dir = tmp_path / "demo" / "runs" / result.run_id
    for path in run_dir.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".md", ".jsonl"}:
            text = path.read_text(encoding="utf-8")
            assert str(tmp_path) not in text, path
            assert "/Users/example" not in text, path


def test_cli_demo_list_and_validate(tmp_path: Path, capsys) -> None:
    output = tmp_path / "demo"
    assert cli.main(["demo", "--output", str(output)]) == 0
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert payload["run_id"].startswith("run-")
    assert payload["analysis_status"] == "partial"
    runs_root = output / "runs"
    assert cli.main(["list", "--output", str(runs_root)]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert len(listing["runs"]) == 1
    assert cli.main(["validate", "--run-id", payload["run_id"], "--output", str(runs_root)]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["status"] == "pass"


def test_cli_analyze_with_replay(tmp_path: Path, capsys) -> None:
    project = write_sample_project(tmp_path / "project")
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(demo_event().model_dump(mode="json")), encoding="utf-8")
    scenario_path = tmp_path / "replay.json"
    scenario_path.write_text(
        json.dumps(demo_scenario().model_dump(mode="json"), ensure_ascii=False),
        encoding="utf-8",
    )
    exit_code = cli.main(
        [
            "analyze",
            "--event",
            str(event_path),
            "--project-root",
            str(project),
            "--replay",
            str(scenario_path),
            "--output",
            str(tmp_path / "runs"),
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["analysis_status"] == "partial"
    assert "some listed materials" in payload["status_explanation"]
    assert payload["missing_evidence"]
    run_dir = tmp_path / "runs" / payload["run_id"]
    assert payload["output_dir"] == str(run_dir)
    assert payload["report_path"] == str(run_dir / "report.md")
    assert payload["context_path"] == str(run_dir / "context.json")
    assert payload["handoff_path"] == str(run_dir / "handoff.md")
    for key in ("report_path", "context_path", "handoff_path"):
        assert Path(payload[key]).is_file()
    assert (run_dir / "report.md").exists()
    assert (run_dir / "config.json").exists()
    assert str(tmp_path) not in (run_dir / "config.json").read_text(encoding="utf-8")
    assert not Path(read_json(run_dir / "run.json")["project_root"]).is_absolute()


def test_cli_replay_without_script_is_a_config_error(tmp_path: Path, capsys) -> None:
    project = write_sample_project(tmp_path / "project")
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(demo_event().model_dump(mode="json")), encoding="utf-8")
    exit_code = cli.main(
        [
            "analyze",
            "--event",
            str(event_path),
            "--project-root",
            str(project),
            "--output",
            str(tmp_path / "runs"),
        ]
    )
    assert exit_code == 2
    error = json.loads(capsys.readouterr().err)
    assert "replay" in error["error"]


def test_mcp_configured_but_unavailable_stays_partial(tmp_path: Path) -> None:
    project = write_sample_project(tmp_path / "project")
    scenario = make_scenario(
        [
            {
                "type": "tools",
                "calls": [{"tool": "read_file", "arguments": {"path": "logs/failure.log"}}],
            },
            {
                "type": "final",
                "analysis": {
                    "schema_version": 1,
                    "summary": "local only",
                    "attribution": "undetermined",
                    "facts": [
                        {
                            "text": "log shows KeyError",
                            "evidence_ids": ["@path:logs/failure.log"],
                        }
                    ],
                    "hypotheses": [],
                    "missing_evidence": [],
                    "suggested_checks": [],
                    "duplicate_candidates": [],
                    "analysis_status": "complete",
                    "stop_reason": "completed",
                },
            },
        ]
    )
    config = Config(
        mcp={
            "enabled": True,
            "command": "definitely-not-a-real-mcp-binary",
            "args": [],
            "repo": "example/repo",
        }
    )
    result = run_new_analysis(
        event=demo_event(),
        project_root=project,
        output_root=tmp_path / "runs",
        config=config,
        replay_scenario=scenario,
    )
    assert result.remote_status == "mcp_command_not_found"
    assert result.analysis.analysis_status.value == "partial"
    store = RunStore.load(tmp_path / "runs", result.run_id)
    checks = {item["check_id"]: item for item in store.read_json("checks.json")["checks"]}
    assert checks["github-history"]["status"] == "error"
    assert checks["project-tests"]["status"] == "not_run"
    report = (store.dir / "report.md").read_text(encoding="utf-8")
    assert "remote history not checked" in report
    assert "mcp_command_not_found" in report


def test_cli_resume_uses_checkpoint_and_stored_replay(tmp_path: Path, capsys) -> None:
    from agent_triage.errors import RunInterrupted
    from helpers import run_replay

    with pytest.raises(RunInterrupted):
        run_replay(tmp_path, interrupt_after_rounds=1)
    run_dir = next(path for path in (tmp_path / "runs").iterdir() if path.name.startswith("run-"))
    exit_code = cli.main(
        ["resume", "--run-id", run_dir.name, "--output", str(tmp_path / "runs")]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["resumed"] is True
    assert payload["budget_used"]["model_requests_used"] == 4
    assert (run_dir / "analysis.json").exists()


def test_demo_replay_file_is_stored_for_resume(tmp_path: Path) -> None:
    result = run_demo(tmp_path / "demo")
    run_dir = tmp_path / "demo" / "runs" / result.run_id
    replay = read_json(run_dir / "replay.json")
    assert replay["name"] == "demo-keyerror-items"
    meta = read_json(run_dir / "run.json")
    assert not Path(meta["project_root"]).is_absolute()
    assert meta["project_root"]

def test_missing_replay_does_not_leave_an_empty_run(tmp_path: Path) -> None:
    from agent_triage.errors import ConfigError
    from helpers import make_event

    project = write_sample_project(tmp_path / "project")
    with pytest.raises(ConfigError):
        run_new_analysis(
            event=make_event(),
            project_root=project,
            output_root=tmp_path / "runs",
            config=Config(),
            replay_scenario=None,
        )
    assert not (tmp_path / "runs").exists()


def test_invalid_utf8_symptom_file_is_an_input_error(tmp_path: Path, capsys) -> None:
    project = write_sample_project(tmp_path / "project")
    (project / "logs" / "bad.log").write_bytes(b"\xff\xfe\x00bad")
    exit_code = cli.main(
        [
            "analyze",
            "--symptom-file",
            str(project / "logs" / "bad.log"),
            "--goal",
            "test",
            "--project-id",
            "sample-project",
            "--project-root",
            str(project),
            "--replay",
            "unused.json",
            "--output",
            str(tmp_path / "runs"),
        ]
    )
    assert exit_code == 2
    assert "UTF-8" in capsys.readouterr().err


def test_cli_policy_manifest_is_code_derived(capsys) -> None:
    assert cli.main(["policy"]) == 0
    payload = json.loads(capsys.readouterr().out)
    manifest = payload["manifest"]
    assert manifest["policy_version"] == "policy-v3"
    assert len(manifest["policy_hash"]) == 64
    assert len(manifest["prompt_hash"]) == 64
    assert [rule["rule_id"] for rule in payload["rules"]] == [f"R0{index}" for index in range(1, 10)]
