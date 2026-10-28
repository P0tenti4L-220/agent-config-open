"""Live smoke test.  Not run by default; requires owner-provided credentials.

This file is deliberately explicit about what it needs.  Without a model key,
a GitHub token and a designated test repository it skips, and the acceptance
status remains ``not_run`` rather than being reported as a pass.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent_triage.acceptance import Target, build_target
from agent_triage.config import Config, load_config
from agent_triage.live_bundle import verify_live_bundle, write_live_bundle
from agent_triage.livecheck import verify_issue_chain
from agent_triage.pipeline import run_new_analysis
from helpers import make_event, write_sample_project


@pytest.mark.live
def test_live_model_and_github_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = os.environ.get("AGENT_TRIAGE_EVAL_CONFIG")
    if config_path:
        config = load_config(Path(config_path))
    else:
        config = Config(
            provider="anthropic",
            model=os.environ.get("AGENT_TRIAGE_MODEL", ""),
            mcp={
                "enabled": True,
                "command": os.environ.get("AGENT_TRIAGE_MCP_COMMAND", "github-mcp-server"),
                "args": ["stdio", "--read-only", "--tools", "search_issues,issue_read"],
                "repo": os.environ.get("AGENT_TRIAGE_TEST_REPO", ""),
            },
        )
    required = {
        config.credential_env: os.environ.get(config.credential_env),
        "AGENT_TRIAGE_TEST_ISSUE": os.environ.get("AGENT_TRIAGE_TEST_ISSUE"),
        "AGENT_TRIAGE_ISSUE_MARKER": os.environ.get("AGENT_TRIAGE_ISSUE_MARKER"),
    }
    if config.mcp.enabled:
        required[config.mcp.token_env or "GITHUB_PERSONAL_ACCESS_TOKEN"] = os.environ.get(
            config.mcp.token_env or "GITHUB_PERSONAL_ACCESS_TOKEN"
        )
    missing = [name for name, value in required.items() if not value]
    if not config.mcp.enabled or not config.mcp.repo:
        missing.append("enabled MCP config with repo")
    if config.provider not in {"anthropic", "deepseek"}:
        missing.append("anthropic or deepseek provider")
    if missing:
        pytest.skip("live acceptance requires: " + ", ".join(missing))

    project = write_sample_project(tmp_path / "live-project")
    issue_ref = os.environ["AGENT_TRIAGE_TEST_ISSUE"]
    marker = os.environ["AGENT_TRIAGE_ISSUE_MARKER"]
    event = make_event(
        event_id="live-smoke-event",
        goal=(
            "Use the provided tools to read the failure evidence, then search the configured "
            f"test repository for issue {issue_ref}. After the search, you must call "
            f"github_read_issue with issue_ref {issue_ref} and cite the marker from the issue "
            f"body (the frozen marker is {marker!r}) in an evidence-backed analysis. Search "
            "results alone do not count as reading the issue. Do not claim the remote history "
            "is complete."
        ),
    )
    result = run_new_analysis(
        event=event,
        project_root=project,
        output_root=tmp_path / "live-runs",
        config=config,
        replay_scenario=None,
    )
    run_dir = tmp_path / "live-runs" / result.run_id
    assert result.remote_status == "available"
    assert result.analysis.analysis_status.value in {"complete", "partial"}
    summary = verify_issue_chain(run_dir, issue_ref=issue_ref, marker=marker)
    assert summary["read_call_id"]
    assert summary["cited_claims"]
    assert (run_dir / "report.md").exists()

    bundle_root = Path(
        os.environ.get("AGENT_TRIAGE_LIVE_BUNDLE_DIR", str(tmp_path / "live-bundles"))
    )
    bundle_dir = bundle_root / result.run_id
    target_payload = os.environ.get("AGENT_TRIAGE_EVAL_TARGET")
    if target_payload:
        target = Target.model_validate_json(target_payload)
    else:
        target = build_target(repo_root, live_config=config)
    write_live_bundle(
        run_dir,
        bundle_dir,
        issue_ref=issue_ref,
        marker=marker,
        target=target,
    )
    verify_live_bundle(bundle_dir, issue_ref=issue_ref, marker=marker)
