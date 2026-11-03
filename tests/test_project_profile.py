"""Offline tests for the sourced project profile."""

from __future__ import annotations

from pathlib import Path

from agent_triage.contracts import DependencyMapping, IssueResearchConfig
from agent_triage.project_profile import build_project_profile
from agent_triage.tools import Workspace
from tests.helpers import make_event


def _project(tmp_path: Path) -> Workspace:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "example-application"',
                'version = "0.1.0"',
                'requires-python = ">=3.11"',
                'dependencies = ["client-lib>=1.0,<2", "requests==2.31.0"]',
            ]
        ),
        encoding="utf-8",
    )
    (project / "uv.lock").write_text(
        "\n".join(
            [
                "version = 1",
                "[[package]]",
                'name = "client-lib"',
                'version = "1.4.2"',
                "[[package]]",
                'name = "requests"',
                'version = "2.31.0"',
            ]
        ),
        encoding="utf-8",
    )
    git = project / ".git"
    git.mkdir()
    (git / "config").write_text(
        "\n".join(
            [
                '[remote "origin"]',
                "\turl = git@github.com:example/application.git",
                "\tfetch = +refs/heads/*:refs/remotes/origin/*",
            ]
        ),
        encoding="utf-8",
    )
    return Workspace(project)


def test_profile_reads_declared_and_locked_versions_with_sources(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    config = IssueResearchConfig(
        mode="layered",
        max_scope="public",
        dependencies=[DependencyMapping(package="client-lib", import_names=["client_lib"], repo="upstream/client")],
    )
    profile = build_project_profile(
        workspace=workspace,
        config=config,
        event=make_event(),
        project_repo="example/application",
    )
    assert profile.project_repo == "example/application"
    assert profile.repo_source == "config_remote_match"
    by_package = {item.package: item for item in profile.dependencies}
    dependency = by_package["client-lib"]
    assert dependency.declared == ">=1.0,<2"
    assert dependency.locked == "1.4.2"
    assert dependency.upstream_repo == "upstream/client"
    assert dependency.observed is None
    assert "pyproject.toml" in profile.source_hashes
    assert "uv.lock" in profile.source_hashes
    assert by_package["requests"].upstream_repo is None
    assert any("dependency_repo_mapping_missing:requests" == gap for gap in profile.gaps)


def test_profile_reports_remote_mismatch_without_widening(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    config = IssueResearchConfig(mode="layered", max_scope="current")
    profile = build_project_profile(
        workspace=workspace,
        config=config,
        event=make_event(),
        project_repo="other/application",
    )
    assert profile.project_repo == "other/application"
    assert profile.repo_source == "config_remote_mismatch"
    assert profile.local_remote == "example/application"
    assert any("configured repository differs" in gap for gap in profile.gaps)


def test_profile_keeps_unknown_when_manifests_are_absent(tmp_path: Path) -> None:
    project = tmp_path / "empty"
    project.mkdir()
    profile = build_project_profile(
        workspace=Workspace(project),
        config=IssueResearchConfig(mode="layered", max_scope="current"),
        event=make_event(environment={"language": None, "runtime_version": None}),
        project_repo=None,
    )
    assert profile.project_repo is None
    assert profile.repo_source == "unresolved"
    assert profile.dependencies == []
    assert "project language could not be established from manifests" in profile.gaps