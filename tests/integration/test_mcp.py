"""Integration tests: real stdio MCP protocol against a synthetic server."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agent_triage.contracts import MCPConfig
from agent_triage.errors import ToolFailure
from agent_triage.mcp_client import ALLOWED_TOOLS, GitHubMCPBackend

SERVER = Path(__file__).resolve().parent / "fake_mcp_server.py"


def _config(mode: str = "normal") -> MCPConfig:
    return MCPConfig(
        enabled=True,
        command=sys.executable,
        args=[str(SERVER), mode],
        repo="example/repo",
        startup_timeout_seconds=15,
        call_timeout_seconds=10,
    )


def _backend(mode: str = "normal") -> GitHubMCPBackend:
    backend = GitHubMCPBackend(_config(mode))
    assert backend.start() is True, backend.unavailable_reason
    return backend


def test_local_mcp_protocol_and_error_adapter() -> None:
    backend = _backend("normal")
    try:
        assert ALLOWED_TOOLS == ("search_issues", "issue_read")
        result = backend.search_issues("KeyError items", 2)
        assert len(result.issues) == 2
        assert result.issues[0]["issue_ref"] == "example/repo#1"
        assert result.total_count == 3
        assert result.complete is False
        assert "repo:example/repo" in result.query_scope
        with pytest.raises(ToolFailure, match="not returned by github_search_issues"):
            backend.read_issue("example/repo#3")
        read = backend.read_issue("example/repo#1")
        assert read.issue["number"] == 1
        assert read.url.startswith("https://")
        assert read.comments and read.comments[0]["body"] == "Synthetic comment body."
        with pytest.raises(ToolFailure, match="outside the configured repository"):
            backend.read_issue("other/repo#1")
    finally:
        backend.close()


def test_iserror_is_not_converted_to_empty_results() -> None:
    backend = _backend("iserror")
    try:
        with pytest.raises(ToolFailure) as info:
            backend.search_issues("KeyError", 3)
        assert info.value.code == "mcp_tool_error"
        assert "permission denied" in info.value.message.lower()
        assert info.value.retryable is False
    finally:
        backend.close()


def test_unknown_schema_and_bad_items_fail_explicitly() -> None:
    backend = _backend("unknown")
    try:
        with pytest.raises(ToolFailure) as info:
            backend.search_issues("KeyError", 3)
        assert info.value.code == "unsupported_result"
    finally:
        backend.close()
    bad = _backend("baditem")
    try:
        with pytest.raises(ToolFailure) as info:
            bad.search_issues("KeyError", 3)
        assert info.value.code == "unsupported_result"
    finally:
        bad.close()


def test_pagination_is_bounded_and_marked_incomplete() -> None:
    backend = _backend("pagination")
    try:
        result = backend.search_issues("KeyError", 3)
        assert len(result.issues) == 3
        assert result.total_count == 10
        assert result.complete is False
    finally:
        backend.close()
    empty = _backend("empty")
    try:
        result = empty.search_issues("nothing", 3)
        assert result.issues == []
        assert result.complete is True
        with pytest.raises(ToolFailure, match="not returned by github_search_issues"):
            empty.read_issue("example/repo#1")
    finally:
        empty.close()


def test_startup_failure_is_reported_not_raised_by_runner() -> None:
    config = MCPConfig(
        enabled=True,
        command="definitely-not-a-real-mcp-binary",
        args=[],
        repo="example/repo",
    )
    backend = GitHubMCPBackend(config)
    assert backend.start() is False
    assert backend.unavailable_reason == "mcp_command_not_found"
    with pytest.raises(ToolFailure) as info:
        backend.search_issues("x", 1)
    assert info.value.code == "mcp_command_not_found"