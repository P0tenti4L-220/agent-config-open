from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from agent_triage.contracts import BudgetLimits
from helpers import make_tool_executor, write_sample_project


def _execute(executor, tool: str, arguments: dict, *, call_id: str = "call-1", turn: int = 0):
    return executor.execute(tool, arguments, call_id=call_id, turn=turn)


def test_read_file_returns_bounded_evidence(tmp_path: Path) -> None:
    executor, store, evidence, budget, project, _ = make_tool_executor(tmp_path)
    result = _execute(executor, "read_file", {"path": "src/check_reader.py", "start_line": 1, "max_lines": 10})
    assert result.status.value == "ok"
    assert result.truncated is False
    assert len(result.evidence_refs) == 1
    item = evidence.require(result.evidence_refs[0])
    assert item.path == "src/check_reader.py"
    assert "payload" in item.snippet
    assert item.sha256 == hashlib.sha256((project / "src/check_reader.py").read_bytes()).hexdigest()
    assert budget.state.tool_attempts_used == 1
    trace = store.read_trace()
    assert len(trace) == 1
    assert trace[0].tool_name == "read_file"
    assert trace[0].arguments == {"path": "src/check_reader.py", "start_line": 1, "max_lines": 10}


def test_path_traversal_credentials_and_shell_chars(tmp_path: Path) -> None:
    executor, store, evidence, _, project, _ = make_tool_executor(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("OUTSIDE_SECRET_CANARY\n", encoding="utf-8")
    (project / "link_out").symlink_to(outside)

    for bad in ("../outside.txt", "/etc/passwd", "src/../../outside.txt"):
        result = _execute(executor, "read_file", {"path": bad})
        assert result.status.value == "error"
        assert result.error_code == "invalid_arguments"

    for sensitive in (".env", "auth.json"):
        result = _execute(executor, "read_file", {"path": sensitive})
        assert result.error_code == "sensitive_path"
        assert result.data["message"]

    result = _execute(executor, "read_file", {"path": "link_out"})
    assert result.error_code == "path_escape"

    result = _execute(executor, "read_file", {"path": "src/check_reader.py; touch pwned"})
    assert result.error_code == "missing"
    assert not (project / "pwned").exists()

    result = _execute(executor, "shell", {"command": "cat .env"})
    assert result.error_code == "unknown_tool"

    trace_text = (store.dir / "trace.jsonl").read_text(encoding="utf-8")
    assert "FAKE_SECRET_CANARY" not in trace_text
    assert "OUTSIDE_SECRET_CANARY" not in trace_text
    assert all(item.path not in (".env", "auth.json") for item in evidence.all())


def test_search_does_not_read_sensitive_files(tmp_path: Path) -> None:
    executor, _, evidence, _, _, _ = make_tool_executor(tmp_path)
    result = _execute(executor, "search_workspace", {"query": "FAKE_SECRET_CANARY", "paths": ["."], "limit": 5})
    assert result.status.value == "ok"
    assert result.data["matches"] == []
    assert result.truncated is False
    assert all(item.path not in (".env", "auth.json") for item in evidence.all())


def test_binary_oversized_and_invalid_utf8_are_readable_errors(tmp_path: Path) -> None:
    executor, _, _, _, project, _ = make_tool_executor(tmp_path)
    (project / "binary.bin").write_bytes(b"abc\x00def")
    (project / "huge.txt").write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    (project / "bad.txt").write_bytes(b"\xff\xfe\xfa")
    assert _execute(executor, "read_file", {"path": "binary.bin"}).error_code == "unsupported_binary"
    assert _execute(executor, "read_file", {"path": "huge.txt"}).error_code == "file_too_large"
    assert _execute(executor, "read_file", {"path": "bad.txt"}).error_code == "invalid_utf8"


def test_long_log_continuation_and_limits(tmp_path: Path) -> None:
    executor, _, evidence, budget, project, _ = make_tool_executor(tmp_path)
    lines = [f"line {index}: filler" for index in range(1, 301)]
    lines[249] = "line 250: ERROR key marker"
    (project / "logs" / "long.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    first = _execute(executor, "read_file", {"path": "logs/long.log", "start_line": 1, "max_lines": 120})
    assert first.status.value == "ok"
    assert first.truncated is True
    assert first.continuation["start_line"] == 121
    assert first.continuation["tool"] == "read_evidence"
    assert first.continuation["cursor"]
    second = _execute(
        executor,
        "read_file",
        {"path": "logs/long.log", "start_line": 241, "max_lines": 120},
        call_id="call-2",
    )
    joined = "\n".join(line["text"] for line in second.data["lines"])
    assert "ERROR key marker" in joined
    assert second.continuation is None
    assert len(evidence.all()) == 2
    assert budget.state.tool_attempts_used == 2

    (project / "logs" / "many.log").write_text("\n".join(["needle"] * 10), encoding="utf-8")
    search = _execute(
        executor,
        "search_workspace",
        {"query": "needle", "paths": ["logs/many.log"], "limit": 2},
        call_id="call-3",
    )
    assert search.status.value == "ok"
    assert len(search.data["matches"]) == 2
    assert search.truncated is True
    assert search.continuation["reason"] == "limit_reached"

    (project / "logs" / "single.log").write_text("x" * 20_000, encoding="utf-8")
    long_line = _execute(
        executor,
        "read_file",
        {"path": "logs/single.log", "start_line": 1, "max_lines": 1},
        call_id="call-4",
    )
    assert long_line.status.value == "ok"
    assert long_line.data["line_truncated"] is True
    assert long_line.truncated is True
    assert long_line.continuation["reason"] == "line_too_long"


def test_migration_missing_file_and_non_git(tmp_path: Path) -> None:
    old_root = write_sample_project(tmp_path / "old-project")
    shutil.copytree(old_root, tmp_path / "new-project")
    old_root.joinpath("data/checks.json").write_text('{"OLD_ROOT_MARKER": true}', encoding="utf-8")
    new_root = tmp_path / "new-project"
    new_root.joinpath("data/checks.json").write_text('{"NEW_ROOT_MARKER": true}', encoding="utf-8")

    from agent_triage.budget import Budget
    from agent_triage.store import LessonStore
    from agent_triage.tools import ToolExecutor, Workspace

    output = tmp_path / "runs"
    output.mkdir()
    executor = ToolExecutor(
        workspace=Workspace(new_root),
        project_id="sample-project",
        budget=Budget(BudgetLimits()),
        lesson_store=LessonStore(output),
        run_id="run-20261024-200000-abcdef",
    )
    read = _execute(executor, "read_file", {"path": "data/checks.json"})
    assert "NEW_ROOT_MARKER" in read.data["lines"][0]["text"]
    assert "OLD_ROOT_MARKER" not in json.dumps(read.data)

    (new_root / "data/checks.json").unlink()
    missing = _execute(executor, "read_file", {"path": "data/checks.json"}, call_id="call-2")
    assert missing.error_code == "missing"
    assert old_root.joinpath("data/checks.json").exists()

    git = _execute(executor, "get_git_context", {"paths": ["src/check_reader.py"]}, call_id="call-3")
    assert git.status.value == "ok"
    assert git.data["commit"] is None
    assert git.data["files"][0]["exists"] is True
    assert len(git.data["files"][0]["sha256"]) == 64


def test_git_context_commit_and_dirty_state(tmp_path: Path) -> None:
    project = write_sample_project(tmp_path / "git-project")
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "fixture",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_NAME": "fixture",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
    }
    subprocess.run(["git", "init", "-q"], cwd=project, check=True, env=env)
    subprocess.run(["git", "add", "."], cwd=project, check=True, env=env)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=project, check=True, env=env)
    from agent_triage.budget import Budget
    from agent_triage.store import LessonStore
    from agent_triage.tools import ToolExecutor, Workspace

    output = tmp_path / "runs"
    output.mkdir()
    executor = ToolExecutor(
        workspace=Workspace(project),
        project_id="sample-project",
        budget=Budget(BudgetLimits()),
        lesson_store=LessonStore(output),
        run_id="run-20261024-200000-abcdef",
    )
    clean = _execute(executor, "get_git_context", {"paths": ["src/check_reader.py"]})
    assert clean.data["commit"] and len(clean.data["commit"]) == 40
    assert clean.data["dirty"] is False
    project.joinpath("src/check_reader.py").write_text("# changed\n", encoding="utf-8")
    dirty = _execute(executor, "get_git_context", {"paths": ["src/check_reader.py"]}, call_id="call-2")
    assert dirty.data["dirty"] is True
    assert dirty.data["files"][0]["sha256"] == hashlib.sha256(
        project.joinpath("src/check_reader.py").read_bytes()
    ).hexdigest()

def test_long_line_after_selected_lines_is_reported_not_silently_skipped(tmp_path: Path) -> None:
    executor, _, _, _, project, _ = make_tool_executor(tmp_path)
    (project / "logs" / "mixed.log").write_text("short line\n" + "x" * 20_000 + "\ntail\n", encoding="utf-8")
    result = _execute(executor, "read_file", {"path": "logs/mixed.log", "start_line": 1, "max_lines": 10})
    assert result.status.value == "ok"
    assert result.data["skipped_long_line"] is True
    assert result.data["line_truncated"] is False
    assert result.truncated is True
    assert result.continuation["reason"] == "line_too_long"
    assert result.continuation["next_line"] == 2
    assert [line["text"] for line in result.data["lines"]] == ["short line"]


def test_tool_timeout_is_bounded_and_reported(tmp_path: Path, monkeypatch) -> None:
    import time

    from agent_triage.contracts import ToolResult, ToolStatus

    executor, _, _, _, _, _ = make_tool_executor(tmp_path)

    def slow_dispatch(tool_name, parsed):
        time.sleep(0.5)
        return ToolResult(status=ToolStatus.OK, duration_ms=500)

    monkeypatch.setattr(executor, "_dispatch", slow_dispatch)
    monkeypatch.setattr(executor.budget, "per_tool_timeout", lambda: 0.1)
    started = time.monotonic()
    result = _execute(executor, "read_file", {"path": "logs/failure.log", "start_line": 1})
    elapsed = time.monotonic() - started
    assert result.error_code == "tool_timeout"
    assert result.retryable is True
    assert elapsed < 0.45
