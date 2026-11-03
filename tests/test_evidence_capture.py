"""Snapshot/continuation boundaries, using the production tool executor."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from agent_triage.contracts import BudgetLimits, EvidenceKind
from agent_triage.errors import StoreError
from agent_triage.store import EvidenceStore
from agent_triage.tools import IssueReadOutcome, TOOL_RESULT_MAX_BYTES, build_evidence
from helpers import make_tool_executor


def _read(executor, evidence_id, *, cursor=None, max_bytes=8192, call_id="page"):
    arguments = {"evidence_id": evidence_id, "max_bytes": max_bytes}
    if cursor is not None:
        arguments["cursor"] = cursor
    return executor.execute("read_evidence", arguments, call_id=call_id, turn=1)


def _text(result) -> str:
    return "".join(window["text"] for window in result.data["evidence_windows"])


def test_long_chinese_single_line_pages_are_contiguous_and_budgeted(tmp_path: Path) -> None:
    executor, store, evidence, budget, project, _ = make_tool_executor(tmp_path, limits=BudgetLimits(max_tool_attempts=80))
    text = "启动记录，" * 1300 + "MID_EXCEPTION：版本二不允许重试。" + "完成信息。" * 900
    (project / "logs/single.log").write_text(text)
    initial = executor.execute("read_file", {"path": "logs/single.log"}, call_id="initial", turn=0)
    root_id = initial.evidence_refs[0]
    root = evidence.require(root_id)
    assert evidence.capture_text(root_id) == text
    assert root.capture_scope == "file" and root.complete_within_scope is True
    assert initial.data["line_truncated"] is True
    assert len(initial.model_dump_json().encode()) <= TOOL_RESULT_MAX_BYTES
    cursor = None
    chunks = []
    positions = []
    before = budget.state.tool_attempts_used
    for n in range(70):
        page = _read(executor, root_id, cursor=cursor, max_bytes=1200, call_id=f"page-{n}")
        assert page.status.value == "ok", page.model_dump()
        assert len(page.model_dump_json().encode()) <= TOOL_RESULT_MAX_BYTES
        chunk = _text(page)
        assert chunk and len(chunk.encode()) <= 1200
        positions.append((page.data["start_byte"], page.data["end_byte"]))
        chunks.append(chunk)
        cursor = page.data["next_cursor"]
        if cursor is None:
            break
    assert "".join(chunks) == text
    assert positions[0][0] == 0 and positions[-1][1] == len(text.encode())
    assert all(a[1] == b[0] for a, b in zip(positions, positions[1:]))
    assert budget.state.tool_attempts_used == before + len(chunks)
    assert sum(t.tool_name == "read_evidence" for t in store.read_trace()) == len(chunks)


def test_repeated_page_deduplicates_and_source_change_keeps_old_snapshot(tmp_path: Path) -> None:
    executor, store, evidence, budget, project, _ = make_tool_executor(tmp_path)
    path = project / "logs/version.log"
    original = "first version\n" * 100
    path.write_text(original)
    initial = executor.execute("read_file", {"path": "logs/version.log", "max_lines": 120}, call_id="read", turn=0)
    root_id = initial.evidence_refs[0]
    first = _read(executor, root_id, max_bytes=100, call_id="page-1")
    count = len(evidence.ids())
    repeated = _read(executor, root_id, max_bytes=100, call_id="page-2")
    assert first.evidence_refs == repeated.evidence_refs
    assert len(evidence.ids()) == count
    next_cursor = first.data["next_cursor"]
    path.write_text("NEW_SOURCE_CANARY\n")
    historical = _read(executor, root_id, cursor=next_cursor, max_bytes=100, call_id="page-3")
    assert historical.status.value == "ok"
    assert historical.data["stale"] is True
    assert "NEW_SOURCE_CANARY" not in _text(historical)
    assert _text(historical).encode() == original.encode()[100:200]
    assert evidence.require(historical.evidence_refs[0]).stale
    assert evidence.require(root_id).stale
    # A repeated previously-current page cannot retain a false current flag.
    old_page = _read(executor, root_id, max_bytes=100, call_id="page-4")
    assert evidence.require(old_page.evidence_refs[0]).stale


def test_initial_tail_exposes_a_cursor_for_the_captured_prefix(tmp_path: Path) -> None:
    executor, _, evidence, _, project, _ = make_tool_executor(tmp_path)
    original = "prefix version 1\ncondition: not for production\nlast line\n"
    (project / "logs/tail.log").write_text(original)
    initial = executor.execute("read_file", {"path": "logs/tail.log", "start_line": 3}, call_id="tail", turn=0)
    root_id = initial.evidence_refs[0]
    assert _text(initial) == "last line\n"
    prefix = _read(executor, root_id, cursor=initial.data["snapshot_start_cursor"], max_bytes=20)
    assert prefix.status.value == "ok" and prefix.data["start_byte"] == 0
    assert _text(prefix) == original[:20]
    assert evidence.require(prefix.evidence_refs[0]).snapshot_sha256 == evidence.require(root_id).snapshot_sha256


@pytest.mark.parametrize("mutation", ["missing", "corrupt"])
def test_missing_or_corrupt_object_never_falls_back_to_live_source(tmp_path: Path, mutation: str) -> None:
    executor, store, evidence, _, project, _ = make_tool_executor(tmp_path)
    initial = executor.execute("read_file", {"path": "logs/failure.log"}, call_id="read", turn=0)
    root = evidence.require(initial.evidence_refs[0])
    obj = store.dir / "evidence/objects" / f"{root.snapshot_sha256}.txt"
    if mutation == "missing":
        obj.unlink()
    else:
        raw = obj.read_bytes()
        obj.write_bytes(b"X" + raw[1:])
    (project / "logs/failure.log").write_text("DO_NOT_SUBSTITUTE_CURRENT_SOURCE\n")
    result = _read(executor, root.evidence_id)
    assert result.error_code == f"snapshot_{mutation}"
    assert not result.evidence_refs
    assert "DO_NOT_SUBSTITUTE" not in result.model_dump_json()


def test_cursor_rejects_other_run_object_version_and_invalid_byte_boundary(tmp_path: Path) -> None:
    executor, _, evidence, _, project, _ = make_tool_executor(tmp_path)
    (project / "logs/utf.log").write_text("中文版本" * 100)
    initial = executor.execute("read_file", {"path": "logs/utf.log"}, call_id="read", turn=0)
    root_id = initial.evidence_refs[0]
    first = _read(executor, root_id, max_bytes=12)
    cursor = first.data["next_cursor"]
    payload = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
    for key, value in [("run", "another-run"), ("source", "0" * 64), ("snapshot", "1" * 64),
                       ("offset", 1), ("offset", True), ("offset", 100000)]:
        changed = {**payload, key: value}
        bad = base64.urlsafe_b64encode(json.dumps(changed).encode()).decode()
        assert _read(executor, root_id, cursor=bad).error_code == "invalid_cursor"
    assert _read(executor, "ev-000000000000").error_code == "unknown_evidence"
    assert executor.execute("read_evidence", {"evidence_id": root_id, "path": "../outside"},
                            call_id="bad-path", turn=1).error_code == "invalid_arguments"


def test_search_excerpt_centers_late_hit_and_archives_only_matched_line(tmp_path: Path) -> None:
    executor, _, evidence, _, project, _ = make_tool_executor(tmp_path)
    line = "甲" * 4000 + "LateFailure" + "乙" * 1000
    (project / "logs/search.log").write_text("UNREAD_OTHER_LINE\n" + line + "\n")
    result = executor.execute("search_workspace", {"query": "LateFailure", "paths": ["logs/search.log"]},
                              call_id="search", turn=0)
    assert result.status.value == "ok"
    item = evidence.require(result.evidence_refs[0])
    assert "LateFailure" in item.snippet and item.snapshot_start_byte > 8192
    assert item.capture_scope == "search_match" and item.complete_within_scope is True
    assert evidence.capture_text(item.evidence_id) == line
    assert "UNREAD_OTHER_LINE" not in evidence.capture_text(item.evidence_id)
    assert any("other file content" in note for note in item.capture_notes)


def test_search_casefold_expansion_keeps_the_original_character_position(tmp_path: Path) -> None:
    executor, _, evidence, _, project, _ = make_tool_executor(tmp_path)
    original = "start " * 1000 + "Straße: original spelling; not normalized"
    (project / "logs/fold.log").write_text(original)
    result = executor.execute("search_workspace", {"query": "STRASSE", "paths": ["logs/fold.log"]},
                              call_id="fold", turn=0)
    assert result.status.value == "ok" and result.evidence_refs
    item = evidence.require(result.evidence_refs[0])
    assert "Straße" in item.snippet and "STRASSE" not in item.snippet
    assert item.snippet.encode() == original.encode()[item.snapshot_start_byte:item.snapshot_end_byte]


def test_archive_index_write_failure_leaves_no_recoverable_pointer(tmp_path: Path, monkeypatch) -> None:
    executor, store, evidence, _, _, _ = make_tool_executor(tmp_path)
    from agent_triage import store as store_module

    original = store_module.atomic_write_json
    def fail_index(path, payload):
        if path == evidence.path:
            raise StoreError("synthetic disk failure")
        return original(path, payload)
    monkeypatch.setattr(store_module, "atomic_write_json", fail_index)
    result = executor.execute("read_file", {"path": "logs/failure.log"}, call_id="read", turn=0)
    assert result.error_code == "archive_failed"
    assert not result.evidence_refs and not evidence.ids()
    assert "recoverable\":true" not in result.model_dump_json()
    assert EvidenceStore(store).all() == []


def test_legacy_snippet_is_readable_but_not_promoted_to_full_capture(tmp_path: Path) -> None:
    executor, _, evidence, _, project, _ = make_tool_executor(tmp_path)
    legacy = build_evidence(kind=EvidenceKind.LOCAL_FILE, project_id="sample-project",
                            path="logs/failure.log", line_start=1, line_end=1,
                            snippet="legacy captured prefix", truncated=True,
                            sha256=hashlib.sha256((project / "logs/failure.log").read_bytes()).hexdigest())
    evidence.add(legacy)
    result = _read(executor, legacy.evidence_id)
    assert result.status.value == "ok"
    assert _text(result) == legacy.snippet
    assert result.data["capture_scope"] == "snippet"
    assert result.data["complete_within_scope"] is False
    assert result.data["next_cursor"] is None


def test_github_partial_response_stays_partial_after_readback(tmp_path: Path) -> None:
    class Backend:
        available = True
        unavailable_reason = None
        def read_issue(self, issue_ref):
            return IssueReadOutcome(issue={"number": 1, "body": "a" * 15000 + "TAIL_MARKER"},
                                    comments=[{"body": "captured comment"}],
                                    url="https://github.com/example/fixture/issues/1", updated_at=None,
                                    comments_error="comments_page_unavailable")
    executor, _, evidence, _, _, _ = make_tool_executor(tmp_path, github=Backend())
    result = executor.execute("github_read_issue", {"issue_ref": "example/fixture#1"}, call_id="issue", turn=0)
    assert result.status.value == "ok"
    assert len(result.model_dump_json().encode()) <= TOOL_RESULT_MAX_BYTES
    root = evidence.require(result.evidence_refs[0])
    assert root.complete_within_scope is False
    assert "TAIL_MARKER" in evidence.capture_text(root.evidence_id)
    assert "comments_page_unavailable" in root.capture_notes
    page = _read(executor, root.evidence_id, max_bytes=500)
    assert page.data["complete_within_scope"] is False
    assert "comments_page_unavailable" in page.data["capture_notes"]


def test_runner_counts_readback_and_stops_repeated_page_without_fake_progress(tmp_path: Path) -> None:
    from helpers import SAMPLE_FILES, base_analysis, final_step, make_scenario, read_json, run_replay, tool_step

    text = SAMPLE_FILES["logs/failure.log"]
    root_id = build_evidence(kind=EvidenceKind.LOCAL_FILE, project_id="sample-project",
                             path="logs/failure.log", line_start=1, line_end=len(text.splitlines()),
                             snippet=text, sha256=hashlib.sha256(text.encode()).hexdigest()).evidence_id
    scenario = make_scenario([
        tool_step(("read_file", {"path": "logs/failure.log"})),
        tool_step(("read_evidence", {"evidence_id": root_id, "max_bytes": 100})),
        tool_step(("read_evidence", {"evidence_id": root_id, "max_bytes": 100})),
        final_step(base_analysis()),
    ])
    result, _, output = run_replay(tmp_path, scenario=scenario)
    run_dir = output / result.run_id
    assert result.budget_used["tool_attempts_used"] == 3
    assert result.analysis.stop_reason.value == "insufficient_evidence"
    index = read_json(run_dir / "evidence/index.json")["evidence"]
    assert len(index) == 2
    investigation = read_json(run_dir / "investigation.json")
    assert investigation["attempt_ledger"][-1]["new_information"] is False
    assert investigation["attempt_ledger"][-1]["call_id"]
    context = read_json(run_dir / "context.json")
    assert len(context["state_projection"]["no_progress_attempts"]) == 1
    assert "成功返回但无新证据" in (run_dir / "handoff.md").read_text()
    traces = [json.loads(line) for line in (run_dir / "trace.jsonl").read_text().splitlines()]
    assert sum(record["tool_name"] == "read_evidence" for record in traces if record["kind"] == "tool") == 2
