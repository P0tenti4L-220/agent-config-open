from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from agent_triage.errors import ValidationFailure
from agent_triage.livecheck import verify_issue_chain
from helpers import read_json, run_replay


URL = "https://github.com/example/repo/issues/7"
MARKER = "FROZEN-ISSUE-MARKER-7"


def _prepare(tmp_path: Path) -> Path:
    result, _, output = run_replay(tmp_path)
    source = output / result.run_id
    target = tmp_path / "case" / "runs" / result.run_id
    target.parent.mkdir(parents=True)
    shutil.copytree(source, target)

    issue_id = "ev-000000000007"
    search_id = "ev-000000000008"
    issue_snippet = f"{MARKER}\nSynthetic issue body."
    search_snippet = '{"number": 7, "title": "Synthetic issue", "url": "%s", "body": "%s"}' % (URL, MARKER)
    evidence_path = target / "evidence" / "index.json"
    index = read_json(evidence_path)
    for evidence_id, snippet in ((issue_id, issue_snippet), (search_id, search_snippet)):
        index["evidence"].append(
            {
                "schema_version": 1,
                "evidence_id": evidence_id,
                "kind": "issue",
                "project_id": "sample-project",
                "url": URL,
                "sha256": hashlib.sha256(snippet.encode("utf-8")).hexdigest(),
                "snippet": snippet,
                "captured_at": "2026-10-25T12:00:00+00:00",
                "stale": False,
            }
        )
    evidence_path.write_text(json.dumps(index), encoding="utf-8")

    fragment = issue_snippet
    digest = hashlib.sha256(fragment.encode("utf-8")).hexdigest()
    request_view = {
        "schema_version": 2,
        "request_id": "req-000099",
        "run_id": result.run_id,
        "turn": 1,
        "provider": "replay",
        "status": "response_received",
        "payload_sha256": "d" * 64,
        "payload_chars": len(fragment) + 100,
        "payload_complete": True,
        "system": "synthetic request",
        "messages": [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "call-read-1"}]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "call-read-1", "content": fragment}],
            },
        ],
        "tool_call_ids": ["call-read-1"],
        "tool_result_ids": ["call-read-1"],
        "evidence_views": [
            {
                "evidence_id": issue_id,
                "source_sha256": hashlib.sha256(issue_snippet.encode("utf-8")).hexdigest(),
                "source": "tool_result",
                "mode": "full",
                "visible_chars": len(fragment),
                "visible_sha256": digest,
                "fragment": fragment,
            }
        ],
        "created_at": "2026-10-25T12:00:00+00:00",
        "responded_at": "2026-10-25T12:00:01+00:00",
    }
    payload_text = json.dumps(
        {"system": request_view["system"], "messages": request_view["messages"]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    request_view["payload_sha256"] = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
    request_view["payload_chars"] = len(payload_text)
    views = target / "request-views"
    views.mkdir(parents=True, exist_ok=True)
    (views / "req-000099.json").write_text(json.dumps(request_view), encoding="utf-8")

    display_path = target / "display.json"
    display = read_json(display_path)
    display["displayed"].append(
        {
            "schema_version": 2,
            "evidence_id": issue_id,
            "turn": 1,
            "source": "tool_result",
            "mode": "full",
            "visible_chars": len(fragment),
            "visible_sha256": digest,
            "request_id": "req-000099",
            "source_sha256": hashlib.sha256(issue_snippet.encode("utf-8")).hexdigest(),
        }
    )
    display_path.write_text(json.dumps(display), encoding="utf-8")

    analysis_path = target / "analysis.json"
    analysis = read_json(analysis_path)
    analysis["facts"].append(
        {
            "claim_id": "claim-fact-900",
            "text": "The designated issue records the same failure.",
            "evidence_ids": [issue_id],
        }
    )
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")

    trace_path = target / "trace.jsonl"
    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    template = dict(records[0])
    for name, call_id, refs in (
        ("github_search_issues", "call-search-1", [search_id]),
        ("github_read_issue", "call-read-1", [issue_id]),
    ):
        record = {
            **template,
            "kind": "tool",
            "tool_name": name,
            "call_id": call_id,
            "status": "ok",
            "error_code": None,
            "arguments": {"issue_ref": "example/repo#7"} if name == "github_read_issue" else {"query": MARKER},
            "evidence_refs": refs,
        }
        records.append(record)
    trace_path.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")

    report_path = target / "report.md"
    report_path.write_text(report_path.read_text(encoding="utf-8") + f"\n- {issue_id} {URL}\n", encoding="utf-8")
    return target


def test_issue_chain_accepts_complete_synthetic_chain(tmp_path: Path) -> None:
    run_dir = _prepare(tmp_path)
    summary = verify_issue_chain(run_dir, issue_ref="example/repo#7", marker=MARKER)
    assert summary["read_call_id"] == "call-read-1"
    assert summary["evidence_id"] == "ev-000000000007"
    assert summary["cited_claims"]


def _mutate(run_dir: Path, kind: str) -> None:
    if kind == "no-read":
        trace = [json.loads(line) for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
        trace = [item for item in trace if item.get("tool_name") != "github_read_issue"]
        (run_dir / "trace.jsonl").write_text("\n".join(json.dumps(item) for item in trace) + "\n", encoding="utf-8")
    elif kind == "wrong-issue":
        path = run_dir / "evidence" / "index.json"
        payload = read_json(path)
        for item in payload["evidence"]:
            if item["evidence_id"] == "ev-000000000007":
                item["url"] = "https://github.com/example/repo/issues/8"
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif kind == "not-displayed":
        path = run_dir / "display.json"
        payload = read_json(path)
        payload["displayed"] = [item for item in payload["displayed"] if item["evidence_id"] != "ev-000000000007"]
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif kind == "not-cited":
        path = run_dir / "analysis.json"
        payload = read_json(path)
        payload["facts"] = [item for item in payload["facts"] if item.get("claim_id") != "claim-fact-900"]
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif kind == "only-search":
        # A run with a search and a read trace but no evidence/display/claim.
        shutil.rmtree(run_dir / "evidence")
        (run_dir / "display.json").unlink()
    elif kind == "stale":
        path = run_dir / "evidence" / "index.json"
        payload = read_json(path)
        for item in payload["evidence"]:
            if item["evidence_id"] == "ev-000000000007":
                item["stale"] = True
                item["stale_reason"] = "source changed"
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif kind == "no-marker":
        path = run_dir / "evidence" / "index.json"
        payload = read_json(path)
        for item in payload["evidence"]:
            if item["evidence_id"] == "ev-000000000007":
                item["snippet"] = "unrelated body"
                item["sha256"] = hashlib.sha256(b"unrelated body").hexdigest()
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif kind == "wrong-read-issue":
        path = run_dir / "trace.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for item in records:
            if item.get("tool_name") == "github_read_issue":
                item["arguments"] = {"issue_ref": "example/repo#999"}
        path.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
    elif kind == "no-evidence-ref":
        path = run_dir / "trace.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for item in records:
            if item.get("tool_name") == "github_read_issue":
                item["evidence_refs"] = []
        path.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
    elif kind == "display-hash":
        path = run_dir / "display.json"
        payload = read_json(path)
        for item in payload["displayed"]:
            if item["evidence_id"] == "ev-000000000007":
                item["visible_sha256"] = "c" * 64
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif kind == "no-request-view":
        shutil.rmtree(run_dir / "request-views")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("no-read", "only searched"),
        ("wrong-issue", "designated issue"),
        ("not-displayed", "never displayed"),
        ("not-cited", "no claim cites"),
        ("only-search", "cannot read"),
        ("stale", "stale"),
        ("no-marker", "frozen marker"),
        ("wrong-read-issue", "different issue_ref"),
        ("no-evidence-ref", "does not reference"),
        ("display-hash", "display proof"),
        ("no-request-view", "request views"),
    ],
)
def test_issue_chain_rejects_missing_links(tmp_path: Path, mutation: str, message: str) -> None:
    run_dir = _prepare(tmp_path)
    _mutate(run_dir, mutation)
    with pytest.raises(ValidationFailure, match=message):
        verify_issue_chain(run_dir, issue_ref="example/repo#7", marker=MARKER)


def test_issue_chain_rejects_read_for_another_issue(tmp_path: Path) -> None:
    run_dir = _prepare(tmp_path)
    _mutate(run_dir, "wrong-read-issue")
    with pytest.raises(ValidationFailure, match="different issue_ref"):
        verify_issue_chain(run_dir, issue_ref="example/repo#7", marker=MARKER)
