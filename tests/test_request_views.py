"""Request-view display provenance (F01).

The regression these tests protect is concrete: a tool may store evidence while
the next context build drops the whole call/result pair.  In that case the model
never saw the body and ``claim-display`` must not be reported as pass.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from agent_triage.config import Config
from agent_triage.contracts import DisplayedEvidence
from agent_triage.models import ReplayModel
from agent_triage.request_views import load_request_views, verify_display_records
from helpers import base_analysis, final_step, make_scenario, read_json, run_replay, tool_step


MARKER = "synthetic-check-1"


def _spy(seen: list[str]):
    original = ReplayModel.next

    def inner(self, context, specs, remaining):
        seen.append(context.system + json.dumps(context.messages, ensure_ascii=False))
        return original(self, context, specs, remaining)

    return inner


def _checks(run_dir: Path) -> dict[str, dict]:
    return {item["check_id"]: item for item in read_json(run_dir / "checks.json")["checks"]}


def test_claim_display_fails_when_body_never_enters_provider_request(tmp_path: Path) -> None:
    seen: list[str] = []
    config = Config(context={"current_evidence_tokens": 0, "working_state_tokens": 0})
    scenario = make_scenario(
        [
            tool_step(("read_file", {"path": "data/checks.json", "start_line": 1, "max_lines": 20})),
            final_step(
                base_analysis(
                    facts=[
                        {
                            "text": "The input has a synthetic check.",
                            "evidence_ids": ["@path:data/checks.json"],
                        }
                    ],
                    hypotheses=[],
                )
            ),
        ]
    )
    with patch.object(ReplayModel, "next", _spy(seen)):
        result, _, output = run_replay(tmp_path, scenario=scenario, config=config)
    run_dir = output / result.run_id
    assert all(MARKER not in request for request in seen)
    checks = _checks(run_dir)
    assert checks["claim-display"]["status"] == "fail"
    assert "never displayed" in checks["claim-display"]["scope"]

    display = read_json(run_dir / "display.json")
    assert display["schema_version"] == 2
    assert all(item["schema_version"] == 2 for item in display["displayed"])
    assert all(item["evidence_id"] for item in display["displayed"])


def test_display_records_are_bound_to_received_request_views(tmp_path: Path) -> None:
    result, _, output = run_replay(tmp_path)
    run_dir = output / result.run_id
    display = read_json(run_dir / "display.json")
    records = [DisplayedEvidence.model_validate(item) for item in display["displayed"]]
    views = load_request_views(run_dir)
    assert records
    assert views
    assert {record.request_id for record in records} <= {view.request_id for view in views}
    problems, mode = verify_display_records(records, views, _evidence(run_dir))
    assert problems == []
    assert mode == "request_view"
    checks = _checks(run_dir)
    assert checks["claim-display"]["status"] == "pass"

    # The stored fragment is enough to recompute the recorded hash; the request
    # view also carries the tool call ids used by the live chain check.
    for view in views:
        if view.status != "response_received":
            continue
        for item in view.evidence_views:
            assert hashlib.sha256(item.fragment.encode("utf-8")).hexdigest() == item.visible_sha256
    assert any(view.tool_result_ids for view in views)


def test_legacy_display_v1_is_not_request_view_proof(tmp_path: Path) -> None:
    result, _, output = run_replay(tmp_path)
    run_dir = output / result.run_id
    display = read_json(run_dir / "display.json")
    for item in display["displayed"]:
        item.pop("request_id", None)
        item.pop("source_sha256", None)
        item["schema_version"] = 1
    display["schema_version"] = 1
    (run_dir / "display.json").write_text(json.dumps(display), encoding="utf-8")

    from agent_triage.cli import main

    assert main(["validate", "--run-id", result.run_id, "--output", str(output)]) == 0
    checks = _checks(run_dir)
    assert checks["claim-display"]["status"] == "not_run"
    assert "legacy v1" in checks["claim-display"]["scope"]


def test_earlier_request_view_counts_after_the_pair_is_compacted(tmp_path: Path) -> None:
    config = Config(
        context={
            "current_evidence_tokens": 0,
            "working_state_tokens": 240,
        }
    )
    scenario = make_scenario(
        [
            tool_step(("read_file", {"path": "logs/failure.log", "start_line": 1, "max_lines": 20})),
            tool_step(("read_file", {"path": "src/check_reader.py", "start_line": 1, "max_lines": 20})),
            final_step(
                base_analysis(
                    facts=[
                        {
                            "text": "The failure log contains the KeyError.",
                            "evidence_ids": ["@path:logs/failure.log"],
                        }
                    ],
                    hypotheses=[],
                )
            ),
        ]
    )
    result, _, output = run_replay(tmp_path, scenario=scenario, config=config)
    run_dir = output / result.run_id
    checks = _checks(run_dir)
    assert checks["claim-display"]["status"] == "pass"
    views = load_request_views(run_dir)
    first = next(
        view
        for view in views
        if any(item.evidence_id for item in view.evidence_views)
    )
    assert first.evidence_views
    # The final request is allowed to omit the old pair; the earlier confirmed
    # request remains valid display proof for the final citation.
    assert [view.turn for view in views] == sorted(view.turn for view in views)


def _evidence(run_dir: Path):
    from agent_triage.contracts import Evidence

    payload = read_json(run_dir / "evidence" / "index.json")
    return [Evidence.model_validate(item) for item in payload["evidence"]]

def test_incomplete_payload_snapshot_cannot_prove_display(tmp_path: Path) -> None:
    result, _, output = run_replay(tmp_path)
    run_dir = output / result.run_id
    display = read_json(run_dir / "display.json")
    record = next(item for item in display["displayed"] if item.get("request_id"))
    view_path = run_dir / "request-views" / f"{record['request_id']}.json"
    view = read_json(view_path)
    view["payload_complete"] = False
    view_path.write_text(json.dumps(view), encoding="utf-8")
    problems, mode = verify_display_records(
        [DisplayedEvidence.model_validate(item) for item in display["displayed"]],
        load_request_views(run_dir),
        _evidence(run_dir),
    )
    assert mode == "request_view"
    assert any("incomplete payload" in problem for problem in problems)


def test_display_source_must_match_request_view(tmp_path: Path) -> None:
    result, _, output = run_replay(tmp_path)
    run_dir = output / result.run_id
    display = read_json(run_dir / "display.json")
    record = next(item for item in display["displayed"] if item.get("request_id"))
    record["source"] = "context" if record.get("source") != "context" else "tool_result"
    problems, _ = verify_display_records(
        [DisplayedEvidence.model_validate(item) for item in display["displayed"]],
        load_request_views(run_dir),
        _evidence(run_dir),
    )
    assert any("does not contain the recorded view" in problem for problem in problems)


def test_request_ids_do_not_collide_after_a_gap(tmp_path: Path) -> None:
    from agent_triage.request_views import RequestViewStore

    store = RequestViewStore(tmp_path)
    store.directory.mkdir()
    (store.directory / "req-000001.json").write_text("{}", encoding="utf-8")
    (store.directory / "req-000003.json").write_text("{}", encoding="utf-8")
    view = store.begin(
        run_id="run-test",
        turn=1,
        provider="replay",
        system="system",
        messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        evidence=[],
    )
    assert view.request_id == "req-000004"
    assert (store.directory / "req-000004.json").is_file()
