from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_triage import cli
from agent_triage.contracts import Analysis, DisplayedEvidence
from agent_triage.errors import ValidationFailure
from agent_triage.review import (
    ExternalReview,
    analysis_content_hash,
    evidence_snapshot_hash,
    import_external_review,
    load_external_review,
    sanitize_model_review,
    view_ref,
)
from agent_triage.store import RunStore
from helpers import (
    base_analysis,
    final_step,
    make_scenario,
    read_json,
    run_replay,
    tool_step,
)


def _reviewed_analysis() -> dict:
    payload = base_analysis()
    payload["facts"][0]["support_review"] = {
        "status": "pass",
        "reviewer": "owner",
        "method": "manual",
        "scope": "looks right",
        "reviewed_at": "2026-10-25T12:00:00+00:00",
    }
    payload["hypotheses"][0]["support_review"] = {
        "status": "pass",
        "reviewer": "owner",
        "method": "manual",
        "scope": "looks right",
        "reviewed_at": "2026-10-25T12:00:00+00:00",
    }
    return payload


def test_model_cannot_self_authorize_support_review(tmp_path: Path) -> None:
    """Before the boundary fix, this replay put `claim-support=pass` into checks.

    The replay now keeps the diagnostic content but drops the self-declared
    review before the analysis is stored.  This is a regression test, not a
    real human review.
    """
    scenario = make_scenario(
        [
            tool_step(("read_file", {"path": "logs/failure.log", "start_line": 1, "max_lines": 20})),
            tool_step(("read_file", {"path": "src/check_reader.py", "start_line": 1, "max_lines": 20})),
            tool_step(("read_file", {"path": "data/checks.json", "start_line": 1, "max_lines": 20})),
            final_step(_reviewed_analysis()),
        ]
    )
    result, _, output = run_replay(tmp_path, scenario=scenario)
    run_dir = output / result.run_id
    analysis = read_json(run_dir / "analysis.json")
    for claim in [*analysis["facts"], *analysis["hypotheses"]]:
        assert claim["support_review"]["status"] == "not_run"
        assert "model-supplied" in claim["support_review"]["note"]
        assert claim["support_review"]["reviewer"] is None
    checks = {item["check_id"]: item for item in read_json(run_dir / "checks.json")["checks"]}
    assert checks["claim-support"]["status"] == "not_run"
    status = read_json(run_dir / "review-status.json")
    assert status["overall"] == "not_run"
    assert status["reviews"] == []
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "模型边界降级" in report

    # A second guard: sanitizing twice is idempotent and preserves content.
    parsed = Analysis.model_validate(analysis)
    clean, changed = sanitize_model_review(parsed)
    assert changed == 0
    assert clean.summary == parsed.summary


def _run_and_load(tmp_path: Path):
    result, _, output = run_replay(tmp_path)
    store = RunStore.load(output, result.run_id)
    analysis = Analysis.model_validate(store.read_json("analysis.json"))
    display_raw = store.read_json("display.json")["displayed"]
    displayed = [DisplayedEvidence.model_validate(item) for item in display_raw]
    evidence = store.read_json("evidence/index.json")["evidence"]
    return result, output, store, analysis, displayed, evidence


def _review_payload(store, analysis, displayed, evidence, *, claims=None, review_id="review-test-001") -> dict:
    from agent_triage.contracts import Evidence

    evidence_models = [Evidence.model_validate(item) for item in evidence]
    fact_view = next(
        view_ref(item) for item in displayed if item.evidence_id == analysis.facts[0].evidence_ids[0]
    )
    default_claims = [
        {
            "claim_id": analysis.facts[0].claim_id,
            "verdict": "pass",
            "rationale": "The quoted traceback line shows the observed KeyError.",
            "evidence_refs": [{"evidence_id": analysis.facts[0].evidence_ids[0], "view_ref": fact_view}],
        },
        {
            "claim_id": analysis.hypotheses[0].claim_id,
            "verdict": "pass",
            "rationale": "The hypothesis is plausible and still labelled as unproven.",
            "evidence_refs": [],
        },
    ]
    return {
        "schema_version": 1,
        "review_id": review_id,
        "run_id": store.run_id,
        "analysis_content_hash": analysis_content_hash(analysis),
        "evidence_snapshot_hash": evidence_snapshot_hash(evidence_models, displayed),
        "reviewer": "owner",
        "method": "human",
        "reviewed_at": "2026-10-25T12:00:00+00:00",
        "rubric_version": "mvp-review-v1",
        "claims": claims if claims is not None else default_claims,
        "next_action": {"verdict": "pass", "rationale": "Run the suggested adapter check."},
        "unresolved_questions": [],
    }


def test_external_review_binding_and_staleness(tmp_path: Path) -> None:
    result, output, store, analysis, displayed, evidence = _run_and_load(tmp_path)
    payload = _review_payload(store, analysis, displayed, evidence)
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    original_analysis_bytes = (store.dir / "analysis.json").read_bytes()

    assert cli.main(
        ["record-review", "--run-id", store.run_id, "--review", str(review_path), "--output", str(output)]
    ) == 0
    view = read_json(store.dir / "review-status.json")
    assert view["overall"] == "pass"
    assert view["claims"][analysis.facts[0].claim_id]["status"] == "pass"
    assert view["claims"][analysis.facts[0].claim_id]["reviewer"] == "owner"
    # Importing the review does not rewrite the diagnostic content.
    assert (store.dir / "analysis.json").read_bytes() == original_analysis_bytes

    assert cli.main(["validate", "--run-id", store.run_id, "--output", str(output)]) == 0
    checks = {item["check_id"]: item for item in read_json(store.dir / "checks.json")["checks"]}
    assert checks["claim-support"]["status"] == "pass"

    # An unknown claim, a duplicate claim, and an expired hash are rejected.
    unknown = json.loads(json.dumps(payload))
    unknown["claims"][0]["claim_id"] = "claim-fact-999"
    unknown_path = tmp_path / "unknown.json"
    unknown_path.write_text(json.dumps(unknown), encoding="utf-8")
    with pytest.raises(ValidationFailure, match="unknown claim"):
        import_external_review(
            store=store,
            analysis=analysis,
            evidence=[__import__("agent_triage.contracts", fromlist=["Evidence"]).Evidence.model_validate(item) for item in evidence],
            displayed=displayed,
            review=load_external_review(unknown_path),
            source_path=unknown_path,
        )

    duplicate = json.loads(json.dumps(payload))
    duplicate["claims"].append(dict(duplicate["claims"][0]))
    with pytest.raises(ValidationError):
        ExternalReview.model_validate(duplicate)

    expired = json.loads(json.dumps(payload))
    expired["analysis_content_hash"] = "0" * 64
    expired_path = tmp_path / "expired.json"
    expired_path.write_text(json.dumps(expired), encoding="utf-8")
    with pytest.raises(ValidationFailure, match="different analysis content"):
        import_external_review(
            store=store,
            analysis=analysis,
            evidence=[__import__("agent_triage.contracts", fromlist=["Evidence"]).Evidence.model_validate(item) for item in evidence],
            displayed=displayed,
            review=load_external_review(expired_path),
            source_path=expired_path,
        )

    # Changing the analysis makes the imported review stale; the old record is
    # kept, but it no longer counts as current support.
    changed = analysis.model_copy(update={"summary": analysis.summary + " (updated)"})
    store.write_json("analysis.json", changed.model_dump(mode="json"))
    # validate is not a completion oracle: not_run stays non-blocking.
    assert cli.main(["validate", "--run-id", store.run_id, "--output", str(output)]) == 0
    view = read_json(store.dir / "review-status.json")
    assert view["overall"] == "not_run"
    assert payload["review_id"] in view["stale_review_ids"]
    checks = {item["check_id"]: item for item in read_json(store.dir / "checks.json")["checks"]}
    assert checks["claim-support"]["status"] == "not_run"

    # A hypothesis pass never resolves the run or proves the root cause.
    run_meta = read_json(store.dir / "run.json")
    assert run_meta["status"] != "resolved"


def test_partial_review_leaves_unlisted_claims_not_run(tmp_path: Path) -> None:
    result, output, store, analysis, displayed, evidence = _run_and_load(tmp_path)
    payload = _review_payload(store, analysis, displayed, evidence)
    payload["claims"] = payload["claims"][:1]  # only the fact is reviewed
    review_path = tmp_path / "partial.json"
    review_path.write_text(json.dumps(payload), encoding="utf-8")
    assert cli.main(
        ["record-review", "--run-id", store.run_id, "--review", str(review_path), "--output", str(output)]
    ) == 0
    view = read_json(store.dir / "review-status.json")
    assert view["overall"] == "not_run"
    assert view["claims"][analysis.hypotheses[0].claim_id]["status"] == "not_run"
    assert cli.main(["validate", "--run-id", store.run_id, "--output", str(output)]) == 0
    checks = {item["check_id"]: item for item in read_json(store.dir / "checks.json")["checks"]}
    assert checks["claim-support"]["status"] == "not_run"


def test_tool_evidence_context_links_are_reviewable(tmp_path: Path) -> None:
    scenario = make_scenario(
        [
            tool_step(("read_file", {"path": "../etc/passwd", "start_line": 1, "max_lines": 5})),
            tool_step(("read_file", {"path": ".env", "start_line": 1, "max_lines": 5})),
            tool_step(("read_file", {"path": "logs/failure.log", "start_line": 1, "max_lines": 20})),
            tool_step(("read_file", {"path": "src/check_reader.py", "start_line": 1, "max_lines": 20})),
            tool_step(("read_file", {"path": "data/checks.json", "start_line": 1, "max_lines": 20})),
            final_step(base_analysis()),
        ]
    )
    result, _, output = run_replay(tmp_path, scenario=scenario)
    run_dir = output / result.run_id
    trace = [json.loads(line) for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    tool_trace = [item for item in trace if item["kind"] == "tool"]
    assert any(item["error_code"] in {"path_escape", "invalid_arguments"} for item in tool_trace)
    assert any(item["error_code"] == "sensitive_path" for item in tool_trace)
    failed = next(item for item in tool_trace if item["error_code"] in {"path_escape", "invalid_arguments"})
    assert failed["call_id"]
    assert failed["attempt"] == 1
    # The failed attempt is retained and followed by successful reads.
    assert any(item["status"] == "ok" and item["evidence_refs"] for item in tool_trace)

    display = read_json(run_dir / "display.json")["displayed"]
    refs = {view_ref(DisplayedEvidence.model_validate(item)) for item in display}
    assert refs, "the model must have seen at least one bounded view"
    analysis = Analysis.model_validate(read_json(run_dir / "analysis.json"))
    cited = {evidence_id for fact in analysis.facts for evidence_id in fact.evidence_ids}
    displayed_ids = {item["evidence_id"] for item in display}
    assert cited <= displayed_ids
    for item in display:
        assert item["visible_sha256"]
        assert item["visible_chars"] >= 0

    # Context and handoff point back to the same evidence ids.
    packet = read_json(run_dir / "context.json")
    indexed = {item["evidence_id"] for item in packet["evidence_index"]}
    assert cited <= indexed
    handoff = (run_dir / "handoff.md").read_text(encoding="utf-8")
    for evidence_id in cited:
        assert evidence_id in handoff
    # No hidden reasoning or duplicated raw context is stored.
    serialized = json.dumps(trace, ensure_ascii=False).lower()
    assert "chain-of-thought" not in serialized and '"reasoning"' not in serialized


def test_legacy_self_declared_review_on_disk_is_not_trusted(tmp_path: Path) -> None:
    result, output, store, analysis, displayed, evidence = _run_and_load(tmp_path)
    legacy = analysis.model_copy(deep=True)
    from agent_triage.contracts import SupportReview

    for fact in legacy.facts:
        fact.support_review = SupportReview(
            status="pass",
            reviewer="owner",
            method="manual",
            scope="self-declared",
            reviewed_at="2026-10-25T12:00:00+00:00",
        )
    store.write_json("analysis.json", legacy.model_dump(mode="json"))
    assert cli.main(["validate", "--run-id", store.run_id, "--output", str(output)]) == 0
    checks = {item["check_id"]: item for item in read_json(store.dir / "checks.json")["checks"]}
    assert checks["claim-support"]["status"] == "not_run"
    assert "no current external review" in checks["claim-support"]["scope"]


def test_review_becomes_stale_when_evidence_changes(tmp_path: Path) -> None:
    result, output, store, analysis, displayed, evidence = _run_and_load(tmp_path)
    payload = _review_payload(store, analysis, displayed, evidence)
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(payload), encoding="utf-8")
    assert cli.main(
        ["record-review", "--run-id", store.run_id, "--review", str(review_path), "--output", str(output)]
    ) == 0
    evidence_path = store.dir / "evidence" / "index.json"
    index = read_json(evidence_path)
    changed = index["evidence"][0]
    changed["snippet"] += "\nchanged"
    raw = changed["snippet"].encode()
    # Keep the new capture internally consistent. Old request/display/review
    # records still bind the previous source and must not verify the new one.
    digest = __import__("hashlib").sha256(raw).hexdigest()
    changed.update(sha256=digest, snapshot_sha256=digest, snapshot_bytes=len(raw),
                   snapshot_start_byte=0, snapshot_end_byte=len(raw))
    (store.dir / "evidence/objects" / f"{digest}.txt").write_bytes(raw)
    evidence_path.write_text(json.dumps(index), encoding="utf-8")
    assert cli.main(["validate", "--run-id", store.run_id, "--output", str(output)]) == 1
    view = read_json(store.dir / "review-status.json")
    assert view["overall"] == "not_run"
    assert payload["review_id"] in view["stale_review_ids"]
    checks = {item["check_id"]: item for item in read_json(store.dir / "checks.json")["checks"]}
    assert checks["claim-support"]["status"] == "not_run"
    assert checks["claim-display"]["status"] == "fail"
