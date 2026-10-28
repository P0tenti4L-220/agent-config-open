from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_triage.acceptance import (
    CapabilityCatalog,
    load_capabilities,
    load_execution_results,
    validate_capabilities,
)
from agent_triage.eval import load_cases, run_eval
from agent_triage.store import atomic_write_json
from agent_triage.tools import TOOL_NAMES, tool_specs
from helpers import base_analysis, final_step, make_scenario, run_replay, tool_step


CAPABILITIES = Path("docs/agent-capabilities.json")


def test_capability_catalog_rejects_dangling_and_duplicate_checks(tmp_path: Path) -> None:
    cases = load_cases(Path("tests/cases"))
    known = {case.id for case in cases}
    catalog = load_capabilities(CAPABILITIES)
    assert validate_capabilities(catalog, known_check_ids=known) == []
    assert [item.id for item in catalog.capabilities] == ["F01", "F02", "F03", "F04", "F05", "F06"]

    dangling = catalog.model_dump(mode="json")
    dangling["capabilities"][0]["check_ids"].append("AC99")
    assert any("AC99" in item for item in validate_capabilities(CapabilityCatalog.model_validate(dangling), known_check_ids=known))

    duplicate = catalog.model_dump(mode="json")
    duplicate["capabilities"][1]["id"] = "F01"
    assert any("unique" in item for item in validate_capabilities(CapabilityCatalog.model_validate(duplicate), known_check_ids=known))

    # A declaration cannot smuggle in a hand-written result.
    with pytest.raises(ValidationError):
        CapabilityCatalog.model_validate(
            {
                **catalog.model_dump(mode="json"),
                "capabilities": [
                    {**catalog.capabilities[0].model_dump(mode="json"), "status": "pass"},
                    *[item.model_dump(mode="json") for item in catalog.capabilities[1:]],
                ],
            }
        )

    architecture = Path("docs/agent-architecture.md").read_text(encoding="utf-8")
    assert "docs/agent-capabilities.json" in architecture
    assert "accept" in architecture and "doctor" in architecture
    contracts = Path("docs/runtime-contracts.md").read_text(encoding="utf-8")
    assert "request-views" in contracts
    assert "docs/agent-architecture.md" in contracts


def test_parent_manual_split_does_not_inherit_routing_pass(tmp_path: Path) -> None:
    cases = load_cases(Path("tests/cases"))
    by_id = {case.id: case for case in cases}
    assert by_id["AC15"].source == "aggregate"
    assert by_id["AC15"].children == ["AC15.live", "AC15.quality"]
    assert by_id["AC15.live"].level == "live"
    assert by_id["AC15.quality"].source == "human_review"
    assert by_id["NC20"].children == ["NC20.routing", "NC20.manual"]
    assert by_id["NC20.routing"].pytest
    assert by_id["NC20.manual"].source == "human_review"

    case_dir = tmp_path / "cases"
    case_dir.mkdir()
    atomic_write_json(
        case_dir / "R1.json",
        {
            "schema_version": 2,
            "id": "R1",
            "title": "routing",
            "level": "offline",
            "source": "pytest",
            "pytest": ["tests/test_contracts.py::test_event_accepts_documented_sample"],
            "expectation": "routing test only",
        },
    )
    for name in ("M1", "P1"):
        atomic_write_json(
            case_dir / f"{name}.json",
            {
                "schema_version": 2,
                "id": name,
                "title": name,
                "level": "human" if name == "M1" else "offline",
                "source": "human_review" if name == "M1" else "aggregate",
                "children": [] if name == "M1" else ["R1", "M1"],
                "expectation": "manual" if name == "M1" else "parent",
            },
        )
    payload = run_eval(cases_dir=case_dir, output_root=tmp_path / "eval")
    by_id = {case["id"]: case for case in payload["cases"]}
    assert by_id["R1"]["status"] == "pass"
    assert by_id["M1"]["status"] == "not_run"
    assert by_id["P1"]["status"] == "not_run"

    # A legacy v1 AC15 pass cannot create the new child evidence.
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    atomic_write_json(
        legacy_dir / "eval.json",
        {"schema_version": 1, "cases": [{"id": "AC15", "status": "pass"}]},
    )
    results, _ = load_execution_results(legacy_dir)
    assert {item.check_id for item in results} == {"AC15"}
    assert all(item.check_id != "AC15.live" for item in results)


def test_acceptance_commands_are_not_model_tools(tmp_path: Path) -> None:
    names = {spec.name for spec in tool_specs()}
    assert len(TOOL_NAMES) == 6
    assert names == set(TOOL_NAMES)
    for forbidden in ("accept", "doctor", "record-review", "shell", "bash"):
        assert forbidden not in names

    scenario = make_scenario(
        [
            tool_step(
                ("accept", {"profile": "offline"}),
                ("doctor", {"profile": "offline"}),
                ("record-review", {"run_id": "run-20261025-000000-abcdef"}),
                ("shell", {"command": "git reset --hard"}),
            ),
            final_step(
                base_analysis(
                    hypotheses=[],
                    facts=[],
                    missing_evidence=["no tool was allowed"],
                    analysis_status="partial",
                    stop_reason="insufficient_evidence",
                )
            ),
        ]
    )
    result, _, output = run_replay(tmp_path, scenario=scenario)
    trace = [
        json.loads(line)
        for line in (output / result.run_id / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rejected = [item for item in trace if item["kind"] == "tool" and item["error_code"] == "unknown_tool"]
    assert len(rejected) == 4
    assert not (tmp_path / "acceptance.json").exists()
