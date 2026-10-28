"""Self-contained synthetic demo. No credentials, network or real project is used."""

from __future__ import annotations

from pathlib import Path

from .contracts import Event
from .models import ReplayScenario
from .pipeline import run_new_analysis
from .runner import RunResult
from .store import atomic_write_text

FAILURE_LOG = """2026-10-24T20:00:00+08:00 INFO loading check records from data/checks.json
2026-10-24T20:00:00+08:00 ERROR failed to read check records
Traceback (most recent call last):
  File "src/check_reader.py", line 7, in load_checks
    return payload["items"]
KeyError: 'items'
"""

READER = """import json


def load_checks(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["items"]
"""

DATA = """{
  "checks": [
    {"name": "synthetic-check-1", "status": "pass"}
  ]
}
"""


def demo_event() -> Event:
    return Event(
        schema_version=1,
        event_id="demo-event-001",
        occurred_at="2026-10-24T20:00:00+08:00",
        project_id="sample-project",
        rule_version="v0.1.0",
        goal="读取检查记录，定位 KeyError 的来源并整理下一步验证材料",
        symptom="KeyError: 'items'",
        error_files=["logs/failure.log"],
        related_files=["src/check_reader.py", "data/checks.json"],
        attempts=[],
        environment={"language": "python", "runtime_version": None},
        source={"kind": "synthetic_fixture", "session_id": None},
        note="synthetic demo; generated locally, never committed",
    )


def write_demo_project(project_root: Path) -> None:
    project_root.mkdir(parents=True, exist_ok=True)
    atomic_write_text(project_root / "logs" / "failure.log", FAILURE_LOG)
    atomic_write_text(project_root / "src" / "check_reader.py", READER)
    atomic_write_text(project_root / "data" / "checks.json", DATA)


def demo_scenario() -> ReplayScenario:
    return ReplayScenario.model_validate(
        {
            "schema_version": 1,
            "name": "demo-keyerror-items",
            "steps": [
                {
                    "type": "tools",
                    "calls": [
                        {
                            "tool": "read_file",
                            "arguments": {"path": "logs/failure.log", "start_line": 1, "max_lines": 40},
                        }
                    ],
                },
                {
                    "type": "tools",
                    "calls": [
                        {
                            "tool": "read_file",
                            "arguments": {"path": "src/check_reader.py", "start_line": 1, "max_lines": 40},
                        }
                    ],
                },
                {
                    "type": "tools",
                    "calls": [
                        {
                            "tool": "read_file",
                            "arguments": {"path": "data/checks.json", "start_line": 1, "max_lines": 40},
                        }
                    ],
                },
                {
                    "type": "tools",
                    "calls": [
                        {
                            "tool": "search_lessons",
                            "arguments": {"query": "KeyError items checks adapter", "limit": 3},
                        }
                    ],
                },
                {
                    "type": "final",
                    "analysis": {
                        "schema_version": 1,
                        "summary": "日志显示读取检查记录时在 payload['items'] 触发 KeyError；输入顶层提供的是 checks。",
                        "attribution": "public_workflow",
                        "facts": [
                            {
                                "text": "失败日志显示 src/check_reader.py 第 7 行访问 payload['items'] 时抛出 KeyError: 'items'",
                                "evidence_ids": ["@path:logs/failure.log", "@path:src/check_reader.py"],
                            },
                            {
                                "text": "data/checks.json 的顶层键是 checks，没有 items 键",
                                "evidence_ids": ["@path:data/checks.json"],
                            },
                        ],
                        "hypotheses": [
                            {
                                "text": "读取器缺少按格式版本选择字段的适配层；输入与读取器预期不一致",
                                "supporting_evidence_ids": ["@path:src/check_reader.py", "@path:data/checks.json"],
                                "contradicting_evidence_ids": [],
                                "open_questions": ["是否存在其他格式版本的检查记录样例？", "items 是否为旧版字段？"],
                            }
                        ],
                        "missing_evidence": [],
                        "suggested_checks": [
                            {
                                "check_id": "adapter-normal-case",
                                "description": "用正常样例验证按版本选择适配器后能读取 checks",
                                "target_paths": ["src/check_reader.py"],
                                "status": "not_run",
                            },
                            {
                                "check_id": "adapter-invalid-type",
                                "description": "验证字段类型错误和未知格式版本会显式失败",
                                "target_paths": ["src/check_reader.py"],
                                "status": "not_run",
                            },
                        ],
                        "duplicate_candidates": [],
                        "analysis_status": "complete",
                        "stop_reason": "completed",
                    },
                },
            ],
        }
    )


def run_demo(output_root: Path) -> RunResult:
    output_root = output_root.resolve()
    project_root = output_root / "project"
    write_demo_project(project_root)
    from .config import Config

    return run_new_analysis(
        event=demo_event(),
        project_root=project_root,
        output_root=output_root / "runs",
        config=Config(),
        replay_scenario=demo_scenario(),
    )