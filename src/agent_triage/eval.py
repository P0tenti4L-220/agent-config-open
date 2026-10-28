"""Offline acceptance catalog runner.

Each case declares its evidence source.  ``pytest`` cases run the production
tests (not a re-implementation), ``aggregate`` cases combine declared children,
and ``human_review`` / ``run_artifact`` cases are never fabricated here: they
remain ``not_run`` until an external record exists.

Every evaluation is written to its own ``eval-<id>/`` directory and carries a
target provenance record (code commit, tree digest, config, corpus and check
definition hashes).  ``acceptance.py`` refuses to count a result whose target no
longer matches the current working tree.
"""

from __future__ import annotations

import os
import secrets
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .acceptance import (
    ExecutionResult,
    Target,
    _sha256_file,
    build_target,
    versions_manifest,
)
from .case_manifest import (
    EvalCase,
    case_definition_hash,
    case_definition_payload,
    load_cases,
    load_manifest,
)
from .config import Config
from .contracts import CheckStatus
from .errors import InputError, TriageError
from .store import atomic_write_json, atomic_write_text, utcnow


def _evaluation_id() -> str:
    return f"eval-{utcnow().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"


def _node_outcome(testcase: ET.Element) -> tuple[str, str | None]:
    for child in testcase:
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "failure":
            return "failed", (child.get("message") or "")[:500]
        if tag == "error":
            return "error", (child.get("message") or "")[:500]
        if tag == "skipped":
            return "skipped", (child.get("message") or "")[:500]
    return "passed", None


def _junit_cases(path: Path) -> list[ET.Element]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return []
    return list(root.iter("testcase"))


def _case_matches(expected: str, testcase: ET.Element) -> bool:
    if "::" not in expected:
        return False
    file_part, name_part = expected.split("::", 1)
    name_part = name_part.split("::")[-1]
    classname = (testcase.get("classname") or "").replace(".", "/")
    name = testcase.get("name") or ""
    class_suffix = file_part.rsplit("/", 1)[0] + "/" + file_part.rsplit("/", 1)[-1].replace(".py", "")
    if not classname.endswith(class_suffix):
        return False
    if "[" in name_part:
        return name == name_part
    return name == name_part or name.startswith(name_part + "[")


def _parse_junit(path: Path, nodes: list[str]) -> tuple[list[dict[str, Any]], str | None]:
    cases = _junit_cases(path)
    if not cases:
        return [], "structured report is missing or has no testcase elements"
    results: list[dict[str, Any]] = []
    for node in nodes:
        matches = [case for case in cases if _case_matches(node, case)]
        if not matches:
            results.append({"nodeid": node, "outcome": "missing", "duration_ms": None, "message": None})
            continue
        outcomes = [_node_outcome(matched) for matched in matches]
        if any(item[0] == "failed" for item in outcomes):
            outcome, message = next(item for item in outcomes if item[0] == "failed")
        elif any(item[0] == "error" for item in outcomes):
            outcome, message = next(item for item in outcomes if item[0] == "error")
        elif any(item[0] == "skipped" for item in outcomes):
            outcome, message = next(item for item in outcomes if item[0] == "skipped")
        else:
            outcome, message = "passed", None
        duration = sum(float(matched.get("time") or 0) for matched in matches)
        results.append(
            {
                "nodeid": node,
                "outcome": outcome,
                "duration_ms": int(duration * 1000) if duration else None,
                "message": message,
            }
        )
    return results, None


def _status_from_nodes(nodes: list[dict[str, Any]]) -> tuple[str, str | None, str | None]:
    outcomes = [node["outcome"] for node in nodes]
    if "missing" in outcomes:
        return "error", "not_collected", "a declared pytest node did not produce a structured result"
    if "error" in outcomes:
        return "error", "test_error", "at least one collected test errored"
    if "failed" in outcomes:
        return "fail", "assertion_failed", "at least one collected test failed"
    if "skipped" in outcomes:
        return "not_run", "skipped", "at least one test was skipped; a skip is not a pass"
    return "pass", None, None


def _run_pytest_case(
    case: EvalCase,
    *,
    evaluation_id: str,
    project_root: Path,
    evidence_dir: Path,
    target: Target,
    include_live: bool,
    config_path: Path | None = None,
) -> dict[str, Any]:
    started = utcnow()
    base: dict[str, Any] = {
        "id": case.id,
        "title": case.title,
        "level": case.level,
        "source": case.source,
        "expectation": case.expectation,
        "children": case.children,
        "attempt_id": "1",
        "case_definition_hash": case_definition_hash(case),
        "artifacts": [],
        "nodes": [],
        "interpreter_version": sys.version.split()[0],
        "import_root_checked": True,
    }
    if case.level == "live" and not include_live:
        base.update(
            status="not_run",
            reason_code="live_not_requested",
            reason="live model/GitHub run requires explicit --include-live and owner credentials",
            duration_ms=0,
        )
        return base
    if case.level == "human":
        base.update(
            status="not_run",
            reason_code="human_record_required",
            reason="human review is imported with an external record; eval never fabricates it",
            duration_ms=0,
        )
        return base

    log_path = evidence_dir / "logs" / f"{case.id}.log"
    junit_path = evidence_dir / "junit" / f"{case.id}.xml"
    junit_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable,
        "-m",
        "pytest",
        *case.pytest,
        "-q",
        f"--junitxml={junit_path}",
    ]
    if include_live and case.level == "live":
        # pytest-level opt-in: credentials alone must never enter a live run.
        argv.append("--run-live")
    base["argv"] = argv
    env = dict(os.environ)
    if include_live and case.level == "live":
        env["AGENT_TRIAGE_LIVE_BUNDLE_DIR"] = str((evidence_dir / "live").resolve())
        env["AGENT_TRIAGE_EVAL_TARGET"] = target.model_dump_json()
        if config_path is not None:
            env["AGENT_TRIAGE_EVAL_CONFIG"] = str(config_path.resolve())
    started_monotonic = time.monotonic()
    try:
        completed = subprocess.run(
            argv,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
            env=env,
        )
        duration = int((time.monotonic() - started_monotonic) * 1000)
        atomic_write_text(
            log_path,
            f"$ {' '.join(argv)}\nexit_code={completed.returncode}\n\n"
            f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}\n",
        )
        nodes, report_error = _parse_junit(junit_path, case.pytest)
        status, reason_code, reason = _status_from_nodes(nodes) if not report_error else ("error", "missing_report", report_error)
        if completed.returncode not in (0, 1) and status == "pass":
            status, reason_code, reason = "error", "pytest_exit", f"pytest exited with {completed.returncode}"
        if status == "fail" and completed.returncode == 0:
            status, reason_code, reason = "error", "exit_mismatch", "structured failure but pytest exited 0"
        if completed.returncode == 0 and status in {"fail", "error"} and not report_error:
            # Structured report disagrees with the process exit; treat the
            # structured evidence as authoritative but keep the discrepancy.
            reason = (reason or "") + f"; pytest exit code was {completed.returncode}"
        exit_code = completed.returncode
    except subprocess.TimeoutExpired:
        duration = int((time.monotonic() - started_monotonic) * 1000)
        status, reason_code, reason, exit_code, nodes = (
            "error",
            "timeout",
            "pytest did not finish within 600 seconds",
            None,
            [],
        )
        atomic_write_text(log_path, f"$ {' '.join(argv)}\ntimeout after 600s\n")
    except OSError as exc:
        duration = int((time.monotonic() - started_monotonic) * 1000)
        status, reason_code, reason, exit_code, nodes = (
            "error",
            "start_failed",
            f"could not start pytest: {type(exc).__name__}",
            None,
            [],
        )
        atomic_write_text(log_path, f"$ {' '.join(argv)}\nstart_failed: {type(exc).__name__}\n")

    artifacts: list[dict[str, Any]] = []
    for path, media in ((log_path, "text/plain"), (junit_path, "application/xml")):
        if path.is_file():
            artifacts.append(
                {
                    "relative_path": path.relative_to(evidence_dir).as_posix(),
                    "sha256": _sha256_file(path),
                    "media_type": media,
                    "size_bytes": path.stat().st_size,
                }
            )
    finished = utcnow()
    base.update(
        status=status,
        reason_code=reason_code,
        reason=reason,
        duration_ms=duration,
        exit_code=exit_code,
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        artifacts=artifacts,
        nodes=nodes,
    )
    # Validate the shape here, so a malformed execution cannot silently reach
    # the evidence directory.
    ExecutionResult.model_validate(
        {
            "evaluation_id": evaluation_id,
            "check_id": case.id,
            "attempt_id": "1",
            "status": status,
            "reason_code": reason_code,
            "reason": reason,
            "source_type": "pytest",
            "case_definition_hash": case_definition_hash(case),
            "target": target.model_dump(mode="json"),
            "execution": {
                "started_at": base.get("started_at"),
                "finished_at": base.get("finished_at"),
                "argv": argv,
                "exit_code": exit_code,
                "interpreter_version": base["interpreter_version"],
                "import_root_checked": True,
                "executor_id": "agent-triage eval",
            },
            "artifacts": artifacts,
            "nodes": nodes,
        }
    )
    return base


def _aggregate_status(children: list[str], by_id: dict[str, dict[str, Any]]) -> tuple[str, str | None, str | None]:
    statuses = [by_id[child]["status"] for child in children]
    if any(status == "error" for status in statuses):
        return "error", "child_error", "at least one child check errored"
    if any(status == "fail" for status in statuses):
        return "fail", "child_failed", "at least one child check failed"
    if all(status == "pass" for status in statuses):
        return "pass", None, None
    return "not_run", "children_incomplete", "not every child check has current passing evidence"


def run_eval(
    *,
    cases_dir: Path,
    output_root: Path,
    include_live: bool = False,
    project_root: Path | None = None,
    config: Config | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    if include_live and config is None:
        raise InputError("live evaluation requires a verified configuration (config)")
    cases = load_cases(cases_dir)
    manifest = load_manifest(cases_dir)
    root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    evaluation_id = _evaluation_id()
    evidence_dir = (output_root.resolve() / evaluation_id)
    evidence_dir.mkdir(parents=True, exist_ok=False)
    target = build_target(
        root,
        live_config=config if include_live else None,
        exclude=[evidence_dir],
    )
    atomic_write_json(
        evidence_dir / "cases.json",
        {
            "schema_version": 1,
            "manifest_digest": manifest.digest,
            "cases": [case_definition_payload(case) for case in cases],
            "case_hashes": {case.id: case_definition_hash(case) for case in cases},
        },
    )

    by_id = {case.id: case for case in cases}
    results: dict[str, dict[str, Any]] = {}
    for case in cases:
        if case.source == "aggregate":
            continue
        if case.source in {"human_review", "run_artifact"}:
            results[case.id] = {
                "id": case.id,
                "title": case.title,
                "level": case.level,
                "source": case.source,
                "expectation": case.expectation,
                "children": [],
                "attempt_id": "external",
                "case_definition_hash": case_definition_hash(case),
                "artifacts": [],
                "nodes": [],
                "status": "not_run",
                "reason_code": "external_record_required",
                "reason": f"{case.source} evidence must be imported from an external record",
                "duration_ms": 0,
            }
            continue
        results[case.id] = _run_pytest_case(
            case,
            evaluation_id=evaluation_id,
            project_root=root,
            evidence_dir=evidence_dir,
            target=target,
            include_live=include_live,
            config_path=config_path,
        )

    if include_live:
        _import_live_bundles(
            evidence_dir=evidence_dir,
            project_root=root,
            results=results,
            by_id=by_id,
            evaluation_id=evaluation_id,
        )

    for case in cases:
        if case.source != "aggregate":
            continue
        status, reason_code, reason = _aggregate_status(case.children, results)
        results[case.id] = {
            "id": case.id,
            "title": case.title,
            "level": case.level,
            "source": "aggregate",
            "expectation": case.expectation,
            "children": case.children,
            "attempt_id": "aggregate",
            "case_definition_hash": case_definition_hash(case),
            "artifacts": [],
            "nodes": [],
            "status": status,
            "reason_code": reason_code,
            "reason": reason,
            "duration_ms": 0,
        }

    ordered = [results[case.id] for case in cases]
    summary = {
        "total": len(ordered),
        "pass": sum(1 for item in ordered if item["status"] == "pass"),
        "fail": sum(1 for item in ordered if item["status"] == "fail"),
        "error": sum(1 for item in ordered if item["status"] == "error"),
        "not_run": sum(1 for item in ordered if item["status"] == "not_run"),
    }
    if summary["fail"] or summary["error"]:
        overall = "fail"
    elif summary["not_run"]:
        overall = "partial"
    else:
        overall = "pass"

    import agent_triage

    import_root = Path(agent_triage.__file__).resolve().parent
    payload = {
        "schema_version": 2,
        "evaluation_id": evaluation_id,
        "kind": "executed_evidence",
        "scope": "offline_and_local_mcp" if not include_live else "including_live",
        "overall": overall,
        "summary": summary,
        "target": target.model_dump(mode="json"),
        "cases_manifest_digest": manifest.digest,
        "cases": ordered,
    }
    provenance = {
        "schema_version": 1,
        "evaluation_id": evaluation_id,
        "target": target.model_dump(mode="json"),
        "versions": versions_manifest(),
        "interpreter_version": sys.version.split()[0],
        "interpreter_executable": sys.executable,
        "import_root": str(import_root),
        "import_root_checked": root in import_root.parents or import_root == root / "src" / "agent_triage",
        "argv_cases_dir": str(cases_dir),
        "cases_manifest_digest": manifest.digest,
        "config_source": str(config_path.resolve()) if config_path is not None else None,
        "live_config_hash": target.live_config_hash,
        "created_at": utcnow().isoformat(),
    }
    atomic_write_json(evidence_dir / "eval.json", payload)
    atomic_write_json(evidence_dir / "provenance.json", provenance)
    atomic_write_text(evidence_dir / "eval.md", _render_markdown(payload, evidence_dir))
    return {
        **payload,
        "evidence_dir": str(evidence_dir),
        "evidence_dir_name": evaluation_id,
    }


def _import_live_bundles(
    *,
    evidence_dir: Path,
    project_root: Path,
    results: dict[str, dict[str, Any]],
    by_id: dict[str, EvalCase],
    evaluation_id: str | None = None,
) -> None:
    from .live_bundle import import_live_bundle

    case = by_id.get("HC09")
    if case is None:
        return
    bundles = sorted((evidence_dir / "live").glob("*/manifest.json"))
    if not bundles:
        results["HC09"].update(
            status="not_run",
            reason_code="no_live_bundle",
            reason="no live bundle was produced; HC09 cannot be derived from pytest output alone",
        )
        return
    imported = []
    for index, manifest_path in enumerate(bundles, start=1):
        try:
            result, _ = import_live_bundle(
                manifest_path.parent,
                evidence_dir,
                project_root=project_root,
                attempt=str(index),
                evaluation_id=evaluation_id,
            )
        except TriageError as exc:
            results["HC09"].update(
                status="error",
                reason_code="live_bundle_invalid",
                reason=f"{exc.message}" + (f": {exc.detail}" if exc.detail else ""),
            )
            return
        imported.append(result)
    if not imported:
        results["HC09"].update(status="not_run", reason_code="no_live_bundle", reason="no importable live bundle")
        return
    first = imported[0]
    failed = [item for item in imported if item.status != CheckStatus.PASS]
    results["HC09"].update(
        status="error" if failed else "pass",
        reason_code="live_bundle_failed" if failed else None,
        reason=None if not failed else "one or more live bundles did not verify",
        artifacts=[artifact.model_dump(mode="json") for artifact in first.artifacts],
        related_run_ids=list(first.related_run_ids),
        exit_code=first.execution.exit_code,
        argv=list(first.execution.argv or []),
        duration_ms=0,
    )


def _render_markdown(payload: dict[str, Any], evidence_dir: Path) -> str:
    target = payload["target"]
    lines = [
        "# 离线验收汇总",
        "",
        f"- 评估 ID：`{payload['evaluation_id']}`",
        f"- 范围：`{payload['scope']}`",
        f"- 总体：`{payload['overall']}`",
        f"- 目标 commit：`{target.get('code_commit') or 'not-a-git-tree'}`；"
        f"dirty：`{target.get('dirty')}`",
        f"- 树摘要：`{target['relevant_tree_digest'][:16]}…`",
        f"- 通过 {payload['summary']['pass']} / 失败 {payload['summary']['fail']} / "
        f"执行错误 {payload['summary']['error']} / 未运行 {payload['summary']['not_run']}",
        "",
        "| 案例 | 层级 | 来源 | 状态 | 证据 |",
        "|---|---|---|---|---|",
    ]
    for case in payload["cases"]:
        evidence = "、".join(item["relative_path"] for item in case.get("artifacts", [])) or case.get("reason") or "-"
        lines.append(
            f"| {case['id']} {case['title']} | {case['level']} | {case['source']} | "
            f"`{case['status']}` | {evidence} |"
        )
    lines.extend(
        [
            "",
            f"验收只看这一份证据时，使用：`agent-triage accept --profile offline --evidence-dir {evidence_dir.name}`。",
            "eval 的 partial/退出 0 只是执行语义；profile 是否通过由 `accept` 按必需检查判定。",
            "未运行不等于通过。真实模型、GitHub MCP 联调与人工评分需要单独执行并导入记录。",
            "",
        ]
    )
    return "\n".join(lines)