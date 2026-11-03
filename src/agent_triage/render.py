"""Render all human-facing artifacts from the same validated JSON objects."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

from .contracts import (
    Analysis,
    AnalysisStatus,
    Attribution,
    CheckResult,
    CheckStatus,
    ContextPacket,
    DuplicateCandidate,
    Event,
    Evidence,
    EvidenceKind,
    Lesson,
    LessonStatus,
)
from .store import BudgetState, RunMetadata

SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9_-]{10,}|gh[pousr]_[A-Za-z0-9]{10,}|github_pat_[A-Za-z0-9_]{10,}"
    r"|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|Bearer\s+[A-Za-z0-9._-]{10,})"
)
ABSOLUTE_PATH_RE = re.compile(r"(?<![\w:/])/(?:Users|home|private|var|tmp)/[^\s`'\"<>)]+")


def _claim_review_status(review_view: dict | None, claim_id: str | None) -> str | None:
    if not review_view or not claim_id:
        return None
    item = (review_view.get("claims") or {}).get(claim_id)
    return item.get("status") if item else None


def _safe_text(value: str) -> str:
    """Keep public Markdown free of recognizable secrets and local paths."""
    value = SECRET_RE.sub("[REDACTED]", value)
    return ABSOLUTE_PATH_RE.sub("<LOCAL_PATH>", value)


def _evidence_label(evidence: Evidence) -> str:
    if evidence.kind == EvidenceKind.ISSUE:
        return evidence.url or evidence.evidence_id
    location = evidence.path or "workspace"
    if evidence.line_start is not None:
        location += f":{evidence.line_start}-{evidence.line_end}"
    return location


def _status_label(status: AnalysisStatus) -> str:
    return {"complete": "完整（契约完成，不代表问题已修复）", "partial": "局部", "failed": "失败"}[status.value]


def render_report(
    *,
    event: Event,
    analysis: Analysis,
    evidence: list[Evidence],
    checks: list[CheckResult],
    meta: RunMetadata,
    budget: BudgetState,
    remote_status: str,
    versions: dict[str, str] | None = None,
    investigation: Any | None = None,
    playbook: Any | None = None,
    review_view: dict | None = None,
) -> str:
    by_id = {item.evidence_id: item for item in evidence}
    versions = versions or {}
    lines: list[str] = [
        f"# 报错分析：{event.project_id}",
        "",
        f"> 分析状态：**{_status_label(analysis.analysis_status)}**；"
        f"停止原因：`{analysis.stop_reason.value}`；工作流状态：`{meta.status}`；"
        f"远端 GitHub：`{remote_status}`。",
        "> 本报告由程序从 `analysis.json` 渲染。所有“建议检查”均未执行；"
        "格式检查通过不等于根因已复现或问题已修复。",
        "",
        "## 摘要",
        "",
        _safe_text(analysis.summary),
        "",
        "## 任务与症状",
        "",
        f"- 目标：{_safe_text(event.goal)}",
        f"- 症状：`{_safe_text(event.symptom[:1000])}`",
        f"- 工作根别名：`{event.project_id}`（不展示本机绝对路径）",
        f"- 公共规则版本：`{event.rule_version}`",
        f"- policy：`{versions.get('policy_version', 'unknown')}`；prompt：`{versions.get('prompt_version', 'unknown')}`；"
        f"playbook：`{versions.get('playbook_id', 'none')}`@{versions.get('playbook_version', '-')}",
        f"- 事件 ID：`{event.event_id}`；运行 ID：`{meta.run_id}`",
        f"- 相关文件：{_paths(event.error_files, event.related_files)}",
        "",
        "## 调查状态",
        "",
    ]
    if investigation is not None:
        lines.extend(
            [
                f"- 当前问题：{_safe_text(investigation.active_question)}",
                f"- 阶段：`{investigation.phase.value}`；下一步理由：{_safe_text(investigation.next_action_reason)}",
            ]
        )
        if investigation.playbook_id:
            lines.append(f"- 载入流程：`{investigation.playbook_id}@{investigation.playbook_version}`（只载入一份）")
        if investigation.needed_materials:
            lines.append("- 仍需材料：" + "；".join(_safe_text(item) for item in investigation.needed_materials[:8]))
        if investigation.attempt_ledger:
            lines.append(f"- 已记录尝试：{len(investigation.attempt_ledger)} 次（完整记录见 `investigation.json`）")
    else:
        lines.append("- 旧运行没有调查状态记录；按事件、证据和 trace 重建。")
    lines.extend(["", "## 已确认事实", ""])
    if analysis.facts:
        for index, fact in enumerate(analysis.facts, start=1):
            refs = ", ".join(f"[{evidence_id}]" for evidence_id in fact.evidence_ids)
            claim = f" `{fact.claim_id}`" if fact.claim_id else ""
            fact_status = _claim_review_status(review_view, fact.claim_id) or fact.support_review.status.value
            lines.append(
                f"{index}.{claim} {_safe_text(fact.text)} {refs}"
                f"（支持性复核：`{fact_status}`）"
            )
            for evidence_id in fact.evidence_ids:
                item = by_id.get(evidence_id)
                if item is not None:
                    lines.append(f"   - 来源：`{_evidence_label(item)}`（sha256 `{item.sha256[:12]}`）")
    else:
        lines.append("（无带引用的已确认事实）")
    lines.extend(["", "## 根因假设", ""])
    if analysis.hypotheses:
        for index, hypothesis in enumerate(analysis.hypotheses, start=1):
            claim = f" `{hypothesis.claim_id}`" if hypothesis.claim_id else ""
            hypothesis_status = (
                _claim_review_status(review_view, hypothesis.claim_id)
                or hypothesis.support_review.status.value
            )
            lines.append(
                f"{index}. **假设**{claim}：{_safe_text(hypothesis.text)}"
                f"（支持性复核：`{hypothesis_status}`）"
            )
            lines.append(
                f"   - 支持证据：{_refs(hypothesis.supporting_evidence_ids)}；"
                f"反对证据：{_refs(hypothesis.contradicting_evidence_ids)}"
            )
            if hypothesis.open_questions:
                lines.append(f"   - 待确认：{'；'.join(_safe_text(item) for item in hypothesis.open_questions)}")
    else:
        lines.append("（没有形成可检验的假设）")
    lines.extend(["", "## 缺失证据", ""])
    lines.extend([f"- {_safe_text(item)}" for item in analysis.missing_evidence] or ["- 无"])
    lines.extend(["", "## 建议验证（均未执行）", ""])
    if analysis.suggested_checks:
        for check in analysis.suggested_checks:
            targets = f"；对象：{', '.join(check.target_paths)}" if check.target_paths else ""
            lines.append(f"- `{check.status}` `{check.check_id}`：{_safe_text(check.description)}{targets}")
    else:
        lines.append("- 无")
    lines.extend(["", "## 重复与相似候选", ""])
    if analysis.duplicate_candidates:
        for candidate in analysis.duplicate_candidates:
            lines.append(
                f"- `{candidate.source}` {_safe_text(candidate.reference)}：{_safe_text(candidate.title or '未命名')}；"
                f"理由：{_safe_text(candidate.similarity_reason)}；状态：{candidate.state}"
            )
    else:
        lines.append("- 本地指纹和本次查询范围内没有发现候选；这不等于全局不存在重复问题。")
    if analysis.related_issues:
        lines.extend(["", "## 分层 Issue 检索", ""])
        for item in analysis.related_issues:
            assessment = ""
            if item.assessment is not None:
                assessment = (
                    f"；component={item.assessment.component}, symptom={item.assessment.symptom}, "
                    f"version={item.assessment.version}, environment={item.assessment.environment}, "
                    f"trigger={item.assessment.trigger}, fix={item.assessment.fix}"
                )
            lines.append(
                f"- `{item.layer.value}` {_safe_text(item.issue_ref)}：{_safe_text(item.title or '未命名')}；"
                f"适用性：{item.applicability}{assessment}；{_safe_text(item.url)}"
            )
        lines.append("- 适用性由逐项 match/mismatch/unknown 记录；适用线索不等于根因已确认。")
    lines.extend(["", "## 证据支持性（与结构检查分开）", ""])
    if review_view is not None and review_view.get("overall") in {"pass", "fail"}:
        support_status = review_view["overall"]
    else:
        support_status = (
            "not_run" if all(
                claim.support_review.status.value == "not_run"
                for claim in [*analysis.facts, *analysis.hypotheses]
            ) else "recorded per claim"
        )
    lines.append(
        f"- 主张支持性复核：`{support_status}`。结构、引用和展示检查通过，不等于引用片段在语义上支持结论。"
    )
    lines.append("- 需要逐条标注 supported / overstated / unsupported / insufficient 时，用 `record-review` 导入外部复核记录；模型自填与程序都不能自动给 pass。")
    from .review import render_review_section

    lines.extend(render_review_section(review_view))
    if any(
        claim.support_review.note and "model-supplied" in claim.support_review.note
        for claim in [*analysis.facts, *analysis.hypotheses]
    ):
        lines.append("- 模型输出自带的复核字段已在模型边界降级为 `not_run`；导入记录前不构成外部复核。")
    lines.extend(["", "## 上下文遗漏及其影响", ""])
    omissions = investigation.omissions if investigation is not None else []
    if omissions:
        for omission in omissions[:10]:
            affected = "、".join(omission.affected_claim_ids) or "（尚未映射到主张）"
            lines.append(
                f"- {_safe_text(omission.request)}：{_safe_text(omission.reason)}；影响：{affected}"
            )
    else:
        lines.append("- 无关键材料被省略；这不代表所有已采集材料都已展示。")
    lines.extend(["", "## 检查状态", "", "| 检查 | 状态 | 范围 | 引用 |", "|---|---|---|---|"])
    for check in checks:
        lines.append(
            f"| `{check.check_id}` | `{check.status.value}` | {_safe_text(check.scope)} | {check.output_ref or '-'} |"
        )
    lines.extend(
        [
            "",
            "## 运行信息",
            "",
            f"- provider：`{meta.provider}`；模型：`{meta.model}`",
            f"- 模型请求：{budget.model_requests_used}；工具尝试：{budget.tool_attempts_used}；"
            f"纠错：{budget.corrections_used}",
            f"- 上下文计数方式：估计（`count_mode=estimated`）；真实 token 用量另见 trace。",
            f"- commit：`{(evidence[0].commit if evidence else None) or 'null'}`（无证据时为空）",
            "",
            "## 边界",
            "",
            "- 事实只表示“在指定证据片段中被观察到”；根因是否成立仍需复现或反例。",
            "- 本 Agent 没有修改业务源码、没有执行修复、没有发布远端 issue。",
            "- 候选经验不能当作已验证经验；需要 `record-outcome` 且所有者确认。",
            "",
        ]
    )
    return "\n".join(lines)


def render_handoff(
    *,
    packet: ContextPacket,
    analysis: Analysis | None,
    checks: list[CheckResult],
    meta: RunMetadata,
    versions: dict[str, str] | None = None,
    investigation: Any | None = None,
) -> str:
    versions = versions or packet.versions
    lines: list[str] = [
        f"# 交接上下文：{packet.project_id}",
        "",
        f"- 运行：`{packet.run_id}`；事件：`{packet.event_id}`；状态：`{meta.status}`",
        f"- 目标：{_safe_text(packet.goal)}",
        f"- 分析状态：`{analysis.analysis_status.value if analysis else 'interrupted'}`"
        f"（停止原因：`{analysis.stop_reason.value if analysis else 'interrupted'}`）",
        f"- 归属判断：`{analysis.attribution.value if analysis else 'unknown'}`",
        f"- policy：`{versions.get('policy_version', 'unknown')}`（hash `{versions.get('policy_hash', '-')[:12]}`）；"
        f"prompt：`{versions.get('prompt_version', 'unknown')}`；"
        f"playbook：`{versions.get('playbook_id', 'none')}`@{versions.get('playbook_version', '-')}",
        "",
        "## 当前问题与最小下一步",
        "",
    ]
    if investigation is not None:
        lines.append(f"- 接手后先确认：{_safe_text(investigation.active_question)}")
        lines.append(f"- 下一步理由：{_safe_text(investigation.next_action_reason)}")
        if investigation.needed_materials:
            lines.append("- 仍需材料：" + "；".join(_safe_text(item) for item in investigation.needed_materials[:8]))
    if analysis and analysis.suggested_checks:
        lines.append("- 最小验证（`not_run`）：")
        for check in analysis.suggested_checks[:5]:
            targets = f"；对象：{', '.join(check.target_paths)}" if check.target_paths else ""
            lines.append(f"  - `{check.check_id}`：{_safe_text(check.description)}{targets}")
    elif analysis and analysis.missing_evidence:
        lines.append("- 先补材料：" + "；".join(_safe_text(item) for item in analysis.missing_evidence[:5]))
    else:
        lines.append("- 没有可执行的最小下一步；这是交接缺口，不能视为完成。")
    lines.extend(["", "## 已确认事实（附证据）", ""])
    if packet.facts:
        for fact in packet.facts:
            claim = f"`{fact.claim_id}` " if fact.claim_id else ""
            lines.append(
                f"- {claim}{_safe_text(fact.text)} {_refs(fact.evidence_ids)}"
                f"（支持性复核：`{fact.support_review.status.value}`）"
            )
    else:
        lines.append("- 暂无")
    lines.extend(["", "## 假设与反证缺口（不是事实）", ""])
    hypotheses = packet.hypotheses or (analysis.hypotheses if analysis else [])
    if hypotheses:
        for hypothesis in hypotheses:
            claim = f"`{hypothesis.claim_id}` " if hypothesis.claim_id else ""
            lines.append(
                f"- {claim}{_safe_text(hypothesis.text)} 支持：{_refs(hypothesis.supporting_evidence_ids)}；"
                f"反证：{_refs(hypothesis.contradicting_evidence_ids)}"
            )
            if hypothesis.open_questions:
                lines.append(
                    "  - 尚未回答：" + "；".join(_safe_text(item) for item in hypothesis.open_questions)
                )
    else:
        lines.append("- 暂无")
    lines.extend(["", "## 已尝试动作", ""])
    if packet.failed_attempts:
        for attempt in packet.failed_attempts:
            lines.append(f"- {_safe_text(attempt.action)}：`{attempt.status}`（{_safe_text(attempt.reason or '无原因')}）；"
                         f"来源：`{attempt.source_ref or 'legacy: 未记录'}`；重试条件：{_safe_text(attempt.retry_condition)}")
    else:
        lines.append("- 无失败工具调用")
    no_progress = packet.state_projection.get("no_progress_attempts", [])
    if no_progress:
        lines.append(f"- 成功返回但无新证据：{len(no_progress)} 次；仍未解决，明细见 `context.json:state_projection.no_progress_attempts`。")
        for attempt in no_progress[:10]:
            lines.append(f"  - `{attempt.get('call_id')}`：{_safe_text(attempt['action'])}；来源：`{attempt['source_ref']}`")
    lines.extend(["", "## 未执行项", ""])
    lines.extend([f"- {_safe_text(item)}" for item in packet.not_executed] or ["- 无"])
    if analysis and analysis.suggested_checks:
        lines.append("")
        lines.append("建议下一步（未执行）：")
        for check in analysis.suggested_checks:
            lines.append(f"- `{check.check_id}`：{_safe_text(check.description)}")
    lines.extend(["", "## 上下文遗漏与证据支持性", ""])
    omissions = investigation.omissions if investigation is not None else packet.omissions
    if omissions:
        for omission in omissions[:10]:
            affected = "、".join(omission.affected_claim_ids) or "（尚未映射到主张）"
            lines.append(f"- 遗漏 {_safe_text(omission.request)}：{_safe_text(omission.reason)}；影响：{affected}")
        if len(omissions) > 10:
            lines.append(f"- 另有 {len(omissions) - 10} 条未展开，见 `investigation.json:omissions`。")
    else:
        lines.append("- 没有记录关键材料遗漏。")
    support = (
        "not_run"
        if analysis is None
        or all(
            claim.support_review.status.value == "not_run"
            for claim in [*analysis.facts, *analysis.hypotheses]
        )
        else "recorded per claim"
    )
    lines.append(f"- 主张支持性复核：`{support}`。结构与引用检查不等于语义支持成立。")
    lines.extend(["", "## 检查状态", ""])
    for check in checks:
        lines.append(f"- `{check.check_id}`：`{check.status.value}`（{_safe_text(check.scope)}）")
    lines.extend(["", "## 证据索引", "", "| 证据 | 类型 | 位置 | 过期 |", "|---|---|---|---|"])
    for item in packet.evidence_index:
        location = item.url or item.path or "-"
        if item.line_start is not None:
            location += f":{item.line_start}-{item.line_end}"
        lines.append(f"| `{item.evidence_id}` | `{item.kind.value}` | {location} | {item.stale} |")
    lines.extend(
        [
            "",
            "## 恢复方式",
            "",
            "新会话先读本文件、`context.json` 与 `analysis.json`，不要继承旧聊天全文。",
            "如需从检查点继续：",
            "",
            "```bash",
            f"agent-triage resume --run-id {packet.run_id} --output <OUTPUT_ROOT>",
            "```",
            "",
            "证据正文用 evidence ID 回查 `evidence/index.json` 与 `evidence/snippets/`；"
            "已捕获快照位于 `evidence/objects/`。`read_evidence` 只续读同一快照，未捕获部分仍未知。",
            "",
        ]
    )
    return "\n".join(lines)


def render_issue_draft(*, event: Event, analysis: Analysis, evidence: list[Evidence]) -> str:
    by_id = {item.evidence_id: item for item in evidence}
    lines: list[str] = [
        f"# [triage] {_safe_text(event.symptom[:120])}",
        "",
        "> 这是本地草稿，尚未发布。发布前需按 README 查重、脱敏并经所有者确认。",
        "",
        "## 背景",
        "",
        f"- 项目别名：`{event.project_id}`",
        f"- 公共规则版本：`{event.rule_version}`",
        f"- 归属判断：`{analysis.attribution.value}`",
        f"- 症状：`{_safe_text(event.symptom[:1000])}`",
        "",
        "## 摘要",
        "",
        _safe_text(analysis.summary),
        "",
        "## 已观察事实",
        "",
    ]
    for fact in analysis.facts:
        lines.append(f"- {_safe_text(fact.text)} {_refs(fact.evidence_ids)}")
        for evidence_id in fact.evidence_ids:
            item = by_id.get(evidence_id)
            if item is not None and item.kind == EvidenceKind.LOCAL_FILE:
                lines.append(f"  - `{_evidence_label(item)}`: `{_safe_text(item.snippet[:300])}`")
    lines.extend(["", "## 假设与待确认", ""])
    for hypothesis in analysis.hypotheses:
        lines.append(f"- {_safe_text(hypothesis.text)}")
    if not analysis.hypotheses:
        lines.append("- 暂未形成假设")
    lines.extend(["", "## 建议验证（未执行）", ""])
    for check in analysis.suggested_checks:
        lines.append(f"- {_safe_text(check.description)}")
    lines.extend(["", "## 查重说明", ""])
    for candidate in analysis.duplicate_candidates:
        lines.append(
            f"- {candidate.source}: {_safe_text(candidate.reference)}（{_safe_text(candidate.similarity_reason)}）"
        )
    if not analysis.duplicate_candidates:
        lines.append("- 本次本地指纹与允许的远端查询范围内无候选；查询失败或截断时不能据此断言全局无重复。")
    for item in analysis.related_issues:
        lines.append(
            f"- 分层线索 `{item.layer.value}` {_safe_text(item.issue_ref)}：适用性 {item.applicability}"
        )
    lines.extend(["", "## 发布前检查", ""])
    lines.extend(
        [
            "- [ ] 已确认这是公共工作流问题，而不是业务项目缺陷",
            "- [ ] 已删除绝对路径、凭据、真实业务名称和未公开源码",
            "- [ ] 已用 fingerprint 和来源 event_id 查重",
            "- [ ] 发布后回读 URL，并记录状态",
            "",
        ]
    )
    return "\n".join(lines)


def render_lesson_candidate(*, event: Event, analysis: Analysis, run_id: str) -> Lesson | None:
    if analysis.attribution != Attribution.PUBLIC_WORKFLOW:
        return None
    if analysis.analysis_status == AnalysisStatus.FAILED:
        return None
    if not analysis.hypotheses and not analysis.suggested_checks:
        return None
    guidance_parts = [hypothesis.text for hypothesis in analysis.hypotheses]
    guidance_parts.extend(check.description for check in analysis.suggested_checks)
    guidance = "；".join(guidance_parts)[:2000]
    if not guidance:
        return None
    digest = hashlib.sha256(f"{run_id}\x00{event.symptom}\x00{guidance}".encode()).hexdigest()
    return Lesson(
        lesson_id=f"lesson-{digest[:12]}",
        status=LessonStatus.CANDIDATE,
        symptom_pattern=_safe_text(event.symptom[:1000]),
        applicability=(
            f"project_alias={event.project_id}; rule_version={event.rule_version}; "
            f"attribution={analysis.attribution.value}; not verified in this run"
        ),
        guidance=_safe_text(guidance),
        event_ids=[event.event_id],
        issue_urls=[],
        verification=None,
    )


def checks_to_json(checks: list[CheckResult]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "checks": [check.model_dump(mode="json") for check in checks],
        "summary": {
            "pass": sum(1 for check in checks if check.status == CheckStatus.PASS),
            "fail": sum(1 for check in checks if check.status == CheckStatus.FAIL),
            "error": sum(1 for check in checks if check.status == CheckStatus.ERROR),
            "not_run": sum(1 for check in checks if check.status == CheckStatus.NOT_RUN),
            "not_applicable": sum(1 for check in checks if check.status == CheckStatus.NOT_APPLICABLE),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _refs(ids: list[str]) -> str:
    return "、".join(f"[{item}]" for item in ids) if ids else "（无）"


def _paths(errors: list[str], related: list[str]) -> str:
    values = [*errors, *related]
    return "、".join(f"`{item}`" for item in values) if values else "（无）"
