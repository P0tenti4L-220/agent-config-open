# 本地质量实验层（M1/M2）

这是诊断质量实验的本地实现，不是人工验收，也不是平台接入。它复用现有 `run_new_analysis`、六工具、只读 MCP 和 `eval`/`accept` 语义；不引入 LangGraph、向量库或第二套 Agent 执行器。

## 能回答什么

- 同一冻结数据集上，每个 trial 是否完成、失败在哪个阶段；
- 诊断文本的事实/假设数量和“引用证据的事实比例”等确定性指标；
- 工具调用次数、工具错误次数、trial 耗时；
- 同一 variant 重复运行是否稳定；
- 两个版本在相同数据集上逐例配对比较（仅确定性执行指标）。

**不能**回答“诊断是否语义正确”“根因是否真的解决”“成本是否更低”。这些需要人工评分或独立 judge；当前版本没有实现 judge 和 Langfuse 导出。

## 数据集

复用 `quality.DatasetManifest`（schema 1）：

```json
{
  "schema_version": 1,
  "dataset_id": "pilot-v1",
  "revision": "1",
  "cases": [
    {
      "case_id": "QA01",
      "group_id": "g1",
      "split": "pilot",
      "category": "structured-output-mismatch",
      "playbook_id": "structured-output-mismatch",
      "input_ref": {"relative_path": "inputs/QA01/event.json", "sha256": "...", "size_bytes": 123, "media_type": "application/json"},
      "fixture_refs": [{"relative_path": "fixtures/QA01/logs/failure.log", "sha256": "...", "size_bytes": 456, "media_type": "text/plain"}],
      "reference_ref": {"relative_path": "references/QA01.md", "sha256": "...", "size_bytes": 789, "media_type": "text/markdown"},
      "rubric_ref": {"relative_path": "rubrics/QA01.json", "sha256": "...", "size_bytes": 100, "media_type": "application/json"},
      "replay_ref": {"relative_path": "replay/QA01.json", "sha256": "...", "size_bytes": 200, "media_type": "application/json"},
      "origin": "synthetic",
      "source_run_ref": null
    }
  ],
  "digest": "..."
}
```

- 所有 ref 按原始字节 SHA-256 校验；manifest digest 覆盖除 `digest` 外的规范 JSON；符号链接、越界路径、超限文件、重复 case_id 均拒绝。
- `fixtures/<case_id>/` 下的文件按相对路径复制进 trial workspace；被测 Agent 不能读取 `references/`、`rubrics/` 或其他 trial。
- QA01–QA05 的真实材料由使用者提供并确认；合成样例只验证流程。

## 命令

```bash
# 冻结数据集，预登记每个 (case, repeat) slot；先落 spec/plan，再执行
agent-triage experiment plan \
  --dataset local/quality/pilot-v1 \
  --output local/experiments \
  --variant baseline \
  --repetitions 3 \
  --mode replay

# 逐 trial 执行；每个 trial 使用独立 output_root/registry/lessons
agent-triage experiment run \
  --experiment-dir local/experiments/<exp-id> \
  --mode replay

# 确定性评分与本地 JSON/Markdown 报告
agent-triage experiment score --experiment-dir local/experiments/<exp-id>

# 两个完成实验的逐例配对比较（report-only）
agent-triage experiment compare \
  --baseline local/experiments/<base-id> \
  --candidate local/experiments/<cand-id>
```

live 模式必须显式 `--run-live --config`；replay 模式不会启动真实 MCP/judge/平台，也不会因为环境里有凭据而联网。

## 退出码

- `plan`：0=已冻结并登记；4=数据集/配置错误。
- `run`：0=全部 trial 执行成功；1=存在失败分析；2=存在 error/not_run；4=计划/模式/配置错误。
- `score`：0=评分完成且全部 trial 执行成功；1=存在失败 trial；2=仍有 error/not_run；4=证据损坏。评分成功不等于人工诊断质量通过。
- `compare`：0=可比并已生成报告；2=数据集、重复数、provider 模式或 evaluator manifest 不同，拒绝硬比较。

## 状态与分母

- 每个 trial 先写 `trial.json`，状态 `not_run`、原因 `registered`；执行后更新。
- `analysis_status=failed` 写为 trial `fail`（`reason_code=analysis_failed`），保留 run、产物和失败原因；没有生成 analysis 时为 `error`。`complete/partial` 可记执行 `pass`，仍需另外判断诊断质量。
- 已完成的 trial 不会被重跑覆盖；再次 `run` 会创建 `attempt-2` 并保留 `attempt-1`。
- 缺失/错误 trial 在报告中保留分母；均值只覆盖有值的 trial，并显示 `n` 与 `missing`。
- 指标 `execution_status=not_run/error` 时 `value=null`；0 只表示实际观察到 0。
- `code.claims_with_evidence_ratio` 在没有 fact 时不填 1.0，保持 `not_run` 并注明 `n=0`。

失败分析必须保留为失败 trial；测试同时覆盖失败回归样例与在引用前实际读取数据的正常回放。

## 当前限制

- `compare` 只报告确定性执行指标，不做发布判定；`release_decision=not_evaluated`。没有 policy 阈值、统计显著性或成本结论。
- 没有 LLM judge、人工评分回流、Langfuse/LangSmith 导出。
- 没有并发；`max_concurrency=1`，所有 trial 顺序执行。
- live 实验需要另行执行并记录；公开发行不包含任何私人实验数据。
