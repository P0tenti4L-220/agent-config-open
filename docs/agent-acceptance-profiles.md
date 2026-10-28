# 验收 profile 与退出码

`eval` 只负责执行并保存证据；`accept` 只读取证据并对**当前目标**做判定。两者分开，避免“测试跑过”被当成“功能已验收”。

## Profile

| Profile | 继承 | 额外必需证据 | 通过后允许的说法 |
|---|---|---|---|
| `offline` | — | 当前版本 AC01–AC14、NC01–NC26（含 NC20.routing）、HC01–HC08、HC11、HC12、HC14–HC17 | 当前版本的离线契约、回放、本地 MCP 适配和开发交接检查可用 |
| `connected` | offline | AC15.live、HC09 | 指定模型/配置/仓库的真实只读链路通过；不推出诊断普遍正确 |
| `reviewed` | connected | AC15.quality、NC20.manual、HC10、HC13 | 这组冻结小样本达到 MVP 试点标准；不推出生产可靠性 |

父项 `AC15`、`NC20` 只聚合子项，不自己执行命令。旧 v1 结果没有目标来源，读取时标为 `unknown`，不能计入任何 profile。

检查数量：offline 54 项、connected 56 项、reviewed 60 项。执行结果必须绑定当前公开版的代码和配置；本仓库不附带个人运行记录。

## 两轴状态

- 执行状态：`pass / fail / error / not_run / not_applicable`，保留原义。
- 对当前目标是否有效：`current / stale / unknown`。

旧 pass 在代码、配置、语料、检查定义或产物变化后仍是历史 pass，但 `effective_state=stale`，不能计入当前通过。`required` 项不能用 `not_applicable` 绕过。

live 项（AC15.live、HC09）还要求 `live_config_hash`：换模型、换 provider 或换 MCP 配置后，旧 live 结果变 `unknown`，必须在同一配置下重跑。

## 聚合与退出码

1. 任一必需项 `error` → `overall=error`，退出 4。
2. 否则任一必需项 `fail` → `fail`，退出 1。
3. 否则任一项缺失、`not_run`、`stale`、`unknown`、`not_applicable` → `incomplete`，退出 2。
4. 全部必需项 `current + pass` → `pass`，退出 0。

中断按现有约定退出 130。同一检查的多次尝试全部保留；当前失败不会被另一个目标上的旧成功覆盖。

## 来源绑定

每个结果记录：代码 commit、相关树摘要、工作树是否 dirty、配置 hash、policy/prompt/playbook/tool schema 版本、语料摘要、检查定义 hash、解释器版本、实际 argv、退出码、产物相对路径与 sha256。

`accept` 不扫描全盘、不运行测试、不联网。产物必须解析到给定证据目录内；越界路径、符号链接逃逸、缺失快照或 hash 不符按 `error` 处理。

检查来源必须匹配：pytest 检查只接受 `source_type=pytest`；人工检查只接受 `source_type=human_review`；HC09 只接受 `source_type=run_artifact`。来源伪配按 `error` 处理，不会因为退出码为 0 而变成 pass。

案例定义单独绑定：每个结果带 `case_definition_hash`，`accept` 用当前 `tests/cases` 重算。节点、pytest 列表或 expectation 变化即 `stale`；缺 hash 或当前 manifest 中无该案例即 `unknown`。因此替换 AC01 背后的测试不会继续算当前 AC01 pass。

## 人工记录

`connected` / `reviewed` 需要的真实记录必须来自外部。运行级语义复核用 `agent-triage record-review`；验收级人工项用 schema 3，放在证据目录的 `human/<check_id>.json`。两者不是同一个入口，不能互换。

schema 3 不接受“只填合法 hash 字符串”的记录。质量项必须绑定：

- 根 `eval.json` 的 `evaluation_id` 与 `target`；
- 冻结数据集 `dataset_ref` / `dataset_digest` / `dataset_id`，以及 `rubric_ref` / `rubric_digest`；
- 每个 QA 案例的 `bundle_dir`、完整 `artifacts` allowlist、`roles`（event/run/analysis/display/reference）和 `reviewed_subject_hashes`；
- `run.json` 的 `run_id`、provider、config fingerprint 与声明一致；
- `analysis.json` 的原始字节 hash 与 `analysis_content` 语义 hash 分开记录；
- display 记录必须能由 bundle 内的 request-views 与 evidence index 重算。

推荐流程：

```bash
# 1. 把真实运行目录按 <case_id>/<run_id>/ 放好；dataset 用冻结 manifest
agent-triage acceptance-review prepare \
  --evidence-dir local/eval/<evaluation_id> \
  --check AC15.quality \
  --dataset local/quality/pilot-v1 \
  --subjects local/reviews/runs \
  --output local/reviews/AC15.quality.template.json

# 2. 人工填写 reviewer_id / reviewed_at / rubric_version / verdict / rationale；
#    verdict 留空或 pass 与未解决 blocker 并存都会被拒绝/记为 fail。

# 3. 导入时重算全部哈希、allowlist、run/config/case 绑定
agent-triage acceptance-review import \
  --evidence-dir local/eval/<evaluation_id> \
  --review local/reviews/AC15.quality.completed.json
```

`prepare` 只写只读复核包与机器字段，不写 `human/<check>.json`，因此准备完成不等于已验收。`import` 会保留原始记录副本与 receipt；`accept` 每次都会重新核验，不只信任导入时的检查。跨评测记录必须经 `import` 显式导入并保留 `source_evaluation_id`；直接把外来 `evaluation_id` 的记录放进证据目录会按 `error` 处理。

人工判定规则保持原标准：QA01–QA05 恰好五例、至少 4 个 pass 且三类严重 flag 为零；三类 playbook 全部 pass；HC13 识别目标/事实/缺口/下一步。`pass` 与未解决 blocker 并存记为 `fail`；schema v1/v2 记录只作历史，状态为 `unknown`，不能计入当前通过。没有记录时保持 `not_run`。合成记录的单测只证明导入契约，不证明真实人工质量。

## HC09 的 run_artifact 导入

live smoke 实际运行成功后写出 bundle（`manifest.json`、`chain.json`、允许的 run 产物、证据与 request-views）。`eval --include-live` 或 `agent-triage import-live` 会：

1. 校验 bundle 文件清单、路径、大小与 sha256，拒绝未登记文件；
2. 重算 issue 链路：搜索命中指定 issue、唯一读取、`issue_ref` 一致、`evidence_refs` 指向 issue 证据、请求视图中出现 call id 与 result id、证据片段确实进入收到响应的请求、display hash 匹配、claim 引用该证据；
3. 校验 bundle 的 `run_id` 与 `run.json` 一致；
4. 用当前 `tests/cases` 的 HC09 定义计算 `case_definition_hash`，写入 `results/HC09-<attempt>.json`，`source_type=run_artifact`，并绑定 bundle 的 target。

`run_artifact` 的 pass 还要求非空 `related_run_ids` 与可校验的 `artifacts`；因此只写一个 `source_type=run_artifact`、`exit_code=0` 的空结果不会通过。pytest 退出码 0、HC09 的占位 pass 或没有 request-view 的旧 bundle 都不能替代这一步。