# 使用说明

## 离线演示与真实模型

`agent-triage demo --output local/demo` 自动创建合成项目、按预设响应读取材料并输出报告。默认无 MCP，因此分析可能为 `partial`，说明未检查远端；这不是进程失败。

`analyze` 使用真实模型时，复制 `examples/deepseek.toml` 或 `examples/anthropic.toml` 到 `local/config.toml`，在进程环境设置对应 Key，再传 `--config local/config.toml`。Anthropic 样板中的模型 ID 是占位符，必须替换。DeepSeek 样板的模型名是项目已适配的配置值，账号是否支持以服务商为准。

```bash
.venv/bin/agent-triage analyze \
  --event examples/event.json \
  --project-root local/demo/project \
  --config local/config.toml --output local/runs
```

上例先运行 `demo` 生成相应文件。要分析自己的项目，用 `--symptom-file` 替代 `--event`，按 README 指定项目根、项目别名与目标。JSON 事件可额外指定 `related_files`。

真实模型会收到选中的项目片段；API 计费按自己的服务商账号发生。Replay 只验证程序链路。

## 可选 GitHub MCP

从 [GitHub 官方 MCP server](https://github.com/github/github-mcp-server) 安装服务端；适配器面向 v1.8.0 的 `search_issues` / `issue_read` schema。更新服务端后先验证工具发现和返回格式。

复制 `examples/github-mcp.toml` 到 `local/config.toml`，填写可执行路径与 `owner/repo`。Token 放 `GITHUB_PERSONAL_ACCESS_TOKEN`，使用只读且范围最小的权限。服务端按 `--read-only` 和两个允许工具启动；Agent 没有 issue 写入工具。

没有配置、凭据不可用或服务启动失败时，报告保留具体缺口。搜索空结果不等于不存在历史问题。

### 分层 Issue 检索

复制 `examples/github-layered.toml` 后填入 `mcp.repo` 和依赖映射，并保留 `schema_version = 2`。默认的 `mode = "legacy"` 只查当前仓库；`mode = "layered"` 由程序按以下顺序推进：

1. 当前项目仓库：`repo:<mcp.repo>`；
2. 配置中映射、且症状里出现包名或 import 名的依赖上游：`repo:<映射仓库>`；
3. 配置允许时，公共层：`is:public`。

每层先用有界查询搜索，再读取排名靠前的候选正文，再由模型用 `record_issue_assessment` 逐项记录 `component`、`symptom`、`version`、`environment`、`trigger`、`fix` 的 `match`/`mismatch`/`unknown`。程序只接受本次运行搜索回执中出现、已读取且证据已展示的候选；`component`/`symptom` 匹配且没有已知冲突才停止扩大范围。没有评估的候选按“未核对”处理，继续下一层并保留缺口。

依赖仓库只来自配置，不从日志或 Issue 正文推断。`max_searches`、`max_issue_reads`、`max_remote_requests` 是整个运行的上限，与现有工具/模型/时间预算叠加；恢复运行保留层级、候选回执和已用次数。搜索失败、截断或评论未读完都保留为覆盖不足，不会被解释成“没有相关问题”。

## 报告与恢复

终端 JSON 的 `report_path`、`context_path`、`handoff_path` 都是实际产物路径。

| 状态 | 含义 |
|---|---|
| `complete` | 分析完成；根因仍须独立验证 |
| `partial` | 已生成分析，仍有列出的证据缺口 |
| `failed` | 分析失败；进程退出码 1 |

```bash
.venv/bin/agent-triage list --output local/runs
.venv/bin/agent-triage resume --run-id RUN_ID --output local/runs
.venv/bin/agent-triage validate --run-id RUN_ID --output local/runs
.venv/bin/agent-triage policy
```

配置 / 输入 / 版本变化可能拒绝恢复。确认接受源文件和配置变化后才使用 `--accept-changes`；版本变化通常需要新运行。

`validate` 是产物检查，不会执行报告建议的业务命令。`claim-support=not_run` 表示没有独立语义复核。

## 真实联调测试

默认 pytest 跳过 `live`；显式运行需 `--run-live`，以及模型 Key、GitHub Token、配置和指定测试 issue：

```bash
export AGENT_TRIAGE_EVAL_CONFIG=local/config.toml
export AGENT_TRIAGE_TEST_ISSUE='https://github.com/YOUR_OWNER/YOUR_REPO/issues/1'
export AGENT_TRIAGE_ISSUE_MARKER='YOUR_SYNTHETIC_ISSUE_MARKER'
.venv/bin/python -m pytest --run-live tests/integration/test_live_smoke.py -q
```

测试读取现有合成 issue，不会创建或修改 issue。模型仅本地分析成功，不等于此模型 + GitHub 链路已通过。验收与实验用法见 [验收 profile](agent-acceptance-profiles.md) 和 [质量实验](quality-experiments.md)。
