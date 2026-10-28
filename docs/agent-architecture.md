# 诊断 Agent 架构

项目把一次开发失败转换成有证据来源的诊断、上下文包和交接文件。它使用一个模型执行循环；不依赖多 Agent 框架、向量数据库或个人全局配置。

## 执行路径

1. `cli.py` / `ingest.py` 校验事件、日志与项目根。
2. `pipeline.py` 组装模型、可选 MCP、证据与本地存储。
3. `runner.py` 执行有次数和时间预算的模型 / 工具循环。
4. `context.py` 优先保留当前问题、必要证据与反证，并记录省略影响。
5. `claims.py`、`request_views.py`、`lifecycle.py` 检查结构、引用、展示与运行约束。
6. `render.py` 输出报告、上下文与交接材料。

## 模块地图

| 模块 | 责任 |
|---|---|
| `contracts.py`、`config.py` | Pydantic 严格契约、配置和非凭据指纹 |
| `models.py` | Replay、Anthropic、DeepSeek provider |
| `tools.py`、`mcp_client.py` | 六个受限工具与只读 GitHub 协议适配 |
| `policy.py`、`playbooks.py` | 固定规则与三类诊断 playbook |
| `budget.py`、`runner.py` | 请求 / 工具 / 时间预算、重复调用与恢复 |
| `store.py`、`feedback.py`、`review.py` | 运行存储、经验反馈和外部复核 |
| `eval.py`、`acceptance.py`、`case_manifest.py` | 执行案例、证据来源、当前目标验收 |
| `livecheck.py`、`live_bundle.py` | 搜索到读取再到引用的 live 证据链 |
| `quality.py`、`acceptance_review.py`、`experiments/` | 冻结数据集、人工记录和本地实验 |
| `doctor.py`、`handoff.py` | 环境准备度和开发交接检查 |

## 边界

模型仅能使用 `read_file`、`search_text`、`git_status`、`search_lessons`、`github_search_issues`、`github_read_issue`。参数校验和根目录约束由程序实施；模型不能添加任意工具或执行 shell。GitHub 仅查询显式配置的仓库。

日志、源码、issue 与历史经验都作为不可信材料。当前实现包含路径与凭据模式拦截，但证据正文仍进入模型上下文；不能据此声称已彻底隔离提示注入或所有隐私泄漏。

`complete` / `partial` 是分析状态；建议检查、根因复现、语义支持和人工质量评审各自保留独立状态。运行时检查细节见 [runtime-contracts.md](runtime-contracts.md)。

## 启动与验证

最小启动使用 `agent-triage demo`；真实模型配置和结果阅读见 [usage.md](usage.md)。`doctor` 只检查本地环境，不调用模型；`eval` 执行案例；`accept` 只读取已有证据。能力声明位于 `docs/agent-capabilities.json`，声明本身不是验证结果。

公开版不携带个人运行目录或历史验收结果。当前发行的实际验证范围见 [verification.md](verification.md)；未完成的人工质量与跨平台验证保留为 `not_run`。
