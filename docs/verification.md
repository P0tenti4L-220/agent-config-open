# 公开发行验证

本文件只记录公开版自己的运行结果，不引用私人仓库的历史数字。

## 2026-11-03：上下文与快照续读同步

环境：macOS、Python 3.11.9。公开副本使用独立虚拟环境和锁定依赖；没有复制本机配置、凭据或真实运行数据。

| 实际执行 | 退出码 | 观察到的结果 |
|---|---|---|
| `uv sync --frozen --extra dev --offline` | 0 | 从本机缓存安装锁定的 40 个包，导入来自公开副本 |
| `python -m pytest -m 'not live' -q` | 0 | 222 passed、1 deselected；覆盖原文窗口、快照游标、历史展示证明和既有回归样例 |
| `agent-triage demo --output local/sync-demo` | 0 | 生成 report、context、handoff；`partial` 明确保留 GitHub 未配置的缺口 |
| `agent-triage doctor --profile offline --output local/sync-doctor` | 0 | `ready`；网络和真实模型调用仍为 `not_run` |
| `agent-triage eval --cases tests/cases --output local/sync-eval` | 0 | 62 项：54 pass、0 fail、0 error、8 not_run；整体 `partial` |
| `agent-triage accept --profile offline --evidence-dir <evaluation_dir>` | 0 | 整理本记录前的工作快照为 54/54 `current+pass` |
| `python3 scripts/check-public.py` | 0 | 检查 161 个索引内文本文件，无配置模式命中；另人工核对公开范围和文档差异 |
| `git diff --cached --check` | 0 | 暂存差异无空白错误 |
| 本地 Markdown 链接检查（Python） | 0 | 32 个本地文件链接均可解析 |

新增工具为 `read_evidence`，运行时为 `prompt-v3` / `tools-v2`；新展示证明为 v3。证据捕获、窗口和游标的边界见 [context-selection.md](context-selection.md)。原有对抗样例和公开版脱敏适配保留。

验收绑定运行时的 commit 和整个文件快照，修改本记录也会使旧结果失效。更新 `main` 前需对最终提交重新执行 `eval` / `accept`，完整证据保留在本机忽略目录，本表不能代替当前目标检查。以上为本地执行结果；本仓库没有自动 CI 门禁。

本轮真实模型 / GitHub MCP、人工质量评审、配对质量 / 成本 / 耗时实验、发行包构建及 Windows / Linux 运行均为 `not_run`。下方真实 DeepSeek 与构建记录属于首次公开发行，不是本轮重跑结果。

## 2026-10-28：首次公开发行（历史记录）

日期：2026-10-28。环境：macOS、Python 3.11.9；在独立虚拟环境中安装公开副本，没有复用私人项目的 Python 导入路径。

| 实际执行 | 退出码 | 观察到的结果 |
|---|---|---|
| `uv sync --frozen --extra dev` | 0 | 独立安装 40 个包，包括当前项目 |
| `uv lock --check --offline` | 0 | 锁文件与项目元数据一致 |
| `python -m pytest -m 'not live' -q` | 0 | 201 passed、1 deselected；包含本地假 MCP 协议测试 |
| `agent-triage demo --output local/demo` | 0 | 生成 report、context、handoff；`partial` 保留 GitHub 未配置的缺口 |
| `agent-triage doctor --profile offline --output local/doctor` | 0 | `ready`，确认导入来自公开副本；网络和模型调用仍为 `not_run` |
| `agent-triage eval --cases tests/cases --output local/eval` | 0 | 62 项：54 pass、0 fail、0 error、8 not_run；整体 `partial` |
| `uv build --offline --out-dir local/dist` | 0 | 构建 sdist 与 wheel |
| `python3 scripts/check-public.py` | 0 | 检查 155 个暂存文本文件，无配置模式命中；另做来源标识与文档链接检查 |
| DeepSeek 真实 `analyze`，使用 `examples/event.json` 与合成项目 | 0 | 3 次实际模型响应、5 次工具尝试，生成三份主要产物；`partial` 保留生产端契约及远端历史等缺口 |

真实模型检查只发送合成示例，凭据只存在于进程环境，没有复制进公开目录或提交。人工阅读了这次报告，确认它指出 `items` / `checks` 字段差异，并保留生产端约定未知；这不构成五例人工质量验收，`claim-support` 仍为 `not_run`。

`eval` 执行成功与 `accept` 对当前目标的判定是两件事。整理文件会改变目标摘要；发布前应在最终 commit 上重新执行 `eval` / `accept`，其完整证据保存在本机忽略目录，不能用文档里的历史数字替代。

## 未完成项

- 真实模型 + GitHub MCP 的完整链路：`not_run`；本次真实模型检查没有启用 MCP。
- 五例人工质量评分、三类 playbook 人工走查、无聊天历史接手走查：`not_run`。
- Anthropic 真实 API 调用：`not_run`；其 provider 仅有离线契约测试。
- GitHub Actions、分支保护：`not_run`，本仓库未配置。
- Windows / Linux 与其他 Python 版本运行：`not_run`。

这些结果证明安装、合成流程和指定模型的一次实际分析可执行，不证明普遍诊断正确率、成本或性能收益。
