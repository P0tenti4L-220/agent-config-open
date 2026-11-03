# 下一次开发会话入口

公开版是独立的诊断 Agent 项目。请从 README 的离线演示开始，再查看 `docs/agent-architecture.md` 与 `docs/verification.md`。不要依赖私人仓库、原始聊天、个人全局规则或任何机器绝对路径。

## 当前组织

- 运行包仍叫 `agent-triage`，Python 模块为 `agent_triage`，仓库名为 `agent-config-open`。
- 保留 Agent、恢复、证据 / 展示检查、经验、评测和人工复核契约，以及全部合成测试与失败样例。
- 文档面向新用户重写。个人工作规则、备份恢复工具、历史设计与运行统计不属于本发行。
- `local/` 由使用者自己创建，不随 clone 分发。

## 本次同步（2026-11-03）

已同步精确原文窗口、关键调查状态和 `read_evidence` 快照续读，以及相应合成回归样例。该轮工具数为七个；运行时使用 `prompt-v3` / `tools-v2`，新展示证明为 v3，旧 v2 只按旧契约读取。详见 [context-selection.md](context-selection.md) 与本轮 [verification.md](verification.md)。

公开范围仍为诊断 Agent、合成测试和通用文档；未导入个人规则、备份工具、配置或运行数据。本次没有执行真实模型、真实 GitHub MCP 或人工质量实验，均为 `not_run`。

## 本次同步（2026-11-03）：分层 Issue 检索

新增 `project_profile.py`、`github_scope.py`、`issue_research.py`，并由 Runner 在每次模型请求前执行程序计划：

- 项目画像有界读取 `pyproject.toml`、`uv.lock`、`requirements*.txt` 和本地 `.git/config`；声明版本、锁定版本与未知状态分开记录，不执行项目代码。
- 搜索范围由配置上限和控制器决定；模型查询拒绝 `repo:`、`org:`、`is:`、布尔运算、否定和引号。依赖层只使用配置中的“包 → 上游仓库”映射。
- 候选身份从响应中的仓库字段、API/HTML URL 或 `issue_ref` 交叉核对；跨仓库后保留真实 `owner/repo#number`，不再用 `mcp.repo` 拼接。
- 搜索回执绑定 run、层级、范围、查询 hash、调用 id、证据快照与候选列表，写入 checkpoint；`issue-research.json` 记录逐层过程。
- 新工具 `record_issue_assessment` 记录六个维度的 `match`/`mismatch`/`unknown`；程序校验候选已读、证据已展示后提交，`component`/`symptom` 匹配且无冲突才停止扩大。
- 最终 `analysis.related_issues` 与报告中的“分层 Issue 检索”保留层级和适用性；适用线索不升级为已确认根因。

运行时版本为 `policy-v2` / `prompt-v4` / `tools-v3`；`Config v1` 默认仍是单仓库 legacy 行为，layered 模式要求 `schema_version = 2`。真实模型 / MCP 分层联调为 `not_run`。

## 接下来优先处理

1. 用自己的配置分析一个有已知答案的小故障，独立核对事实、证据引用和下一步建议；收集真实使用缺陷。
2. 对需要 GitHub 查询的场景，单独执行指定模型 + MCP 版本 + 合成 issue 的完整 live smoke，保留本机证据。
3. 五例真实人工质量评审与无聊天历史接手走查仍为 `not_run`；不要用离线测试数量替代。
4. 评估加入 CI 的离线测试与发布内容检查。当前没有配置 GitHub Actions 或分支保护，记为 `not_run`。
5. 按需求实测 Windows / Linux 和其他 Python 版本；文档提供路径写法，但跨平台运行仍须实际确认。
6. 继续缩小提示注入与敏感材料进入模型请求的风险。当前模式拦截不是完整的数据防泄漏系统。

涉及产品行为的修复补有意义的回归测试，保留既有失败样本。提交前检查暂存差异和公开内容；报告实际命令、退出码和未执行项。
