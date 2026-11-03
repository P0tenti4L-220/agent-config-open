# 原文窗口、关键状态与快照续读

2026-11-03。本文描述公开版的原文窗口、关键调查状态与已捕获快照续读。功能只作用于诊断 Agent 的运行，不修改业务项目源码。

## 选择原文

`windows.py` 用当前问题、症状与所缺材料生成有界词法查询。英文匹配不改变返回文本；中文保留短语并补充二字串。证据依次按反证、required、关联路径、辅助材料分档；档内优先覆盖不同来源与新增查询词，再比较相关性、采集时间与稳定 ID。

每条材料最多两个窗口。先分配来源份额，避免长片段占满证据层；窗口围绕命中位置保留邻接行，重叠范围合并。无命中使用标有 `fallback_preview` 的首尾预览。已保留在工具协议中的完整片段不重复写进证据层。匹配是启发式，不能保证找全语义反证；显式反证优先级始终保留。

`EvidenceWindow` 记录来源 hash、快照 hash、原始表示、UTF-8 字节区间 `[start,end)`、行号、文本 hash 与选择原因。文本须逐字节等于原始切片。当前自动选择只扫描已入库 snippet；更远的捕获内容经 `read_evidence` 读回成新证据后参与下一轮选择。部分 JSON 只作为文本窗口展示。

必要材料部分展示、完全未展示或过期时，`OmissionImpact` 分别保留 `partially_displayed`、`stored_not_displayed`、`stale`。未捕获内容不能从摘要推断。全部来源指针保存在 `context.json:evidence_index`，窗口与遗漏分开保存。

## 保留关键状态

`state_projection.py` 从事件、调查状态、trace 与预算确定性生成短视图，没有额外摘要模型调用。`ContextPacket.failed_attempts` 复用现有失败记录，新增 call ID、脱敏参数指纹、trace 来源与可确定的重试条件；无依据的条件为 `unknown`。成功返回而没有新证据的尝试单独保留。

失败按与当前问题的相关性选择，同等相关时保留较早的未决项。完整失败列表仍在 context 中，较晚的工具成功不会自动解决旧失败。关键状态先获得 working-state 预算，余量才用于完整的调用与结果对。放不下的明细保留计数和文件字段引用，状态行不会在中途截断；最小状态都放不下时明确返回预算错误。`not_run` 不会从原文自述中升级。

业务 `handoff.md` 使用同一投影，包含失败来源、无进展计数、未执行项与遗漏引用。独立开发交接模型 `DevelopmentHandoff` 本次未扩展；它不是业务失败记录的事实源。

## 归档与续读

归档对象在当前运行目录的 `evidence/objects/<snapshot_sha256>.txt`，引用登记在 `evidence/index.json`。对象先原子写入，索引最后提交；写入失败返回 `archive_failed`，不发布可恢复指针，缓存中也不登记未提交对象。

| 工具来源 | 捕获范围与完整性 |
|---|---|
| `read_file` | 本次已通过原有访问检查、实际读取的 UTF-8 文件，仍受 2 MiB 上限约束；不规范化换行 |
| `search_text`（内部 `search_workspace`） | 仅选中的匹配行，预览围绕命中；不归档扫描过的其他行或文件 |
| GitHub | backend 本次实际返回的内容，保留 `comments_error`；评论分页完整性未知，不声明全量 |
| Git | 只归档子进程实际捕获的有界输出；上游截断仍为不完整 |
| 旧运行 | 只有 snippet 就只读取该 snippet，`capture_scope=snippet`，不补造历史正文 |

第七个工具 `read_evidence` 的接口为：

```text
read_evidence(evidence_id, cursor?, max_bytes=8192)
```

它只访问当前 run 中已登记的快照，不接受路径或 URL，不发远端请求。首次读取关联的初始片段；`next_cursor` 继续同一快照。初始片段前还有已捕获内容时，`snapshot_start_cursor` 可读回快照开头。游标绑定 run、证据身份、来源版本、对象 hash 与 UTF-8 字节位置。窗口中的 `read_cursor` 可定位其原始位置；超长单行不会跳到下一行而丢掉行内内容。

新范围产生绑定来源版本、区间与片段 hash 的证据记录；重复范围去重，同文但不同位置或来源不合并。调用经原有 `ToolExecutor` 计入次数、时间与 trace，重复页不会制造新进展。快照缺失或损坏明确报错；本地源变化时仍读取历史快照并标 `stale`，不会拼接当前源。当前访问边界继续适用。

完整工具结果（含 JSON、元数据和游标）限制为 8 KiB。装不下时以较小的精确窗口替代结构化正文，保留未展开字段与来源 ID 的说明及真实续读操作。Runner 不再二次截断 JSON。单窗口最多 8,000 UTF-8 字节，同时满足旧证明的 8,000 字符上限。

## 展示与版本

新捕获必须提供显式窗口清单；仅有指针不会产生展示记录。`RequestViewStore` 验证切片、位置、hash 以及最终请求中实际存在的正文；收到响应之后才确认展示。两个不连续窗口分别记录，不把中间区域算作已读。历史 v2 请求和展示按旧契约读取，不补造字节范围。

- 运行时提示词：`prompt-v4`；工具：`tools-v3`，八个受限工具（新增 `record_issue_assessment`，只提交适用性判断，不新增 MCP 权限）。
- Evidence 与证据索引：v2；ContextPacket：v2；调查账本新增字段为 v2。
- RequestView 与新窗口展示：v3；窗口契约自身为 v1。
- 旧运行读取兼容，不允许用新运行时版本悄悄续跑旧检查点，沿用现有版本校验。

八个 schema 实测约 1,572 estimated tokens。默认工具层从 1,000 调至 1,600，证据层从 5,000 调至 4,800，总输入预算从 12,000 调至 12,400。输入计量包含实际 system、保留消息的 JSON 转义和单独发送的工具 schema；不在 system 中重复完整 schema。预算错误不会靠最终字符串截断隐藏。

## 验证与边界

新行为测试位于 `tests/test_context_selection.py`、`tests/test_evidence_capture.py`；原有上下文、展示、Runner、provider、复核和 bundle 测试同步覆盖新契约。测试包含实际请求中的中部异常及条件、分离窗口、指针无证明、反证遗漏、旧失败、中文单行续读、跨 run/对象/版本游标、重复页、历史快照、写入失败、上游部分评论和旧 v2 读取。

```sh
rtk proxy .venv/bin/python -m pytest tests/test_context_selection.py tests/test_evidence_capture.py tests/test_context.py tests/test_evidence_tools.py tests/test_request_views.py tests/test_runner.py -q
rtk proxy .venv/bin/python -m pytest -m 'not live' -q
rtk proxy .venv/bin/python -m agent_triage demo --output local/context-selection-smoke
```

测试和回放可验证选择、范围和状态契约。真实模型与搜索的配对质量/成本实验为 `not_run`，不能据此声称诊断质量、token 成本或耗时改善。未引入向量库、FTS5、摘要模型、自动重置或自动执行修复。仓库没有自动 CI 门禁；公开版的实际验证结果见 [verification.md](verification.md)。
