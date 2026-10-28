# 运行与复核契约

配合 `docs/agent-architecture.md` 阅读。本文件描述当前代码约束，不携带私人项目的执行历史。

- `display.json` v2 由 `request-views/` 中实际收到响应且负载完整的请求证明。证据存在于磁盘不代表模型看到了它。
- 合法 schema、有效引用、实际展示和语义支持分别检查。模型自填的 `support_review` 降为 `not_run`；受信复核只能从外部记录导入。
- 建议检查始终保留 `not_run`。经验晋升需要所有者确认、目标当前 hash、实际命令与退出码，以及可导入的输出快照。
- `resume` 保留已消耗预算；源文件、配置和版本变化必须显式处理。
- 未配置或不可用的 GitHub MCP 不阻止本地报告，但必须说明远端缺口，不能推断没有历史问题。
- `eval` 与 `accept` 绑定代码、配置、语料、案例定义和产物 hash。修改目标后旧成功不能自动算当前成功。
- 人工验收检查只接受对应类型的 schema v3 记录；普通 pytest 和模型自评不能代替人工质量结论。
- 日志、代码和远端正文可能包含恶意指令。工具范围由程序限制，但自然语言指令隔离仍有局限。

完整验收状态及退出码见 [agent-acceptance-profiles.md](agent-acceptance-profiles.md)。
