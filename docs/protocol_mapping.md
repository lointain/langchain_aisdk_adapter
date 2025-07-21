---

# LangChain/LangGraph 事件到 AI SDK 协议映射

本文档概述了从 LangChain/LangGraph 事件（`astream_events` v2）到 Vercel AI SDK UI 流协议的明确映射。此映射作为适配器的核心逻辑。

---

## 核心原则

1.  **单一职责**：每个协议都有一个清晰且独特的目的。
2.  **LangGraph 生命周期**：映射完全支持 LangGraph 的基于节点的生命周期，使用 `b:` 和 `c:` 定义每个步骤的清晰边界。
3.  **AI SDK "步骤" 定义**：AI SDK 上下文中的“步骤”被定义为一个完整的交互单元，通常涉及工具使用。简单的 LLM 调用不构成一个步骤。一个步骤由 `b: (步骤开始协议)` 和 `c: (步骤结束协议)` 协议包围。
4.  **完成信号**：`f: (完成原因协议)` 仅在整个链/图执行的末尾发送一次。
5.  **手动控制**：所有协议都可以手动调用，其中 `g: (推理协议)` 仅限手动调用。

---

## 事件到协议映射表

| LangChain/LangGraph 事件 | AI SDK 协议 | 数据 (`JSON.stringify(value)`) | 发送方 | 备注 |
| :----------------------- | :--------------------- | :---------------------------------------------------------------------------------------------------------------- | :--------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `on_chain_start` | `2: (数据协议)` | `{ "type": "chain_start", "name": event.name, ...metadata }` | 自动 | 标记整个过程的开始。传输元数据。 |
| `on_node_start` | `b: (步骤开始协议)` | `{ "name": event.name, "node_id": event.run_id, ...metadata }` | 自动 (LangGraph) | 标记 LangGraph 节点执行的开始，对应于 AI SDK 的“步骤”。 |
| `on_agent_action` | `a: (工具调用协议)` | `{ "toolCallId": run_id, "toolName": action.tool, "args": action.tool_input }` | 自动 | 信号代理已决定调用工具。创建一个工具调用部分。 |
| `on_tool_end` | `9: (工具结果协议)` | `{ "toolCallId": run_id, "result": output.result }` | 自动 | 提供工具执行的结果。更新相应的工具调用部分。 |
| `on_node_end` | `c: (步骤结束协议)` | `{ "node_id": event.run_id }` | 自动 (LangGraph) | 标记 LangGraph 节点执行的结束。 |
| `on_chain_end` | `f: (完成原因协议)` | `{ "finishReason": "stop" \| "length" \| ..., "usage": { ... } }` | 自动 | 标记整个过程的结束。仅针对根链/图发送。 |
| `on_chat_model_stream` | `0: (文本协议)` | `delta_string` | 自动 | 流式传输语言模型的文本增量。 |
| `on_chain_error` / `on_tool_error` / `on_node_error` | `e: (错误协议)` | `error_message_string` | 自动 | 报告执行过程中发生的任何错误。 |
| (手动调用) | `g: (推理协议)` | `reasoning_string` | 手动 | 用于手动提供推理或思考过程。不与特定事件绑定。 |