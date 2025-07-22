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


## 事件到协议映射表 (基于图片内容)

| LangChain/LangGraph 事件 (`astream_events` v2) | AI SDK 协议          | 协议负载/用途说明                                                               |
| :--------------------------------------------- | :------------------- | :---------------------------------------------------------------------------- |
| `on_chain_start`                               | `2: (Data Protocol)` | 流程开始: 标记整个 LangChain 或 LangGraph 流程的开始。负载: `{"custom_type": "chain_start", "run_id": "..."}` |
| `on_node_start (LangGraph)`                    | `2: (Data Protocol)` | 节点开始: 标记 LangGraph 中一个节点的开始。负载: `{"custom_type": "node_start", "node_name": "..."}` |
| `on_agent_action`                              | `f: (Step Start)`    | 步骤开始: 标记一个 Agent 推理步骤的开始。负载可包含步骤的元数据。               |
| `on_agent_action`                              | `9: (Tool Call Protocol)` | 工具调用在 b: 步骤内部，指明要调用的工具和参数。                               |
| `on_tool_end`                                  | `a: (Tool Result Protocol)` | 工具结果在 b: 步骤内部，返回工具执行的结果。                                   |
| `on_tool_end`                                  | `e: (Step End)`      | 步骤结束: 标记一个 Agent 推理步骤的结束。与 b: 成对出现。                       |
| `on_node_end (LangGraph)`                      | `2: (Data Protocol)` | 节点结束: 标记 LangGraph 中一个节点的结束。负载: `{"custom_type": "node_end", "node_name": "..."}` |
| `on_chain_end`                                 | `2: (Data Protocol)` | 流程结束: 标记整个流程的结束，并提供结束原因。                               |
| `on_chat_model_stream`                         | `0: (Text Protocol)` | 流式传输纯文本内容。                                                         |
| `on_chain_error`, `on_tool_error`, `on_node_error` | `3: (Error Protocol)` | 捕获并发送任何在流程、工具或节点中发生的错误。                               |
| `手动调用`                                     | `g: (Reasoning Protocol)` | 用户通过代码手动创建，用于自定义的思考或推理过程。                             |



还有流最终结束的(包括langchain和langgraph） d:{"finishReason":"stop","usage":{"promptTokens":10,"completionTokens":20}}\n
还有几个手动触发的，上面没有提到的， 下文有提就算作可以手动触发， 和g原理四一样的


另外我复制了整个ai-sdk协议

Data Stream Protocol
A data stream follows a special protocol that the AI SDK provides to send information to the frontend.

Each stream part has the format TYPE_ID:CONTENT_JSON\n.


When you provide data streams from a custom backend, you need to set the x-vercel-ai-data-stream header to v1.

The following stream parts are currently supported:

Text Part
The text parts are appended to the message as they are received.

Format: 0:string\n

Example: 0:"example"\n


Reasoning Part
The reasoning parts are appended to the message as they are received. The reasoning part is available through reasoning.

Format: g:string\n

Example: g:"I will open the conversation with witty banter."\n

Redacted Reasoning Part
The redacted reasoning parts contain reasoning data that has been redacted. This is available through redacted_reasoning.

Format: i:{"data": string}\n

Example: i:{"data": "This reasoning has been redacted for security purposes."}\n

Reasoning Signature Part
The reasoning signature parts contain a signature associated with the reasoning. This is available through reasoning_signature.

Format: j:{"signature": string}\n

Example: j:{"signature": "abc123xyz"}\n

Source Part
The source parts are appended to the message as they are received. The source part is available through source.

Format: h:Source\n

Example: h:{"sourceType":"url","id":"source-id","url":"https://example.com","title":"Example"}\n

File Part
The file parts contain binary data encoded as base64 strings along with their MIME type. The file part is available through file.

Format: k:{data:string; mimeType:string}\n

Example: k:{"data":"base64EncodedData","mimeType":"image/png"}\n

Data Part
The data parts are parsed as JSON and appended to the data array as they are received. The data is available through data.

Format: 2:Array<JSONValue>\n

Example: 2:[{"key":"object1"},{"anotherKey":"object2"}]\n


Message Annotation Part
The message annotation parts are appended to the message as they are received. The annotation part is available through annotations.

Format: 8:Array<JSONValue>\n

Example: 8:[{"id":"message-123","other":"annotation"}]\n

Error Part
The error parts are appended to the message as they are received.

Format: 3:string\n

Example: 3:"error message"\n


Tool Call Streaming Start Part
A part indicating the start of a streaming tool call. This part needs to be sent before any tool call delta for that tool call. Tool call streaming is optional, you can use tool call and tool result parts without it.

Format: b:{toolCallId:string; toolName:string}\n

Example: b:{"toolCallId":"call-456","toolName":"streaming-tool"}\n


Tool Call Delta Part
A part representing a delta update for a streaming tool call.

Format: c:{toolCallId:string; argsTextDelta:string}\n

Example: c:{"toolCallId":"call-456","argsTextDelta":"partial arg"}\n


Tool Call Part
A part representing a tool call. When there are streamed tool calls, the tool call part needs to come after the tool call streaming is finished.

Format: 9:{toolCallId:string; toolName:string; args:object}\n

Example: 9:{"toolCallId":"call-123","toolName":"my-tool","args":{"some":"argument"}}\n


Tool Result Part
A part representing a tool result. The result part needs to be sent after the tool call part for that tool call.

Format: a:{toolCallId:string; result:object}\n

Example: a:{"toolCallId":"call-123","result":"tool output"}\n


Start Step Part
A part indicating the start of a step.

It includes the following metadata:

messageId to indicate the id of the message that this step belongs to.
Format: f:{messageId:string}\n

Example: f:{"messageId":"step_123"}\n

Finish Step Part
A part indicating that a step (i.e., one LLM API call in the backend) has been completed.

This part is necessary to correctly process multiple stitched assistant calls, e.g. when calling tools in the backend, and using steps in useChat at the same time.

It includes the following metadata:

FinishReason
Usage for that step.
isContinued to indicate if the step text will be continued in the next step.
The finish step part needs to come at the end of a step.

Format: e:{finishReason:'stop' | 'length' | 'content-filter' | 'tool-calls' | 'error' | 'other' | 'unknown';usage:{promptTokens:number; completionTokens:number;},isContinued:boolean}\n

Example: e:{"finishReason":"stop","usage":{"promptTokens":10,"completionTokens":20},"isContinued":false}\n

Finish Message Part
A part indicating the completion of a message with additional metadata, such as FinishReason and Usage. This part needs to be the last part in the stream.

Format: d:{finishReason:'stop' | 'length' | 'content-filter' | 'tool-calls' | 'error' | 'other' | 'unknown';usage:{promptTokens:number; completionTokens:number;}}\n

Example: d:{"finishReason":"stop","usage":{"promptTokens":10,"completionTokens":20}}\n


The data stream protocol is supported by useChat and useCompletion on the frontend and used by default. useCompletion only supports the text and data stream parts.

On the backend, you can use toDataStreamResponse() from the streamText result object to return a streaming HTTP response.