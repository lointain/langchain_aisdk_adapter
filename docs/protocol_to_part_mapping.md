-----

好的，这是您提供的文档的中文翻译：

-----

# AI SDK 协议、部分和回调系统设计

本文档全面概述了将 LangChain/LangGraph 事件映射到 Vercel AI SDK 的设计。它涵盖了核心协议到部分的映射、回调系统设计以及实现最佳实践。

## 1\. 核心设计原则

整个系统都建立在以下原则之上，以确保其健壮性、兼容性和可扩展性。

### 1.1. 严格遵循 AI SDK 标准

  - 所有接口定义都与 AI SDK 的 `index.d.ts` 完全兼容。
  - `Message` 对象的结构严格匹配 AI SDK 标准。
  - 回调参数类型与 AI SDK 保持一致。

### 1.2. 回调的纯 Hook 模式

  - **回调作为纯 Hook**：它们不替换或修改任何内置的流处理逻辑。
  - **无 Super 调用**：避免 `await super().on_xxx()` 调用以保持隔离。
  - **仅观察和记录**：回调仅用于监听、分析和日志记录等副作用。
  - **不影响主流程**：回调的执行不影响原始流处理逻辑。
  - **独立性**：每个回调都是独立的，互不干扰。

### 1.3. 向后兼容性

  - 保留已弃用的字段以支持旧代码库。
  - 回调参数是可选的，以避免破坏现有功能。
  - 系统支持逐步迁移路径。

### 1.4. 类型安全

  - 使用 Pydantic 进行严格的类型验证。
  - 提供清晰的类型注解和文档。
  - 强制执行运行时类型检查。

### 1.5. 性能优化

  - 异步回调处理，实现非阻塞操作。
  - 最小化对象创建和序列化开销。
  - 原生支持流处理。

### 1.6. 可扩展性

  - 支持自定义回调处理器。
  - 灵活的配置选项。
  - 插件友好的架构。

### 1.7. 命名约定

  - **`AICallbackHandler`**：定义回调处理器标准契约的抽象接口。
  - **`BaseAICallbackHandler`**：一个提供默认方法的基类实现，便于继承和扩展。
  - **`ProcessSafeCallback`**：针对特定场景（例如进程安全）的具体实现。
  - 避免使用“Simple”等模糊术语；使用“Base”清晰表明它是一个基础抽象类。

## 2\. 协议到 UI 部分的映射

UI 部分的生成直接由 AI SDK 流协议驱动。下表详细说明了每个协议消息如何转换为 UI `Part`。

| 协议 | 创建的 UI 部分 | 触发 | 备注 |
| :--- | :--- | :--- | :--- |
| `0: (文本协议)` | `TextUIPart` | 自动/手动 | 流式传输 LLM 的文本输出。可以附加到现有 `TextUIPart` 或创建新的。 |
| `2: (数据协议)` | `DataUIPart` | 自动/手动 | 发送结构化的 JSON 数据以在 UI 中渲染。通常用于任意数据负载。 |
| `a: (工具调用协议)` | `ToolInvocationUIPart` | 自动/手动 | 标记工具调用的开始。创建带有工具名称和参数的 `ToolInvocationUIPart`。`result` 字段初始为空。 |
| `9: (工具输出协议)` | `ToolInvocationUIPart` (更新) | 自动/手动 | 提供工具执行的输出。此协议消息通过 `toolCallId` 查找对应的 `ToolInvocationUIPart` 并填充其 `result` 字段。 |
| `b: (步骤开始协议)` | `StepStartUIPart` | 自动/手动 | 标记“步骤”的开始（一个完整的交互单元，通常涉及工具调用）。创建 `StepStartUIPart`。 |
| `c: (步骤结束协议)` | `StepEndUIPart` | 自动/手动 | 标记“步骤”的结束。创建 `StepEndUIPart`。 |
| `e: (错误协议)` | `ErrorUIPart` | 自动/手动 | 报告处理过程中发生的错误。创建带有错误消息的 `ErrorUIPart`。 |
| `f: (完成协议)` | `FinishUIPart` | 自动/手动 | 标记整个链/图执行成功完成。创建带有完成原因的 `FinishUIPart`。 |
| `g: (推理协议)` | `ReasoningUIPart` | 仅手动 | 用于显示模型的内部推理或思考过程。旨在由开发人员手动调用。 |
| `h: (来源协议)` | `SourceUIPart` | 仅手动 | 提供来源或引用，通常渲染为超链接。旨在手动调用。 |

### 手动触发的 UI 部分

以下 UI 部分符合 AI SDK 规范，但在标准 LangChain/LangGraph 事件流中不会自动生成。它们旨在由开发人员根据需要手动创建和流式传输。

  - **`ReasoningUIPart`**：用于显示模型的内部推理或思考过程。此部分必须手动创建，因为 LangChain 流中没有标准的“推理”事件。
  - **`SourceUIPart`**：用于提供来源或引用，例如通过 RAG 过程检索到的文档。开发人员可以在获取来源后手动创建和流式传输此部分。
  - **`FileUIPart`**：用于文件相关的 UI，例如显示上传的文件或提供下载链接。这通常是根据特定的应用程序逻辑手动创建的。

## 3\. 实现与状态管理

### 3.1. 流类型假设

本适配器专门设计用于处理 LangChain 的 `astream_event` 类型流。这个设计假设简化了实现并提供了以下优势：

- **单一流类型**：to_data_stream_response只需要处理 `astream_event` 格式，避免了多种流类型的复杂性
- **事件驱动架构**：`astream_event` 提供结构化的事件流，包含明确的事件类型、数据和元数据
- **直接映射**：LangChain 事件类型可以直接映射到 AI SDK 协议，减少转换开销
- **更好的可预测性**：统一的事件格式使得错误处理和调试更加容易

### 3.2. 状态管理

为了管理流的状态并确保 Web 服务器环境中的线程安全，我们使用适配器实例模式。

`LangChainAdapter` 在实例化时接收配置和回调处理器，为每个请求创建独立的状态管理器。该管理器负责：

1.  接收 LangChain/LangGraph 事件。
2.  将它们转换为相应的 AI SDK 协议消息。
3.  在流的生命周期内构建 `Message` 对象及其 `parts` 数组。
4.  在适当的时机调用用户提供的回调 Hook。
5.  提供手动调用接口，允许开发者直接添加 UI Parts。

这确保了每个请求都有自己的状态，从而防止并发请求之间的数据泄露或竞态条件。

## 4\. 使用示例

以下是新的适配器实例模式的使用方法。

### 4.1. 基本用法

```python
import time
from typing import Dict, Any, Optional

from langchain_aisdk_adapter import BaseAICallbackHandler, LangChainAdapter
from langchain_aisdk_adapter.callbacks import Message, LanguageModelUsage

# 假设 'config' 和 'create_langchain_stream' 在其他地方定义

# --- 定义回调处理器 ---
class MyCallback(BaseAICallbackHandler):
    async def on_finish(self, message: Message, options: Dict[str, Any]) -> None:
        # 当整个流完成时，调用此 Hook。
        print(f"收到最终消息内容: {message.content}")
        print(f"部分总数: {len(message.parts or [])}")

        usage = options.get('usage')
        if usage:
            print(f"Token 使用量: {usage.totalTokens} tokens")

    async def on_error(self, error: Exception) -> None:
        # 如果发生未处理的异常，调用此 Hook。
        print(f"发生错误: {error}")

# --- 创建适配器实例 ---
callback_handler = MyCallback()
adapter = LangChainAdapter(
    config=config,
    callback=callback_handler
)

# --- 自动模式：处理 LangChain 流 ---
langchain_stream = create_langchain_stream(...)
ai_sdk_stream = adapter.to_data_stream_response(langchain_stream)

async for chunk in ai_sdk_stream:
    # 处理 AI SDK 协议流
    pass

# --- 手动模式：直接添加 UI Parts ---
# 在流处理过程中或独立使用
await adapter.text("这是一段文本")
await adapter.error("发生了错误")
await adapter.reasoning("模型正在思考...")
await adapter.source("参考来源", "https://example.com")
await adapter.tool_call("search", {"query": "AI"}, "tool_123")
await adapter.tool_result("tool_123", "搜索结果...")
```

### 4.2. 高级用法：分析回调

```python
class AnalyticsCallback(BaseAICallbackHandler):
    def __init__(self):
        self.tool_calls = []
        self.reasoning_count = 0

    async def on_tool_call(self, tool_call: Dict[str, Any]) -> None:
        # 纯 Hook: 仅记录工具调用数据用于分析。
        self.tool_calls.append({
            'timestamp': time.time(),
            'tool_name': tool_call.get('toolName'),
            'tool_id': tool_call.get('toolCallId')
        })

    async def on_reasoning(self, reasoning: str) -> None:
        # 记录推理调用次数
        self.reasoning_count += 1
        print(f"推理调用 #{self.reasoning_count}: {reasoning[:50]}...")

    async def on_finish(self, message: Message, options: Dict[str, Any]) -> None:
        # 分析最终消息
        part_types = {}
        for part in message.parts or []:
            part_type = part.type
            part_types[part_type] = part_types.get(part_type, 0) + 1

        print(f"最终部分分析: {part_types}")
        print(f"工具调用总数: {len(self.tool_calls)}")
        print(f"推理调用总数: {self.reasoning_count}")

# 使用分析回调
analytics_callback = AnalyticsCallback()
adapter = LangChainAdapter(
    config=config,
    callback=analytics_callback
)
```

### 4.3. 混合模式：自动 + 手动

```python
# 创建适配器
adapter = LangChainAdapter(config=config, callback=callback_handler)

# 开始自动流处理
langchain_stream = create_langchain_stream(...)
ai_sdk_stream = adapter.to_data_stream_response(langchain_stream)

# 在流处理过程中手动添加额外信息
async def process_stream():
    async for chunk in ai_sdk_stream:
        # 处理自动生成的协议消息
        yield chunk
        
        # 根据需要手动添加推理信息
        if some_condition:
            await adapter.reasoning("基于上下文，我认为...")
        
        # 添加来源引用
        if has_sources:
            await adapter.source("相关文档", "https://docs.example.com")

# 使用混合流
async for chunk in process_stream():
    # 发送到客户端
    pass
```

## 5\. FastAPI 集成示例

在 FastAPI 等 Web 框架中使用适配器时，为每个请求创建独立的适配器实例以确保线程安全。

### 5.1. 基本 FastAPI 集成

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any

# 假设 MyCallback、LangChainAdapter、config 和 create_langchain_stream 已定义

app = FastAPI()

@app.post("/chat/stream")
async def chat_stream(request: Dict[str, Any]):
    """线程安全的流式聊天端点。"""
    try:
        # 为每个请求创建独立的适配器实例
        adapter = LangChainAdapter(
            config=config,
            callback=MyCallback()  # 每个请求一个新的回调实例
        )
        
        langchain_stream = create_langchain_stream(request)
        
        return StreamingResponse(
            adapter.to_data_stream_response(langchain_stream),
            media_type="text/plain; charset=utf-8"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 5.2. 带手动控制的 FastAPI 端点

```python
@app.post("/chat/stream-with-reasoning")
async def chat_stream_with_reasoning(request: Dict[str, Any]):
    """支持手动推理注入的聊天端点。"""
    try:
        adapter = LangChainAdapter(
            config=config,
            callback=AnalyticsCallback()
        )
        
        langchain_stream = create_langchain_stream(request)
        
        async def enhanced_stream():
            # 开始推理
            await adapter.reasoning("开始分析用户请求...")
            
            # 处理主流
            async for chunk in adapter.to_data_stream_response(langchain_stream):
                yield chunk
                
                # 在特定条件下添加额外信息
                if should_add_source(chunk):
                    await adapter.source("相关文档", get_source_url())
            
            # 结束推理
            await adapter.reasoning("分析完成，生成最终回答。")
        
        return StreamingResponse(
            enhanced_stream(),
            media_type="text/plain; charset=utf-8"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 5.3. 依赖注入模式

```python
from fastapi import Depends

def get_adapter() -> LangChainAdapter:
    """依赖注入：为每个请求创建新的适配器。"""
    return LangChainAdapter(
        config=config,
        callback=MyCallback()
    )

@app.post("/chat/stream")
async def chat_stream(
    request: Dict[str, Any],
    adapter: LangChainAdapter = Depends(get_adapter)
):
    """使用依赖注入的聊天端点。"""
    try:
        langchain_stream = create_langchain_stream(request)
        
        return StreamingResponse(
            adapter.to_data_stream_response(langchain_stream),
            media_type="text/plain; charset=utf-8"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/manual/add-reasoning")
async def add_reasoning(
    reasoning_text: str,
    adapter: LangChainAdapter = Depends(get_adapter)
):
    """手动添加推理信息的端点。"""
    try:
        await adapter.reasoning(reasoning_text)
        return {"status": "success", "message": "推理信息已添加"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## 总结

此设计确保与 AI SDK 完全兼容，同时提供灵活的扩展能力。通过严格遵循 AI SDK 标准，我们的回调系统可以无缝集成到任何使用 Vercel AI SDK 的前端应用程序中。

### 主要优势

1.  **完全兼容性**：100% 符合 AI SDK 标准。
2.  **类型安全**：严格的类型定义和验证。
3.  **易于使用**：清晰的 API 和全面的示例。
4.  **高性能**：异步处理和优化的数据结构。
5.  **可扩展性**：支持自定义回调和插件。