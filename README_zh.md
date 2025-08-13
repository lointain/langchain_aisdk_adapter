# LangChain AI SDK 适配器

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/langchain-aisdk-adapter.svg)](https://badge.fury.io/py/langchain-aisdk-adapter)

一个尝试将 LangChain 流式输出转换为 AI SDK 兼容数据流的 Python 适配器。该库致力于在 LangChain 和 AI SDK 协议之间建立桥梁，但可能无法覆盖所有边缘情况和使用场景。

## 特性

- **🔄 协议支持**：基本支持 AI SDK v4 和 v5 协议（可能无法覆盖所有协议细节）
- **📡 流式转换**：尝试将 LangChain `astream_events()` 转换为 AI SDK 数据流
- **🛠️ 工具集成**：有限支持工具调用和工具输出
- **📝 丰富内容**：处理常见内容类型如文本、推理、文件和来源（某些边缘情况可能未覆盖）
- **⚡ FastAPI 集成**：基本的 FastAPI `StreamingResponse` 集成
- **🎯 手动控制**：提供手动事件发射功能
- **🔧 灵活配置**：可配置协议版本和输出格式
- **📊 使用跟踪**：基本的令牌使用和性能监控
- **🌊 平滑流式传输**：内置 `smooth_stream` 功能，提供增强的文本输出平滑处理
- **🔗 扩展回调系统**：全面的回调系统，支持 `onChunk`、`onError`、`onStepFinish`、`onFinish` 和 `onAbort`
- **🧪 实验性功能**：支持 `experimental_transform` 和 `experimental_generateMessageId`

## 已知限制

- **协议兼容性**：虽然我们努力保持兼容性，但某些 AI SDK 功能可能无法完全支持
- **错误处理**：复杂流式场景中的错误情况可能需要额外处理
- **工具复杂性**：高级工具调用模式可能需要自定义实现
- **测试覆盖**：某些边缘情况和复杂场景可能未经充分测试

## 安装

```bash
pip install -i https://test.pypi.org/simple/ langchain-aisdk-adapter
```

## 快速开始

### 基本用法

```python
from langchain_aisdk_adapter import LangChainAdapter
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# 初始化您的 LangChain 模型
llm = ChatOpenAI(model="gpt-4")
messages = [HumanMessage(content="你好，世界！")]

# 转换为 AI SDK 数据流
stream = llm.astream_events(messages, version="v2")
data_stream = LangChainAdapter.to_data_stream(stream)

# 遍历流
async for chunk in data_stream:
    print(chunk)
```

### FastAPI 集成

```python
from fastapi import FastAPI
from langchain_aisdk_adapter import LangChainAdapter
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

app = FastAPI()
llm = ChatOpenAI(model="gpt-4")

@app.post("/api/chat")
async def chat(request: dict):
    messages = [HumanMessage(content=request["message"])]
    stream = llm.astream_events(messages, version="v2")
    
    # 返回 FastAPI StreamingResponse
    return LangChainAdapter.to_data_stream_response(
        stream,
        options={"protocol_version": "v5"}  # 使用 AI SDK v5
    )
```

### 基于上下文的事件发射

```python
from langchain_aisdk_adapter import LangChainAdapter, DataStreamContext

# 创建具有自动上下文管理的数据流
stream = llm.astream_events(messages, version="v2")
data_stream = LangChainAdapter.to_data_stream(
    stream, 
    options={"auto_context": True}
)

# 使用上下文发出自定义事件
context = DataStreamContext.get_current_stream()
if context:
    await context.emit_text_delta("自定义文本")
    await context.emit_source_url("https://example.com", "示例")
    await context.emit_data({"key": "value"})
```

## 核心组件

### LangChainAdapter

提供三个核心方法的主适配器类：

#### `to_data_stream_response()`
将 LangChain 流转换为 FastAPI `StreamingResponse`：

```python
response = LangChainAdapter.to_data_stream_response(
    stream=langchain_stream,
    options={
        "protocol_version": "v5",  # "v4" 或 "v5"
        "output_format": "protocol",  # "chunks" 或 "protocol"
        "auto_events": True,
        "auto_close": True
    }
)
```

#### `to_data_stream()`
将 LangChain 流转换为 `DataStreamWithEmitters`：

```python
data_stream = LangChainAdapter.to_data_stream(
    stream=langchain_stream,
    message_id="custom-id",
    options={"protocol_version": "v4"}
)
```

#### `merge_into_data_stream()`
将 LangChain 流合并到现有数据流写入器中：

```python
from langchain_aisdk_adapter import DataStreamWriter

writer = DataStreamWriter()
await LangChainAdapter.merge_into_data_stream(
    stream=langchain_stream,
    data_stream_writer=writer
)
```

### 协议支持

#### AI SDK v4 协议
- 文本格式：`text/plain; charset=utf-8`
- 头部：`x-vercel-ai-data-stream: v1`
- 格式：`<type>:<data>\n`

#### AI SDK v5 协议
- 文本格式：`text/event-stream`
- 头部：`x-vercel-ai-ui-message-stream: v1`
- 格式：服务器发送事件 (SSE)

### 配置选项

```python
options = {
    "protocol_version": "v5",      # "v4" 或 "v5"
    "output_format": "protocol",   # "chunks" 或 "protocol"
    "auto_events": True,           # 自动发出开始/结束事件
    "auto_close": True,            # 自动关闭流
    "emit_start": True,            # 发出开始事件
    "emit_finish": True            # 发出结束事件
}
```

## 高级功能

### 平滑流式传输

`smooth_stream` 方法提供增强的文本输出平滑处理，支持可配置的分块策略和延迟：

```python
from langchain_aisdk_adapter import LangChainAdapter
import re

# 基于单词的分块，带延迟
smooth_transform = LangChainAdapter.smooth_stream(
    delayInMs=50,
    chunking='word'
)

# 基于行的分块
smooth_transform = LangChainAdapter.smooth_stream(
    delayInMs=100,
    chunking='line'
)

# 自定义正则表达式分块（例如，用于中文文本）
smooth_transform = LangChainAdapter.smooth_stream(
    delayInMs=30,
    chunking=re.compile(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\S')
)

# 自定义分块函数
def custom_chunker(text: str) -> list[str]:
    return text.split('，')

smooth_transform = LangChainAdapter.smooth_stream(
    delayInMs=75,
    chunking=custom_chunker
)

# 与 experimental_transform 一起使用
data_stream = LangChainAdapter.to_data_stream(
    stream=langchain_stream,
    options={
        "experimental_transform": smooth_transform
    }
)
```

### 扩展回调系统

支持 AI SDK 流式事件的全面回调系统：

```python
from langchain_aisdk_adapter import BaseAICallbackHandler

class ExtendedCallback(BaseAICallbackHandler):
    async def on_chunk(self, chunk, **kwargs):
        """每个流块时调用"""
        print(f"收到块：{chunk}")
    
    async def on_step_finish(self, step_result, **kwargs):
        """步骤完成时调用"""
        print(f"步骤完成：{step_result}")
    
    async def on_finish(self, message, options, **kwargs):
        """流式传输完成时调用"""
        print(f"流完成：{message}")
    
    async def on_error(self, error, **kwargs):
        """发生错误时调用"""
        print(f"流错误：{error}")
    
    async def on_abort(self, **kwargs):
        """流式传输中止时调用"""
        print("流已中止")

callbacks = ExtendedCallback()
data_stream = LangChainAdapter.to_data_stream(
    stream=langchain_stream,
    callbacks=callbacks
)
```

### 实验性功能

```python
# 自定义消息 ID 生成
def generate_custom_id():
    return f"custom-{uuid.uuid4()}"

# 使用实验性功能
data_stream = LangChainAdapter.to_data_stream(
    stream=langchain_stream,
    options={
        "experimental_transform": LangChainAdapter.smooth_stream(delayInMs=50),
        "experimental_generateMessageId": generate_custom_id
    }
)
```

### 自定义回调（传统）

```python
from langchain_aisdk_adapter import BaseAICallbackHandler

class CustomCallback(BaseAICallbackHandler):
    async def on_text_delta(self, delta: str, **kwargs):
        print(f"文本增量：{delta}")
    
    async def on_tool_call_start(self, tool_name: str, **kwargs):
        print(f"工具调用开始：{tool_name}")

callbacks = CustomCallback()
data_stream = LangChainAdapter.to_data_stream(
    stream=langchain_stream,
    callbacks=callbacks
)
```

### 上下文管理

```python
# 使用 DataStreamContext 进行事件发射
from langchain_aisdk_adapter import DataStreamContext

# 在回调或处理器中
context = DataStreamContext.get_current_stream()
if context:
    # 发出各种块类型
    await context.emit_text_start("text-1")
    await context.emit_text_delta("你好", "text-1")
    await context.emit_text_end("你好世界", "text-1")
    
    # 发出工具调用
    await context.emit_tool_input_start("tool-1", "search")
    await context.emit_tool_input_available("tool-1", "search", {"query": "AI"})
    await context.emit_tool_output_available("tool-1", "找到结果")
```

### 错误处理

```python
try:
    data_stream = LangChainAdapter.to_data_stream(stream)
    async for chunk in data_stream:
        if chunk.get("type") == "error":
            print(f"错误：{chunk.get('errorText')}")
except Exception as e:
    print(f"流错误：{e}")
```

## 支持的块类型

| 块类型 | AI SDK v4 | AI SDK v5 | 描述 | 备注 |
|--------|-----------|-----------|------|------|
| `text-start` | ✅ | ✅ | 文本生成开始 | 基本支持 |
| `text-delta` | ✅ | ✅ | 文本内容增量 | 测试充分 |
| `text-end` | ✅ | ✅ | 文本生成结束 | 基本支持 |
| `tool-input-start` | ⚠️ | ⚠️ | 工具调用输入开始 | 可能需要改进 |
| `tool-input-delta` | ⚠️ | ⚠️ | 工具调用输入增量 | 测试有限 |
| `tool-input-available` | ⚠️ | ⚠️ | 工具调用输入完成 | 可能需要改进 |
| `tool-output-available` | ⚠️ | ⚠️ | 工具调用输出 | 基本支持 |
| `tool-output-error` | ⚠️ | ⚠️ | 工具调用错误 | 错误处理有限 |
| `reasoning` | ⚠️ | ⚠️ | 推理内容 | 实验性功能 |
| `source-url` | ⚠️ | ⚠️ | 来源 URL 引用 | 基本实现 |
| `source-document` | ⚠️ | ⚠️ | 来源文档 | 基本实现 |
| `file` | ⚠️ | ⚠️ | 文件附件 | 支持有限 |
| `data` | ✅ | ✅ | 自定义数据 | 支持良好 |
| `error` | ⚠️ | ⚠️ | 错误消息 | 基本错误处理 |
| `start-step` | ⚠️ | ⚠️ | 步骤开始 | 实验性功能 |
| `finish-step` | ⚠️ | ⚠️ | 步骤结束 | 实验性功能 |
| `start` | ✅ | ✅ | 流开始 | 支持良好 |
| `finish` | ✅ | ✅ | 流结束 | 支持良好 |

**图例**：✅ 支持良好，⚠️ 基本/实验性支持，❌ 不支持

## 示例

### 完整聊天应用

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_aisdk_adapter import LangChainAdapter
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

app = FastAPI()
llm = ChatOpenAI(model="gpt-4")

class ChatRequest(BaseModel):
    message: str
    protocol_version: str = "v4"

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        messages = [
            SystemMessage(content="你是一个有用的助手。"),
            HumanMessage(content=request.message)
        ]
        
        stream = llm.astream_events(messages, version="v2")
        
        return LangChainAdapter.to_data_stream_response(
            stream=stream,
            options={
                "protocol_version": request.protocol_version,
                "auto_events": True
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 工具集成示例

```python
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_aisdk_adapter import LangChainAdapter

# 设置工具和代理
search = DuckDuckGoSearchRun()
tools = [search]
llm = ChatOpenAI(model="gpt-4")
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# 使用工具调用进行流式处理
stream = agent_executor.astream_events(
    {"input": "搜索最新的 AI 新闻"},
    version="v2"
)

data_stream = LangChainAdapter.to_data_stream(
    stream=stream,
    options={"protocol_version": "v5"}
)

async for chunk in data_stream:
    chunk_type = chunk.get("type")
    if chunk_type == "tool-input-available":
        print(f"工具：{chunk.get('toolName')}")
        print(f"输入：{chunk.get('input')}")
    elif chunk_type == "tool-output-available":
        print(f"输出：{chunk.get('output')}")
```

## API 参考

### 类

- **`LangChainAdapter`**：主适配器类
- **`DataStreamWithEmitters`**：具有手动发射方法的流
- **`DataStreamResponse`**：FastAPI 响应包装器
- **`DataStreamWriter`**：用于合并的流写入器
- **`DataStreamContext`**：基于上下文的流控制
- **`ContextLifecycleManager`**：上下文生命周期管理
- **`BaseAICallbackHandler`**：基础回调处理器
- **`ProtocolStrategy`**：协议策略接口
- **`AISDKv4Strategy`**：AI SDK v4 实现
- **`AISDKv5Strategy`**：AI SDK v5 实现

### 函数



## 贡献

欢迎贡献！本项目仍处于早期开发阶段，有许多方面可以改进：

- 更好的错误处理和边缘情况覆盖
- 更全面的测试
- 性能优化
- 增强协议兼容性
- 文档改进
- 实际使用案例示例

请随时提交 Pull Request 或开启 issue 讨论改进建议。

## 免责声明

本库按现状提供，可能无法覆盖所有使用场景或边缘情况。虽然我们努力与 AI SDK 协议保持兼容，但可能存在差异或缺失功能。建议用户在特定环境中进行充分测试，并将改进贡献回项目。

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 更新日志

### v0.0.1a1
- 初始 alpha 版本
- 基本支持 AI SDK v4 和 v5 协议
- 核心适配器功能（实验性）
- FastAPI 集成（基本）
- 手动事件发射功能
- 有限的工具调用支持
- 基本内容类型处理
- 简单使用跟踪