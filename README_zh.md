# LangChain AI SDK 适配器

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/langchain-aisdk-adapter.svg)](https://badge.fury.io/py/langchain-aisdk-adapter)

一个将 LangChain 流式输出转换为 AI SDK 兼容数据流的 Python 适配器，支持 AI SDK v4 和 v5 协议。

## 特性

- **🔄 协议支持**：完全支持 AI SDK v4 和 v5 协议
- **📡 流式转换**：将 LangChain `astream_events()` 转换为 AI SDK 数据流
- **🛠️ 工具集成**：支持工具调用和工具输出
- **📝 丰富内容**：处理文本、推理、文件、来源和自定义数据
- **⚡ FastAPI 集成**：直接与 FastAPI `StreamingResponse` 集成
- **🎯 手动控制**：使用 `DataStreamWithEmitters` 发出自定义事件
- **🔧 灵活配置**：可配置协议版本和输出格式
- **📊 使用跟踪**：内置令牌使用和性能监控

## 安装

```bash
pip install langchain-aisdk-adapter
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

### 手动事件发射

```python
from langchain_aisdk_adapter import LangChainAdapter

# 创建具有手动控制的数据流
stream = llm.astream_events(messages, version="v2")
data_stream = LangChainAdapter.to_data_stream(stream)

# 发出自定义事件
await data_stream.emit_text_delta("自定义文本")
await data_stream.emit_source_url("https://example.com", "示例")
await data_stream.emit_data({"key": "value"})
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

### 自定义回调

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

### 手动流控制

```python
# 创建手动流控制器
from langchain_aisdk_adapter import ManualStreamController

controller = ManualStreamController()

# 发出各种块类型
await controller.emit_text_start("text-1")
await controller.emit_text_delta("你好", "text-1")
await controller.emit_text_end("你好世界", "text-1")

# 发出工具调用
await controller.emit_tool_input_start("tool-1", "search")
await controller.emit_tool_input_available("tool-1", "search", {"query": "AI"})
await controller.emit_tool_output_available("tool-1", "找到结果")

# 获取流
stream = controller.get_stream()
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

| 块类型 | AI SDK v4 | AI SDK v5 | 描述 |
|--------|-----------|-----------|------|
| `text-start` | ✅ | ✅ | 文本生成开始 |
| `text-delta` | ✅ | ✅ | 文本内容增量 |
| `text-end` | ✅ | ✅ | 文本生成结束 |
| `tool-input-start` | ✅ | ✅ | 工具调用输入开始 |
| `tool-input-delta` | ✅ | ✅ | 工具调用输入增量 |
| `tool-input-available` | ✅ | ✅ | 工具调用输入完成 |
| `tool-output-available` | ✅ | ✅ | 工具调用输出 |
| `tool-output-error` | ✅ | ✅ | 工具调用错误 |
| `reasoning` | ✅ | ✅ | 推理内容 |
| `source-url` | ✅ | ✅ | 来源 URL 引用 |
| `source-document` | ✅ | ✅ | 来源文档 |
| `file` | ✅ | ✅ | 文件附件 |
| `data` | ✅ | ✅ | 自定义数据 |
| `error` | ✅ | ✅ | 错误消息 |
| `start-step` | ✅ | ✅ | 步骤开始 |
| `finish-step` | ✅ | ✅ | 步骤结束 |
| `start` | ✅ | ✅ | 流开始 |
| `finish` | ✅ | ✅ | 流结束 |

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
- **`ManualStreamController`**：手动流控制
- **`BaseAICallbackHandler`**：基础回调处理器
- **`ProtocolStrategy`**：协议策略接口
- **`AISDKv4Strategy`**：AI SDK v4 实现
- **`AISDKv5Strategy`**：AI SDK v5 实现

### 函数

- **`to_data_stream()`**：遗留兼容性函数
- **`to_data_stream_response()`**：遗留兼容性函数
- **`merge_into_data_stream()`**：遗留兼容性函数

## 贡献

欢迎贡献！请随时提交 Pull Request。

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 更新日志

### v0.0.1a1
- 初始版本
- 支持 AI SDK v4 和 v5 协议
- 核心适配器功能
- FastAPI 集成
- 手动事件发射
- 工具调用支持
- 丰富内容类型
- 使用跟踪