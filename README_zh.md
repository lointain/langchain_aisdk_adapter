# LangChain AI SDK 适配器

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Development Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://pypi.org/project/langchain-aisdk-adapter/)

> **⚠️ Alpha 版本提醒**: 本项目目前处于 alpha 阶段。虽然我们努力确保稳定性和可靠性，但请注意 API 可能会发生变化，某些功能可能仍在开发中。我们感谢您的耐心，并欢迎任何反馈来帮助我们改进！

用于连接 LangChain/LangGraph 应用程序与 AI SDK UI 流协议。这个库旨在让开发者更容易地将 LangChain 的强大功能与现代流式接口集成。

## ✨ 特性

我们努力让这个适配器尽可能全面和用户友好：

- **🔄 全面的协议支持**: 支持 15+ AI SDK 协议，包括文本流、工具交互、步骤管理和数据处理
- **⚙️ 智能配置**: 灵活的 `AdapterConfig` 系统让您精确控制生成哪些协议
- **🔒 线程安全的多用户支持**: `ThreadSafeAdapterConfig` 为 Web 应用中的并发请求提供隔离的协议状态
- **🎛️ 动态协议控制**: 上下文管理器可在执行期间临时启用/禁用特定协议
- **🛠️ 丰富的工具支持**: 与 LangChain 工具、代理和函数调用无缝集成
- **📊 使用跟踪**: 内置的流处理统计和监控功能
- **🔒 类型安全**: 完整的 Python 类型提示和 Pydantic 验证
- **🏭 工厂方法**: 在需要时提供便捷的工厂方法进行手动协议生成
- **🔌 可扩展设计**: 易于扩展和定制以适应特定用例

## 🚀 快速开始

我们希望这能让您快速上手：

### 安装

**Alpha 版本**: 此包已在 Test PyPI 上发布，供早期测试和反馈使用。

```bash
# 从 Test PyPI 基础安装
pip install --index-url https://test.pypi.org/simple/ langchain-aisdk-adapter

# 包含示例（包括 LangChain、LangGraph、OpenAI）
pip install --index-url https://test.pypi.org/simple/ langchain-aisdk-adapter[examples]

# 包含 Web 框架支持（包括 FastAPI、Uvicorn）
pip install --index-url https://test.pypi.org/simple/ langchain-aisdk-adapter[web]

# 开发版本（包括测试和代码检查工具）
pip install --index-url https://test.pypi.org/simple/ langchain-aisdk-adapter[dev]
```

**注意**: 由于这是 Test PyPI 上的 alpha 版本，您可能需要先从主 PyPI 安装依赖项：

```bash
# 先安装依赖项，然后安装包
pip install langchain-core langchain-openai pydantic
pip install --index-url https://test.pypi.org/simple/ langchain-aisdk-adapter
```

### 基础用法

这里是一个简单的入门示例：

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_aisdk_adapter import LangChainAdapter

# 创建您的 LangChain 模型
llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)

# 创建流
stream = llm.astream([HumanMessage(content="你好，世界！")])

# 转换为 AI SDK 格式 - 就是这么简单！
ai_sdk_stream = LangChainAdapter.to_data_stream_response(stream)

# 在您的应用程序中使用
async for chunk in ai_sdk_stream:
    print(chunk, end="", flush=True)
```

### 配置选项

我们包含了几个预设配置来简化常见用例：

```python
from langchain_aisdk_adapter import LangChainAdapter, AdapterConfig, ThreadSafeAdapterConfig

# 最小输出 - 仅基本文本和数据
config = AdapterConfig.minimal()

# 专注于工具交互
config = AdapterConfig.tools_only()

# 启用所有功能（默认）
config = AdapterConfig.comprehensive()

# 自定义配置
config = AdapterConfig(
    enable_text=True,
    enable_data=True,
    enable_tool_calls=True,
    enable_steps=False,  # 禁用步骤跟踪
    enable_reasoning=False  # 禁用推理输出
)

stream = LangChainAdapter.to_data_stream_response(your_stream, config=config)

# 协议暂停/恢复功能
with config.pause_protocols(['0', '2']):  # 临时禁用文本和数据协议
    # 在此块中，不会生成文本和数据协议
    restricted_stream = LangChainAdapter.to_data_stream_response(some_stream, config=config)
    async for chunk in restricted_stream:
        # 只会发出工具调用、结果和步骤
        print(chunk)
# 上下文结束后协议自动恢复

# 多用户应用的线程安全配置
safe_config = ThreadSafeAdapterConfig()

# 每个请求都有隔离的协议状态
with safe_config.protocols(['0', '9', 'a']):  # 仅启用文本、工具调用和结果
    stream = LangChainAdapter.to_data_stream_response(your_stream, config=safe_config)
    # 此配置不会影响其他并发请求
```

## 📋 协议支持状态

我们将支持的协议分为三类，帮助您了解可用功能以及它们的触发条件：

### 🟢 自动支持的协议

这些协议会从 LangChain/LangGraph 事件中自动生成，具有特定的触发条件：

#### **`0:` (文本协议)**
**触发条件**: 当 LLM 产生流式文本内容时生成
**格式**: `0:"流式文本内容"`
**发生时机**: 
- 在 `llm.astream()` 调用期间
- 当 LangGraph 节点产生文本输出时
- 任何来自语言模型的流式文本

#### **`2:` (数据协议)**
**触发条件**: 为结构化数据和元数据生成
**格式**: `2:[{"key":"value"}]`
**发生时机**:
- LangGraph 节点元数据和中间结果
- 工具执行元数据
- 来自 LangChain 回调的自定义数据

#### **`9:` (工具调用协议)**
**触发条件**: 当工具被调用时生成
**格式**: `9:{"toolCallId":"call_123","toolName":"search","args":{"query":"test"}}`
**发生时机**:
- LangChain 代理工具调用
- LangGraph 工具节点执行
- 聊天模型中的函数调用

#### **`a:` (工具结果协议)**
**触发条件**: 当工具执行完成时生成
**格式**: `a:{"toolCallId":"call_123","result":"工具输出"}`
**发生时机**:
- 工具执行成功后
- 在任何 `9:` 协议之后
- 成功和错误结果都会生成

#### **`b:` (工具调用流开始协议)**
**触发条件**: 在流式工具调用开始时生成
**格式**: `b:{"toolCallId":"call_123","toolName":"search"}`
**发生时机**:
- 在工具参数流开始之前
- 仅适用于支持流式参数的工具

#### **`d:` (完成消息协议)** ⚠️ **仅限 LangGraph**
**触发条件**: **仅在 LangGraph 工作流中**当消息完成时生成
**格式**: `d:{"finishReason":"stop","usage":{"promptTokens":10,"completionTokens":20}}`
**发生时机**:
- **LangGraph 工作流消息完成**
- **不会在基础 LangChain 流中生成**
- LangGraph 节点执行结束
- 包含可用的使用统计信息

#### **`e:` (完成步骤协议)** 🔄 **增强支持**
**触发条件**: 当主要工作流组件完成执行时生成
**格式**: `e:{"stepId":"step_123","finishReason":"completed"}`
**发生时机**:
- **LangGraph 工作流步骤完成** (主要用例)
- **LangChain 代理执行** (AgentExecutor, ReActAgent, ChatAgent 等)
- **基于链的工作流** (LLMChain, SequentialChain, RouterChain 等)
- **具有特定标签的组件** (agent, chain, executor, workflow, multi_agent)
- 多步骤过程和推理步骤的结束

#### **`f:` (开始步骤协议)** 🔄 **增强支持**
**触发条件**: 当主要工作流组件开始执行时生成
**格式**: `f:{"stepId":"step_123","stepType":"agent_action"}`
**发生时机**:
- **LangGraph 工作流步骤开始** (主要用例)
- **LangChain 代理执行** (AgentExecutor, ReActAgent, ChatAgent 等)
- **基于链的工作流** (LLMChain, SequentialChain, RouterChain 等)
- **LangGraph 组件** (LangGraph, CompiledGraph, StateGraph 等)
- **具有特定标签的组件** (agent, chain, executor, workflow, multi_agent, langgraph, graph)
- 多步骤过程和推理步骤的开始

> 💡 **重要说明**: 
> - 协议 `d:`、`e:` 和 `f:` 是 **LangGraph 专用的**，不会出现在基础 LangChain 流中
> - 所有自动支持的协议都可以通过 `AdapterConfig` 单独启用或禁用
> - 确切格式可能会根据底层 LangChain/LangGraph 事件结构而有所不同

### 🟡 仅手动支持的协议

这些协议需要使用我们的工厂方法手动生成：

#### **`g:` (推理协议)** ⚠️ **仅限手动支持**
**用途**: 传输 AI 推理过程和思维链
**格式**: `g:{"reasoning":"让我一步步思考这个问题...","confidence":0.85}`
**手动创建**:
```python
from langchain_aisdk_adapter import AISDKFactory

# 创建推理协议
reasoning_part = AISDKFactory.create_reasoning_part(
    reasoning="让我分析用户的请求...",
    confidence=0.9
)
print(f"g:{reasoning_part.model_dump_json()}")
```
**使用场景**: 思维链推理、决策解释、置信度评分

#### **`c:` (工具调用增量协议)** ⚠️ **仅限手动支持**
**用途**: 在工具调用执行期间流式传输增量更新
**格式**: `c:{"toolCallId":"call_123","delta":{"function":{"arguments":"{\"query\":\"hello\"}"}},"index":0}`
**手动创建**:
```python
from langchain_aisdk_adapter import AISDKFactory

# 创建工具调用增量
delta_part = AISDKFactory.create_tool_call_delta_part(
    tool_call_id="call_123",
    delta={"function": {"arguments": '{"query":"hello"}'}},
    index=0
)
print(f"c:{delta_part.model_dump_json()}")
```
**使用场景**: 实时工具执行反馈、流式函数调用

#### **`8:` (消息注解协议)** ⚠️ **仅限手动支持**
**用途**: 为消息添加元数据和注解
**格式**: `8:{"annotations":[{"type":"citation","text":"来源: 维基百科"}],"metadata":{"confidence":0.95}}`
**手动创建**:
```python
from langchain_aisdk_adapter import AISDKFactory

# 创建消息注解
annotation_part = AISDKFactory.create_message_annotation_part(
    annotations=[{"type": "citation", "text": "来源: 维基百科"}],
    metadata={"confidence": 0.95}
)
print(f"8:{annotation_part.model_dump_json()}")
```
**使用场景**: 来源引用、置信度评分、内容元数据

#### **`h:` (来源协议)** ✅ **手动支持**
**手动创建**: 使用 `create_source_part(url, title=None)` 或 `AISDKFactory.source(url, title=None)`
**格式**: `h:{"url":"https://example.com","title":"文档标题"}`
**使用场景**: 文档引用、引用跟踪、来源归属

#### **`i:` (删节推理协议)** ✅ **手动支持**
**手动创建**: 使用 `create_redacted_reasoning_part(data)` 或 `AISDKFactory.redacted_reasoning(data)`
**格式**: `i:{"data":"[已删节] 推理内容"}`
**使用场景**: 隐私合规的推理输出、内容过滤

#### **`j:` (推理签名协议)** ✅ **手动支持**
**手动创建**: 使用 `create_reasoning_signature_part(signature)` 或 `AISDKFactory.reasoning_signature(signature)`
**格式**: `j:{"signature":"signature_abc123"}`
**使用场景**: 推理验证、模型签名、真实性跟踪

#### **`k:` (文件协议)** ✅ **手动支持**
**手动创建**: 使用 `create_file_part(data, mime_type)` 或 `AISDKFactory.file(data, mime_type)`
**格式**: `k:{"data":"base64_encoded_data","mimeType":"image/png"}`
**使用场景**: 文件附件、二进制数据传输、文档共享

### 🔴 当前不支持的协议

我们正在努力支持这些协议，但目前还不可用：

- **`1:` (函数调用)**: 与 LangChain 的工具系统架构不同
- **`4:` (工具调用流)**: 需要当前 LangChain 版本中不可用的流式参数支持
- **`5:` (工具调用流部分)**: 与上述限制相同
- **`6:` (工具调用流增量)**: 与上述限制相同
- **`7:` (工具调用流完成)**: 与上述限制相同

## 🛠️ 手动协议生成

对于需要手动实现的协议，我们提供了便捷的工厂方法：

```python
from langchain_aisdk_adapter.factory import AISDKFactory

# 创建工厂实例
factory = AISDKFactory()

# 生成推理协议
reasoning_part = factory.reasoning(
    content="让我一步步思考这个问题..."
)

# 生成来源协议
source_part = factory.source(
    url="https://example.com/document",
    title="重要文档"
)

# 生成删节推理协议
redacted_part = factory.redacted_reasoning(
    data="[已删节] 敏感推理内容"
)

# 生成推理签名协议
signature_part = factory.reasoning_signature(
    signature="model_signature_abc123"
)

# 生成文件协议
file_part = factory.file(
    data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
    mime_type="image/png"
)

# 生成消息注解
annotation_part = factory.annotation(
    message_id="msg_123",
    annotation_type="confidence",
    value={"score": 0.95}
)

# 生成工具调用增量（用于流式参数）
tool_delta_part = factory.tool_call_delta(
    tool_call_id="call_123",
    name="search",
    args_delta='{"query": "人工智能'  # 部分 JSON
)
```

### 工厂实例使用

使用简化的工厂实例快速创建协议：

```python
from langchain_aisdk_adapter import factory

# 创建各种协议部分
text_part = factory.text("来自 LangChain 的问候！")
data_part = factory.data({"temperature": 0.7, "max_tokens": 100})
error_part = factory.error("连接超时")
reasoning_part = factory.reasoning("基于上下文，我应该...")
source_part = factory.source(
    url="https://docs.langchain.com",
    title="LangChain 文档"
)

# 在流式响应中使用
async def stream_with_factory():
    yield text_part
    yield reasoning_part
    yield data_part
```

**为什么需要手动实现？**

由于技术限制，我们不得不将某些协议设为手动：
- **推理内容**: 不同的 LLM 使用不同的推理格式，无法自动标准化
- **工具调用增量**: LangChain 的工具系统不提供流式参数生成
- **消息注解**: LangChain 缺乏标准化的消息元数据事件系统
- **来源跟踪**: 文档来源信息需要显式的应用级实现
- **内容过滤**: 删节推理需要自定义隐私和安全策略
- **文件处理**: 二进制文件处理和编码在不同实现中差异很大

## 🌐 Web 集成示例

我们为 Web 框架提供了全面的示例：

### FastAPI 集成

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_aisdk_adapter import LangChainAdapter

app = FastAPI()

@app.post("/chat")
async def chat(message: str):
    # 您的 LangChain 设置
    stream = llm.astream([HumanMessage(content=message)])
    
    return StreamingResponse(
        LangChainAdapter.to_data_stream_response(stream),
        media_type="text/plain"
    )
```

### 多轮对话

处理带有消息历史的多轮对话：

```python
from langchain_core.messages import HumanMessage, AIMessage
from langchain_aisdk_adapter import LangChainAdapter

async def multi_turn_chat():
    conversation_history = []
    
    # 第一轮
    user_input = "什么是机器学习？"
    conversation_history.append(HumanMessage(content=user_input))
    
    response_content = ""
    stream = llm.astream(conversation_history)
    ai_sdk_stream = LangChainAdapter.to_data_stream_response(stream)
    
    async for chunk in ai_sdk_stream:
        if chunk.startswith('0:'):
            # 从协议中提取文本内容
            text_content = chunk[2:].strip('"')
            response_content += text_content
        yield chunk
    
    conversation_history.append(AIMessage(content=response_content))
    
    # 第二轮
    user_input = "能给我一个例子吗？"
    conversation_history.append(HumanMessage(content=user_input))
    
    stream = llm.astream(conversation_history)
    ai_sdk_stream = LangChainAdapter.to_data_stream_response(stream)
    
    async for chunk in ai_sdk_stream:
        yield chunk
```

有关包括代理集成、工具使用和错误处理的完整示例，请查看 `web/` 目录。

## 🧪 使用示例

以下是展示适配器不同使用方式的全面示例：

### 基础 LangChain 流式处理

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_aisdk_adapter import LangChainAdapter

# 简单流式示例
llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)
stream = llm.astream([HumanMessage(content="给我讲个笑话")])

# 转换为 AI SDK 格式
ai_sdk_stream = LangChainAdapter.to_data_stream_response(stream)

# 处理流
async for chunk in ai_sdk_stream:
    print(chunk, end="", flush=True)
    # 输出: 0:"为什么鸡要过马路？"
    #      0:" 因为它想到对面去！"
```

### LangChain 与工具

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_aisdk_adapter import LangChainAdapter

# 定义工具
@tool
def get_weather(city: str) -> str:
    """获取城市的天气信息。"""
    return f"{city}的天气是晴天，25°C"

# 创建代理
llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的助手。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_openai_functions_agent(llm, [get_weather], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[get_weather])

# 带工具的流式处理
stream = agent_executor.astream({"input": "巴黎的天气怎么样？"})
ai_sdk_stream = LangChainAdapter.to_data_stream_response(stream)

async for chunk in ai_sdk_stream:
    print(chunk)
    # 输出包括:
    # 9:{"toolCallId":"call_123","toolName":"get_weather","args":{"city":"巴黎"}}
    # a:{"toolCallId":"call_123","result":"巴黎的天气是晴天，25°C"}
    # 0:"巴黎的天气是晴天，25°C"
```

### LangGraph 工作流（包含步骤协议）

```python
from langgraph import StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_aisdk_adapter import LangChainAdapter
from typing import TypedDict, List

class State(TypedDict):
    messages: List[HumanMessage | AIMessage]

def chat_node(state: State):
    llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

# 创建工作流
workflow = StateGraph(State)
workflow.add_node("chat", chat_node)
workflow.set_entry_point("chat")
workflow.set_finish_point("chat")

app = workflow.compile()

# 流式处理 LangGraph 工作流
stream = app.astream({"messages": [HumanMessage(content="你好！")]})
ai_sdk_stream = LangChainAdapter.to_data_stream_response(stream)

async for chunk in ai_sdk_stream:
    print(chunk)
    # 输出包括 LangGraph 特定协议:
    # f:{"stepId":"step_123","stepType":"node_execution"}
    # 0:"你好！我今天能为您做些什么？"
    # e:{"stepId":"step_123","finishReason":"completed"}
    # d:{"finishReason":"stop","usage":{"promptTokens":10,"completionTokens":15}}
```

### 自定义配置示例

```python
from langchain_aisdk_adapter import LangChainAdapter, AdapterConfig

# 仅文本输出
config = AdapterConfig(
    enable_text=True,
    enable_data=False,
    enable_tool_calls=False,
    enable_steps=False
)

# 仅工具交互
config = AdapterConfig.tools_only()

# 除步骤外的所有功能（适用于基础 LangChain）
config = AdapterConfig(
    enable_text=True,
    enable_data=True,
    enable_tool_calls=True,
    enable_tool_results=True,
    enable_steps=False  # 禁用 LangGraph 特定协议
)

stream = LangChainAdapter.to_data_stream_response(your_stream, config=config)
```

### 多用户应用的线程安全配置

```python
from langchain_aisdk_adapter import ThreadSafeAdapterConfig
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

# 为 FastAPI 创建线程安全配置
safe_config = ThreadSafeAdapterConfig()

app = FastAPI()

@app.post("/chat")
async def chat(message: str):
    """每个请求都有隔离的协议状态"""
    stream = llm.astream([HumanMessage(content=message)])
    
    # 线程安全：每个请求都有隔离的配置
    return StreamingResponse(
        LangChainAdapter.to_data_stream_response(stream, config=safe_config),
        media_type="text/plain"
    )

@app.post("/chat-minimal")
async def chat_minimal(message: str):
    """仅为此请求临时禁用某些协议"""
    stream = llm.astream([HumanMessage(content=message)])
    
    # 使用上下文管理器临时修改协议
    with safe_config.pause_protocols(['2', '9', 'a']):  # 禁用数据和工具
        return StreamingResponse(
            LangChainAdapter.to_data_stream_response(stream, config=safe_config),
            media_type="text/plain"
        )
    # 上下文结束后协议自动恢复

@app.post("/chat-selective")
async def chat_selective(message: str):
    """仅为此请求启用特定协议"""
    stream = llm.astream([HumanMessage(content=message)])
    
    # 仅启用文本和数据协议
    with safe_config.protocols(['0', '2']):
        return StreamingResponse(
            LangChainAdapter.to_data_stream_response(stream, config=safe_config),
            media_type="text/plain"
        )
```

### 协议上下文管理

```python
from langchain_aisdk_adapter import AdapterConfig, ThreadSafeAdapterConfig

# 带上下文管理的常规配置
config = AdapterConfig()

# 临时禁用特定协议
with config.pause_protocols(['0', '2']):  # 暂停文本和数据
    # 只会生成工具调用和结果
    stream = LangChainAdapter.to_data_stream_response(some_stream, config=config)
    async for chunk in stream:
        print(chunk)  # 没有文本或数据协议
# 协议自动恢复

# 仅启用特定协议
with config.protocols(['0', '9', 'a']):  # 仅文本、工具调用和结果
    stream = LangChainAdapter.to_data_stream_response(some_stream, config=config)
    async for chunk in stream:
        print(chunk)  # 仅指定的协议

# 并发应用的线程安全版本
safe_config = ThreadSafeAdapterConfig()

# 每个上下文都按请求/线程隔离
with safe_config.protocols(['0']):  # 仅文本
    # 这不会影响其他并发请求
    stream = LangChainAdapter.to_data_stream_response(stream1, config=safe_config)

# 支持嵌套上下文
with safe_config.pause_protocols(['2']):
    with safe_config.protocols(['0', '9']):
        # 仅文本和工具调用，数据被暂停
        stream = LangChainAdapter.to_data_stream_response(stream2, config=safe_config)
```

### 错误处理

```python
from langchain_aisdk_adapter import LangChainAdapter
import asyncio

async def safe_streaming():
    try:
        stream = llm.astream([HumanMessage(content="你好")])
        ai_sdk_stream = LangChainAdapter.to_data_stream_response(stream)
        
        async for chunk in ai_sdk_stream:
            print(chunk, end="", flush=True)
            
    except Exception as e:
        print(f"流式处理过程中出错: {e}")
        # 适当处理错误

asyncio.run(safe_streaming())
```

### 与回调集成

```python
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_aisdk_adapter import LangChainAdapter

class CustomCallback(AsyncCallbackHandler):
    async def on_llm_start(self, serialized, prompts, **kwargs):
        print("LLM 开始")
    
    async def on_llm_end(self, response, **kwargs):
        print("LLM 结束")

# 使用回调
llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=True, callbacks=[CustomCallback()])
stream = llm.astream([HumanMessage(content="你好")])
ai_sdk_stream = LangChainAdapter.to_data_stream_response(stream)

# 适配器将捕获回调事件作为数据协议
async for chunk in ai_sdk_stream:
    print(chunk)
    # 可能包括: 2:[{"event":"llm_start","timestamp":"..."}]
```

## 🧪 测试

我们认真对待测试并保持高覆盖率：

```bash
# 安装开发依赖
pip install langchain-aisdk-adapter[dev]

# 运行带覆盖率的测试
pytest tests/ -v --cov=src --cov-report=term-missing

# 当前覆盖率: 98%
```

## 📚 API 参考

### LangChainAdapter

主要适配器类：

```python
class LangChainAdapter:
    @staticmethod
    async def to_data_stream_response(
        stream: AsyncIterator,
        config: Optional[AdapterConfig] = None
    ) -> AsyncIterator[str]:
        """将 LangChain 流转换为 AI SDK 格式"""
```

### AdapterConfig

用于控制协议生成的配置类：

```python
class AdapterConfig:
    enable_text: bool = True
    enable_data: bool = True
    enable_tool_calls: bool = True
    enable_tool_results: bool = True
    enable_steps: bool = True
    enable_reasoning: bool = False  # 仅手动
    enable_annotations: bool = False  # 仅手动
    enable_files: bool = False  # 仅手动
    
    @classmethod
    def minimal(cls) -> "AdapterConfig": ...
    
    @classmethod
    def tools_only(cls) -> "AdapterConfig": ...
    
    @classmethod
    def comprehensive(cls) -> "AdapterConfig": ...
    
    @contextmanager
    def pause_protocols(self, protocol_types: List[str]):
        """临时禁用特定协议类型"""
    
    @contextmanager
    def protocols(self, protocol_types: List[str]):
        """仅启用特定协议类型"""
```

### ThreadSafeAdapterConfig

多用户应用的线程安全配置包装器：

```python
class ThreadSafeAdapterConfig:
    def __init__(self, base_config: Optional[AdapterConfig] = None):
        """使用可选的基础配置初始化"""
    
    def is_protocol_enabled(self, protocol_type: str) -> bool:
        """检查协议是否启用（线程安全）"""
    
    @contextmanager
    def pause_protocols(self, protocol_types: List[str]):
        """线程安全的上下文管理器，临时禁用协议"""
    
    @contextmanager
    def protocols(self, protocol_types: List[str]):
        """线程安全的上下文管理器，仅启用特定协议"""
```

**主要特性：**
- **线程隔离**: 每个请求/线程都有隔离的协议状态
- **上下文管理**: 支持嵌套上下文管理器
- **FastAPI 就绪**: 完美适用于多用户 Web 应用
- **基础配置支持**: 可以包装现有的 AdapterConfig 实例

### AISDKFactory

手动协议创建的工厂类：

```python
class AISDKFactory:
    @staticmethod
    def create_reasoning_part(
        reasoning: str,
        confidence: Optional[float] = None
    ) -> ReasoningPartContent:
        """创建推理协议部分"""
    
    @staticmethod
    def create_tool_call_delta_part(
        tool_call_id: str,
        delta: Dict[str, Any],
        index: int = 0
    ) -> ToolCallDeltaPartContent:
        """创建工具调用增量协议部分"""
    
    @staticmethod
    def create_message_annotation_part(
        annotations: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> MessageAnnotationPartContent:
        """创建消息注释协议部分"""
    
    @staticmethod
    def create_source_part(
        source_id: str,
        source_type: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SourcePartContent:
        """创建来源协议部分"""
    
    @staticmethod
    def create_file_part(
        file_id: str,
        file_name: str,
        file_type: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FilePartContent:
         """创建文件协议部分"""

### 工厂函数（向后兼容）

用于创建协议部分的便捷函数：

```python
# 基础协议创建
create_text_part(text: str) -> AISDKPartEmitter
create_data_part(data: Any) -> AISDKPartEmitter
create_error_part(error: str) -> AISDKPartEmitter

# 工具相关协议
create_tool_call_part(tool_call_id: str, tool_name: str, args: Dict) -> AISDKPartEmitter
create_tool_result_part(tool_call_id: str, result: str) -> AISDKPartEmitter
create_tool_call_streaming_start_part(tool_call_id: str, tool_name: str) -> AISDKPartEmitter

# 步骤协议
create_start_step_part(message_id: str) -> AISDKPartEmitter
create_finish_step_part(finish_reason: str, **kwargs) -> AISDKPartEmitter
create_finish_message_part(finish_reason: str, **kwargs) -> AISDKPartEmitter

# 高级协议
create_redacted_reasoning_part(reasoning: str) -> AISDKPartEmitter
create_reasoning_signature_part(signature: str) -> AISDKPartEmitter

# 通用工厂
create_ai_sdk_part(protocol_type: str, content: Any) -> AISDKPartEmitter
```

### 工厂实例

具有简化方法的便捷工厂实例：

```python
from langchain_aisdk_adapter import factory

# 简化的工厂方法
text_part = factory.text("你好世界")
data_part = factory.data(["键", "值"])
error_part = factory.error("出现错误")
reasoning_part = factory.reasoning("让我想想...")
source_part = factory.source(url="https://example.com", title="示例")
```

### 配置实例

常见用例的预配置实例：

```python
from langchain_aisdk_adapter import default_config, safe_config

# 默认配置实例
default_config: AdapterConfig

# 线程安全配置实例
safe_config: ThreadSafeAdapterConfig
```

## 🤝 贡献

我们欢迎贡献！这个项目仍处于 alpha 阶段，所以有很大的改进空间：

1. Fork 仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 进行更改并添加测试
4. 确保测试通过 (`pytest tests/`)
5. 提交 Pull Request

请随时：
- 报告错误和问题
- 建议新功能
- 改进文档
- 添加更多示例
- 提高测试覆盖率

## 📄 许可证

本项目采用 Apache License 2.0 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件。

## 🙏 致谢

我们感谢：
- LangChain 团队提供的优秀框架
- AI SDK 社区提供的流协议规范
- 所有帮助改进这个项目的贡献者和用户

## 📞 支持

如果您遇到任何问题或有疑问：

- 📋 [提交问题](https://github.com/lointain/langchain_aisdk_adapter/issues)
- 📖 [查看文档](https://github.com/lointain/langchain_aisdk_adapter#readme)
- 💬 [开始讨论](https://github.com/lointain/langchain_aisdk_adapter/discussions)

我们感谢您在我们继续改进这个 alpha 版本时的耐心！

---

*为 LangChain 和 AI SDK 社区用 ❤️ 制作*