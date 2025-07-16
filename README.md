# LangChain AI SDK Adapter

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Development Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://pypi.org/project/langchain-aisdk-adapter/)

> **⚠️ Alpha Release Notice**: This project is currently in alpha stage. While we strive for stability and reliability, please be aware that APIs may change and some features might still be under development. We appreciate your patience and welcome any feedback to help us improve!

A thoughtfully designed Python adapter that bridges LangChain/LangGraph applications with the AI SDK UI Stream Protocol. This library aims to make it easier for developers to integrate LangChain's powerful capabilities with modern streaming interfaces.

## ✨ Features

We've tried to make this adapter as comprehensive and user-friendly as possible:

- **🔄 Comprehensive Protocol Support**: Supports 15+ AI SDK protocols including text streaming, tool interactions, step management, and data handling
- **⚙️ Intelligent Configuration**: Flexible `AdapterConfig` system allows you to control exactly which protocols are generated
- **⏸️ Protocol Control**: Advanced pause/resume functionality for temporarily disabling specific protocols during execution
- **🔒 Thread-Safe Multi-User Support**: `ThreadSafeAdapterConfig` provides isolated protocol state for concurrent requests in web applications
- **🎛️ Dynamic Protocol Control**: Context managers for temporarily enabling/disabling specific protocols during execution
- **🛠️ Rich Tool Support**: Seamless integration with LangChain tools, agents, and function calling
- **🔍 Extended Step Detection**: Enhanced recognition of LangChain agents, chains, and LangGraph components for better step tracking
- **📊 Usage Tracking**: Built-in statistics and monitoring for stream processing
- **🔒 Type Safety**: Complete Python type hints and Pydantic validation
- **🏭 Factory Methods**: Convenient factory methods for manual protocol generation when needed
- **🌐 Web Framework Ready**: Optimized for FastAPI, Flask, and other web frameworks
- **🔌 Extensible Design**: Easy to extend and customize for specific use cases

## 🚀 Quick Start

We hope this gets you up and running quickly:

### Installation

```bash
# Basic installation
pip install langchain-aisdk-adapter

# With examples (includes LangChain, LangGraph, OpenAI)
pip install langchain-aisdk-adapter[examples]

# With web framework support (includes FastAPI, Uvicorn)
pip install langchain-aisdk-adapter[web]

# For development (includes testing and linting tools)
pip install langchain-aisdk-adapter[dev]
```

### Basic Usage

Here's a simple example to get you started:

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_aisdk_adapter import LangChainAdapter

# Create your LangChain model
llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)

# Create a stream
stream = llm.astream([HumanMessage(content="Hello, world!")])

# Convert to AI SDK format - it's that simple!
ai_sdk_stream = LangChainAdapter.to_data_stream_response(stream)

# Use in your application
async for chunk in ai_sdk_stream:
    print(chunk, end="", flush=True)
```

### Configuration Options

We've included several preset configurations to make common use cases easier:

```python
from langchain_aisdk_adapter import LangChainAdapter, AdapterConfig, ThreadSafeAdapterConfig

# Minimal output - just essential text and data
config = AdapterConfig.minimal()

# Focus on tool interactions
config = AdapterConfig.tools_only()

# Everything enabled (default)
config = AdapterConfig.comprehensive()

# Custom configuration
config = AdapterConfig(
    enable_text=True,
    enable_data=True,
    enable_tool_calls=True,
    enable_steps=False,  # Disable step tracking
    enable_reasoning=False  # Disable reasoning output
)

stream = LangChainAdapter.to_data_stream_response(your_stream, config=config)

# Protocol pause/resume functionality
with config.pause_protocols(['0', '2']):  # Temporarily disable text and data protocols
    # During this block, text and data protocols won't be generated
    restricted_stream = LangChainAdapter.to_data_stream_response(some_stream, config=config)
    async for chunk in restricted_stream:
        # Only tool calls, results, and steps will be emitted
        print(chunk)
# Protocols are automatically restored after the context

# Thread-safe configuration for multi-user applications
safe_config = ThreadSafeAdapterConfig()

# Each request gets isolated protocol state
with safe_config.protocols(['0', '9', 'a']):  # Only text, tool calls, and results
    stream = LangChainAdapter.to_data_stream_response(your_stream, config=safe_config)
    # This configuration won't affect other concurrent requests
```

## 📋 Protocol Support Status

We've organized the supported protocols into three categories to help you understand what's available and when they're triggered:

### 🟢 Automatically Supported Protocols

These protocols are generated automatically from LangChain/LangGraph events with specific trigger conditions:

#### **`0:` (Text Protocol)**
**Trigger Condition**: Generated when LLM produces streaming text content
**Format**: `0:"streaming text content"`
**When it occurs**: 
- During `llm.astream()` calls
- When LangGraph nodes produce text output
- Any streaming text from language models

#### **`2:` (Data Protocol)**
**Trigger Condition**: Generated for structured data and metadata
**Format**: `2:[{"key":"value"}]`
**When it occurs**:
- LangGraph node metadata and intermediate results
- Tool execution metadata
- Custom data from LangChain callbacks

#### **`9:` (Tool Call Protocol)**
**Trigger Condition**: Generated when tools are invoked
**Format**: `9:{"toolCallId":"call_123","toolName":"search","args":{"query":"test"}}`
**When it occurs**:
- LangChain agent tool invocations
- LangGraph tool node executions
- Function calling in chat models

#### **`a:` (Tool Result Protocol)**
**Trigger Condition**: Generated when tool execution completes
**Format**: `a:{"toolCallId":"call_123","result":"tool output"}`
**When it occurs**:
- After successful tool execution
- Following any `9:` protocol
- Both successful and error results

#### **`b:` (Tool Call Stream Start Protocol)**
**Trigger Condition**: Generated at the beginning of streaming tool calls
**Format**: `b:{"toolCallId":"call_123","toolName":"search"}`
**When it occurs**:
- Before tool parameter streaming begins
- Only for tools that support streaming parameters

#### **`d:` (Finish Message Protocol)** ⚠️ **LangGraph Only**
**Trigger Condition**: **Only generated in LangGraph workflows** when a message is completed
**Format**: `d:{"finishReason":"stop","usage":{"promptTokens":10,"completionTokens":20}}`
**When it occurs**:
- **LangGraph workflow message completion**
- **NOT generated in basic LangChain streams**
- End of LangGraph node execution
- Contains usage statistics when available

#### **`e:` (Finish Step Protocol)** 🔄 **Enhanced Support**
**Trigger Condition**: Generated when major workflow components complete execution
**Format**: `e:{"stepId":"step_123","finishReason":"completed"}`
**When it occurs**:
- **LangGraph workflow step completion** (primary use case)
- **LangChain agent execution** (AgentExecutor, ReActAgent, ChatAgent, etc.)
- **Chain-based workflows** (LLMChain, SequentialChain, RouterChain, etc.)
- **Components with specific tags** (agent, chain, executor, workflow, multi_agent)
- End of multi-step processes and reasoning steps

#### **`f:` (Start Step Protocol)** 🔄 **Enhanced Support**
**Trigger Condition**: Generated when major workflow components begin execution
**Format**: `f:{"stepId":"step_123","stepType":"agent_action"}`
**When it occurs**:
- **LangGraph workflow step initiation** (primary use case)
- **LangChain agent execution** (AgentExecutor, ReActAgent, ChatAgent, etc.)
- **Chain-based workflows** (LLMChain, SequentialChain, RouterChain, etc.)
- **LangGraph components** (LangGraph, CompiledGraph, StateGraph, etc.)
- **Components with specific tags** (agent, chain, executor, workflow, multi_agent, langgraph, graph)
- Beginning of multi-step processes and reasoning steps

> 💡 **Important Notes**: 
> - Protocols `d:`, `e:`, and `f:` are **LangGraph-specific** and will not appear in basic LangChain streams
> - All automatically supported protocols can be individually enabled or disabled through `AdapterConfig`
> - The exact format may vary based on the underlying LangChain/LangGraph event structure

### 🟡 Manual Support Only Protocols

These protocols require manual generation using our factory methods:

#### **`g:` (Reasoning Protocol)** ⚠️ **Manual Support Only**
**Purpose**: Transmits AI reasoning process and thought chains
**Format**: `g:{"reasoning":"Let me think about this step by step...","confidence":0.85}`
**Manual Creation**:
```python
from langchain_aisdk_adapter import AISDKFactory

# Create reasoning protocol
reasoning_part = AISDKFactory.create_reasoning_part(
    reasoning="Let me analyze the user's request...",
    confidence=0.9
)
print(f"g:{reasoning_part.model_dump_json()}")
```
**Use Cases**: Chain-of-thought reasoning, decision explanations, confidence scoring

#### **`c:` (Tool Call Delta Protocol)** ⚠️ **Manual Support Only**
**Purpose**: Streams incremental updates during tool call execution
**Format**: `c:{"toolCallId":"call_123","delta":{"function":{"arguments":"{\"query\":\"hello\"}"}},"index":0}`
**Manual Creation**:
```python
from langchain_aisdk_adapter import AISDKFactory

# Create tool call delta
delta_part = AISDKFactory.create_tool_call_delta_part(
    tool_call_id="call_123",
    delta={"function": {"arguments": '{"query":"hello"}'}},
    index=0
)
print(f"c:{delta_part.model_dump_json()}")
```
**Use Cases**: Real-time tool execution feedback, streaming function calls

#### **`8:` (Message Annotations Protocol)** ⚠️ **Manual Support Only**
**Purpose**: Adds metadata and annotations to messages
**Format**: `8:{"annotations":[{"type":"citation","text":"Source: Wikipedia"}],"metadata":{"confidence":0.95}}`
**Manual Creation**:
```python
from langchain_aisdk_adapter import AISDKFactory

# Create message annotation
annotation_part = AISDKFactory.create_message_annotation_part(
    annotations=[{"type": "citation", "text": "Source: Wikipedia"}],
    metadata={"confidence": 0.95}
)
print(f"8:{annotation_part.model_dump_json()}")
```
**Use Cases**: Source citations, confidence scores, content metadata

#### **`h:` (Source Protocol)** ✅ **Manual Support**
**Manual Creation**: Use `create_source_part(url, title=None)` or `AISDKFactory.source(url, title=None)`
**Format**: `h:{"url":"https://example.com","title":"Document Title"}`
**Use Cases**: Document references, citation tracking, source attribution

#### **`i:` (Redacted Reasoning Protocol)** ✅ **Manual Support**
**Manual Creation**: Use `create_redacted_reasoning_part(data)` or `AISDKFactory.redacted_reasoning(data)`
**Format**: `i:{"data":"[REDACTED] reasoning content"}`
**Use Cases**: Privacy-compliant reasoning output, content filtering

#### **`j:` (Reasoning Signature Protocol)** ✅ **Manual Support**
**Manual Creation**: Use `create_reasoning_signature_part(signature)` or `AISDKFactory.reasoning_signature(signature)`
**Format**: `j:{"signature":"signature_abc123"}`
**Use Cases**: Reasoning verification, model signatures, authenticity tracking

#### **`k:` (File Protocol)** ✅ **Manual Support**
**Manual Creation**: Use `create_file_part(data, mime_type)` or `AISDKFactory.file(data, mime_type)`
**Format**: `k:{"data":"base64_encoded_data","mimeType":"image/png"}`
**Use Cases**: File attachments, binary data transmission, document sharing

#### **`h:` (Source Protocol)** ✅ **Manual Support**
**Manual Creation**: Use `create_source_part(url, title=None)` or `AISDKFactory.source(url, title=None)`
**Format**: `h:{"url":"https://example.com","title":"Document Title"}`
**Use Cases**: Document references, citation tracking, source attribution

#### **`i:` (Redacted Reasoning Protocol)** ✅ **Manual Support**
**Manual Creation**: Use `create_redacted_reasoning_part(data)` or `AISDKFactory.redacted_reasoning(data)`
**Format**: `i:{"data":"[REDACTED] reasoning content"}`
**Use Cases**: Privacy-compliant reasoning output, content filtering

#### **`j:` (Reasoning Signature Protocol)** ✅ **Manual Support**
**Manual Creation**: Use `create_reasoning_signature_part(signature)` or `AISDKFactory.reasoning_signature(signature)`
**Format**: `j:{"signature":"signature_abc123"}`
**Use Cases**: Reasoning verification, model signatures, authenticity tracking

#### **`k:` (File Protocol)** ✅ **Manual Support**
**Manual Creation**: Use `create_file_part(data, mime_type)` or `AISDKFactory.file(data, mime_type)`
**Format**: `k:{"data":"base64_encoded_data","mimeType":"image/png"}`
**Use Cases**: File attachments, binary data transmission, document sharing

### 🔴 Currently Unsupported Protocols

We're working on these, but they're not yet available:

- **`1:` (Function Call)**: Different from LangChain's tool system architecture
- **`4:` (Tool Call Stream)**: Requires streaming parameter support not available in current LangChain versions
- **`5:` (Tool Call Stream Part)**: Same limitation as above
- **`6:` (Tool Call Stream Delta)**: Same limitation as above
- **`7:` (Tool Call Stream Finish)**: Same limitation as above

## 🛠️ Manual Protocol Generation

For protocols that need manual implementation, we've provided convenient factory methods:

```python
from langchain_aisdk_adapter.factory import AISDKFactory

# Create factory instance
factory = AISDKFactory()

# Generate reasoning protocol
reasoning_part = factory.reasoning(
    content="Let me think about this step by step..."
)

# Generate source protocol
source_part = factory.source(
    url="https://example.com/document",
    title="Important Document"
)

# Generate redacted reasoning protocol
redacted_part = factory.redacted_reasoning(
    data="[REDACTED] sensitive reasoning content"
)

# Generate reasoning signature protocol
signature_part = factory.reasoning_signature(
    signature="model_signature_abc123"
)

# Generate file protocol
file_part = factory.file(
    data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
    mime_type="image/png"
)

# Generate message annotation
annotation_part = factory.annotation(
    message_id="msg_123",
    annotation_type="confidence",
    value={"score": 0.95}
)

# Generate tool call delta (for streaming parameters)
tool_delta_part = factory.tool_call_delta(
    tool_call_id="call_123",
    name="search",
    args_delta='{"query": "artificial intel'  # partial JSON
)

# Use factory instance for quick protocol creation
from langchain_aisdk_adapter import factory

# Simplified factory methods
text_part = factory.text("Hello from LangChain!")
data_part = factory.data({"temperature": 0.7, "max_tokens": 100})
error_part = factory.error("Connection timeout")
reasoning_part = factory.reasoning("Based on the context, I should...")
source_part = factory.source(
    url="https://docs.langchain.com",
    title="LangChain Documentation"
)

# Use in streaming responses
async def stream_with_factory():
    yield text_part
    yield reasoning_part
    yield data_part
```

**Why Manual Implementation?**

We've had to make some protocols manual due to technical limitations:
- **Reasoning content**: Different LLMs use varying reasoning formats that can't be automatically standardized
- **Tool call deltas**: LangChain's tool system doesn't provide streaming parameter generation
- **Message annotations**: LangChain lacks a standardized event system for message metadata
- **Source tracking**: Document source information requires explicit application-level implementation
- **Content filtering**: Redacted reasoning needs custom privacy and security policies
- **File handling**: Binary file processing and encoding varies significantly across different implementations

## 🌐 Web Integration Examples

We've included comprehensive examples for web frameworks:

### FastAPI Integration

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_aisdk_adapter import LangChainAdapter

app = FastAPI()

@app.post("/chat")
async def chat(message: str):
    # Your LangChain setup here
    stream = llm.astream([HumanMessage(content=message)])
    
    return StreamingResponse(
        LangChainAdapter.to_data_stream_response(stream),
        media_type="text/plain"
    )
```

### Multi-turn Conversations

Handle multi-turn conversations with message history:

```python
from langchain_core.messages import HumanMessage, AIMessage
from langchain_aisdk_adapter import LangChainAdapter

async def multi_turn_chat():
    conversation_history = []
    
    # First turn
    user_input = "What is machine learning?"
    conversation_history.append(HumanMessage(content=user_input))
    
    response_content = ""
    stream = llm.astream(conversation_history)
    ai_sdk_stream = LangChainAdapter.to_data_stream_response(stream)
    
    async for chunk in ai_sdk_stream:
        if chunk.startswith('0:'):
            # Extract text content from protocol
            text_content = chunk[2:].strip('"')
            response_content += text_content
        yield chunk
    
    conversation_history.append(AIMessage(content=response_content))
    
    # Second turn
    user_input = "Can you give me an example?"
    conversation_history.append(HumanMessage(content=user_input))
    
    stream = llm.astream(conversation_history)
    ai_sdk_stream = LangChainAdapter.to_data_stream_response(stream)
    
    async for chunk in ai_sdk_stream:
        yield chunk
```

For complete examples including agent integration, tool usage, and error handling, please check the `web/` directory.

## 🧪 Usage Examples

Here are comprehensive examples showing different ways to use the adapter:

### Basic LangChain Streaming

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_aisdk_adapter import LangChainAdapter

# Simple streaming example
llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)
stream = llm.astream([HumanMessage(content="Tell me a joke")])

# Convert to AI SDK format
ai_sdk_stream = LangChainAdapter.to_data_stream_response(stream)

# Process the stream
async for chunk in ai_sdk_stream:
    print(chunk, end="", flush=True)
    # Output: 0:"Why did the chicken cross the road?"
    #         0:" To get to the other side!"
```

### LangChain with Tools

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_aisdk_adapter import LangChainAdapter

# Define a tool
@tool
def get_weather(city: str) -> str:
    """Get weather information for a city."""
    return f"The weather in {city} is sunny and 25°C"

# Create agent
llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_openai_functions_agent(llm, [get_weather], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[get_weather])

# Stream with tools
stream = agent_executor.astream({"input": "What's the weather in Paris?"})
ai_sdk_stream = LangChainAdapter.to_data_stream_response(stream)

async for chunk in ai_sdk_stream:
    print(chunk)
    # Output includes:
    # 9:{"toolCallId":"call_123","toolName":"get_weather","args":{"city":"Paris"}}
    # a:{"toolCallId":"call_123","result":"The weather in Paris is sunny and 25°C"}
    # 0:"The weather in Paris is sunny and 25°C"
```

### LangGraph Workflow (with Step Protocols)

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

# Create workflow
workflow = StateGraph(State)
workflow.add_node("chat", chat_node)
workflow.set_entry_point("chat")
workflow.set_finish_point("chat")

app = workflow.compile()

# Stream LangGraph workflow
stream = app.astream({"messages": [HumanMessage(content="Hello!")]})
ai_sdk_stream = LangChainAdapter.to_data_stream_response(stream)

async for chunk in ai_sdk_stream:
    print(chunk)
    # Output includes LangGraph-specific protocols:
    # f:{"stepId":"step_123","stepType":"node_execution"}
    # 0:"Hello! How can I help you today?"
    # e:{"stepId":"step_123","finishReason":"completed"}
    # d:{"finishReason":"stop","usage":{"promptTokens":10,"completionTokens":15}}
```

### Custom Configuration Examples

```python
from langchain_aisdk_adapter import LangChainAdapter, AdapterConfig

# Only text output
config = AdapterConfig(
    enable_text=True,
    enable_data=False,
    enable_tool_calls=False,
    enable_steps=False
)

# Only tool interactions
config = AdapterConfig.tools_only()

# Everything except steps (for basic LangChain)
config = AdapterConfig(
    enable_text=True,
    enable_data=True,
    enable_tool_calls=True,
    enable_tool_results=True,
    enable_steps=False  # Disable LangGraph-specific protocols
)

stream = LangChainAdapter.to_data_stream_response(your_stream, config=config)
```

### Thread-Safe Configuration for Multi-User Applications

```python
from langchain_aisdk_adapter import ThreadSafeAdapterConfig
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

# Create thread-safe configuration for FastAPI
safe_config = ThreadSafeAdapterConfig()

app = FastAPI()

@app.post("/chat")
async def chat(message: str):
    """Each request gets isolated protocol state"""
    stream = llm.astream([HumanMessage(content=message)])
    
    # Thread-safe: each request has isolated configuration
    return StreamingResponse(
        LangChainAdapter.to_data_stream_response(stream, config=safe_config),
        media_type="text/plain"
    )

@app.post("/chat-minimal")
async def chat_minimal(message: str):
    """Temporarily disable certain protocols for this request only"""
    stream = llm.astream([HumanMessage(content=message)])
    
    # Use context manager to temporarily modify protocols
    with safe_config.pause_protocols(['2', '9', 'a']):  # Disable data and tools
        return StreamingResponse(
            LangChainAdapter.to_data_stream_response(stream, config=safe_config),
            media_type="text/plain"
        )
    # Protocols automatically restored after context

@app.post("/chat-selective")
async def chat_selective(message: str):
    """Enable only specific protocols for this request"""
    stream = llm.astream([HumanMessage(content=message)])
    
    # Enable only text and data protocols
    with safe_config.protocols(['0', '2']):
        return StreamingResponse(
            LangChainAdapter.to_data_stream_response(stream, config=safe_config),
            media_type="text/plain"
        )
```

### Protocol Context Management

```python
from langchain_aisdk_adapter import AdapterConfig, ThreadSafeAdapterConfig

# Regular config with context management
config = AdapterConfig()

# Temporarily disable specific protocols
with config.pause_protocols(['0', '2']):  # Pause text and data
    # Only tool calls and results will be generated
    stream = LangChainAdapter.to_data_stream_response(some_stream, config=config)
    async for chunk in stream:
        print(chunk)  # No text or data protocols
# Protocols automatically restored

# Enable only specific protocols
with config.protocols(['0', '9', 'a']):  # Only text, tool calls, and results
    stream = LangChainAdapter.to_data_stream_response(some_stream, config=config)
    async for chunk in stream:
        print(chunk)  # Only specified protocols

# Thread-safe version for concurrent applications
safe_config = ThreadSafeAdapterConfig()

# Each context is isolated per request/thread
with safe_config.protocols(['0']):  # Text only
    # This won't affect other concurrent requests
    stream = LangChainAdapter.to_data_stream_response(stream1, config=safe_config)

# Nested contexts are supported
with safe_config.pause_protocols(['2']):
    with safe_config.protocols(['0', '9']):
        # Only text and tool calls, data is paused
        stream = LangChainAdapter.to_data_stream_response(stream2, config=safe_config)
```

### Error Handling

```python
from langchain_aisdk_adapter import LangChainAdapter
import asyncio

async def safe_streaming():
    try:
        stream = llm.astream([HumanMessage(content="Hello")])
        ai_sdk_stream = LangChainAdapter.to_data_stream_response(stream)
        
        async for chunk in ai_sdk_stream:
            print(chunk, end="", flush=True)
            
    except Exception as e:
        print(f"Error during streaming: {e}")
        # Handle errors appropriately

asyncio.run(safe_streaming())
```

### Integration with Callbacks

```python
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_aisdk_adapter import LangChainAdapter

class CustomCallback(AsyncCallbackHandler):
    async def on_llm_start(self, serialized, prompts, **kwargs):
        print("LLM started")
    
    async def on_llm_end(self, response, **kwargs):
        print("LLM finished")

# Use with callbacks
llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=True, callbacks=[CustomCallback()])
stream = llm.astream([HumanMessage(content="Hello")])
ai_sdk_stream = LangChainAdapter.to_data_stream_response(stream)

# The adapter will capture callback events as data protocols
async for chunk in ai_sdk_stream:
    print(chunk)
    # May include: 2:[{"event":"llm_start","timestamp":"..."}]
```

## 🧪 Testing

We take testing seriously and maintain high coverage:

```bash
# Install development dependencies
pip install langchain-aisdk-adapter[dev]

# Run tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Current coverage: 98%
```

## 📚 API Reference

### LangChainAdapter

The main adapter class:

```python
class LangChainAdapter:
    @staticmethod
    async def to_data_stream_response(
        stream: AsyncIterator,
        config: Optional[AdapterConfig] = None
    ) -> AsyncIterator[str]:
        """Convert LangChain stream to AI SDK format"""
```

### AdapterConfig

Configuration class for controlling protocol generation:

```python
class AdapterConfig:
    enable_text: bool = True
    enable_data: bool = True
    enable_tool_calls: bool = True
    enable_tool_results: bool = True
    enable_steps: bool = True
    enable_reasoning: bool = False  # Manual only
    enable_annotations: bool = False  # Manual only
    enable_files: bool = False  # Manual only
    
    @classmethod
    def minimal(cls) -> "AdapterConfig": ...
    
    @classmethod
    def tools_only(cls) -> "AdapterConfig": ...
    
    @classmethod
    def comprehensive(cls) -> "AdapterConfig": ...
    
    @contextmanager
    def pause_protocols(self, protocol_types: List[str]):
        """Temporarily disable specific protocol types"""
    
    @contextmanager
    def protocols(self, protocol_types: List[str]):
        """Enable only specific protocol types"""

### AISDKFactory

Factory class for manual protocol creation:

```python
class AISDKFactory:
    @staticmethod
    def create_reasoning_part(
        reasoning: str,
        confidence: Optional[float] = None
    ) -> ReasoningPartContent:
        """Create reasoning protocol part"""
    
    @staticmethod
    def create_tool_call_delta_part(
        tool_call_id: str,
        delta: Dict[str, Any],
        index: int = 0
    ) -> ToolCallDeltaPartContent:
        """Create tool call delta protocol part"""
    
    @staticmethod
    def create_message_annotation_part(
        annotations: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> MessageAnnotationPartContent:
        """Create message annotation protocol part"""
    
    @staticmethod
    def create_source_part(
        source_id: str,
        source_type: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SourcePartContent:
        """Create source protocol part"""
    
    @staticmethod
    def create_file_part(
        file_id: str,
        file_name: str,
        file_type: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FilePartContent:
        """Create file protocol part"""

### Factory Functions (Backward Compatibility)

Convenience functions for creating protocol parts:

```python
# Basic protocol creation
create_text_part(text: str) -> AISDKPartEmitter
create_data_part(data: Any) -> AISDKPartEmitter
create_error_part(error: str) -> AISDKPartEmitter

# Tool-related protocols
create_tool_call_part(tool_call_id: str, tool_name: str, args: Dict) -> AISDKPartEmitter
create_tool_result_part(tool_call_id: str, result: str) -> AISDKPartEmitter
create_tool_call_streaming_start_part(tool_call_id: str, tool_name: str) -> AISDKPartEmitter

# Step protocols
create_start_step_part(message_id: str) -> AISDKPartEmitter
create_finish_step_part(finish_reason: str, **kwargs) -> AISDKPartEmitter
create_finish_message_part(finish_reason: str, **kwargs) -> AISDKPartEmitter

# Advanced protocols
create_redacted_reasoning_part(reasoning: str) -> AISDKPartEmitter
create_reasoning_signature_part(signature: str) -> AISDKPartEmitter

# Generic factory
create_ai_sdk_part(protocol_type: str, content: Any) -> AISDKPartEmitter
```

### Factory Instance

Convenience factory instance with simplified methods:

```python
from langchain_aisdk_adapter import factory

# Simplified factory methods
text_part = factory.text("Hello world")
data_part = factory.data(["key", "value"])
error_part = factory.error("Something went wrong")
reasoning_part = factory.reasoning("Let me think...")
source_part = factory.source(url="https://example.com", title="Example")
```

### Configuration Instances

Pre-configured instances for common use cases:

```python
from langchain_aisdk_adapter import default_config, safe_config

# Default configuration instance
default_config: AdapterConfig

# Thread-safe configuration instance
safe_config: ThreadSafeAdapterConfig
```
```

### ThreadSafeAdapterConfig

Thread-safe configuration wrapper for multi-user applications:

```python
class ThreadSafeAdapterConfig:
    def __init__(self, base_config: Optional[AdapterConfig] = None):
        """Initialize with optional base configuration"""
    
    def is_protocol_enabled(self, protocol_type: str) -> bool:
        """Check if protocol is enabled (thread-safe)"""
    
    @contextmanager
    def pause_protocols(self, protocol_types: List[str]):
        """Thread-safe context manager to temporarily disable protocols"""
    
    @contextmanager
    def protocols(self, protocol_types: List[str]):
        """Thread-safe context manager to enable only specific protocols"""
```

**Key Features:**
- **Thread Isolation**: Each request/thread gets isolated protocol state
- **Context Management**: Supports nested context managers
- **FastAPI Ready**: Perfect for multi-user web applications
- **Base Config Support**: Can wrap existing AdapterConfig instances

## 🤝 Contributing

We welcome contributions! This project is still in alpha, so there's plenty of room for improvement:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Ensure tests pass (`pytest tests/`)
5. Submit a pull request

Please feel free to:
- Report bugs and issues
- Suggest new features
- Improve documentation
- Add more examples
- Enhance test coverage

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We're grateful to:
- The LangChain team for their excellent framework
- The AI SDK community for the streaming protocol specification
- All contributors and users who help make this project better

## 📞 Support

If you encounter any issues or have questions:

- 📋 [Open an issue](https://github.com/lointain/langchain_aisdk_adapter/issues)
- 📖 [Check the documentation](https://github.com/lointain/langchain_aisdk_adapter#readme)
- 💬 [Start a discussion](https://github.com/lointain/langchain_aisdk_adapter/discussions)

We appreciate your patience as we continue to improve this alpha release!

---

*Made with ❤️ for the LangChain and AI SDK communities*
