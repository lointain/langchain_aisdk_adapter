# LangChain AI SDK Adapter

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/langchain-aisdk-adapter.svg)](https://badge.fury.io/py/langchain-aisdk-adapter)

A Python adapter that converts LangChain streaming outputs to AI SDK compatible data streams, supporting both AI SDK v4 and v5 protocols.

## Features

- **🔄 Protocol Support**: Full support for AI SDK v4 and v5 protocols
- **📡 Streaming Conversion**: Convert LangChain `astream_events()` to AI SDK data streams
- **🛠️ Tool Integration**: Support for tool calls and tool outputs
- **📝 Rich Content**: Handle text, reasoning, files, sources, and custom data
- **⚡ FastAPI Integration**: Direct integration with FastAPI `StreamingResponse`
- **🎯 Manual Control**: Emit custom events with `DataStreamWithEmitters`
- **🔧 Flexible Configuration**: Configurable protocol versions and output formats
- **📊 Usage Tracking**: Built-in token usage and performance monitoring

## Installation

```bash
pip install langchain-aisdk-adapter
```

## Quick Start

### Basic Usage

```python
from langchain_aisdk_adapter import LangChainAdapter
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Initialize your LangChain model
llm = ChatOpenAI(model="gpt-4")
messages = [HumanMessage(content="Hello, world!")]

# Convert to AI SDK data stream
stream = llm.astream_events(messages, version="v2")
data_stream = LangChainAdapter.to_data_stream(stream)

# Iterate over the stream
async for chunk in data_stream:
    print(chunk)
```

### FastAPI Integration

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
    
    # Return FastAPI StreamingResponse
    return LangChainAdapter.to_data_stream_response(
        stream,
        options={"protocol_version": "v5"}  # Use AI SDK v5
    )
```

### Manual Event Emission

```python
from langchain_aisdk_adapter import LangChainAdapter

# Create data stream with manual control
stream = llm.astream_events(messages, version="v2")
data_stream = LangChainAdapter.to_data_stream(stream)

# Emit custom events
await data_stream.emit_text_delta("Custom text")
await data_stream.emit_source_url("https://example.com", "Example")
await data_stream.emit_data({"key": "value"})
```

## Core Components

### LangChainAdapter

The main adapter class providing three core methods:

#### `to_data_stream_response()`
Converts LangChain streams to FastAPI `StreamingResponse`:

```python
response = LangChainAdapter.to_data_stream_response(
    stream=langchain_stream,
    options={
        "protocol_version": "v5",  # "v4" or "v5"
        "output_format": "protocol",  # "chunks" or "protocol"
        "auto_events": True,
        "auto_close": True
    }
)
```

#### `to_data_stream()`
Converts LangChain streams to `DataStreamWithEmitters`:

```python
data_stream = LangChainAdapter.to_data_stream(
    stream=langchain_stream,
    message_id="custom-id",
    options={"protocol_version": "v4"}
)
```

#### `merge_into_data_stream()`
Merges LangChain streams into existing data stream writers:

```python
from langchain_aisdk_adapter import DataStreamWriter

writer = DataStreamWriter()
await LangChainAdapter.merge_into_data_stream(
    stream=langchain_stream,
    data_stream_writer=writer
)
```

### Protocol Support

#### AI SDK v4 Protocol
- Text format: `text/plain; charset=utf-8`
- Header: `x-vercel-ai-data-stream: v1`
- Format: `<type>:<data>\n`

#### AI SDK v5 Protocol
- Text format: `text/event-stream`
- Header: `x-vercel-ai-ui-message-stream: v1`
- Format: Server-Sent Events (SSE)

### Configuration Options

```python
options = {
    "protocol_version": "v5",      # "v4" or "v5"
    "output_format": "protocol",   # "chunks" or "protocol"
    "auto_events": True,           # Auto emit start/finish events
    "auto_close": True,            # Auto close stream
    "emit_start": True,            # Emit start events
    "emit_finish": True            # Emit finish events
}
```

## Advanced Features

### Custom Callbacks

```python
from langchain_aisdk_adapter import BaseAICallbackHandler

class CustomCallback(BaseAICallbackHandler):
    async def on_text_delta(self, delta: str, **kwargs):
        print(f"Text delta: {delta}")
    
    async def on_tool_call_start(self, tool_name: str, **kwargs):
        print(f"Tool call started: {tool_name}")

callbacks = CustomCallback()
data_stream = LangChainAdapter.to_data_stream(
    stream=langchain_stream,
    callbacks=callbacks
)
```

### Manual Stream Control

```python
# Create manual stream controller
from langchain_aisdk_adapter import ManualStreamController

controller = ManualStreamController()

# Emit various chunk types
await controller.emit_text_start("text-1")
await controller.emit_text_delta("Hello", "text-1")
await controller.emit_text_end("Hello World", "text-1")

# Emit tool calls
await controller.emit_tool_input_start("tool-1", "search")
await controller.emit_tool_input_available("tool-1", "search", {"query": "AI"})
await controller.emit_tool_output_available("tool-1", "Results found")

# Get the stream
stream = controller.get_stream()
```

### Error Handling

```python
try:
    data_stream = LangChainAdapter.to_data_stream(stream)
    async for chunk in data_stream:
        if chunk.get("type") == "error":
            print(f"Error: {chunk.get('errorText')}")
except Exception as e:
    print(f"Stream error: {e}")
```

## Supported Chunk Types

| Chunk Type | AI SDK v4 | AI SDK v5 | Description |
|------------|-----------|-----------|-------------|
| `text-start` | ✅ | ✅ | Text generation start |
| `text-delta` | ✅ | ✅ | Text content delta |
| `text-end` | ✅ | ✅ | Text generation end |
| `tool-input-start` | ✅ | ✅ | Tool call input start |
| `tool-input-delta` | ✅ | ✅ | Tool call input delta |
| `tool-input-available` | ✅ | ✅ | Tool call input complete |
| `tool-output-available` | ✅ | ✅ | Tool call output |
| `tool-output-error` | ✅ | ✅ | Tool call error |
| `reasoning` | ✅ | ✅ | Reasoning content |
| `source-url` | ✅ | ✅ | Source URL reference |
| `source-document` | ✅ | ✅ | Source document |
| `file` | ✅ | ✅ | File attachment |
| `data` | ✅ | ✅ | Custom data |
| `error` | ✅ | ✅ | Error message |
| `start-step` | ✅ | ✅ | Step start |
| `finish-step` | ✅ | ✅ | Step finish |
| `start` | ✅ | ✅ | Stream start |
| `finish` | ✅ | ✅ | Stream finish |

## Examples

### Complete Chat Application

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
            SystemMessage(content="You are a helpful assistant."),
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

### Tool Integration Example

```python
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_aisdk_adapter import LangChainAdapter

# Setup tools and agent
search = DuckDuckGoSearchRun()
tools = [search]
llm = ChatOpenAI(model="gpt-4")
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Stream with tool calls
stream = agent_executor.astream_events(
    {"input": "Search for latest AI news"},
    version="v2"
)

data_stream = LangChainAdapter.to_data_stream(
    stream=stream,
    options={"protocol_version": "v5"}
)

async for chunk in data_stream:
    chunk_type = chunk.get("type")
    if chunk_type == "tool-input-available":
        print(f"Tool: {chunk.get('toolName')}")
        print(f"Input: {chunk.get('input')}")
    elif chunk_type == "tool-output-available":
        print(f"Output: {chunk.get('output')}")
```

## API Reference

### Classes

- **`LangChainAdapter`**: Main adapter class
- **`DataStreamWithEmitters`**: Stream with manual emit methods
- **`DataStreamResponse`**: FastAPI response wrapper
- **`DataStreamWriter`**: Stream writer for merging
- **`ManualStreamController`**: Manual stream control
- **`BaseAICallbackHandler`**: Base callback handler
- **`ProtocolStrategy`**: Protocol strategy interface
- **`AISDKv4Strategy`**: AI SDK v4 implementation
- **`AISDKv5Strategy`**: AI SDK v5 implementation

### Functions

- **`to_data_stream()`**: Legacy compatibility function
- **`to_data_stream_response()`**: Legacy compatibility function
- **`merge_into_data_stream()`**: Legacy compatibility function

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

### v0.0.1a1
- Initial release
- Support for AI SDK v4 and v5 protocols
- Core adapter functionality
- FastAPI integration
- Manual event emission
- Tool call support
- Rich content types
- Usage tracking
