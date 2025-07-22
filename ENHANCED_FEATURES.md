# Enhanced LangChain to AI SDK Adapter Features

## Overview

The enhanced adapter provides complete AI SDK UI Stream Protocol compatibility with advanced features including tool invocation events, step control events, and AI SDK compatible callbacks.

## New Features

### 1. Complete AI SDK Data Stream Protocol Support

The new `to_data_stream()` function provides full compatibility with the AI SDK Data Stream Protocol:

```python
from langchain_aisdk_adapter import to_data_stream, BaseAICallbackHandler

# Enhanced adapter with full protocol support
async for chunk in to_data_stream(langchain_stream, callbacks=callback_handler):
    print(chunk)
```

### 2. Tool Invocation Events

Supports complete tool lifecycle tracking:

- `tool-input-start`: When a tool call begins
- `tool-input-available`: When tool input parameters are available
- `tool-output-available`: When tool execution completes

```python
# Example tool events
{
    "type": "tool-input-start",
    "toolCallId": "tool_call_1",
    "toolName": "get_weather"
}

{
    "type": "tool-input-available",
    "toolCallId": "tool_call_1",
    "toolName": "get_weather",
    "input": {"location": "Tokyo"}
}

{
    "type": "tool-output-available",
    "toolCallId": "tool_call_1",
    "output": "The weather in Tokyo is sunny with 22°C."
}
```

### 3. Step Control Events

Provides workflow step tracking:

- `start-step`: Beginning of a processing step
- `finish-step`: Completion of a processing step

```python
# Step control events
{"type": "start-step"}
{"type": "finish-step"}
```

### 4. Stream Control Events

Manages overall stream lifecycle:

- `start`: Stream initialization with message ID
- `finish`: Stream completion

```python
# Stream control events
{
    "type": "start",
    "messageId": "msg_123"
}

{"type": "finish"}
```

### 5. AI SDK Compatible Callbacks

New `BaseAICallbackHandler` class for advanced callback handling:

```python
from langchain_aisdk_adapter import BaseAICallbackHandler, Message

class MyCallbackHandler(BaseAICallbackHandler):
    async def on_finish(self, message: Message, options: Dict[str, Any]) -> None:
        """Called when stream completes with final message."""
        print(f"Final message: {message.content}")
        print(f"Tool calls: {len([p for p in message.parts if isinstance(p, ToolInvocationUIPart)])}")
    
    async def on_error(self, error: Exception) -> None:
        """Called when an error occurs."""
        print(f"Error: {error}")

# Use with enhanced adapter
async for chunk in to_data_stream(stream, callbacks=MyCallbackHandler()):
    process_chunk(chunk)
```

## Message Structure

The enhanced adapter provides rich message structure with parts:

```python
class Message:
    id: str
    content: str  # Aggregated text content
    role: str     # "assistant"
    parts: List[UIPart]  # Structured parts

# Part types
class TextUIPart:
    type: "text"
    text: str

class ToolInvocationUIPart:
    type: "tool-invocation"
    toolInvocation: ToolInvocation

class ToolInvocation:
    state: "call" | "result"
    toolCallId: str
    toolName: str
    args: Any
    result: Optional[Any]
```

## Usage Examples

### Basic Enhanced Usage

```python
import asyncio
from langchain_aisdk_adapter import to_data_stream

async def basic_example():
    # Your LangChain stream
    langchain_stream = agent_executor.astream({"input": "What's the weather?"})
    
    # Convert to AI SDK format
    async for chunk in to_data_stream(langchain_stream):
        print(f"Event: {chunk['type']}")
        if chunk['type'] == 'text-delta':
            print(f"Text: {chunk['delta']}")
        elif chunk['type'] == 'tool-input-available':
            print(f"Tool: {chunk['toolName']} - {chunk['input']}")

asyncio.run(basic_example())
```

### Advanced Callback Usage

```python
from langchain_aisdk_adapter import BaseAICallbackHandler, to_data_stream

class AdvancedCallbackHandler(BaseAICallbackHandler):
    def __init__(self):
        self.tool_count = 0
        self.text_length = 0
    
    async def on_finish(self, message, options):
        # Analyze final message
        tool_parts = [p for p in message.parts if p.type == "tool-invocation"]
        text_parts = [p for p in message.parts if p.type == "text"]
        
        print(f"Completed with {len(tool_parts)} tools and {len(text_parts)} text parts")
        print(f"Final content length: {len(message.content)}")
        
        # Log tool usage
        for tool_part in tool_parts:
            tool = tool_part.toolInvocation
            print(f"Tool {tool.toolName}: {tool.state}")
    
    async def on_error(self, error):
        print(f"Stream error: {error}")

# Use advanced callbacks
callback_handler = AdvancedCallbackHandler()
async for chunk in to_data_stream(langchain_stream, callbacks=callback_handler):
    # Process chunks...
    pass
```

### Legacy Compatibility

The enhanced adapter maintains backward compatibility:

```python
from langchain_aisdk_adapter import to_data_stream, StreamCallbacks

# Legacy StreamCallbacks still work
legacy_callbacks = StreamCallbacks(
    on_start=lambda: print("Started"),
    on_token=lambda token: print(f"Token: {token}"),
    on_final=lambda text: print(f"Final: {text}")
)

# Will use legacy adapter internally
async for chunk in to_data_stream(stream, callbacks=legacy_callbacks):
    print(chunk)
```

## Event Flow Example

Typical event sequence for an agent with tool calls:

```
1. start (messageId: "msg_123")
2. start-step
3. text-start (id: "text_1")
4. text-delta ("I'll help you with that...")
5. tool-input-start (toolName: "get_weather", toolCallId: "tool_1")
6. tool-input-available (input: {"location": "Tokyo"})
7. tool-output-available (output: "Sunny, 22°C")
8. text-delta ("The weather is sunny...")
9. text-end (id: "text_1")
10. finish-step
11. finish
12. [AI SDK Callback: on_finish called]
```

## Migration Guide

### From `to_ui_message_stream` to `to_data_stream`

```python
# Old way
from langchain_aisdk_adapter import to_ui_message_stream
async for chunk in to_ui_message_stream(stream, callbacks):
    process_chunk(chunk)

# New enhanced way
from langchain_aisdk_adapter import to_data_stream, BaseAICallbackHandler

class MyHandler(BaseAICallbackHandler):
    async def on_finish(self, message, options):
        # Rich message analysis
        pass

async for chunk in to_data_stream(stream, callbacks=MyHandler()):
    process_chunk(chunk)
```

### Callback Migration

```python
# Old StreamCallbacks
old_callbacks = StreamCallbacks(
    on_start=lambda: print("Started"),
    on_final=lambda text: print(f"Final: {text}")
)

# New AI SDK Callbacks
class NewCallbacks(BaseAICallbackHandler):
    async def on_finish(self, message, options):
        print(f"Final: {message.content}")
        # Plus access to structured parts, tool calls, etc.
```

## Benefits

1. **Complete Protocol Support**: Full AI SDK UI Stream Protocol compatibility
2. **Rich Tool Tracking**: Detailed tool invocation lifecycle events
3. **Structured Messages**: Access to parsed message parts and tool results
4. **Advanced Callbacks**: Async callbacks with rich message context
5. **Backward Compatibility**: Existing code continues to work
6. **Better Debugging**: Detailed event tracking and error handling
7. **Production Ready**: Robust error handling and resource management

## Performance

The enhanced adapter is designed for efficiency:

- Minimal overhead over the original adapter
- Efficient event processing and memory usage
- Async-first design for high concurrency
- Proper resource cleanup and error handling

For high-throughput scenarios, the enhanced adapter provides the same performance characteristics as the original while adding comprehensive protocol support.