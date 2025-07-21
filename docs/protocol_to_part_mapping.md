# AI SDK Protocol, Parts, and Callback System Design

This document provides a comprehensive overview of the design for mapping LangChain/LangGraph events to the Vercel AI SDK. It covers the core protocol-to-part mapping, the callback system design, and implementation best practices.

## 1. Core Design Principles

The entire system is built upon the following principles to ensure robustness, compatibility, and extensibility.

### 1.1. Strict Adherence to AI SDK Standard
- All interface definitions are fully compatible with the AI SDK's `index.d.ts`.
- The `Message` object structure strictly matches the AI SDK standard.
- Callback parameter types are consistent with the AI SDK.

### 1.2. Pure Hook Mode for Callbacks
- **Callbacks as Pure Hooks**: They do not replace or modify any built-in stream processing logic.
- **No Super Calls**: Avoid `await super().on_xxx()` calls to maintain isolation.
- **Observation and Recording Only**: Callbacks are used for side effects like listening, analytics, and logging.
- **No Impact on Main Flow**: Callback execution does not affect the original stream processing logic.
- **Independence**: Each callback is independent and does not interfere with others.

### 1.3. Backward Compatibility
- Deprecated fields are retained to support older codebases.
- Callback parameters are optional to avoid breaking existing functionality.
- The system supports a gradual migration path.

### 1.4. Type Safety
- Pydantic is used for strict type validation.
- Clear type annotations and documentation are provided.
- Runtime type checking is enforced.

### 1.5. Performance Optimization
- Asynchronous callback handling for non-blocking operations.
- Minimized object creation and serialization overhead.
- Native support for stream processing.

### 1.6. Extensibility
- Support for custom callback handlers.
- Flexible configuration options.
- A plugin-friendly architecture.

### 1.7. Naming Conventions
- **`AICallbackHandler`**: The abstract interface defining the standard contract for callback handlers.
- **`BaseAICallbackHandler`**: A base implementation class providing default methods, making it easy to inherit and extend.
- **`ProcessSafeCallback`**: A specific implementation for special scenarios (e.g., process safety).
- Avoid vague terms like "Simple"; use "Base" to clearly indicate it's a foundational abstract class.

## 2. Protocol to UI Part Mapping

The generation of UI Parts is directly driven by the AI SDK streaming protocol. The following table details how each protocol message translates into a UI `Part`.

| Protocol | UI Part Created | Trigger | Notes |
| :--- | :--- | :--- | :--- |
| `0: (Text Protocol)` | `TextUIPart` | Auto/Manual | Streams the LLM's text output. Can be appended to an existing `TextUIPart` or create a new one. |
| `2: (Data Protocol)` | `DataUIPart` | Auto/Manual | Sends structured JSON data to be rendered in the UI. Typically used for arbitrary data payloads. |
| `a: (Tool Call Protocol)` | `ToolInvocationUIPart` | Auto/Manual | Signals the start of a tool call. Creates a `ToolInvocationUIPart` with the tool name and arguments. The `result` field is initially empty. |
| `9: (Tool Output Protocol)` | `ToolInvocationUIPart` (Update) | Auto/Manual | Provides the output of a tool execution. This protocol message finds the corresponding `ToolInvocationUIPart` (by `toolCallId`) and fills in its `result` field. |
| `b: (Step Start Protocol)` | `StepStartUIPart` | Auto/Manual | Marks the beginning of a "Step" (a complete interaction unit, usually involving a tool call). Creates a `StepStartUIPart`. |
| `c: (Step End Protocol)` | `StepEndUIPart` | Auto/Manual | Marks the end of a "Step". Creates a `StepEndUIPart`. |
| `e: (Error Protocol)` | `ErrorUIPart` | Auto/Manual | Reports an error that occurred during the process. Creates an `ErrorUIPart` with the error message. |
| `f: (Finish Protocol)` | `FinishUIPart` | Auto/Manual | Signals the successful completion of the entire chain/graph execution. Creates a `FinishUIPart` with a finish reason. |
| `g: (Reasoning Protocol)` | `ReasoningUIPart` | Manual Only | Used to display the model's internal reasoning or thought process. Intended to be called manually by the developer. |
| `h: (Source Protocol)` | `SourceUIPart` | Manual Only | Provides a source or reference, typically rendered as a hyperlink. Intended to be called manually. |

### Manually Triggered UI Parts

The following UI Parts are compliant with the AI SDK specification but are not automatically generated in the standard LangChain/LangGraph event stream. They are intended to be created and streamed manually by the developer as needed.

-   **`ReasoningUIPart`**: Used to display the model's internal reasoning or thought process. This part must be created manually, as there is no standard "reasoning" event in the LangChain stream.
-   **`SourceUIPart`**: Used to provide sources or references, such as documents retrieved by a RAG process. Developers can manually create and stream this part after obtaining the sources.
-   **`FileUIPart`**: Used for file-related UI, such as displaying an uploaded file or providing a download link. This is typically created manually based on specific application logic.

## 3. Implementation and State Management

To manage the stream's state and ensure thread safety in web server environments, we use a factory method pattern.

The `LangChainAdapter.to_data_stream_response` method acts as a factory. For each incoming request, it creates a new, isolated state manager. This manager is responsible for:
1.  Receiving LangChain/LangGraph events.
2.  Translating them into the corresponding AI SDK protocol messages.
3.  Constructing the `Message` object and its `parts` array over the lifecycle of the stream.
4.  Invoking the user-provided callback hooks at the appropriate times.

This ensures that each request has its own state, preventing data leakage or race conditions between concurrent requests.

## 4. Usage Examples

Here is how to use the `AICallbackHandler` in practice.

```python
import time
from typing import Dict, Any, Optional

from langchain_aisdk_adapter import BaseAICallbackHandler, LangChainAdapter
from langchain_aisdk_adapter.callbacks import Message, LanguageModelUsage

# Assume 'config' and 'create_langchain_stream' are defined elsewhere

# --- Basic Usage --- 
# A simple callback to print the final message and usage stats.
class MyCallback(BaseAICallbackHandler):
    async def on_finish(self, message: Message, options: Dict[str, Any]) -> None:
        # This hook is called when the entire stream is finished.
        print(f"Received final message content: {message.content}")
        print(f"Total number of parts: {len(message.parts or [])}")
        
        usage = options.get('usage')
        if usage:
            print(f"Token Usage: {usage.totalTokens} tokens")
    
    async def on_error(self, error: Exception) -> None:
        # This hook is called if an unhandled exception occurs.
        print(f"An error occurred: {error}")

# --- Advanced Usage: Pure Hook for Analytics ---
# This callback only records analytics without affecting the stream.
class AnalyticsCallback(BaseAICallbackHandler):
    def __init__(self):
        self.tool_calls = []
    
    async def on_tool_call(self, tool_call: Dict[str, Any]) -> None:
        # Pure Hook: Only records tool call data for analytics.
        # It does not call any parent method or alter behavior.
        self.tool_calls.append({
            'timestamp': time.time(),
            'tool_name': tool_call.get('toolName'),
            'tool_id': tool_call.get('toolCallId')
        })
    
    async def on_finish(self, message: Message, options: Dict[str, Any]) -> None:
        # Pure Hook: Only performs analysis on the final message.
        part_types = {}
        for part in message.parts or []:
            part_type = part.type
            part_types[part_type] = part_types.get(part_type, 0) + 1
        
        print(f"Final parts analysis: {part_types}")
        print(f"Total tool calls: {len(self.tool_calls)}")
        
        usage = options.get('usage')
        if usage:
            print(f"Token Usage: Total={usage.totalTokens}, Prompt={usage.promptTokens}, Completion={usage.completionTokens}")

# --- Integrating the Callback ---
# The callback instance is passed to the factory method.
langchain_stream = create_langchain_stream(...)
callback_handler = MyCallback()

ai_sdk_stream = LangChainAdapter.to_data_stream_response(
    langchain_stream,
    config=config,
    callback=callback_handler
)

async for chunk in ai_sdk_stream:
    # Process the AI SDK protocol stream
    pass
```

## 5. FastAPI Process Safety Example

When using the callback system in a multi-process or multi-threaded environment like FastAPI, it's crucial to ensure safety by creating a new callback instance for each request.

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

# Assume MyCallback, LangChainAdapter, config, and create_langchain_stream are defined

app = FastAPI()

@app.post("/chat/stream")
async def chat_stream(request: Dict[str, Any]):
    """A streaming chat endpoint that is thread-safe."""
    try:
        langchain_stream = create_langchain_stream(request)
        
        # The factory method pattern ensures thread safety.
        # A new instance of MyCallback is created for each request,
        # and to_data_stream_response creates an isolated state for it.
        return StreamingResponse(
            LangChainAdapter.to_data_stream_response(
                langchain_stream,
                config=config,
                callback=MyCallback()  # A new instance per request is key
            ),
            media_type="text/plain; charset=utf-8"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

```

## Summary

This design ensures full compatibility with the AI SDK while providing flexible extension capabilities. By strictly adhering to the AI SDK standard, our callback system can be seamlessly integrated into any frontend application using the Vercel AI SDK.

### Key Advantages

1.  **Full Compatibility**: 100% consistent with the AI SDK standard.
2.  **Type Safety**: Strict type definitions and validation.
3.  **Ease of Use**: A clean API with comprehensive examples.
4.  **High Performance**: Asynchronous processing and optimized data structures.
5.  **Extensibility**: Support for custom callbacks and plugins.