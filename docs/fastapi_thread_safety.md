# FastAPI Thread Safety Guide

## Problem

In FastAPI applications with multiple concurrent users, sharing a single `AdapterConfig` instance can lead to protocol state interference between different requests. When one user pauses protocols or uses selective protocol tracking, it affects all other users.

## Solution: ThreadSafeAdapterConfig

The `ThreadSafeAdapterConfig` class provides thread-safe protocol state isolation using Python's `contextvars` module. Each request gets its own isolated configuration context.

## Basic Usage

```python
from fastapi import FastAPI
from langchain_aisdk_adapter import ThreadSafeAdapterConfig, safe_config

app = FastAPI()

@app.post("/chat")
async def chat(request: ChatRequest):
    # Create request-scoped configuration
    thread_safe_config = ThreadSafeAdapterConfig.for_request(safe_config)
    
    # Use context managers safely
    with thread_safe_config.pause_protocols(['2', '6']):
        stream = llm.astream(messages)
        return StreamingResponse(
            LangChainAdapter.to_data_stream_response(
                stream, 
                config=thread_safe_config.base_config
            ),
            media_type="text/plain"
        )
```

## Key Features

### 1. Request Isolation
Each call to `ThreadSafeAdapterConfig.for_request()` creates an isolated configuration context:

```python
# User A's request
config_a = ThreadSafeAdapterConfig.for_request(safe_config)
with config_a.pause_protocols(['2']):
    # Only affects User A
    pass

# User B's request (concurrent)
config_b = ThreadSafeAdapterConfig.for_request(safe_config)
with config_b.protocols(['0', '3']):
    # Only affects User B, User A is unaffected
    pass
```

### 2. Selective Protocol Tracking

```python
@app.post("/chat-minimal")
async def chat_minimal(request: ChatRequest):
    thread_safe_config = ThreadSafeAdapterConfig.for_request(safe_config)
    
    # Only enable text content and finish reason protocols
    with thread_safe_config.protocols(['0', '3']):
        stream = llm.astream(messages)
        return StreamingResponse(
            LangChainAdapter.to_data_stream_response(
                stream, 
                config=thread_safe_config.base_config
            ),
            media_type="text/plain"
        )
```

### 3. Protocol Pausing

```python
@app.post("/chat-no-steps")
async def chat_no_steps(request: ChatRequest):
    thread_safe_config = ThreadSafeAdapterConfig.for_request(safe_config)
    
    # Disable step-related protocols for this request
    with thread_safe_config.pause_protocols(['2', '6']):
        stream = llm.astream(messages)
        return StreamingResponse(
            LangChainAdapter.to_data_stream_response(
                stream, 
                config=thread_safe_config.base_config
            ),
            media_type="text/plain"
        )
```

## Best Practices

1. **Always use `for_request()`**: Create a new thread-safe config for each request
2. **Use context managers**: Always use `with` statements for temporary protocol changes
3. **Pass `base_config`**: Use `thread_safe_config.base_config` when calling the adapter
4. **Don't share instances**: Never share `ThreadSafeAdapterConfig` instances between requests

## Example: Complete FastAPI Application

See `web/fastapi_example.py` for a complete working example with multiple endpoints demonstrating different thread-safe configuration patterns.

## Protocol Reference

- `'0'`: Text content
- `'1'`: Text delta
- `'2'`: Step start
- `'3'`: Finish reason
- `'4'`: Usage information
- `'5'`: Error handling
- `'6'`: Step completion
- `'7'`: Tool calls
- `'8'`: Tool results
- `'9'`: Data content
- `'a'`: Message annotations
- `'b'`: Message metadata
- `'c'`: Response metadata
- `'d'`: Roundtrip metadata
- `'e'`: Warning messages
- `'f'`: Request information
- `'g'`: Response information
- `'h'`: Tool call streaming
- `'i'`: Tool call delta
- `'j'`: Tool result streaming
- `'k'`: Tool result delta