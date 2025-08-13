# Streaming Modules

This document describes the optional streaming modules that have been separated from the core LangChain AI SDK Adapter functionality.

## Overview

The streaming functionality has been modularized into separate, optional components:

- **`stream_text.py`** - AI SDK compatible text streaming
- **`smooth_stream.py`** - Configurable chunk smoothing and timing control

## Modules

### stream_text.py

Provides AI SDK compatible text streaming functionality.

#### Functions

- **`stream_text(model, messages, **options)`** - Main streaming function
  - Returns a `StreamTextResult` object with `data_stream` attribute
  - Compatible with AI SDK v4/v5 protocols
  - Supports callbacks, tools, and transformations

- **`stream_text_response(model, messages, **options)`** - FastAPI-compatible streaming response function
  - Returns a `DataStreamResponse` for direct use in FastAPI endpoints
  - Includes HTTP headers and status code configuration
  - Built on top of `stream_text` functionality

#### Usage

```python
from langchain_aisdk_adapter.stream_text import stream_text, stream_text_response
from langchain_openai import ChatOpenAI
from fastapi import FastAPI

model = ChatOpenAI()
messages = [{"role": "user", "content": "Hello!"}]

# Basic streaming
result = stream_text(model=model, messages=messages)

async for chunk in result.data_stream:
    print(chunk)

# FastAPI streaming response
app = FastAPI()

@app.post("/chat/stream")
async def chat_stream(request):
    return stream_text_response(
        model=model,
        messages=request.messages,
        system="You are helpful",
        temperature=0.7,
        headers={"X-Custom-Header": "value"}
    )
```

### smooth_stream.py

Provides smooth streaming with configurable chunking and timing.

#### Classes

- **`SmoothStreamTransformer`** - Core transformer for smooth streaming

#### Functions

- **`smooth_stream(stream, delay_in_ms=10, chunking='word')`** - Transform stream with smooth chunking
- **`create_smooth_text_stream(text, delay_in_ms=10, chunking='word')`** - Create smooth stream from static text

#### Chunking Options

- `'word'` - Split by words (default)
- `'line'` - Split by lines
- `re.Pattern` - Split using regex pattern
- `Callable[[str], List[str]]` - Custom chunking function

#### Usage

```python
from langchain_aisdk_adapter.smooth_stream import smooth_stream, create_smooth_text_stream
import re

# Word-based chunking
smooth = smooth_stream(text_stream, delay_in_ms=50, chunking='word')

# Line-based chunking
smooth = smooth_stream(text_stream, delay_in_ms=100, chunking='line')

# Regex-based chunking
pattern = re.compile(r'[.!?]+\s*')
smooth = smooth_stream(text_stream, delay_in_ms=200, chunking=pattern)

# Custom function chunking
def custom_chunker(text: str) -> List[str]:
    return text.split(',')

smooth = smooth_stream(text_stream, delay_in_ms=75, chunking=custom_chunker)

# Direct text streaming
text = "Hello world! This is a test."
smooth = create_smooth_text_stream(text, delay_in_ms=100, chunking='word')

async for chunk in smooth:
    print(chunk, end='', flush=True)
```

## Integration with stream_text

You can combine both modules using the `experimental_transform` parameter:

```python
from langchain_aisdk_adapter.stream_text import stream_text
from langchain_aisdk_adapter.smooth_stream import smooth_stream

result = stream_text(
    model=model,
    messages=messages,
    experimental_transform=lambda stream: smooth_stream(
        stream, 
        chunking="word",
        delay_in_ms=50
    )
)

async for chunk in result.data_stream:
    print(chunk)
```

## Import Aliases

For convenience, the main package exports these functions with aliases:

```python
from langchain_aisdk_adapter import (
    stream_text_func,      # alias for stream_text
    smooth_stream_func,    # alias for smooth_stream
    create_smooth_text_stream  # direct import
)
```

## Design Philosophy

These modules are designed as **optional components** that extend the core adapter functionality:

1. **Separation of Concerns** - Streaming logic is separate from core adapter
2. **Optional Dependencies** - Can be used independently or together
3. **AI SDK Compatibility** - Maintains compatibility with AI SDK patterns
4. **Flexible Configuration** - Extensive customization options

## Examples

See `example_usage.py` for complete working examples of all streaming functionality.

## Migration from Core Methods

If you were previously using static methods on `LangChainAdapter`:

```python
# Old way (deprecated)
from langchain_aisdk_adapter import LangChainAdapter
result = LangChainAdapter.stream_text(model, messages)

# New way (recommended)
from langchain_aisdk_adapter.stream_text import stream_text
result = stream_text(model, messages)
```

The new modular approach provides better separation of concerns and more flexible usage patterns.