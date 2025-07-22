"""LangChain to AI SDK adapter implementation."""

import asyncio
from typing import AsyncGenerator, AsyncIterable, Optional, Union

from .callbacks import CallbacksTransformer, StreamCallbacks
from .models import (
    LangChainAIMessageChunk,
    LangChainMessageContentComplex,
    LangChainStreamEvent,
    LangChainStreamInput,
    UIMessageChunk,
)


def _extract_text_from_ai_message_chunk(chunk: LangChainAIMessageChunk) -> str:
    """Extract text content from LangChain AI message chunk.
    
    Args:
        chunk: LangChain AI message chunk
        
    Returns:
        Extracted text content
    """
    # Handle both dict-like and object-like chunks
    if isinstance(chunk, dict):
        content = chunk.get("content", "")
    else:
        # Handle AIMessageChunk object
        content = getattr(chunk, "content", "")
    
    if isinstance(content, str):
        return content
    
    # Handle complex content (list of content items)
    text_parts = []
    if hasattr(content, '__iter__') and not isinstance(content, str):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
    
    return "".join(text_parts)


async def _process_langchain_stream(
    stream: AsyncIterable[LangChainStreamInput],
) -> AsyncGenerator[str, None]:
    """Process LangChain stream and extract text content.
    
    Args:
        stream: Input stream of LangChain events/chunks/strings
        
    Yields:
        Text content extracted from the stream
    """
    async for value in stream:
        # Handle string stream (e.g., from StringOutputParser)
        if isinstance(value, str):
            yield value
            continue
        
        # Handle LangChain stream events v2
        if isinstance(value, dict) and "event" in value:
            event: LangChainStreamEvent = value
            if event["event"] == "on_chat_model_stream":
                chunk_data = event.get("data", {})
                chunk = chunk_data.get("chunk")
                if chunk:
                    text = _extract_text_from_ai_message_chunk(chunk)
                    if text:
                        yield text
            continue
        
        # Handle AI message chunk stream
        if isinstance(value, dict) and "content" in value:
            chunk: LangChainAIMessageChunk = value
            text = _extract_text_from_ai_message_chunk(chunk)
            if text:
                yield text


async def _apply_callbacks(
    stream: AsyncIterable[str],
    callbacks: Optional[StreamCallbacks] = None,
) -> AsyncGenerator[str, None]:
    """Apply callbacks to text stream.
    
    Args:
        stream: Input text stream
        callbacks: Optional callbacks configuration
        
    Yields:
        Text chunks with callbacks applied
    """
    transformer = CallbacksTransformer(callbacks)
    
    try:
        async for chunk in stream:
            processed_chunk = await transformer.transform(chunk)
            yield processed_chunk
    finally:
        await transformer.finish()


async def _convert_to_ui_message_chunks(
    stream: AsyncIterable[str],
    message_id: str = "1",
) -> AsyncGenerator[UIMessageChunk, None]:
    """Convert text stream to UI message chunks.
    
    Args:
        stream: Input text stream
        message_id: Message ID for the chunks
        
    Yields:
        UI message chunks in AI SDK format
    """
    # Emit text-start chunk
    yield {"type": "text-start", "id": message_id}
    
    # Emit text-delta chunks for each text piece
    async for chunk in stream:
        if chunk:  # Only emit non-empty chunks
            yield {"type": "text-delta", "id": message_id, "delta": chunk}
    
    # Emit text-end chunk
    yield {"type": "text-end", "id": message_id}


async def to_ui_message_stream(
    stream: AsyncIterable[LangChainStreamInput],
    callbacks: Optional[StreamCallbacks] = None,
    message_id: str = "1",
) -> AsyncGenerator[UIMessageChunk, None]:
    """Convert LangChain output streams to AI SDK UI Message Stream.
    
    The following streams are supported:
    - LangChainAIMessageChunk streams (LangChain model.stream output)
    - string streams (LangChain StringOutputParser output)
    - LangChainStreamEvent streams (LangChain stream events v2)
    
    Args:
        stream: Input stream from LangChain
        callbacks: Optional callbacks for stream lifecycle events
        message_id: ID for the generated message chunks (default: "1")
        
    Yields:
        UI message chunks compatible with AI SDK
        
    Example:
        ```python
        from langchain_aisdk_adapter import to_ui_message_stream
        
        # Convert LangChain stream to AI SDK format
        async for chunk in to_ui_message_stream(langchain_stream):
            print(chunk)
        ```
    """
    # Step 1: Process LangChain stream to extract text
    text_stream = _process_langchain_stream(stream)
    
    # Step 2: Apply callbacks if provided
    if callbacks:
        text_stream = _apply_callbacks(text_stream, callbacks)
    
    # Step 3: Convert to UI message chunks
    async for chunk in _convert_to_ui_message_chunks(text_stream, message_id):
        yield chunk