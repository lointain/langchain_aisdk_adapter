"""LangChain to AI SDK adapter providing core conversion methods."""

import uuid
from typing import AsyncIterable, AsyncGenerator, Optional, Dict, Any

from .models import LangChainStreamInput, UIMessageChunk
from .stream_processor import StreamProcessor
from .data_stream import DataStreamWithEmitters, DataStreamResponse, DataStreamWriter
from .callbacks import BaseAICallbackHandler


class LangChainAdapter:
    """LangChain to AI SDK adapter providing three core methods."""
    
    @staticmethod
    async def to_data_stream_response(
        stream: AsyncIterable[LangChainStreamInput],
        callbacks: Optional[BaseAICallbackHandler] = None,
        message_id: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        status: int = 200,
        options: Optional[Dict[str, Any]] = None
    ) -> DataStreamResponse:
        """Convert LangChain stream to FastAPI StreamingResponse.
        
        Args:
            stream: LangChain async stream from astream_events()
            callbacks: Callback handlers for stream events
            message_id: Optional message ID for tracking
            headers: HTTP headers for the response
            status: HTTP status code
            options: Control options (auto_events, emit_start, emit_finish, etc.)
        
        Returns:
            DataStreamResponse: StreamingResponse-compatible object
        """
        # Create data stream
        data_stream = LangChainAdapter.to_data_stream(
            stream, callbacks, message_id, options
        )
        
        # Convert to response
        return DataStreamResponse(
            data_stream,
            headers=headers or {"Content-Type": "text/plain; charset=utf-8"},
            status=status
        )
    
    @staticmethod
    def to_data_stream(
        stream: AsyncIterable[LangChainStreamInput],
        callbacks: Optional[BaseAICallbackHandler] = None,
        message_id: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> DataStreamWithEmitters:
        """Convert LangChain stream to DataStreamWithEmitters.
        
        Args:
            stream: LangChain async stream from astream_events()
            callbacks: Callback handlers for stream events
            message_id: Optional message ID for tracking
            options: Control options (auto_events, emit_start, emit_finish, etc.)
        
        Returns:
            DataStreamWithEmitters: Stream object with emit methods
        """
        # Parse options
        opts = options or {}
        auto_events = opts.get("auto_events", True)
        auto_close = opts.get("auto_close", True)
        message_id = message_id or str(uuid.uuid4())
        
        # Create processor
        processor = StreamProcessor(
            message_id=message_id,
            auto_events=auto_events,
            callbacks=callbacks
        )
        
        # Create the async generator
        async def stream_generator():
            async for chunk in processor.process_stream(stream):
                yield chunk
        
        # Return wrapped stream with emit methods, passing the message builder and callbacks
        return DataStreamWithEmitters(
            stream_generator(), 
            message_id, 
            auto_close, 
            processor.message_builder,
            callbacks
        )
    
    @staticmethod
    async def merge_into_data_stream(
        stream: AsyncIterable[LangChainStreamInput],
        data_stream_writer: DataStreamWriter,
        callbacks: Optional[BaseAICallbackHandler] = None,
        message_id: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> None:
        """Merge LangChain stream into existing data stream.
        
        Args:
            stream: LangChain async stream from astream_events()
            data_stream_writer: Existing stream writer to merge into
            callbacks: Callback handlers for stream events
            message_id: Optional message ID for tracking
            options: Control options (auto_events, emit_start, emit_finish, etc.)
        
        Returns:
            None: Merges events into the provided data_stream_writer
        """
        # Convert to data stream
        data_stream = LangChainAdapter.to_data_stream(
            stream, callbacks, message_id, options
        )
        
        # Write to existing stream
        async for chunk in data_stream:
            await data_stream_writer.write(chunk)


# Legacy functions for backward compatibility
async def to_data_stream(
    stream: AsyncIterable[LangChainStreamInput],
    callbacks: Optional[BaseAICallbackHandler] = None,
    message_id: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None
) -> AsyncGenerator[UIMessageChunk, None]:
    """Legacy function - use LangChainAdapter.to_data_stream instead."""
    data_stream = LangChainAdapter.to_data_stream(stream, callbacks, message_id, options)
    async for chunk in data_stream:
        yield chunk


async def to_data_stream_response(
    stream: AsyncIterable[LangChainStreamInput],
    callbacks: Optional[BaseAICallbackHandler] = None,
    message_id: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    status: int = 200
) -> DataStreamResponse:
    """Convert LangChain output streams to AI SDK Data Stream Response.
    
    This function creates a response wrapper around the data stream,
    similar to the Node.js AI SDK's toDataStreamResponse function.
    
    Args:
        stream: Input stream from LangChain
        callbacks: Optional callbacks (BaseAICallbackHandler)
        message_id: Optional message ID (auto-generated if not provided)
        options: Control options (auto_events, auto_close, emit_start, emit_finish, etc.)
        headers: Optional HTTP headers
        status: HTTP status code
        
    Returns:
        DataStreamResponse object containing the converted stream
        
    Example:
        ```python
        from langchain_aisdk_adapter import to_data_stream_response
        
        # Convert LangChain stream to response format
        response = await to_data_stream_response(langchain_stream)
        
        # Use in web framework (e.g., FastAPI)
        async for chunk in response:
            yield chunk
        ```
    """
    return await LangChainAdapter.to_data_stream_response(
        stream, callbacks, message_id, headers, status, options
    )


async def merge_into_data_stream(
    stream: AsyncIterable[LangChainStreamInput],
    data_stream_writer: DataStreamWriter,
    callbacks: Optional[BaseAICallbackHandler] = None,
    message_id: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None
) -> None:
    """Merge LangChain output streams into an existing data stream.
    
    This function merges a LangChain stream into an existing data stream writer,
    similar to the Node.js AI SDK's mergeIntoDataStream function.
    
    Args:
        stream: Input stream from LangChain
        data_stream_writer: Target data stream writer
        callbacks: Optional callbacks (BaseAICallbackHandler)
        message_id: Optional message ID (auto-generated if not provided)
        options: Control options (auto_events, auto_close, emit_start, emit_finish, etc.)
        
    Example:
        ```python
        from langchain_aisdk_adapter import merge_into_data_stream, DataStreamWriter
        
        # Create a data stream writer
        writer = DataStreamWriter()
        
        # Merge LangChain stream into the writer
        await merge_into_data_stream(langchain_stream, writer)
        
        # Get all chunks
        chunks = writer.get_chunks()
        ```
    """
    await LangChainAdapter.merge_into_data_stream(
        stream, data_stream_writer, callbacks, message_id, options
    )