"""LangChain to AI SDK adapter providing core conversion methods."""

import uuid
from typing import AsyncIterable, AsyncGenerator, Optional, Dict, Any, Literal, TypedDict

from .models import LangChainStreamInput, UIMessageChunk
from .stream_processor import StreamProcessor
from .data_stream import DataStreamWithEmitters, DataStreamResponse, DataStreamWriter
from .callbacks import BaseAICallbackHandler

try:
    from fastapi.responses import StreamingResponse
except ImportError:
    # Fallback for when FastAPI is not available
    StreamingResponse = None


class AdapterOptions(TypedDict, total=False):
    """Options for controlling adapter behavior.
    
    Attributes:
        protocol_version: AI SDK protocol version ('v4' or 'v5'), defaults to 'v4'
        auto_events: Whether to automatically emit start/finish events, defaults to True
        auto_close: Whether to automatically close the stream, defaults to True
        emit_start: Whether to emit start events, defaults to True
        emit_finish: Whether to emit finish events, defaults to True
    """
    protocol_version: Literal['v4', 'v5']
    auto_events: bool
    auto_close: bool
    emit_start: bool
    emit_finish: bool


class LangChainAdapter:
    """LangChain to AI SDK adapter providing three core methods."""
    
    @staticmethod
    def to_data_stream_response(
        stream: AsyncIterable[LangChainStreamInput],
        callbacks: Optional[BaseAICallbackHandler] = None,
        message_id: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        status: int = 200,
        options: Optional[AdapterOptions] = None
    ) -> StreamingResponse:
        """Convert LangChain stream to FastAPI StreamingResponse.
        
        Args:
            stream: LangChain async stream from astream_events()
            callbacks: Callback handlers for stream events
            message_id: Optional message ID for tracking
            headers: HTTP headers for the response
            status: HTTP status code
            options: Control options including:
                - protocol_version: 'v4' (default) or 'v5'
                - auto_events: Whether to emit start/finish events (default: True)
                - auto_close: Whether to auto-close stream (default: True)
                - emit_start: Whether to emit start events (default: True)
                - emit_finish: Whether to emit finish events (default: True)
        
        Returns:
            StreamingResponse: FastAPI StreamingResponse with protocol format
        """
        # Parse options
        opts = options or {}
        protocol_version = opts.get("protocol_version", "v4")  # Default to v4
        
        # Get DataStreamWithEmitters with protocol format
        data_stream = LangChainAdapter.to_data_stream(
            stream=stream,
            callbacks=callbacks,
            message_id=message_id,
            options=opts
        )
        
        # Get protocol-specific headers
        from .protocol_strategy import ProtocolConfig
        protocol_config = ProtocolConfig(protocol_version)
        protocol_headers = protocol_config.strategy.get_headers()
        if headers:
            protocol_headers.update(headers)
        
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            content=data_stream,
            headers=protocol_headers,
            status_code=status
        )
    

    
    @staticmethod
    def to_data_stream(
        stream: AsyncIterable[LangChainStreamInput],
        callbacks: Optional[BaseAICallbackHandler] = None,
        message_id: Optional[str] = None,
        options: Optional[AdapterOptions] = None
    ) -> DataStreamWithEmitters:
        """Convert LangChain stream to DataStreamWithEmitters.
        
        Args:
            stream: LangChain async stream from astream_events()
            callbacks: Callback handlers for stream events
            message_id: Optional message ID for tracking
            options: Control options including:
                - protocol_version: 'v4' (default) or 'v5'
                - auto_events: Whether to emit start/finish events (default: True)
                - auto_close: Whether to auto-close stream (default: True)
                - emit_start: Whether to emit start events (default: True)
                - emit_finish: Whether to emit finish events (default: True)

        
        Returns:
            DataStreamWithEmitters: Stream object with emit methods
        """
        # Parse options
        opts = options or {}
        protocol_version = opts.get("protocol_version", "v4")  # Default to v4
        auto_events = opts.get("auto_events", True)
        auto_close = opts.get("auto_close", True)

        message_id = message_id or str(uuid.uuid4())
        
        # Create stream processor
        processor = StreamProcessor(
            message_id=message_id,
            auto_events=auto_events,
            callbacks=callbacks,
            protocol_version=protocol_version
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
            callbacks,
            protocol_version,
            "protocol",  # Always use protocol format
            processor  # Pass processor instance for usage tracking
        )
    
    @staticmethod
    async def merge_into_data_stream(
        stream: AsyncIterable[LangChainStreamInput],
        data_stream_writer: DataStreamWriter,
        callbacks: Optional[BaseAICallbackHandler] = None,
        message_id: Optional[str] = None,
        options: Optional[AdapterOptions] = None
    ) -> None:
        """Merge LangChain stream into existing data stream.
        
        Args:
            stream: LangChain async stream from astream_events()
            data_stream_writer: Existing stream writer to merge into
            callbacks: Callback handlers for stream events
            message_id: Optional message ID for tracking
            options: Control options including:
                - protocol_version: 'v4' (default) or 'v5'
                - output_format: 'chunks' (default) or 'protocol'
                - auto_events: Whether to emit start/finish events (default: True)
                - auto_close: Whether to auto-close stream (default: True)
                - emit_start: Whether to emit start events (default: True)
                - emit_finish: Whether to emit finish events (default: True)
        
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
    options: Optional[AdapterOptions] = None
) -> AsyncGenerator[UIMessageChunk, None]:
    """Legacy function - use LangChainAdapter.to_data_stream instead."""
    data_stream = LangChainAdapter.to_data_stream(stream, callbacks, message_id, options)
    async for chunk in data_stream:
        yield chunk


def to_data_stream_response(
    stream: AsyncIterable[LangChainStreamInput],
    callbacks: Optional[BaseAICallbackHandler] = None,
    message_id: Optional[str] = None,
    options: Optional[AdapterOptions] = None,
    headers: Optional[Dict[str, str]] = None,
    status: int = 200
) -> StreamingResponse:
    """Convert LangChain output streams to AI SDK Data Stream Response.
    
    This function creates a response wrapper around the data stream,
    similar to the Node.js AI SDK's toDataStreamResponse function.
    
    Args:
        stream: Input stream from LangChain
        callbacks: Optional callbacks (BaseAICallbackHandler)
        message_id: Optional message ID (auto-generated if not provided)
        options: Control options including protocol_version ('v4'/'v5'), auto_events, auto_close, etc.
        headers: Optional HTTP headers
        status: HTTP status code
        
    Returns:
        StreamingResponse object containing the converted stream
        
    Example:
        ```python
        from langchain_aisdk_adapter import to_data_stream_response
        
        # Convert LangChain stream to response format
        response = to_data_stream_response(langchain_stream)
        
        # Use in web framework (e.g., FastAPI)
        return response
        ```
    """
    return LangChainAdapter.to_data_stream_response(
        stream, callbacks, message_id, headers, status, options
    )


async def merge_into_data_stream(
    stream: AsyncIterable[LangChainStreamInput],
    data_stream_writer: DataStreamWriter,
    callbacks: Optional[BaseAICallbackHandler] = None,
    message_id: Optional[str] = None,
    options: Optional[AdapterOptions] = None
) -> None:
    """Merge LangChain output streams into an existing data stream.
    
    This function merges a LangChain stream into an existing data stream writer,
    similar to the Node.js AI SDK's mergeIntoDataStream function.
    
    Args:
        stream: Input stream from LangChain
        data_stream_writer: Target data stream writer
        callbacks: Optional callbacks (BaseAICallbackHandler)
        message_id: Optional message ID (auto-generated if not provided)
        options: Control options including protocol_version ('v4'/'v5'), auto_events, auto_close, etc.
        
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