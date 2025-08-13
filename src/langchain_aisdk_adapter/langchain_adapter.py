"""LangChain to AI SDK adapter providing core conversion methods."""

import asyncio
import logging
import uuid
import re
from typing import AsyncIterable, AsyncGenerator, Optional, Dict, Any, Literal, TypedDict, Callable, Awaitable, Union, List

from .models import LangChainStreamInput, UIMessageChunk
from .stream_processor import StreamProcessor
from .data_stream import DataStreamWithEmitters, DataStreamResponse, DataStreamWriter
from .callbacks import BaseAICallbackHandler
from .context import DataStreamContext
from .lifecycle import ContextLifecycleManager

try:
    from langchain_core.language_models import BaseLanguageModel
    from langchain_core.messages import BaseMessage
    from langchain_core.tools import BaseTool
    from langchain_core.runnables import Runnable
except ImportError:
    # Fallback for when LangChain is not available
    BaseLanguageModel = None
    BaseMessage = None
    BaseTool = None
    Runnable = None

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
        auto_context: Whether to automatically set up context, defaults to True
        experimental_transform: Stream transformer function for smoothing output
        experimental_generateMessageId: Function to generate custom message IDs
    """
    protocol_version: Literal['v4', 'v5']
    auto_events: bool
    auto_close: bool
    emit_start: bool
    emit_finish: bool
    auto_context: bool
    experimental_transform: Callable[[AsyncIterable[Any]], AsyncIterable[Any]]
    experimental_generateMessageId: Callable[[], str]


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
                - auto_context: Whether to automatically set up context (default: True)
        
        Returns:
            StreamingResponse: FastAPI StreamingResponse with protocol format
        """
        # Parse options
        opts = options or {}
        protocol_version = opts.get("protocol_version", "v4")  # Default to v4
        auto_context = opts.get("auto_context", True)
        
        # Create stream processor
        message_id = message_id or str(uuid.uuid4())
        processor = StreamProcessor(
            message_id=message_id,
            auto_events=opts.get("auto_events", True),
            callbacks=callbacks,
            protocol_version=protocol_version
        )
        
        # Create the async generator with automatic context management
        async def stream_generator():
            if auto_context:
                # Create a temporary DataStreamWithEmitters for context
                temp_stream = DataStreamWithEmitters(
                    processor.process_stream(stream),
                    message_id,
                    opts.get("auto_close", True),
                    processor.message_builder,
                    callbacks,
                    protocol_version,
                    "protocol",
                    processor
                )
                
                # Set up context and process with lifecycle management
                async with ContextLifecycleManager.managed_context(temp_stream):
                    async for chunk in processor.process_stream(stream):
                        yield chunk
            else:
                # Process without context management
                async for chunk in processor.process_stream(stream):
                    yield chunk
        
        # Get protocol-specific headers
        from .protocol_strategy import ProtocolConfig
        protocol_config = ProtocolConfig(protocol_version)
        protocol_headers = protocol_config.strategy.get_headers()
        if headers:
            protocol_headers.update(headers)
        
        return DataStreamResponse(
            stream=stream_generator(),
            protocol_version=protocol_version,
            headers=headers,
            status=status
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
                - auto_context: Whether to automatically set up context (default: True)
                - experimental_transform: Stream transformer function for smoothing output
                - experimental_generateMessageId: Function to generate custom message IDs

        
        Returns:
            DataStreamWithEmitters: Stream object with emit methods
        """
        # Parse options
        opts = options or {}
        protocol_version = opts.get("protocol_version", "v4")  # Default to v4
        auto_events = opts.get("auto_events", True)
        auto_close = opts.get("auto_close", True)
        auto_context = opts.get("auto_context", True)
        experimental_transform = opts.get("experimental_transform")
        experimental_generateMessageId = opts.get("experimental_generateMessageId")

        # Use custom message ID generator if provided
        if experimental_generateMessageId:
            message_id = message_id or experimental_generateMessageId()
        else:
            message_id = message_id or str(uuid.uuid4())
        
        # Create stream processor
        processor = StreamProcessor(
            message_id=message_id,
            auto_events=auto_events,
            callbacks=callbacks,
            protocol_version=protocol_version
        )
        
        # Create the async generator with automatic context management
        async def stream_generator():
            try:
                processed_stream = processor.process_stream(stream)
                
                # Apply experimental transform if provided
                if experimental_transform:
                    processed_stream = experimental_transform(processed_stream)
                
                async for chunk in processed_stream:
                    yield chunk
            except GeneratorExit:
                # Generator is being closed, log for debugging and re-raise to ensure proper cleanup
                logging.debug(f"LangChainAdapter.stream_generator: Generator exit for message {message_id}")
                raise
        
        # Create wrapped stream with emit methods
        data_stream = DataStreamWithEmitters(
            stream_generator(), 
            message_id, 
            auto_close, 
            processor.message_builder,
            callbacks,
            protocol_version,
            "protocol",  # Always use protocol format
            processor  # Pass processor instance for usage tracking
        )
        
        # Automatically set up context if enabled
        if auto_context:
            DataStreamContext.set_current_stream(data_stream)
            # Note: Lifecycle management is handled by the stream itself
            # Background processing is not needed for basic functionality
        
        return data_stream
    
    @staticmethod
    async def _process_stream_with_lifecycle(
        stream: AsyncIterable[LangChainStreamInput],
        data_stream: DataStreamWithEmitters,
        processor: StreamProcessor
    ) -> None:
        """Process stream with automatic lifecycle management.
        
        Args:
            stream: Original LangChain stream
            data_stream: DataStreamWithEmitters instance
            processor: StreamProcessor instance
        """
        try:
            # Use context lifecycle manager
            async with ContextLifecycleManager.managed_context(data_stream):
                # Process the stream
                async for chunk in processor.process_stream(stream):
                    # Stream processing is handled by the DataStreamWithEmitters
                    pass
        except Exception as e:
            # Emit error if context is available
            context = DataStreamContext.get_current_stream()
            if context:
                await context.emit_error(str(e))
            raise
        finally:
            # Clean up context
            DataStreamContext.clear_current_stream()
    
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