"""LangChain to AI SDK adapter providing core conversion methods."""

import asyncio
import uuid
import re
from typing import AsyncIterable, AsyncGenerator, Optional, Dict, Any, Literal, TypedDict, Callable, Awaitable, Union

from .models import LangChainStreamInput, UIMessageChunk
from .stream_processor import StreamProcessor
from .data_stream import DataStreamWithEmitters, DataStreamResponse, DataStreamWriter
from .callbacks import BaseAICallbackHandler
from .context import DataStreamContext
from .lifecycle import ContextLifecycleManager

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
        
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            content=stream_generator(),
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
            processed_stream = processor.process_stream(stream)
            
            # Apply experimental transform if provided
            if experimental_transform:
                processed_stream = experimental_transform(processed_stream)
            
            async for chunk in processed_stream:
                yield chunk
        
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
    
    @staticmethod
    def smooth_stream(
        stream: AsyncIterable[str],
        delay_in_ms: int = 15,
        chunking: Union[Literal['word', 'line'], re.Pattern, Callable[[str], list[str]]] = 'word'
    ) -> AsyncIterable[str]:
        """Create a smooth stream transformer that chunks and delays text output.
        
        This method implements AI SDK's smoothStream functionality for creating
        smoother text streaming experiences by controlling the rate and chunking
        of text output.
        
        Based on AI SDK smoothStream:
        https://ai-sdk.dev/docs/reference/ai-sdk-core/smooth-stream
        
        Args:
            stream: Input async iterable of text chunks
            delay_in_ms: Delay between chunks in milliseconds (default: 15ms)
            chunking: Chunking strategy:
                - 'word': Split by whitespace (default)
                - 'line': Split by line breaks
                - re.Pattern: Split using custom regex pattern
                - Callable: Custom chunking function
        
        Returns:
            AsyncIterable[str]: Smoothed stream with controlled chunking and timing
        
        Example:
            ```python
            # Basic word-based smoothing
            smooth = LangChainAdapter.smooth_stream(text_stream)
            
            # Custom delay and line-based chunking
            smooth = LangChainAdapter.smooth_stream(
                text_stream, 
                delay_in_ms=25, 
                chunking='line'
            )
            
            # Custom regex chunking (e.g., for Chinese text)
            import re
            chinese_pattern = re.compile(r'[\u4e00-\u9fff]+|[^\u4e00-\u9fff]+')
            smooth = LangChainAdapter.smooth_stream(
                text_stream,
                chunking=chinese_pattern
            )
            ```
        """
        return _SmoothStreamTransformer(stream, delay_in_ms, chunking)


class _SmoothStreamTransformer:
    """Internal transformer class for smooth streaming functionality.
    
    This class implements the core logic for AI SDK's smoothStream,
    providing controlled chunking and timing for text streams.
    """
    
    def __init__(
        self,
        stream: AsyncIterable[str],
        delay_in_ms: int,
        chunking: Union[Literal['word', 'line'], re.Pattern, Callable[[str], list[str]]]
    ):
        self.stream = stream
        self.delay_in_ms = delay_in_ms
        self.chunking = chunking
        self._buffer = ""
    
    def __aiter__(self):
        return self._transform()
    
    async def _transform(self) -> AsyncGenerator[str, None]:
        """Transform the input stream with smooth chunking and delays."""
        try:
            async for chunk in self.stream:
                if not chunk:
                    continue
                
                self._buffer += chunk
                
                # Process complete chunks from buffer
                chunks = self._chunk_text(self._buffer)
                
                # Keep the last incomplete chunk in buffer if needed
                if chunks:
                    # Yield all but the last chunk (which might be incomplete)
                    for i, text_chunk in enumerate(chunks[:-1]):
                        if text_chunk.strip():  # Only yield non-empty chunks
                            yield text_chunk
                            if self.delay_in_ms > 0:
                                await asyncio.sleep(self.delay_in_ms / 1000.0)
                    
                    # Handle the last chunk
                    last_chunk = chunks[-1]
                    if self._is_complete_chunk(last_chunk):
                        if last_chunk.strip():
                            yield last_chunk
                            if self.delay_in_ms > 0:
                                await asyncio.sleep(self.delay_in_ms / 1000.0)
                        self._buffer = ""
                    else:
                        # Keep incomplete chunk in buffer
                        self._buffer = last_chunk
            
            # Yield any remaining buffer content
            if self._buffer.strip():
                yield self._buffer
                
        except Exception as e:
            # Ensure any remaining buffer is yielded on error
            if self._buffer.strip():
                yield self._buffer
            raise e
    
    def _chunk_text(self, text: str) -> list[str]:
        """Chunk text according to the specified chunking strategy."""
        if not text:
            return []
        
        if callable(self.chunking) and not isinstance(self.chunking, re.Pattern):
            # Custom chunking function
            return self.chunking(text)
        elif isinstance(self.chunking, re.Pattern):
             # Regex pattern chunking
             chunks = self.chunking.findall(text)
             # Filter out empty chunks
             chunks = [chunk for chunk in chunks if chunk]
             return chunks if chunks else [text]
        elif self.chunking == 'line':
            # Line-based chunking
            lines = text.split('\n')
            # Preserve line breaks except for the last line
            result = []
            for i, line in enumerate(lines[:-1]):
                result.append(line + '\n')
            if lines[-1]:  # Add last line if not empty
                result.append(lines[-1])
            return result
        else:  # Default to 'word' chunking
            # Word-based chunking (split by whitespace)
            # Use regex to preserve whitespace
            chunks = re.findall(r'\S+\s*', text)
            return chunks if chunks else [text]
    
    def _is_complete_chunk(self, chunk: str) -> bool:
        """Determine if a chunk is complete based on chunking strategy."""
        if not chunk:
            return True
        
        if callable(self.chunking) and not isinstance(self.chunking, re.Pattern):
            # For custom functions, assume chunks are complete
            return True
        elif isinstance(self.chunking, re.Pattern):
            # For regex patterns, check if chunk matches the pattern
            return bool(self.chunking.match(chunk))
        elif self.chunking == 'line':
            # Line is complete if it ends with newline or is the final chunk
            return chunk.endswith('\n')
        else:  # 'word' chunking
            # Word is complete if it ends with whitespace
            return chunk[-1].isspace() if chunk else True