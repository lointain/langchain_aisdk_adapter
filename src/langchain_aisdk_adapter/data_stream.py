"""Data stream classes for AI SDK compatibility."""

import asyncio
import json
import uuid
from typing import AsyncGenerator, Dict, Any, Optional, List

from fastapi.responses import StreamingResponse

from .protocol_strategy import ProtocolConfig
from .text_processing_adapter import TextProcessingAdapter

from .models import (
    UIMessageChunk,
    UIMessageChunkFile,
    UIMessageChunkData,
    UIMessageChunkSourceUrl,
    UIMessageChunkSourceDocument,
    UIMessageChunkReasoning,
    UIMessageChunkError,
    UIMessageChunkToolInputStart,
    UIMessageChunkToolInputDelta,
    UIMessageChunkToolInputAvailable,
    UIMessageChunkToolOutputAvailable,
    UIMessageChunkToolOutputError,
    UIMessageChunkReasoningStart,
    UIMessageChunkReasoningDelta,
    UIMessageChunkReasoningEnd,
    UIMessageChunkReasoningPartFinish,
    UIMessageChunkStartStep,
    UIMessageChunkFinishStep,
    UIMessageChunkStart,
    UIMessageChunkFinish,
    UIMessageChunkMessageStart,
    UIMessageChunkMessageEnd,
    UIMessageChunkAbort,
    UIMessageChunkMessageMetadata
)
from .message_builder import MessageBuilder
from .protocol_generator import ProtocolGenerator
from .callbacks import BaseAICallbackHandler


class DataStreamWithEmitters:
    """Data stream wrapper that provides emit methods for manual control.
    
    This class wraps an AsyncGenerator and provides emit_* methods
    for manual event emission, allowing direct method calls on the stream object.
    """
    
    def __init__(
        self,
        stream_generator: AsyncGenerator[UIMessageChunk, None],
        message_id: str,
        auto_close: bool = True,
        message_builder: Optional[MessageBuilder] = None,
        callbacks: Optional[BaseAICallbackHandler] = None,
        protocol_version: str = "v4",
        output_format: str = "chunks",  # "chunks" or "protocol"
        stream_processor: Optional[Any] = None  # StreamProcessor instance for usage tracking
    ):
        self._stream_generator = stream_generator
        self._message_id = message_id
        self._manual_queue: asyncio.Queue = asyncio.Queue()
        self._closed = False
        self._stream_started = False
        self._auto_close = auto_close
        self._message_builder = message_builder
        self._callbacks = callbacks
        self._protocol_version = protocol_version
        self._output_format = output_format
        self._stream_processor = stream_processor
        
        # Initialize protocol components if protocol output is enabled
        if self._output_format == "protocol":
            from .protocol_strategy import ProtocolConfig
            from .text_processing_adapter import TextProcessingAdapter
            self._protocol_config = ProtocolConfig(protocol_version)
            self._text_adapter = TextProcessingAdapter(protocol_version)
    
    async def emit_file(
        self, 
        url: str,
        mediaType: str,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a file chunk."""
        chunk = UIMessageChunkFile(
            type="file",
            url=url,
            mediaType=mediaType
        )
        await self._emit_manual_chunk(chunk)
    
    async def emit_data(
        self, 
        data: Dict[str, Any],
        data_type: str = "data-custom",
        message_id: Optional[str] = None
    ) -> None:
        """Emit a data chunk."""
        chunk = UIMessageChunkData(
            type=data_type,
            data=data
        )
        await self._emit_manual_chunk(chunk)
    
    async def emit_source_url(
        self, 
        url: str,
        title: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a source-url chunk."""
        chunk = UIMessageChunkSourceUrl(
            type="source-url",
            sourceId=str(uuid.uuid4()),
            url=url,
            title=title
        )
        await self._emit_manual_chunk(chunk)
    
    async def emit_source_document(
        self, 
        source_id: str,
        media_type: str,
        title: str,
        filename: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a source-document chunk."""
        chunk = UIMessageChunkSourceDocument(
            type="source-document",
            sourceId=source_id,
            mediaType=media_type,
            title=title,
            filename=filename
        )
        await self._emit_manual_chunk(chunk)
    
    async def emit_reasoning(
        self, 
        text: str,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a reasoning chunk."""
        chunk = UIMessageChunkReasoning(
            type="reasoning",
            text=text
        )
        await self._emit_manual_chunk(chunk)
    
    async def emit_error(
        self, 
        error_text: str,
        message_id: Optional[str] = None
    ) -> None:
        """Emit an error chunk."""
        chunk = UIMessageChunkError(
            type="error",
            errorText=error_text
        )
        await self._emit_manual_chunk(chunk)
    
    # === Text-related emit methods ===
    async def emit_text_start(
        self,
        text_id: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a text-start chunk using unified protocol generator."""
        chunk = ProtocolGenerator.create_text_start(
            text_id or str(uuid.uuid4()),
            self._protocol_version
        )
        await self._emit_manual_chunk(chunk)
    
    async def emit_text_delta(
        self,
        delta: str,
        text_id: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a text-delta chunk using unified protocol generator."""
        chunk = ProtocolGenerator.create_text_delta(
            text_id or str(uuid.uuid4()),
            delta,
            self._protocol_version
        )
        await self._emit_manual_chunk(chunk)
    
    async def emit_text_end(
        self,
        text: str,
        text_id: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a text-end chunk using unified protocol generator."""
        chunk = ProtocolGenerator.create_text_end(
            text_id or str(uuid.uuid4()),
            text,
            self._protocol_version
        )
        await self._emit_manual_chunk(chunk)
    
    # === Tool-related emit methods ===
    async def emit_tool_input_start(
        self,
        tool_call_id: str,
        tool_name: str,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a tool-input-start chunk."""
        chunk = UIMessageChunkToolInputStart(
            type="tool-input-start",
            toolCallId=tool_call_id,
            toolName=tool_name
        )
        await self._emit_manual_chunk(chunk)
    
    async def emit_tool_input_delta(
        self,
        tool_call_id: str,
        delta: str,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a tool-input-delta chunk."""
        chunk = UIMessageChunkToolInputDelta(
            type="tool-input-delta",
            toolCallId=tool_call_id,
            delta=delta
        )
        await self._emit_manual_chunk(chunk)
    
    async def emit_tool_input_available(
        self,
        tool_call_id: str,
        tool_name: str,
        input_data: Any,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a tool-input-available chunk."""
        chunk = UIMessageChunkToolInputAvailable(
            type="tool-input-available",
            toolCallId=tool_call_id,
            toolName=tool_name,
            input=input_data
        )
        await self._emit_manual_chunk(chunk)
    
    async def emit_tool_output_available(
        self,
        tool_call_id: str,
        output: Any,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a tool-output-available chunk."""
        chunk = UIMessageChunkToolOutputAvailable(
            type="tool-output-available",
            toolCallId=tool_call_id,
            output=output
        )
        await self._emit_manual_chunk(chunk)
    
    async def emit_tool_output_error(
        self,
        tool_call_id: str,
        error_text: str,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a tool-output-error chunk."""
        chunk = UIMessageChunkToolOutputError(
            type="tool-output-error",
            toolCallId=tool_call_id,
            errorText=error_text
        )
        await self._emit_manual_chunk(chunk)
    
    # === Reasoning-related emit methods ===
    async def emit_reasoning_start(
        self,
        reasoning_id: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a reasoning-start chunk."""
        chunk = UIMessageChunkReasoningStart(
            type="reasoning-start",
            id=reasoning_id or str(uuid.uuid4())
        )
        await self._emit_manual_chunk(chunk)
    
    async def emit_reasoning_delta(
        self,
        delta: str,
        reasoning_id: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a reasoning-delta chunk."""
        chunk = UIMessageChunkReasoningDelta(
            type="reasoning-delta",
            id=reasoning_id or str(uuid.uuid4()),
            delta=delta
        )
        await self._emit_manual_chunk(chunk)
    
    async def emit_reasoning_end(
        self,
        reasoning_id: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a reasoning-end chunk."""
        chunk = UIMessageChunkReasoningEnd(
            type="reasoning-end",
            id=reasoning_id or str(uuid.uuid4())
        )
        await self._emit_manual_chunk(chunk)
    
    async def emit_reasoning_part_finish(
        self,
        reasoning_id: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a reasoning-part-finish chunk."""
        chunk = UIMessageChunkReasoningPartFinish(
            type="reasoning-part-finish",
            id=reasoning_id or str(uuid.uuid4())
        )
        await self._emit_manual_chunk(chunk)
    
    # === Step-related emit methods ===
    async def emit_start_step(
        self,
        step_type: str,
        step_id: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a start-step chunk."""
        chunk = UIMessageChunkStartStep(
            type="start-step",
            stepType=step_type,
            stepId=step_id or str(uuid.uuid4())
        )
        await self._emit_manual_chunk(chunk)
    
    async def emit_finish_step(
        self,
        step_type: str,
        step_id: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a finish-step chunk."""
        chunk = UIMessageChunkFinishStep(
            type="finish-step",
            stepType=step_type,
            stepId=step_id or str(uuid.uuid4())
        )
        await self._emit_manual_chunk(chunk)
    
    # === Message lifecycle emit methods ===
    async def emit_start(
        self,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a start chunk."""
        chunk = UIMessageChunkStart(
            type="start",
            messageId=message_id or self._message_id
        )
        await self._emit_manual_chunk(chunk)
    
    async def emit_finish(
        self,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a finish chunk."""
        chunk = UIMessageChunkFinish(
            type="finish",
            messageId=message_id or self._message_id
        )
        await self._emit_manual_chunk(chunk)
    
    async def emit_message_start(
        self,
        role: str,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a message-start chunk."""
        chunk = UIMessageChunkMessageStart(
            type="message-start",
            messageId=message_id or self._message_id,
            role=role
        )
        await self._emit_manual_chunk(chunk)
    
    async def emit_message_end(
        self,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a message-end chunk."""
        chunk = UIMessageChunkMessageEnd(
            type="message-end",
            messageId=message_id or self._message_id
        )
        await self._emit_manual_chunk(chunk)
    
    async def emit_abort(
        self,
        reason: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> None:
        """Emit an abort chunk."""
        chunk = UIMessageChunkAbort(
            type="abort",
            reason=reason
        )
        await self._emit_manual_chunk(chunk)
    
    async def emit_message_metadata(
        self,
        metadata: Dict[str, Any],
        message_id: Optional[str] = None
    ) -> None:
        """Emit a message-metadata chunk."""
        chunk = UIMessageChunkMessageMetadata(
            type="message-metadata",
            messageId=message_id or self._message_id,
            metadata=metadata
        )
        await self._emit_manual_chunk(chunk)
    
    async def _emit_manual_chunk(self, chunk: UIMessageChunk) -> None:
        """Emit a manual chunk to the queue."""
        if self._closed:
            raise RuntimeError("Cannot emit to closed stream")
        
        # Mark chunk as manual for processing in __aiter__
        chunk_with_meta = dict(chunk)
        chunk_with_meta['_is_manual'] = True
        
        await self._manual_queue.put(chunk_with_meta)
    
    async def __aiter__(self):
        """Async iterator that merges automatic and manual chunks."""
        # Use a queue to merge both streams
        merged_queue: asyncio.Queue = asyncio.Queue()
        
        async def auto_producer():
            try:
                async for chunk in self._stream_generator:
                    await merged_queue.put(("auto", chunk))
            except Exception as e:
                await merged_queue.put(("error", e))
            finally:
                await merged_queue.put(("auto_end", None))
        
        async def manual_producer():
            try:
                while not self._closed:
                    try:
                        chunk = await asyncio.wait_for(self._manual_queue.get(), timeout=0.1)
                        if chunk is None:
                            break
                        await merged_queue.put(("manual", chunk))
                    except asyncio.TimeoutError:
                        # If auto stream has ended and auto_close is True, end manual stream too
                        if auto_task.done() and self._auto_close:
                            break
                        continue
            except Exception as e:
                await merged_queue.put(("error", e))
            finally:
                await merged_queue.put(("manual_end", None))
        
        # Start both producers
        auto_task = asyncio.create_task(auto_producer())
        manual_task = asyncio.create_task(manual_producer())
        
        auto_ended = False
        manual_ended = False
        
        try:
            while not (auto_ended and manual_ended):
                try:
                    source, chunk = await asyncio.wait_for(merged_queue.get(), timeout=1.0)
                    
                    if source == "auto_end":
                        auto_ended = True
                        continue
                    elif source == "manual_end":
                        manual_ended = True
                        continue
                    elif source == "error":
                        # Re-raise the error
                        raise chunk
                    elif source in ("auto", "manual"):
                        # Remove meta flag if present
                        clean_chunk = dict(chunk)
                        clean_chunk.pop('_is_manual', None)
                        
                        # For auto chunks (from StreamProcessor), don't reprocess through message_builder
                        # as they have already been processed. Only process manual chunks.
                        if source == "manual" and self._message_builder:
                            # Get newly generated parts from message builder for manual chunks only
                            new_parts = await self._message_builder.add_chunk(clean_chunk)
                            
                            # Create chunk with parts
                            chunk_with_parts = dict(clean_chunk)
                            if new_parts:
                                # Convert UIPart objects to dictionaries for JSON serialization
                                parts_dicts = []
                                for part in new_parts:
                                    if hasattr(part, '__dict__'):
                                        parts_dicts.append(part.__dict__)
                                    else:
                                        parts_dicts.append(part)
                                chunk_with_parts['parts'] = parts_dicts
                            
                            final_chunk = chunk_with_parts
                        else:
                            # For auto chunks, just yield them as-is since they're already processed
                            final_chunk = clean_chunk
                        
                        # Output chunk based on format preference
                        if self._output_format == "protocol":
                            # Format chunk for protocol output
                            formatted_text = self._format_chunk_for_protocol(final_chunk)
                            if formatted_text:
                                yield formatted_text
                        else:
                            # Output raw chunk
                            yield final_chunk
                        
                except asyncio.TimeoutError:
                    # Check if both tasks are done
                    if auto_task.done() and manual_task.done():
                        break
                    continue
        finally:
            # Send termination marker for protocol output
            if self._output_format == "protocol":
                termination_text = self._send_termination_marker()
                if termination_text:
                    yield termination_text
            
            # Clean up tasks
            if not auto_task.done():
                auto_task.cancel()
            if not manual_task.done():
                manual_task.cancel()
            
            # Wait for tasks to complete
            try:
                await asyncio.gather(auto_task, manual_task, return_exceptions=True)
            except Exception:
                pass
    
    def _format_chunk_for_protocol(self, chunk: UIMessageChunk) -> Optional[str]:
        """Format a chunk for protocol output."""
        if self._output_format != "protocol":
            return None
            
        # Handle text sequence management for different protocols
        chunk_type = chunk.get("type") if isinstance(chunk, dict) else getattr(chunk, "type", None)
        
        # Check if we need to finish current text sequence
        if (self._text_adapter.is_text_active() and 
            chunk_type not in ["text-delta", "text-start", "text-end"]):
            # Finish current text sequence before processing non-text chunk
            for finish_chunk in self._text_adapter.finish_text_sequence():
                formatted_chunk = self._protocol_config.strategy.format_chunk(finish_chunk)
                if formatted_chunk:
                    return formatted_chunk
        
        # Format the chunk using protocol strategy
        return self._protocol_config.strategy.format_chunk(chunk)
    
    def _send_termination_marker(self) -> Optional[str]:
        """Send protocol-specific termination marker."""
        if self._output_format != "protocol":
            return None
            
        # Finish any remaining text sequence
        if self._text_adapter.is_text_active():
            for finish_chunk in self._text_adapter.finish_text_sequence():
                formatted_chunk = self._protocol_config.strategy.format_chunk(finish_chunk)
                if formatted_chunk:
                    return formatted_chunk
        
        # Get usage information from stream processor if available
        usage_info = None
        if self._stream_processor and hasattr(self._stream_processor, 'current_usage'):
            usage_info = self._stream_processor.current_usage
        
        # Send protocol-specific termination marker with usage info
        return self._protocol_config.strategy.get_termination_marker(usage_info)
    
    def _get_termination_text(self) -> str:
        """Get termination text for protocol output."""
        if self._output_format != "protocol":
            return ""
        return self._protocol_config.strategy.get_termination_marker()

    @property
    def message_id(self) -> str:
        """Get the message ID."""
        return self._message_id
    
    @property
    def protocol_version(self) -> str:
        """Get the protocol version."""
        return self._protocol_version
    
    @property
    def output_format(self) -> str:
        """Get the output format."""
        return self._output_format
    
    async def close(self):
        """Close the stream."""
        self._closed = True
        
        # Call on_finish callback if available
        if self._callbacks and isinstance(self._callbacks, BaseAICallbackHandler):
            try:
                # Build final message from message builder
                if self._message_builder:
                    final_message = self._message_builder.build_message()
                    await self._callbacks.on_finish(final_message, {})
            except Exception as e:
                # Don't re-raise to avoid breaking the stream
                pass
        
        await self._manual_queue.put(None)
    



class DataStreamResponse(StreamingResponse):
    """FastAPI StreamingResponse wrapper for AI SDK Data Stream Protocol.
    
    This class wraps an AI SDK data stream as a FastAPI StreamingResponse,
    automatically formatting UIMessageChunk objects for AI SDK v4/v5 compatibility.
    """
    
    def __init__(
        self,
        stream: AsyncGenerator[UIMessageChunk, None],
        protocol_version: str = "v4",  # Default to v4
        headers: Optional[Dict[str, str]] = None,
        status: int = 200
    ):
        """Initialize DataStreamResponse.
        
        Args:
            stream: The data stream generator
            protocol_version: Protocol version ("v4" or "v5")
            headers: Optional HTTP headers
            status: HTTP status code
        """
        # Create protocol configuration
        self.protocol_config = ProtocolConfig(protocol_version)
        
        # Convert UIMessageChunk stream to protocol-specific text stream
        text_stream = self._convert_to_protocol_stream(stream)
        
        # Get protocol-specific headers
        protocol_headers = self.protocol_config.strategy.get_headers()
        if headers:
            protocol_headers.update(headers)
        
        super().__init__(
            content=text_stream,
            headers=protocol_headers,
            status_code=status
        )
    
    async def _convert_to_protocol_stream(
        self, 
        data_stream: AsyncGenerator[UIMessageChunk, None]
    ) -> AsyncGenerator[str, None]:
        """Convert UIMessageChunk stream to protocol-specific format."""
        text_adapter = TextProcessingAdapter(self.protocol_config.version)
        
        async for chunk in data_stream:
            # Handle text sequence management for different protocols
            chunk_type = chunk.get("type") if isinstance(chunk, dict) else getattr(chunk, "type", None)
            
            # Check if we need to finish current text sequence
            if (text_adapter.is_text_active() and 
                chunk_type not in ["text-delta", "text-start", "text-end"]):
                # Finish current text sequence before processing non-text chunk
                for finish_chunk in text_adapter.finish_text_sequence():
                    formatted_chunk = self.protocol_config.strategy.format_chunk(finish_chunk)
                    if formatted_chunk:
                        yield formatted_chunk
            
            # Format the chunk using protocol strategy
            formatted_chunk = self.protocol_config.strategy.format_chunk(chunk)
            if formatted_chunk:
                yield formatted_chunk
        
        # Finish any remaining text sequence
        if text_adapter.is_text_active():
            for finish_chunk in text_adapter.finish_text_sequence():
                formatted_chunk = self.protocol_config.strategy.format_chunk(finish_chunk)
                if formatted_chunk:
                    yield formatted_chunk
        
        # Send protocol-specific termination marker
        yield self.protocol_config.strategy.get_termination_marker()
    



class DataStreamWriter:
    """Writer interface for merging streams into existing data streams.
    
    This class provides a writer interface similar to the Node.js AI SDK's
    DataStreamWriter for merging multiple streams.
    """
    
    def __init__(self):
        """Initialize DataStreamWriter."""
        self._chunks: List[UIMessageChunk] = []
        self._closed = False
        self._queue: asyncio.Queue = asyncio.Queue()
    
    async def write(self, chunk: UIMessageChunk) -> None:
        """Write a chunk to the data stream.
        
        Args:
            chunk: UI message chunk to write
        """
        if self._closed:
            raise RuntimeError("Cannot write to closed stream")
        self._chunks.append(chunk)
        await self._queue.put(chunk)
    
    async def close(self) -> None:
        """Close the data stream writer."""
        self._closed = True
        await self._queue.put(None)  # Sentinel value to signal end
    
    def get_chunks(self) -> List[UIMessageChunk]:
        """Get all written chunks.
        
        Returns:
            List of UI message chunks
        """
        return self._chunks.copy()
    
    async def __aiter__(self):
        """Async iterator for the stream."""
        while True:
            chunk = await self._queue.get()
            if chunk is None:  # End of stream
                break
            yield chunk
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()