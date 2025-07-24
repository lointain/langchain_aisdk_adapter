"""Manual stream controller for AI SDK compatibility.

This module provides manual control over data streams, allowing developers
to emit UIMessageChunk events programmatically.
"""

import asyncio
import uuid
from typing import Any, Dict, List, Optional, Union

from .models import (
    UIMessageChunk,
    UIMessageChunkStart,
    UIMessageChunkFinish,
    UIMessageChunkTextStart,
    UIMessageChunkTextDelta,
    UIMessageChunkTextEnd,
    UIMessageChunkError,
    UIMessageChunkToolInputStart,
    UIMessageChunkToolInputDelta,
    UIMessageChunkToolInputAvailable,
    UIMessageChunkToolOutputAvailable,
    UIMessageChunkToolOutputError,
    UIMessageChunkReasoning,
    UIMessageChunkReasoningStart,
    UIMessageChunkReasoningDelta,
    UIMessageChunkReasoningEnd,
    UIMessageChunkReasoningPartFinish,
    UIMessageChunkSourceUrl,
    UIMessageChunkSourceDocument,
    UIMessageChunkFile,
    UIMessageChunkData,
    UIMessageChunkStartStep,
    UIMessageChunkFinishStep,
    UIMessageChunkMessageStart,
    UIMessageChunkMessageEnd,
    UIMessageChunkAbort,
    UIMessageChunkMessageMetadata
)


class ManualStreamController:
    """Manual stream controller for emitting UIMessageChunk events.
    
    This class provides methods to manually emit various types of UIMessageChunk
    events, giving developers full control over the streaming process.
    """
    
    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._closed = False
    
    async def emit(self, chunk: UIMessageChunk) -> None:
        """Emit a UIMessageChunk."""
        if self._closed:
            raise RuntimeError("Cannot emit to closed stream")
        await self._queue.put(chunk)
    
    async def emit_start(self, message_id: Optional[str] = None) -> None:
        """Emit a start chunk."""
        chunk = UIMessageChunkStart(
            type="start",
            messageId=message_id or str(uuid.uuid4())
        )
        await self.emit(chunk)
    
    async def emit_finish(
        self, 
        message_id: Optional[str] = None,
        finish_reason: str = "stop",
        usage: Optional[Dict[str, Any]] = None
    ) -> None:
        """Emit a finish chunk."""
        chunk = UIMessageChunkFinish(
            type="finish",
            messageId=message_id or str(uuid.uuid4()),
            finishReason=finish_reason,
            usage=usage
        )
        await self.emit(chunk)
    
    async def emit_text_start(
        self, 
        text_id: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a text-start chunk."""
        chunk = UIMessageChunkTextStart(
            type="text-start",
            id=text_id or str(uuid.uuid4()),
            messageId=message_id or str(uuid.uuid4())
        )
        await self.emit(chunk)
    
    async def emit_text_delta(
        self, 
        text_delta: str,
        text_id: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a text-delta chunk."""
        chunk = UIMessageChunkTextDelta(
            type="text-delta",
            id=text_id or str(uuid.uuid4()),
            messageId=message_id or str(uuid.uuid4()),
            textDelta=text_delta
        )
        await self.emit(chunk)
    
    async def emit_text_end(
        self, 
        text: str,
        text_id: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a text-end chunk."""
        chunk = UIMessageChunkTextEnd(
            type="text-end",
            id=text_id or str(uuid.uuid4()),
            messageId=message_id or str(uuid.uuid4()),
            text=text
        )
        await self.emit(chunk)
    
    async def emit_error(
        self, 
        error_text: str,
        message_id: Optional[str] = None
    ) -> None:
        """Emit an error chunk."""
        chunk = UIMessageChunkError(
            type="error",
            messageId=message_id or str(uuid.uuid4()),
            errorText=error_text
        )
        await self.emit(chunk)
    
    async def emit_tool_input_start(
        self, 
        tool_call_id: str,
        tool_name: str,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a tool-input-start chunk."""
        chunk = UIMessageChunkToolInputStart(
            type="tool-input-start",
            messageId=message_id or str(uuid.uuid4()),
            toolCallId=tool_call_id,
            toolName=tool_name
        )
        await self.emit(chunk)
    
    async def emit_tool_input_delta(
        self, 
        tool_call_id: str,
        input_text_delta: str,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a tool-input-delta chunk."""
        chunk = UIMessageChunkToolInputDelta(
            type="tool-input-delta",
            messageId=message_id or str(uuid.uuid4()),
            toolCallId=tool_call_id,
            inputTextDelta=input_text_delta
        )
        await self.emit(chunk)
    
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
            messageId=message_id or str(uuid.uuid4()),
            toolCallId=tool_call_id,
            toolName=tool_name,
            input=input_data
        )
        await self.emit(chunk)
    
    async def emit_tool_output_available(
        self, 
        tool_call_id: str,
        output: Any,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a tool-output-available chunk."""
        chunk = UIMessageChunkToolOutputAvailable(
            type="tool-output-available",
            messageId=message_id or str(uuid.uuid4()),
            toolCallId=tool_call_id,
            output=output
        )
        await self.emit(chunk)
    
    async def emit_tool_output_error(
        self, 
        tool_call_id: str,
        error_text: str,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a tool-output-error chunk."""
        chunk = UIMessageChunkToolOutputError(
            type="tool-output-error",
            messageId=message_id or str(uuid.uuid4()),
            toolCallId=tool_call_id,
            errorText=error_text
        )
        await self.emit(chunk)
    
    async def emit_reasoning(
        self, 
        text: str,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a reasoning chunk."""
        chunk = UIMessageChunkReasoning(
            type="reasoning",
            messageId=message_id or str(uuid.uuid4()),
            text=text
        )
        await self.emit(chunk)
    
    async def emit_reasoning_start(
        self, 
        reasoning_id: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a reasoning-start chunk."""
        chunk = UIMessageChunkReasoningStart(
            type="reasoning-start",
            id=reasoning_id or str(uuid.uuid4()),
            messageId=message_id or str(uuid.uuid4())
        )
        await self.emit(chunk)
    
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
            messageId=message_id or str(uuid.uuid4()),
            delta=delta
        )
        await self.emit(chunk)
    
    async def emit_reasoning_end(
        self, 
        reasoning_id: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a reasoning-end chunk."""
        chunk = UIMessageChunkReasoningEnd(
            type="reasoning-end",
            id=reasoning_id or str(uuid.uuid4()),
            messageId=message_id or str(uuid.uuid4())
        )
        await self.emit(chunk)
    
    async def emit_reasoning_part_finish(
        self, 
        reasoning_id: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a reasoning-part-finish chunk."""
        chunk = UIMessageChunkReasoningPartFinish(
            type="reasoning-part-finish",
            id=reasoning_id or str(uuid.uuid4()),
            messageId=message_id or str(uuid.uuid4())
        )
        await self.emit(chunk)
    
    async def emit_source_url(
        self, 
        url: str,
        description: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a source-url chunk."""
        chunk = UIMessageChunkSourceUrl(
            type="source-url",
            messageId=message_id or str(uuid.uuid4()),
            url=url,
            description=description
        )
        await self.emit(chunk)
    
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
            messageId=message_id or str(uuid.uuid4()),
            sourceId=source_id,
            mediaType=media_type,
            title=title,
            filename=filename
        )
        await self.emit(chunk)
    
    async def emit_file(
        self, 
        url: str,
        media_type: str,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a file chunk."""
        chunk = UIMessageChunkFile(
            type="file",
            messageId=message_id or str(uuid.uuid4()),
            url=url,
            mediaType=media_type
        )
        await self.emit(chunk)
    
    async def emit_data(
        self, 
        data: Dict[str, Any],
        message_id: Optional[str] = None
    ) -> None:
        """Emit a data chunk."""
        chunk = UIMessageChunkData(
            type="data",
            messageId=message_id or str(uuid.uuid4()),
            data=data
        )
        await self.emit(chunk)
    
    async def emit_start_step(
        self, 
        step_type: str,
        step_id: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a start-step chunk."""
        chunk = UIMessageChunkStartStep(
            type="start-step",
            messageId=message_id or str(uuid.uuid4()),
            stepType=step_type,
            stepId=step_id or str(uuid.uuid4())
        )
        await self.emit(chunk)
    
    async def emit_finish_step(
        self, 
        step_type: str,
        step_id: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a finish-step chunk."""
        chunk = UIMessageChunkFinishStep(
            type="finish-step",
            messageId=message_id or str(uuid.uuid4()),
            stepType=step_type,
            stepId=step_id or str(uuid.uuid4())
        )
        await self.emit(chunk)
    
    async def emit_message_start(
        self, 
        role: str,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a message-start chunk."""
        chunk = UIMessageChunkMessageStart(
            type="message-start",
            messageId=message_id or str(uuid.uuid4()),
            role=role
        )
        await self.emit(chunk)
    
    async def emit_message_end(
        self, 
        message_id: Optional[str] = None
    ) -> None:
        """Emit a message-end chunk."""
        chunk = UIMessageChunkMessageEnd(
            type="message-end",
            messageId=message_id or str(uuid.uuid4())
        )
        await self.emit(chunk)
    
    async def emit_abort(
        self, 
        reason: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> None:
        """Emit an abort chunk."""
        chunk = UIMessageChunkAbort(
            type="abort",
            messageId=message_id or str(uuid.uuid4()),
            reason=reason
        )
        await self.emit(chunk)
    
    async def emit_message_metadata(
        self, 
        metadata: Dict[str, Any],
        message_id: Optional[str] = None
    ) -> None:
        """Emit a message-metadata chunk."""
        chunk = UIMessageChunkMessageMetadata(
            type="message-metadata",
            messageId=message_id or str(uuid.uuid4()),
            metadata=metadata
        )
        await self.emit(chunk)
    
    async def close(self) -> None:
        """Close the stream."""
        self._closed = True
        await self._queue.put(None)  # Sentinel value
    
    async def __aiter__(self):
        """Async iterator for the stream."""
        while True:
            chunk = await self._queue.get()
            if chunk is None:  # End of stream
                break
            yield chunk