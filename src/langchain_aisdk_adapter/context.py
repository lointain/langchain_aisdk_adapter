"""Unified data stream context manager.

Provides thread-safe data stream context management and unified emit interface.
"""

import uuid
from contextvars import ContextVar
from typing import Optional, Dict, Any, Union, List

# Global context variable - using ContextVar for thread safety
# ContextVar is provided by Python 3.7+ for async and multi-threaded environments
# Each async task or thread has independent context copies, ensuring data isolation
_current_data_stream: ContextVar[Optional['DataStreamWithEmitters']] = ContextVar(
    'current_data_stream', 
    default=None
)

# Performance monitoring counter
_context_operations_count: ContextVar[int] = ContextVar(
    'context_operations_count',
    default=0
)


class DataStreamContext:
    """Unified data stream context manager.
    
    Provides thread-safe data stream context management and all emit functionality.
    """
    
    # ==================== Context Management ====================
    
    @staticmethod
    def set_current_stream(stream: 'DataStreamWithEmitters') -> None:
        """Set current data stream.
        
        Args:
            stream: DataStreamWithEmitters instance
        """
        _current_data_stream.set(stream)
    
    @staticmethod
    def get_current_stream() -> Optional['DataStreamWithEmitters']:
        """Get current data stream.
        
        Returns:
            Current DataStreamWithEmitters instance, or None if not set
        """
        return _current_data_stream.get()
    
    @staticmethod
    def clear_current_stream() -> None:
        """Clear current data stream."""
        _current_data_stream.set(None)
    
    @staticmethod
    def has_stream() -> bool:
        """Check if there is a current data stream.
        
        Returns:
            bool: True if there is a data stream, False otherwise
        """
        return DataStreamContext.get_current_stream() is not None
    
    # ==================== Basic Emit Functionality ====================
    
    @staticmethod
    async def emit_data(data: Dict[str, Any]) -> bool:
        """Send data to current data stream.
        
        Args:
            data: Data dictionary to send
            
        Returns:
            bool: True if sent successfully, False if no stream or send failed
        """
        stream = DataStreamContext.get_current_stream()
        if stream:
            try:
                await stream.emit_raw_data(data)
                return True
            except Exception:
                # Silently handle errors to avoid affecting main tool logic
                return False
        return False
    
    # ==================== Message Lifecycle ====================
    
    @staticmethod
    async def emit_start(message_id: Optional[str] = None) -> bool:
        """Send start message.
        
        Args:
            message_id: Message ID, auto-generated if not provided
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        return await DataStreamContext.emit_data({
            "type": "start",
            "messageId": message_id or str(uuid.uuid4())
        })
    
    @staticmethod
    async def emit_finish(
        message_id: Optional[str] = None,
        finish_reason: str = "stop",
        usage: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send finish message.
        
        Args:
            message_id: Message ID, auto-generated if not provided
            finish_reason: Finish reason, defaults to "stop"
            usage: Usage statistics
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        data = {
            "type": "finish",
            "messageId": message_id or str(uuid.uuid4()),
            "finishReason": finish_reason
        }
        if usage:
            data["usage"] = usage
        
        return await DataStreamContext.emit_data(data)
    
    @staticmethod
    async def emit_message_start(
        role: str,
        message_id: Optional[str] = None
    ) -> bool:
        """Send message start.
        
        Args:
            role: Message role (user, assistant, system)
            message_id: Message ID, auto-generated if not provided
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        return await DataStreamContext.emit_data({
            "type": "message-start",
            "messageId": message_id or str(uuid.uuid4()),
            "role": role
        })
    
    @staticmethod
    async def emit_message_end(message_id: Optional[str] = None) -> bool:
        """Send message end.
        
        Args:
            message_id: Message ID, auto-generated if not provided
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        return await DataStreamContext.emit_data({
            "type": "message-end",
            "messageId": message_id or str(uuid.uuid4())
        })
    
    # ==================== Text Processing ====================
    
    @staticmethod
    async def emit_text_start(
        text_id: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> bool:
        """Send text start.
        
        Args:
            text_id: Text ID, auto-generated if not provided
            message_id: Message ID, auto-generated if not provided
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        return await DataStreamContext.emit_data({
            "type": "text-start",
            "id": text_id or str(uuid.uuid4()),
            "messageId": message_id or str(uuid.uuid4())
        })
    
    @staticmethod
    async def emit_text_delta(
        text_delta: str,
        text_id: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> bool:
        """Send text delta.
        
        Args:
            text_delta: Text delta content
            text_id: Text ID, auto-generated if not provided
            message_id: Message ID, auto-generated if not provided
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        return await DataStreamContext.emit_data({
            "type": "text-delta",
            "id": text_id or str(uuid.uuid4()),
            "messageId": message_id or str(uuid.uuid4()),
            "textDelta": text_delta
        })
    
    @staticmethod
    async def emit_text_end(
        text: str,
        text_id: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> bool:
        """Send text end.
        
        Args:
            text: Complete text content
            text_id: Text ID, auto-generated if not provided
            message_id: Message ID, auto-generated if not provided
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        return await DataStreamContext.emit_data({
            "type": "text-end",
            "id": text_id or str(uuid.uuid4()),
            "messageId": message_id or str(uuid.uuid4()),
            "text": text
        })
    
    # ==================== Tool Calls ====================
    
    @staticmethod
    async def emit_tool_input_start(
        tool_call_id: str,
        tool_name: str,
        message_id: Optional[str] = None
    ) -> bool:
        """Send tool input start.
        
        Args:
            tool_call_id: Tool call ID
            tool_name: Tool name
            message_id: Message ID, auto-generated if not provided
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        return await DataStreamContext.emit_data({
            "type": "tool-input-start",
            "messageId": message_id or str(uuid.uuid4()),
            "toolCallId": tool_call_id,
            "toolName": tool_name
        })
    
    @staticmethod
    async def emit_tool_input_delta(
        tool_call_id: str,
        input_text_delta: str,
        message_id: Optional[str] = None
    ) -> bool:
        """Send tool input delta.
        
        Args:
            tool_call_id: Tool call ID
            input_text_delta: Input text delta
            message_id: Message ID, auto-generated if not provided
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        return await DataStreamContext.emit_data({
            "type": "tool-input-delta",
            "messageId": message_id or str(uuid.uuid4()),
            "toolCallId": tool_call_id,
            "inputTextDelta": input_text_delta
        })
    
    @staticmethod
    async def emit_tool_input_available(
        tool_call_id: str,
        tool_name: str,
        input_data: Any,
        message_id: Optional[str] = None
    ) -> bool:
        """Send tool input available.
        
        Args:
            tool_call_id: Tool call ID
            tool_name: Tool name
            input_data: Input data
            message_id: Message ID, auto-generated if not provided
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        return await DataStreamContext.emit_data({
            "type": "tool-input-available",
            "messageId": message_id or str(uuid.uuid4()),
            "toolCallId": tool_call_id,
            "toolName": tool_name,
            "input": input_data
        })
    
    @staticmethod
    async def emit_tool_output_available(
        tool_call_id: str,
        output: Any,
        message_id: Optional[str] = None
    ) -> bool:
        """Send tool output available.
        
        Args:
            tool_call_id: Tool call ID
            output: Output data
            message_id: Message ID, auto-generated if not provided
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        return await DataStreamContext.emit_data({
            "type": "tool-output-available",
            "messageId": message_id or str(uuid.uuid4()),
            "toolCallId": tool_call_id,
            "output": output
        })
    
    @staticmethod
    async def emit_tool_output_error(
        tool_call_id: str,
        error_text: str,
        message_id: Optional[str] = None
    ) -> bool:
        """Send tool output error.
        
        Args:
            tool_call_id: Tool call ID
            error_text: Error text
            message_id: Message ID, auto-generated if not provided
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        return await DataStreamContext.emit_data({
            "type": "tool-output-error",
            "messageId": message_id or str(uuid.uuid4()),
            "toolCallId": tool_call_id,
            "errorText": error_text
        })
    
    # ==================== Reasoning Process ====================
    
    @staticmethod
    async def emit_reasoning(
        text: str,
        message_id: Optional[str] = None
    ) -> bool:
        """Send reasoning content.
        
        Args:
            text: Reasoning text
            message_id: Message ID, auto-generated if not provided
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        return await DataStreamContext.emit_data({
            "type": "reasoning",
            "messageId": message_id or str(uuid.uuid4()),
            "text": text
        })
    
    @staticmethod
    async def emit_reasoning_start(
        reasoning_id: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> bool:
        """Send reasoning start.
        
        Args:
            reasoning_id: Reasoning ID, auto-generated if not provided
            message_id: Message ID, auto-generated if not provided
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        return await DataStreamContext.emit_data({
            "type": "reasoning-start",
            "id": reasoning_id or str(uuid.uuid4()),
            "messageId": message_id or str(uuid.uuid4())
        })
    
    @staticmethod
    async def emit_reasoning_delta(
        delta: str,
        reasoning_id: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> bool:
        """Send reasoning delta.
        
        Args:
            delta: Reasoning delta content
            reasoning_id: Reasoning ID, auto-generated if not provided
            message_id: Message ID, auto-generated if not provided
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        return await DataStreamContext.emit_data({
            "type": "reasoning-delta",
            "id": reasoning_id or str(uuid.uuid4()),
            "messageId": message_id or str(uuid.uuid4()),
            "delta": delta
        })
    
    @staticmethod
    async def emit_reasoning_end(
        reasoning_id: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> bool:
        """Send reasoning end.
        
        Args:
            reasoning_id: Reasoning ID, auto-generated if not provided
            message_id: Message ID, auto-generated if not provided
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        return await DataStreamContext.emit_data({
            "type": "reasoning-end",
            "id": reasoning_id or str(uuid.uuid4()),
            "messageId": message_id or str(uuid.uuid4())
        })
    
    # ==================== Error Handling ====================
    
    @staticmethod
    async def emit_error(
        error_text: str,
        message_id: Optional[str] = None
    ) -> bool:
        """Send error message.
        
        Args:
            error_text: Error text
            message_id: Message ID, auto-generated if not provided
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        return await DataStreamContext.emit_data({
            "type": "error",
            "messageId": message_id or str(uuid.uuid4()),
            "errorText": error_text
        })
    
    @staticmethod
    async def emit_abort(
        reason: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> bool:
        """Send abort message.
        
        Args:
            reason: Abort reason
            message_id: Message ID, auto-generated if not provided
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        data = {
            "type": "abort",
            "messageId": message_id or str(uuid.uuid4())
        }
        if reason:
            data["reason"] = reason
        
        return await DataStreamContext.emit_data(data)
    
    # ==================== Metadata and Attachments ====================
    
    @staticmethod
    async def emit_message_metadata(
        metadata: Dict[str, Any],
        message_id: Optional[str] = None
    ) -> bool:
        """Send message metadata.
        
        Args:
            metadata: Metadata dictionary
            message_id: Message ID, auto-generated if not provided
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        return await DataStreamContext.emit_data({
            "type": "message-metadata",
            "messageId": message_id or str(uuid.uuid4()),
            "metadata": metadata
        })
    
    @staticmethod
    async def emit_file(
        url: str,
        media_type: str,
        message_id: Optional[str] = None
    ) -> bool:
        """Send file.
        
        Args:
            url: File URL
            media_type: Media type
            message_id: Message ID, auto-generated if not provided
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        return await DataStreamContext.emit_data({
            "type": "file",
            "messageId": message_id or str(uuid.uuid4()),
            "url": url,
            "mediaType": media_type
        })
    
    @staticmethod
    async def emit_source_url(
        url: str,
        description: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> bool:
        """Send source URL.
        
        Args:
            url: Source URL
            description: Description
            message_id: Message ID, auto-generated if not provided
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        data = {
            "type": "source-url",
            "messageId": message_id or str(uuid.uuid4()),
            "url": url
        }
        if description:
            data["description"] = description
        
        return await DataStreamContext.emit_data(data)
    
    # ==================== Step Control ====================
    
    @staticmethod
    async def emit_start_step(
        step_type: str,
        step_id: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> bool:
        """Send step start.
        
        Args:
            step_type: Step type
            step_id: Step ID, auto-generated if not provided
            message_id: Message ID, auto-generated if not provided
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        return await DataStreamContext.emit_data({
            "type": "start-step",
            "messageId": message_id or str(uuid.uuid4()),
            "stepType": step_type,
            "stepId": step_id or str(uuid.uuid4())
        })
    
    @staticmethod
    async def emit_finish_step(
        step_type: str,
        step_id: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> bool:
        """Send step finish.
        
        Args:
            step_type: Step type
            step_id: Step ID, auto-generated if not provided
            message_id: Message ID, auto-generated if not provided
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        return await DataStreamContext.emit_data({
            "type": "finish-step",
            "messageId": message_id or str(uuid.uuid4()),
            "stepType": step_type,
            "stepId": step_id or str(uuid.uuid4())
        })