#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangChain to AI SDK Adapter - V1 Implementation

This module provides functionality to convert LangChain output streams
to AI SDK Data Stream Protocol compatible streams according to the v1 specification.
"""

import asyncio
import json
import uuid
from typing import (
    Any, Dict, List, Optional, Union, AsyncGenerator, AsyncIterable,
    Callable, Awaitable, Tuple
)
from fastapi.responses import StreamingResponse
from datetime import datetime

from .models import (
    LangChainStreamInput,
    UIMessageChunk,
    UIMessageChunkStart,
    UIMessageChunkFinish,
    UIMessageChunkStartStep,
    UIMessageChunkFinishStep,
    UIMessageChunkTextStart,
    UIMessageChunkTextDelta,
    UIMessageChunkTextEnd,
    UIMessageChunkToolInputStart,
    UIMessageChunkToolInputDelta,
    UIMessageChunkToolInputAvailable,
    UIMessageChunkToolOutputAvailable,
    UIMessageChunkToolOutputError,
    UIMessageChunkError,
    UIMessageChunkFile,
    UIMessageChunkData,
    UIMessageChunkReasoning,
    UIMessageChunkReasoningStart,
    UIMessageChunkReasoningDelta,
    UIMessageChunkReasoningEnd,
    UIMessageChunkSourceUrl,
    UIMessageChunkSourceDocument,
    UIMessageChunkAbort,
    UIMessageChunkMessageMetadata,
    LangChainStreamEvent,
    LangChainAIMessageChunk,
)
from .callbacks import (
    BaseAICallbackHandler,
    CallbacksTransformer,
    Message,
    UIPart,
    TextUIPart,
    ToolInvocationUIPart,
    ToolInvocation,
    StepStartUIPart,
)


class MessageBuilder:
    """Builds Message objects from UIMessageChunk events.
    
    This class accumulates UIMessageChunk events and constructs
    a complete Message object with proper parts and content.
    """

    def __init__(self, message_id: Optional[str] = None):
        self.message_id = message_id or str(uuid.uuid4())
        self.parts: List[UIPart] = []
        self.content = ""
        self.created_at = datetime.now()
        self._lock = asyncio.Lock()
        self._new_parts: List[UIPart] = []
        self._current_text_parts = {}  # Track current TextUIPart objects by ID

    async def add_chunk(self, chunk: UIMessageChunk) -> List[UIPart]:
        """Add a UIMessageChunk and update the message state.
        
        This method processes all types of UIMessageChunk and converts them to appropriate UIPart objects.
        It ensures that parts are accumulated regardless of auto_events setting.
        
        Returns:
            List of newly generated UIPart objects for this chunk.
        """
        async with self._lock:
            self._new_parts.clear()  # Clear previous new parts
            chunk_type = chunk.get("type")
            
            # Skip chunks without type
            if not chunk_type:
                return self._new_parts.copy()
            
            # Handle step-related chunks
            if chunk_type == "start-step":
                # Import StepStartUIPart here to avoid circular imports
                from .callbacks import StepStartUIPart
                # Only create step-start part if we don't already have one at the end
                # This prevents duplicate step-start parts
                if not self.parts or not (hasattr(self.parts[-1], 'type') and self.parts[-1].type == "step-start"):
                    part = StepStartUIPart(type="step-start")
                    self.parts.append(part)
                    self._new_parts.append(part)
            
            # Handle text-related chunks - SIMPLIFIED IMPLEMENTATION
            elif chunk_type == "text-start":
                # text-start: 创建新的TextUIPart对象
                text_id = chunk.get("id", "default")
                text_part = TextUIPart(type="text", text="")
                self._current_text_parts[text_id] = text_part
            elif chunk_type == "text-delta":
                # text-delta: 累加text到对应的TextUIPart对象
                text_id = chunk.get("id", "default")
                delta = chunk.get("textDelta", chunk.get("delta", ""))
                if text_id in self._current_text_parts:
                    self._current_text_parts[text_id].text += delta
            elif chunk_type == "text-end":
                # text-end: 将TextUIPart对象放入parts数组
                text_id = chunk.get("id", "default")
                if text_id in self._current_text_parts:
                    text_part = self._current_text_parts[text_id]
                    if text_part.text.strip():  # 只有非空文本才添加
                        self.parts.append(text_part)
                        self._new_parts.append(text_part)
                    # 清理当前文本部分
                    del self._current_text_parts[text_id]
            
            # Handle tool-related chunks
            elif chunk_type == "tool-input-start":
                # Tool input start doesn't create a part immediately
                pass
            elif chunk_type == "tool-input-delta":
                # Tool input delta doesn't create a part immediately
                pass
            elif chunk_type == "tool-input-available":
                # Store tool information but don't create part yet
                # Only create part when tool has result (state="result")
                tool_call_id = chunk.get("toolCallId", "")
                tool_input = chunk.get("input", {})
                args_dict = tool_input if isinstance(tool_input, dict) else {"input": tool_input}
                
                # Store tool info for later use
                if not hasattr(self, '_pending_tools'):
                    self._pending_tools = {}
                self._pending_tools[tool_call_id] = {
                    "toolName": chunk.get("toolName", ""),
                    "args": args_dict
                }
            elif chunk_type == "tool-output-available":
                # Create tool invocation part only when result is available
                tool_call_id = chunk.get("toolCallId", "")
                if hasattr(self, '_pending_tools') and tool_call_id in self._pending_tools:
                    # Check if we already have a tool invocation with this toolCallId
                    existing_tool = None
                    for part in self.parts:
                        if (hasattr(part, 'toolInvocation') and 
                            hasattr(part.toolInvocation, 'toolCallId') and 
                            part.toolInvocation.toolCallId == tool_call_id):
                            existing_tool = part
                            break
                    
                    # Only create new tool invocation if it doesn't exist
                    if existing_tool is None:
                        tool_info = self._pending_tools[tool_call_id]
                        
                        # Calculate step number based on existing tool invocations
                        step = sum(1 for part in self.parts if hasattr(part, 'toolInvocation'))
                        
                        tool_invocation = ToolInvocation(
                            state="result",
                            step=step,
                            toolCallId=tool_call_id,
                            toolName=tool_info["toolName"],
                            args=tool_info["args"],
                            result=chunk.get("output")
                        )
                        part = ToolInvocationUIPart(
                            type="tool-invocation",
                            toolInvocation=tool_invocation
                        )
                        self.parts.append(part)
                        self._new_parts.append(part)
                        
                        # Clean up pending tool
                        del self._pending_tools[tool_call_id]
            elif chunk_type == "tool-output-error":
                # Create tool invocation part with error state
                tool_call_id = chunk.get("toolCallId", "")
                if hasattr(self, '_pending_tools') and tool_call_id in self._pending_tools:
                    # Check if we already have a tool invocation with this toolCallId
                    existing_tool = None
                    for part in self.parts:
                        if (hasattr(part, 'toolInvocation') and 
                            hasattr(part.toolInvocation, 'toolCallId') and 
                            part.toolInvocation.toolCallId == tool_call_id):
                            existing_tool = part
                            break
                    
                    # Only create new tool invocation if it doesn't exist
                    if existing_tool is None:
                        tool_info = self._pending_tools[tool_call_id]
                        
                        # Calculate step number based on existing tool invocations
                        step = sum(1 for part in self.parts if hasattr(part, 'toolInvocation'))
                        
                        tool_invocation = ToolInvocation(
                            state="error",
                            step=step,
                            toolCallId=tool_call_id,
                            toolName=tool_info["toolName"],
                            args=tool_info["args"],
                            result={"error": chunk.get("errorText", "Unknown error")}
                        )
                        part = ToolInvocationUIPart(
                            type="tool-invocation",
                            toolInvocation=tool_invocation
                        )
                        self.parts.append(part)
                        self._new_parts.append(part)
                        
                        # Clean up pending tool
                        del self._pending_tools[tool_call_id]
            
            # Handle reasoning chunks (if supported in the future)
            elif chunk_type in ["reasoning", "reasoning-start", "reasoning-delta", "reasoning-end"]:
                # For now, treat reasoning as text content
                if chunk_type == "reasoning":
                    reasoning_text = chunk.get("text", "")
                    if reasoning_text:
                        self.content += reasoning_text
                        part = TextUIPart(type="text", text=reasoning_text)
                        self.parts.append(part)
                        self._new_parts.append(part)
            
            # Handle file chunks
            elif chunk_type == "file":
                # Import FileUIPart here to avoid circular imports
                from .callbacks import FileUIPart
                # Map from AI SDK file chunk to FileUIPart
                file_part = FileUIPart(
                    type="file",
                    url=chunk.get("url", ""),
                    mediaType=chunk.get("mediaType", "")
                )
                self.parts.append(file_part)
                self._new_parts.append(file_part)
            
            # Handle data chunks (type can be any string starting with 'data-')
            elif chunk_type.startswith("data") or chunk_type == "data":
                # Data chunks don't have a specific UIPart type in the current implementation
                # They are handled as metadata or ignored for now
                pass
            
            # Handle source chunks
            elif chunk_type == "source-url":
                # Import SourceUIPart here to avoid circular imports
                from .callbacks import SourceUIPart
                # Map from AI SDK source chunk to SourceUIPart
                source_data = {
                    "url": chunk.get("url", ""),
                    "description": chunk.get("description")
                }
                source_part = SourceUIPart(
                    type="source",
                    source=source_data
                )
                self.parts.append(source_part)
                self._new_parts.append(source_part)
            elif chunk_type == "source-document":
                # Import SourceUIPart here to avoid circular imports
                from .callbacks import SourceUIPart
                # Map from AI SDK source document chunk to SourceUIPart
                source_data = {
                    "sourceId": chunk.get("sourceId", ""),
                    "mediaType": chunk.get("mediaType", ""),
                    "title": chunk.get("title", ""),
                    "filename": chunk.get("filename")
                }
                source_part = SourceUIPart(
                    type="source",
                    source=source_data
                )
                self.parts.append(source_part)
                self._new_parts.append(source_part)
            
            # Handle error chunks
            elif chunk_type == "error":
                # Import ErrorUIPart here to avoid circular imports
                from .callbacks import ErrorUIPart
                error_part = ErrorUIPart(
                    type="error",
                    error=chunk.get("errorText", "Error occurred")
                )
                self.parts.append(error_part)
                self._new_parts.append(error_part)
                # Also add to content for backward compatibility
                error_text = chunk.get("errorText", "Error occurred")
                self.content += f"\n[Error: {error_text}]"
            
            # Handle step chunks (already handled above)
            # elif chunk_type == "start-step":
            #     # Already handled above
            
            # Handle other chunk types (start, finish, etc.)
            elif chunk_type in ["start", "finish", "finish-step", "abort"]:
                # These don't create parts but are important for protocol flow
                pass
            
            return self._new_parts.copy()

    def build_message(self) -> Message:
        """Build the final Message object."""
        return Message(
            id=self.message_id,
            createdAt=self.created_at,
            content=self.content,
            role="assistant",
            parts=self.parts.copy()
        )


class ProtocolGenerator:
    """Unified protocol generator for all chunk types."""
    
    @staticmethod
    def create_text_start(text_id: str) -> UIMessageChunkTextStart:
        """Create text-start protocol chunk."""
        return {
            "type": "text-start",
            "id": text_id
        }
    
    @staticmethod
    def create_text_delta(text_id: str, delta: str) -> UIMessageChunkTextDelta:
        """Create text-delta protocol chunk."""
        return {
            "type": "text-delta",
            "id": text_id,
            "delta": delta
        }
    
    @staticmethod
    def create_text_end(text_id: str, text: str = "") -> UIMessageChunkTextEnd:
        """Create text-end protocol chunk."""
        return {
            "type": "text-end",
            "id": text_id,
            "text": text
        }


class StreamProcessor:
    """Core stream processing logic for converting LangChain events to AI SDK format."""

    def __init__(
        self,
        message_id: str,
        auto_events: bool = True,
        callbacks: Optional[BaseAICallbackHandler] = None
    ):
        self.message_id = message_id
        self.auto_events = auto_events
        self.callbacks = callbacks
        self.message_builder = MessageBuilder(message_id)
        self._lock = asyncio.Lock()
        self._current_text_id: Optional[str] = None
        self._tool_calls: Dict[str, Dict[str, Any]] = {}
        self._accumulated_text = ""
        
    async def process_stream(
        self,
        stream: AsyncIterable[LangChainStreamInput],
        callbacks: Optional[BaseAICallbackHandler] = None,
    ) -> AsyncGenerator[UIMessageChunk, None]:
        """Process LangChain stream and generate AI SDK compatible events."""
        
        # Use passed callbacks or fallback to instance callbacks
        active_callbacks = callbacks or self.callbacks
        
        # Initialize state variables
        self.current_step_active = False
        self.llm_generation_complete = False
        self.has_text_started = False
        self.tool_completed_in_current_step = False
        self.need_new_step_for_text = False
        self.step_count = 0
        self.text_id = f"text-{uuid.uuid4()}"
        self.tool_calls = {}
        
        try:
            # Create and process start event only if auto_events is True
            if self.auto_events:
                start_chunk = self._create_start_event()
                await self.message_builder.add_chunk(start_chunk)
                yield start_chunk
            
            # Process stream events
            async for event in self._process_langchain_events(stream):
                # Only accumulate parts and yield events if auto_events is True
                if self.auto_events:
                    await self.message_builder.add_chunk(event)
                    yield event
            
            # Create and process final finish-step if there's an active step and LLM generation is complete
            # This handles the case where LLM generates only text without tool calls
            if self.current_step_active and self.llm_generation_complete and self.auto_events:
                finish_step_chunk = self._create_finish_step_event()
                await self.message_builder.add_chunk(finish_step_chunk)
                yield finish_step_chunk
                self.current_step_active = False
                self.llm_generation_complete = False
                
            # Always handle AI SDK callbacks if provided, regardless of auto_events
            if isinstance(active_callbacks, BaseAICallbackHandler):
                await self._handle_ai_sdk_callbacks(active_callbacks)
            
            # Create and process finish event only if auto_events is True
            if self.auto_events:
                finish_chunk = self._create_finish_event()
                await self.message_builder.add_chunk(finish_chunk)
                yield finish_chunk
                
        except Exception as e:
            if isinstance(active_callbacks, BaseAICallbackHandler):
                await active_callbacks.on_error(e)
            raise
    
    async def _process_langchain_events(
        self, 
        stream: AsyncIterable[LangChainStreamInput]
    ) -> AsyncGenerator[UIMessageChunk, None]:
        """Process individual LangChain events and convert to AI SDK format."""
        
        async for value in stream:
            # Handle string stream (direct text output)
            if isinstance(value, str):
                async for chunk in self._handle_incremental_text(value):
                    yield chunk
                continue
            
            # Handle LangChain stream events v2
            if isinstance(value, dict) and "event" in value:
                event: LangChainStreamEvent = value
                async for chunk in self._handle_stream_event(event):
                    yield chunk
                continue
            
            # Handle AI message chunk stream (but skip text processing here to avoid duplication)
            # Text content from AI message chunks will be handled by the event stream processing
            if isinstance(value, dict) and "content" in value:
                # Skip text processing here to avoid duplication with event stream
                # The text content will be processed through on_chat_model_stream events
                continue
    
    # Text protocol methods removed - use ProtocolGenerator instead
    
    def _create_tool_input_start_chunk(self, tool_call_id: str, tool_name: str) -> UIMessageChunkToolInputStart:
        """Create tool input start chunk."""
        return {
            "type": "tool-input-start",
            "toolCallId": tool_call_id,
            "toolName": tool_name
        }
    
    def _create_tool_output_available_chunk(self, tool_call_id: str, output: Any) -> UIMessageChunkToolOutputAvailable:
        """Create tool output available chunk."""
        return {
            "type": "tool-output-available",
            "toolCallId": tool_call_id,
            "output": output
        }
    
    def _create_start_step_chunk(self) -> UIMessageChunkStartStep:
        """Create start step chunk."""
        return {
            "type": "start-step"
        }
    
    def _create_finish_step_chunk(self) -> UIMessageChunkFinishStep:
        """Create finish step chunk."""
        return {
            "type": "finish-step"
        }
    
    def _create_start_chunk(self) -> UIMessageChunkStart:
        """Create start chunk."""
        chunk = {
            "type": "start",
            "messageId": self.message_id
        }
        return chunk
    
    def _create_finish_chunk(self) -> UIMessageChunkFinish:
        """Create finish chunk."""
        return {
            "type": "finish"
        }
    
    def _create_error_chunk(self, error_text: str) -> UIMessageChunkError:
        """Create error chunk."""
        return {
            "type": "error",
            "errorText": error_text
        }
    
    async def _call_callback(self, callback: Optional[Callable], *args) -> None:
        """Safely call a callback function, handling both sync and async callbacks."""
        if callback is None:
            return
        
        try:
            result = callback(*args)
            if hasattr(result, '__await__'):
                await result
        except Exception:
            # Silently ignore callback errors to prevent stream interruption
            pass
    
    async def _handle_stream_event(self, event: LangChainStreamEvent) -> AsyncGenerator[UIMessageChunk, None]:
        """Handle specific LangChain stream events."""
        
        event_type = event["event"]
        data = event.get("data", {})
        
        # Handle chat model events
        if event_type == "on_chat_model_start":
            async for chunk in self._handle_chat_model_start(data):
                yield chunk
        elif event_type == "on_chat_model_stream":
            chunk_data = data.get("chunk")
            if chunk_data:
                text = self._extract_text_from_chunk(chunk_data)
                if text:
                    # LangChain chunks are incremental, use them directly as delta
                    self._accumulated_text += text
                    async for ui_chunk in self._handle_incremental_text(text):
                        yield ui_chunk
        elif event_type == "on_chat_model_end":
            async for chunk in self._handle_chat_model_end(data):
                yield chunk
        
        # Handle chain events that might contain tool information
        elif event_type == "on_chain_stream":
            async for chunk in self._handle_chain_stream(data):
                yield chunk
        elif event_type == "on_chain_end":
            async for chunk in self._handle_chain_end(data):
                yield chunk
        
        # Handle tool events (direct tool calls)
        elif event_type == "on_tool_start":
            async for chunk in self._handle_tool_start(data):
                yield chunk
        elif event_type == "on_tool_end":
            async for chunk in self._handle_tool_end(data):
                yield chunk
    
    async def _handle_chat_model_start(self, data: Dict[str, Any]) -> AsyncGenerator[UIMessageChunk, None]:
        """Handle chat model start event."""
        if not self.current_step_active:
            self.step_count += 1
            self.text_id = f"text-{uuid.uuid4()}"
            self.has_text_started = False
            self._accumulated_text = ""
            yield {"type": "start-step"}
            self.current_step_active = True
    
    async def _handle_chat_model_end(self, data: Dict[str, Any]) -> AsyncGenerator[UIMessageChunk, None]:
        """Handle chat model end event."""
        # End text if it was started
        if self.has_text_started:
            yield ProtocolGenerator.create_text_end(self.text_id, self._accumulated_text)
            self.has_text_started = False
        
        # Mark that LLM generation is complete
        # We don't finish the step here because we need to wait for all tool calls to complete
        # The step will be finished either:
        # 1. In _process_intermediate_steps after all tools are processed
        # 2. In the final cleanup if no tools were called
        self.llm_generation_complete = True
        # after all tool outputs are available
    
    async def _handle_tool_start(self, data: Dict[str, Any]) -> AsyncGenerator[UIMessageChunk, None]:
        """Handle tool start event."""
        # Skip direct tool events if data is incomplete
        # LangChain's direct tool events often lack proper tool information
        # We'll rely on intermediate_steps processing instead
        if not data or not data.get("name") and not data.get("serialized", {}).get("name"):
            return
        
        run_id = data.get("run_id", str(uuid.uuid4()))
        
        # Try different possible field names for tool name with more comprehensive extraction
        name = (
            data.get("name") or 
            data.get("tool_name") or 
            data.get("tool") or
            (data.get("serialized", {}).get("name")) or
            (data.get("serialized", {}).get("id", [""])[-1] if isinstance(data.get("serialized", {}).get("id"), list) else "") or
            (data.get("metadata", {}).get("name")) or
            "unknown_tool"
        )
        
        # Additional extraction from nested structures
        if name == "unknown_tool" and "serialized" in data:
            serialized = data["serialized"]
            if isinstance(serialized, dict):
                # Try to extract from various nested locations
                name = (
                    serialized.get("_type") or
                    serialized.get("class_name") or
                    (serialized.get("kwargs", {}).get("name")) or
                    name
                )
        
        # Skip if we still can't determine the tool name
        if name == "unknown_tool":
            return
        
        inputs = data.get("inputs", {})
        
        tool_call_id = str(run_id)
        
        # Store tool call info
        self.tool_calls[tool_call_id] = {
            "name": name,
            "inputs": inputs,
            "state": "call"
        }
        
        # Emit tool input start
        yield {
            "type": "tool-input-start",
            "toolCallId": tool_call_id,
            "toolName": name
        }
        
        # Emit tool input delta (serialize the input as JSON)
        import json
        input_json = json.dumps(inputs if isinstance(inputs, dict) else {"input": inputs})
        yield {
            "type": "tool-input-delta",
            "toolCallId": tool_call_id,
            "inputTextDelta": input_json
        }
        
        # Emit tool input available
        yield {
            "type": "tool-input-available",
            "toolCallId": tool_call_id,
            "toolName": name,
            "input": inputs
        }
        
        # Tool invocation will be handled by MessageBuilder
    
    async def _handle_tool_end(self, data: Dict[str, Any]) -> AsyncGenerator[UIMessageChunk, None]:
        """Handle tool end event."""
        run_id = data.get("run_id", "")
        outputs = data.get("outputs", {})
        
        tool_call_id = str(run_id)
        
        # Only process if we have a corresponding tool start event
        if tool_call_id in self.tool_calls:
            # Update tool call state
            self.tool_calls[tool_call_id]["outputs"] = outputs
            self.tool_calls[tool_call_id]["state"] = "result"
            
            # Emit tool output available
            yield {
                "type": "tool-output-available",
                "toolCallId": tool_call_id,
                "output": outputs
            }
            
            # Tool invocation update will be handled by MessageBuilder
            
            # Mark that we have completed a tool call in this step
            self.tool_completed_in_current_step = True
    
    async def _handle_chain_stream(self, data: Dict[str, Any]) -> AsyncGenerator[UIMessageChunk, None]:
        """Handle chain stream event that might contain tool information."""
        chunk = data.get("chunk", {})
        
        # Check for intermediate_steps in the chunk
        if isinstance(chunk, dict) and "intermediate_steps" in chunk:
            async for ui_chunk in self._process_intermediate_steps(chunk["intermediate_steps"]):
                yield ui_chunk
    
    async def _handle_chain_end(self, data: Dict[str, Any]) -> AsyncGenerator[UIMessageChunk, None]:
        """Handle chain end event that might contain tool information."""
        input_data = data.get("input", {})
        
        # Check for intermediate_steps in the input
        if isinstance(input_data, dict) and "intermediate_steps" in input_data:
            async for ui_chunk in self._process_intermediate_steps(input_data["intermediate_steps"]):
                yield ui_chunk
    
    async def _process_intermediate_steps(self, intermediate_steps) -> AsyncGenerator[UIMessageChunk, None]:
        """Process intermediate steps to extract tool calls."""
        if not intermediate_steps:
            return
            
        for step in intermediate_steps:
            if isinstance(step, tuple) and len(step) == 2:
                action, result = step
                
                # Check if this is an AgentAction with tool information
                if hasattr(action, 'tool') and hasattr(action, 'tool_input'):
                    tool_name = action.tool
                    tool_input = action.tool_input
                    
                    # Create a unique tool call ID based on the action
                    tool_call_id = f"tool_{abs(hash(str(action)))}"
                    
                    # Note: Direct tool events are now skipped when incomplete,
                    # so intermediate_steps processing is the primary source of tool events
                    
                    # Only process if we haven't seen this tool call before
                    if tool_call_id not in self.tool_calls:
                        # Store tool call info
                        self.tool_calls[tool_call_id] = {
                            "name": tool_name,
                            "inputs": tool_input,
                            "outputs": result,
                            "state": "result"
                        }
                        
                        # Emit tool input start
                        yield {
                            "type": "tool-input-start",
                            "toolCallId": tool_call_id,
                            "toolName": tool_name
                        }
                        
                        # Emit tool input delta (serialize the input as JSON)
                        import json
                        input_json = json.dumps(tool_input if isinstance(tool_input, dict) else {"input": tool_input})
                        yield {
                            "type": "tool-input-delta",
                            "toolCallId": tool_call_id,
                            "inputTextDelta": input_json
                        }
                        
                        # Emit tool input available
                        yield {
                            "type": "tool-input-available",
                            "toolCallId": tool_call_id,
                            "toolName": tool_name,
                            "input": tool_input
                        }
                        
                        # Emit tool output available
                        yield {
                            "type": "tool-output-available",
                            "toolCallId": tool_call_id,
                            "output": result
                        }
                        
                        # Tool invocation will be handled by MessageBuilder
                        
                        # Mark that we have completed a tool call in this step
                        self.tool_completed_in_current_step = True
        
        # After processing all intermediate steps (all tools in this LLM cycle),
        # finish the current step if we have tool calls and LLM generation is complete
        if self.tool_completed_in_current_step and self.current_step_active and self.llm_generation_complete:
            yield {"type": "finish-step"}
            self.current_step_active = False
            self.tool_completed_in_current_step = False
            self.need_new_step_for_text = True
            self.llm_generation_complete = False
    
    async def _handle_incremental_text(self, text: str) -> AsyncGenerator[UIMessageChunk, None]:
        """Handle incremental text content using unified protocol generator."""
        if not text:
            return
        
        # Start text if not already started
        if not self.has_text_started:
            yield ProtocolGenerator.create_text_start(self.text_id)
            self.has_text_started = True
        
        # Send text delta
        yield ProtocolGenerator.create_text_delta(self.text_id, text)
    
    def _extract_text_from_chunk(self, chunk: LangChainAIMessageChunk) -> str:
        """Extract text content from LangChain AI message chunk."""
        if isinstance(chunk, dict):
            content = chunk.get("content", "")
        else:
            content = getattr(chunk, "content", "")
        
        if isinstance(content, str):
            return content
        
        # Handle complex content
        text_parts = []
        if hasattr(content, '__iter__') and not isinstance(content, str):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
        
        return "".join(text_parts)
    
    def _create_start_event(self) -> UIMessageChunkStart:
        """Create stream start event."""
        return {
            "type": "start",
            "messageId": self.message_id
        }
    
    def _create_finish_event(self) -> UIMessageChunkFinish:
        """Create stream finish event."""
        return {
            "type": "finish"
        }
    
    async def _handle_ai_sdk_callbacks(self, callback_handler: BaseAICallbackHandler) -> None:
        """Handle AI SDK compatible callbacks."""
        # Use MessageBuilder to get the final message with all parts (including manual ones)
        message = self.message_builder.build_message()
        
        # Filter out empty TextUIParts from message parts
        filtered_parts = []
        for part in message.parts:
            if isinstance(part, TextUIPart):
                # Only include TextUIPart if it has non-empty text
                if part.text and part.text.strip():
                    filtered_parts.append(part)
            else:
                # Include all non-TextUIPart parts
                filtered_parts.append(part)
        
        # Update message with filtered parts
        message.parts = filtered_parts
        
        # Create options with usage info (if available)
        options = {}
        
        # Call on_finish callback
        await callback_handler.on_finish(message, options)


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
        callbacks: Optional[BaseAICallbackHandler] = None
    ):
        self._stream_generator = stream_generator
        self._message_id = message_id
        self._manual_queue: asyncio.Queue = asyncio.Queue()
        self._closed = False
        self._stream_started = False
        self._auto_close = auto_close
        self._message_builder = message_builder
        self._callbacks = callbacks
    
    async def emit_file(
        self, 
        url: str,
        mediaType: str,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a file chunk."""
        from .models import UIMessageChunkFile
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
        from .models import UIMessageChunkData
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
        from .models import UIMessageChunkSourceUrl
        import uuid
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
        from .models import UIMessageChunkSourceDocument
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
        from .models import UIMessageChunkReasoning
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
        from .models import UIMessageChunkError
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
        import uuid
        chunk = ProtocolGenerator.create_text_start(
            text_id or str(uuid.uuid4())
        )
        await self._emit_manual_chunk(chunk)
    
    async def emit_text_delta(
        self,
        delta: str,
        text_id: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a text-delta chunk using unified protocol generator."""
        import uuid
        chunk = ProtocolGenerator.create_text_delta(
            text_id or str(uuid.uuid4()),
            delta
        )
        await self._emit_manual_chunk(chunk)
    
    async def emit_text_end(
        self,
        text: str,
        text_id: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> None:
        """Emit a text-end chunk using unified protocol generator."""
        import uuid
        chunk = ProtocolGenerator.create_text_end(
            text_id or str(uuid.uuid4())
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
        from .models import UIMessageChunkToolInputStart
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
        from .models import UIMessageChunkToolInputDelta
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
        from .models import UIMessageChunkToolInputAvailable
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
        from .models import UIMessageChunkToolOutputAvailable
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
        from .models import UIMessageChunkToolOutputError
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
        from .models import UIMessageChunkReasoningStart
        import uuid
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
        from .models import UIMessageChunkReasoningDelta
        import uuid
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
        from .models import UIMessageChunkReasoningEnd
        import uuid
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
        from .models import UIMessageChunkReasoningPartFinish
        import uuid
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
        from .models import UIMessageChunkStartStep
        import uuid
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
        from .models import UIMessageChunkFinishStep
        import uuid
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
        from .models import UIMessageChunkStart
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
        from .models import UIMessageChunkFinish
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
        from .models import UIMessageChunkMessageStart
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
        from .models import UIMessageChunkMessageEnd
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
        from .models import UIMessageChunkAbort
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
        from .models import UIMessageChunkMessageMetadata
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
                        # Log error but continue
                        continue
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
                            
                            yield chunk_with_parts
                        else:
                            # For auto chunks, just yield them as-is since they're already processed
                            yield clean_chunk
                        
                except asyncio.TimeoutError:
                    # Check if both tasks are done
                    if auto_task.done() and manual_task.done():
                        break
                    continue
        finally:
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
    ) -> "DataStreamResponse":
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
        data_stream_writer: "DataStreamWriter",
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


# Legacy function for backward compatibility
async def to_data_stream(
    stream: AsyncIterable[LangChainStreamInput],
    callbacks: Optional[BaseAICallbackHandler] = None,
    message_id: Optional[str] = None,
) -> AsyncGenerator[UIMessageChunk, None]:
    """Legacy function - use LangChainAdapter.to_data_stream instead."""
    async for chunk in LangChainAdapter.to_data_stream(stream, callbacks, message_id):
        yield chunk


class DataStreamResponse(StreamingResponse):
    """FastAPI StreamingResponse wrapper for AI SDK Data Stream Protocol.
    
    This class wraps an AI SDK data stream as a FastAPI StreamingResponse,
    automatically formatting UIMessageChunk objects as Server-Sent Events (SSE).
    """
    
    def __init__(
        self,
        stream: AsyncGenerator[UIMessageChunk, None],
        headers: Optional[Dict[str, str]] = None,
        status: int = 200
    ):
        """Initialize DataStreamResponse.
        
        Args:
            stream: The data stream generator
            headers: Optional HTTP headers
            status: HTTP status code
        """
        # Convert UIMessageChunk stream to text stream
        text_stream = self._convert_to_text_stream(stream)
        
        # Set default headers for SSE
        default_headers = {
            "Content-Type": "text/plain; charset=utf-8",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
        if headers:
            default_headers.update(headers)
        
        super().__init__(
            content=text_stream,
            headers=default_headers,
            status_code=status
        )
    
    async def _convert_to_text_stream(
        self, 
        data_stream: AsyncGenerator[UIMessageChunk, None]
    ) -> AsyncGenerator[str, None]:
        """Convert UIMessageChunk to SSE text format."""
        async for chunk in data_stream:
            # Convert chunk to JSON string
            if hasattr(chunk, 'dict'):
                chunk_dict = chunk.dict()
            elif isinstance(chunk, dict):
                chunk_dict = chunk
            else:
                chunk_dict = {"type": "error", "errorText": "Invalid chunk format"}
            
            # Format as SSE
            json_str = json.dumps(chunk_dict, ensure_ascii=False)
            yield f"data: {json_str}\n\n"


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
    data_stream = to_data_stream(stream, callbacks, message_id, options)
    return DataStreamResponse(data_stream, headers, status)


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


async def merge_into_data_stream(
    stream: AsyncIterable[LangChainStreamInput],
    data_stream_writer: DataStreamWriter,
    callbacks: Optional[BaseAICallbackHandler] = None,
    message_id: Optional[str] = None
) -> None:
    """Merge LangChain output streams into an existing data stream.
    
    This function merges a LangChain stream into an existing data stream writer,
    similar to the Node.js AI SDK's mergeIntoDataStream function.
    
    Args:
        stream: Input stream from LangChain
        data_stream_writer: Target data stream writer
        callbacks: Optional callbacks (BaseAICallbackHandler)
        message_id: Optional message ID (auto-generated if not provided)
        
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
    data_stream = to_data_stream(stream, callbacks, message_id)
    
    async for chunk in data_stream:
        await data_stream_writer.write(chunk)