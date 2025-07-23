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
        self.created_at = datetime.utcnow()
        self._lock = asyncio.Lock()

    async def add_chunk(self, chunk: UIMessageChunk) -> None:
        """Add a UIMessageChunk and update the message state."""
        async with self._lock:
            # Update content based on chunk type
            if chunk.get("type") == "start-step":
                # Add step start part
                self.parts.append(StepStartUIPart(
                    type="step-start"
                ))
            elif chunk.get("type") == "text-delta":
                self.content += chunk.get("delta", "")
                # Add text part
                self.parts.append(TextUIPart(
                    type="text",
                    text=chunk.get("delta", "")
                ))
            elif chunk.get("type") == "tool-input-available":
                # Add tool invocation part
                tool_input = chunk.get("input", {})
                # Ensure args is always a dictionary
                args_dict = tool_input if isinstance(tool_input, dict) else {"input": tool_input}
                tool_invocation = ToolInvocation(
                    state="call",
                    toolCallId=chunk.get("toolCallId", ""),
                    toolName=chunk.get("toolName", ""),
                    args=args_dict
                )
                self.parts.append(ToolInvocationUIPart(
                    type="tool-invocation",
                    toolInvocation=tool_invocation
                ))
            elif chunk.get("type") == "tool-output-available":
                # Update existing tool invocation with result
                tool_call_id = chunk.get("toolCallId", "")
                for part in self.parts:
                    if (hasattr(part, 'toolInvocation') and 
                        part.toolInvocation.toolCallId == tool_call_id):
                        part.toolInvocation.state = "result"
                        part.toolInvocation.result = chunk.get("output")
                        break

    def build_message(self) -> Message:
        """Build the final Message object."""
        return Message(
            id=self.message_id,
            createdAt=self.created_at,
            content=self.content,
            role="assistant",
            parts=self.parts.copy()
        )


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
        self.message_parts = []
        self.aggregated_text = ""
        
        try:
            # Emit start event
            start_chunk = self._create_start_event()
            await self.message_builder.add_chunk(start_chunk)
            yield start_chunk
            
            # Process stream events
            async for event in self._process_langchain_events(stream):
                await self.message_builder.add_chunk(event)
                yield event
            
            # Emit final finish-step if there's an active step and LLM generation is complete
            # This handles the case where LLM generates only text without tool calls
            if self.current_step_active and self.llm_generation_complete:
                finish_step_chunk = self._create_finish_step_event()
                await self.message_builder.add_chunk(finish_step_chunk)
                yield finish_step_chunk
                self.current_step_active = False
                self.llm_generation_complete = False
                
            # Emit finish event and handle callbacks
            finish_chunk = self._create_finish_event()
            await self.message_builder.add_chunk(finish_chunk)
            yield finish_chunk
            
            # Handle AI SDK callbacks if provided
            if isinstance(active_callbacks, BaseAICallbackHandler):
                await self._handle_ai_sdk_callbacks(active_callbacks)
                
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
                async for chunk in self._handle_text_content(value):
                    yield chunk
                continue
            
            # Handle LangChain stream events v2
            if isinstance(value, dict) and "event" in value:
                event: LangChainStreamEvent = value
                async for chunk in self._handle_stream_event(event):
                    yield chunk
                continue
            
            # Handle AI message chunk stream
            if isinstance(value, dict) and "content" in value:
                chunk: LangChainAIMessageChunk = value
                text = self._extract_text_from_chunk(chunk)
                if text:
                    async for ui_chunk in self._handle_text_content(text):
                        yield ui_chunk
    
    def _create_text_start_chunk(self, text_id: str) -> UIMessageChunkTextStart:
        """Create text start chunk."""
        return {
            "type": "text-start",
            "id": text_id
        }
    
    def _create_text_delta_chunk(self, text_id: str, delta: str) -> UIMessageChunkTextDelta:
        """Create text delta chunk."""
        return {
            "type": "text-delta",
            "id": text_id,
            "delta": delta
        }
    
    def _create_text_end_chunk(self, text_id: str) -> UIMessageChunkTextEnd:
        """Create text end chunk."""
        return {
            "type": "text-end",
            "id": text_id
        }
    
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
                    async for ui_chunk in self._handle_text_content(text):
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
            yield self._create_start_step_event()
            self.current_step_active = True
    
    async def _handle_chat_model_end(self, data: Dict[str, Any]) -> AsyncGenerator[UIMessageChunk, None]:
        """Handle chat model end event."""
        # End text if it was started
        if self.has_text_started:
            yield self._create_text_end_event()
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
        yield self._create_tool_input_start_event(tool_call_id, name)
        
        # Emit tool input delta (serialize the input as JSON)
        import json
        input_json = json.dumps(inputs if isinstance(inputs, dict) else {"input": inputs})
        yield self._create_tool_input_delta_event(tool_call_id, input_json)
        
        # Emit tool input available
        yield self._create_tool_input_available_event(tool_call_id, name, inputs)
        
        # Add tool invocation to message parts
        tool_invocation = ToolInvocation(
            state="call",
            step=self.step_count - 1,  # Set step number
            toolCallId=tool_call_id,
            toolName=name,
            args=inputs
        )
        self.message_parts.append(ToolInvocationUIPart(toolInvocation=tool_invocation))
    
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
            yield self._create_tool_output_available_event(tool_call_id, outputs)
            
            # Update tool invocation in message parts
            for part in self.message_parts:
                if (isinstance(part, ToolInvocationUIPart) and 
                    part.toolInvocation.toolCallId == tool_call_id):
                    part.toolInvocation.result = outputs
                    part.toolInvocation.state = "result"
                    part.toolInvocation.step = self.step_count - 1  # Set step number
                    break
            
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
                        yield self._create_tool_input_start_event(tool_call_id, tool_name)
                        
                        # Emit tool input delta (serialize the input as JSON)
                        import json
                        input_json = json.dumps(tool_input if isinstance(tool_input, dict) else {"input": tool_input})
                        yield self._create_tool_input_delta_event(tool_call_id, input_json)
                        
                        # Emit tool input available
                        yield self._create_tool_input_available_event(tool_call_id, tool_name, tool_input)
                        
                        # Emit tool output available
                        yield self._create_tool_output_available_event(tool_call_id, result)
                        
                        # Add tool invocation to message parts
                        # Ensure args is always a dictionary
                        args_dict = tool_input if isinstance(tool_input, dict) else {"input": tool_input}
                        
                        tool_invocation = ToolInvocation(
                            state="result",
                            step=self.step_count - 1,  # Set step number
                            toolCallId=tool_call_id,
                            toolName=tool_name,
                            args=args_dict,
                            result=result
                        )
                        self.message_parts.append(ToolInvocationUIPart(toolInvocation=tool_invocation))
                        
                        # Mark that we have completed a tool call in this step
                        self.tool_completed_in_current_step = True
        
        # After processing all intermediate steps (all tools in this LLM cycle),
        # finish the current step if we have tool calls and LLM generation is complete
        if self.tool_completed_in_current_step and self.current_step_active and self.llm_generation_complete:
            yield self._create_finish_step_event()
            self.current_step_active = False
            self.tool_completed_in_current_step = False
            self.need_new_step_for_text = True
            self.llm_generation_complete = False
    
    async def _handle_text_content(self, text: str) -> AsyncGenerator[UIMessageChunk, None]:
        """Handle text content and emit appropriate events."""
        if not text:
            return
        
        # If we need a new step for text generation after tool completion
        if self.need_new_step_for_text and not self.current_step_active:
            yield self._create_start_step_event()
            self.current_step_active = True
            self.need_new_step_for_text = False
        
        # Ensure we have an active step for text generation
        if not self.current_step_active:
            yield self._create_start_step_event()
            self.current_step_active = True
            
        # Start text if not already started
        if not self.has_text_started:
            yield self._create_text_start_event()
            self.has_text_started = True
            
            # Add text part to message only when we have actual text content
            # Initialize with the first text content
            self.message_parts.append(TextUIPart(text=text))
            self.aggregated_text = text
        else:
            # Update existing text part - find the most recent TextUIPart
            for i in range(len(self.message_parts) - 1, -1, -1):
                part = self.message_parts[i]
                if isinstance(part, TextUIPart):
                    part.text = self.aggregated_text + text
                    break
            # Update aggregated text
            self.aggregated_text += text
        
        # Emit text delta
        yield self._create_text_delta_event(text)
    
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
    
    def _create_start_step_event(self) -> UIMessageChunkStartStep:
        """Create step start event."""
        self.step_count += 1
        # Add StepStartUIPart to message parts
        self.message_parts.append(StepStartUIPart())
        
        # Reset text state for new step
        self.text_id = f"text-{uuid.uuid4()}"
        self.has_text_started = False
        
        return {
            "type": "start-step"
        }
    
    def _create_finish_step_event(self) -> UIMessageChunkFinishStep:
        """Create step finish event."""
        return {
            "type": "finish-step"
        }
    
    def _create_text_start_event(self) -> UIMessageChunkTextStart:
        """Create text start event."""
        return {
            "type": "text-start",
            "id": self.text_id
        }
    
    def _create_text_delta_event(self, delta: str) -> UIMessageChunkTextDelta:
        """Create text delta event."""
        return {
            "type": "text-delta",
            "id": self.text_id,
            "delta": delta
        }
    
    def _create_text_end_event(self) -> UIMessageChunkTextEnd:
        """Create text end event."""
        return {
            "type": "text-end",
            "id": self.text_id
        }
    
    def _create_tool_input_start_event(self, tool_call_id: str, tool_name: str) -> UIMessageChunkToolInputStart:
        """Create tool input start event."""
        return {
            "type": "tool-input-start",
            "toolCallId": tool_call_id,
            "toolName": tool_name
        }
    
    def _create_tool_input_delta_event(
        self, 
        tool_call_id: str, 
        input_text_delta: str
    ) -> UIMessageChunkToolInputDelta:
        """Create tool input delta event."""
        return {
            "type": "tool-input-delta",
            "toolCallId": tool_call_id,
            "inputTextDelta": input_text_delta
        }
    
    def _create_tool_input_available_event(
        self, 
        tool_call_id: str, 
        tool_name: str, 
        inputs: Any
    ) -> UIMessageChunkToolInputAvailable:
        """Create tool input available event."""
        return {
            "type": "tool-input-available",
            "toolCallId": tool_call_id,
            "toolName": tool_name,
            "input": inputs
        }
    
    def _create_tool_output_available_event(
        self, 
        tool_call_id: str, 
        outputs: Any
    ) -> UIMessageChunkToolOutputAvailable:
        """Create tool output available event."""
        return {
            "type": "tool-output-available",
            "toolCallId": tool_call_id,
            "output": outputs
        }
    
    async def _handle_ai_sdk_callbacks(self, callback_handler: BaseAICallbackHandler) -> None:
        """Handle AI SDK compatible callbacks."""
        # Filter out empty TextUIParts from message parts
        filtered_parts = []
        for part in self.message_parts:
            if isinstance(part, TextUIPart):
                # Only include TextUIPart if it has non-empty text
                if part.text and part.text.strip():
                    filtered_parts.append(part)
            else:
                # Include all non-TextUIPart parts
                filtered_parts.append(part)
        
        # Create final message
        message = Message(
            id=self.message_id,
            content=self.aggregated_text,
            role="assistant",
            parts=filtered_parts
        )
        
        # Create options with usage info (if available)
        options = {}
        
        # Call on_finish callback
        await callback_handler.on_finish(message, options)


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
    async def to_data_stream(
        stream: AsyncIterable[LangChainStreamInput],
        callbacks: Optional[BaseAICallbackHandler] = None,
        message_id: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[UIMessageChunk, None]:
        """Convert LangChain stream to UIMessageChunk generator.
        
        Args:
            stream: LangChain async stream from astream_events()
            callbacks: Callback handlers for stream events
            message_id: Optional message ID for tracking
            options: Control options (auto_events, emit_start, emit_finish, etc.)
        
        Returns:
            AsyncGenerator[UIMessageChunk, None]: Stream of UI message chunks
        """
        # Parse options
        opts = options or {}
        auto_events = opts.get("auto_events", True)
        message_id = message_id or str(uuid.uuid4())
        
        # Create processor
        processor = StreamProcessor(
            message_id=message_id,
            auto_events=auto_events,
            callbacks=callbacks
        )
        
        # Process and yield chunks
        async for chunk in processor.process_stream(stream):
            yield chunk
    
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
    data_stream = to_data_stream(stream, callbacks, message_id)
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