#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced LangChain to AI SDK adapter implementation.

This module provides a comprehensive adapter that supports:
- Tool invocation events (tool-input-start, tool-input-available, tool-output-available)
- Step control events (start-step, finish-step)
- Stream control events (start, finish)
- Complete AI SDK UI Stream Protocol compatibility
"""

import asyncio
import json
import uuid
from typing import AsyncGenerator, AsyncIterable, Optional, Union, Dict, Any, List

from .callbacks import CallbacksTransformer, StreamCallbacks
from .ai_sdk_callbacks import BaseAICallbackHandler, Message, UIPart, TextUIPart, ToolInvocationUIPart, ToolInvocation, StepStartUIPart
from .models import (
    LangChainAIMessageChunk,
    LangChainMessageContentComplex,
    LangChainStreamEvent,
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
    UIMessageChunkToolInputAvailable,
    UIMessageChunkToolOutputAvailable,
)


class EnhancedStreamProcessor:
    """Enhanced stream processor that handles complete AI SDK protocol."""
    
    def __init__(self, message_id: Optional[str] = None):
        self.message_id = message_id or str(uuid.uuid4())
        self.text_id = str(uuid.uuid4())
        self.step_count = 0
        self.current_step_active = False
        self.tool_calls: Dict[str, Dict[str, Any]] = {}
        self.aggregated_text = ""
        self.message_parts: List[UIPart] = []
        self.has_started = False
        self.has_text_started = False
        
    async def process_stream(
        self,
        stream: AsyncIterable[LangChainStreamInput],
        callbacks: Optional[Union[StreamCallbacks, BaseAICallbackHandler]] = None,
    ) -> AsyncGenerator[UIMessageChunk, None]:
        """Process LangChain stream and generate AI SDK compatible events."""
        
        try:
            # Emit start event
            yield self._create_start_event()
            
            # Process stream events
            async for event in self._process_langchain_events(stream):
                yield event
                
            # Emit finish event and handle callbacks
            yield self._create_finish_event()
            
            # Handle AI SDK callbacks if provided
            if isinstance(callbacks, BaseAICallbackHandler):
                await self._handle_ai_sdk_callbacks(callbacks)
                
        except Exception as e:
            if isinstance(callbacks, BaseAICallbackHandler):
                await callbacks.on_error(e)
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
        if self.current_step_active:
            # End text if it was started
            if self.has_text_started:
                yield self._create_text_end_event()
                self.has_text_started = False
            
            yield self._create_finish_step_event()
            self.current_step_active = False
    
    async def _handle_tool_start(self, data: Dict[str, Any]) -> AsyncGenerator[UIMessageChunk, None]:
        """Handle tool start event."""
        run_id = data.get("run_id", str(uuid.uuid4()))
        
        # Try different possible field names for tool name
        name = (
            data.get("name") or 
            data.get("tool_name") or 
            data.get("tool") or
            (data.get("serialized", {}).get("name")) or
            "unknown_tool"
        )
        
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
        
        # Emit tool input available
        yield self._create_tool_input_available_event(tool_call_id, name, inputs)
        
        # Add tool invocation to message parts
        tool_invocation = ToolInvocation(
            state="call",
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
                    break
    
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
                        
                        # Emit tool input available
                        yield self._create_tool_input_available_event(tool_call_id, tool_name, tool_input)
                        
                        # Emit tool output available
                        yield self._create_tool_output_available_event(tool_call_id, result)
                        
                        # Add tool invocation to message parts
                        # Ensure args is always a dictionary
                        args_dict = tool_input if isinstance(tool_input, dict) else {"input": tool_input}
                        
                        tool_invocation = ToolInvocation(
                            state="result",
                            toolCallId=tool_call_id,
                            toolName=tool_name,
                            args=args_dict,
                            result=result
                        )
                        self.message_parts.append(ToolInvocationUIPart(toolInvocation=tool_invocation))
    
    async def _handle_text_content(self, text: str) -> AsyncGenerator[UIMessageChunk, None]:
        """Handle text content and emit appropriate events."""
        if not text:
            return
            
        # Start text if not already started
        if not self.has_text_started:
            yield self._create_text_start_event()
            self.has_text_started = True
            
            # Add text part to message
            self.message_parts.append(TextUIPart(text=""))
        
        # Emit text delta
        yield self._create_text_delta_event(text)
        
        # Update aggregated text
        self.aggregated_text += text
        
        # Update text part in message
        for part in self.message_parts:
            if isinstance(part, TextUIPart):
                part.text = self.aggregated_text
                break
    
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
        self.message_parts.append(StepStartUIPart())
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
        # Create final message
        message = Message(
            id=self.message_id,
            content=self.aggregated_text,
            role="assistant",
            parts=self.message_parts
        )
        
        # Create options with usage info (if available)
        options = {}
        
        # Call on_finish callback
        await callback_handler.on_finish(message, options)


async def to_data_stream(
    stream: AsyncIterable[LangChainStreamInput],
    callbacks: Optional[Union[StreamCallbacks, BaseAICallbackHandler]] = None,
    message_id: Optional[str] = None,
) -> AsyncGenerator[UIMessageChunk, None]:
    """Convert LangChain output streams to AI SDK Data Stream Protocol.
    
    This enhanced adapter supports the complete AI SDK UI Stream Protocol including:
    - Tool invocation events (tool-input-start, tool-input-available, tool-output-available)
    - Step control events (start-step, finish-step)
    - Stream control events (start, finish)
    - AI SDK compatible callbacks
    
    Args:
        stream: Input stream from LangChain
        callbacks: Optional callbacks (StreamCallbacks or BaseAICallbackHandler)
        message_id: Optional message ID (auto-generated if not provided)
        
    Yields:
        UI message chunks compatible with AI SDK Data Stream Protocol
        
    Example:
        ```python
        from langchain_aisdk_adapter import to_data_stream, BaseAICallbackHandler
        
        class MyCallbackHandler(BaseAICallbackHandler):
            async def on_finish(self, message, options):
                print(f"Final message: {message.content}")
                print(f"Tool calls: {len([p for p in message.parts if p.type == 'tool-invocation'])}")
        
        # Convert LangChain stream to AI SDK format with callbacks
        async for chunk in to_data_stream(langchain_stream, callbacks=MyCallbackHandler()):
            print(chunk)
        ```
    """
    processor = EnhancedStreamProcessor(message_id)
    
    # Handle legacy StreamCallbacks by using the original adapter
    if isinstance(callbacks, StreamCallbacks):
        from .adapter import to_ui_message_stream
        async for chunk in to_ui_message_stream(stream, callbacks):
            yield chunk
    else:
        # Use enhanced processing with AI SDK callbacks
        async for chunk in processor.process_stream(stream, callbacks):
            yield chunk