"""Stream processor for converting LangChain events to AI SDK format."""

import asyncio
import uuid
from typing import AsyncGenerator, AsyncIterable, Dict, Any, Optional, Callable

from .models import (
    LangChainStreamInput,
    LangChainStreamEvent,
    LangChainAIMessageChunk,
    UIMessageChunk,
    UIMessageChunkStart,
    UIMessageChunkFinish,
    UIMessageChunkStartStep,
    UIMessageChunkFinishStep,
    UIMessageChunkToolInputStart,
    UIMessageChunkToolOutputAvailable,
    UIMessageChunkError
)
from .message_builder import MessageBuilder
from .protocol_generator import ProtocolGenerator
from .callbacks import BaseAICallbackHandler


class StreamProcessor:
    """Core stream processing logic for converting LangChain events to AI SDK format."""

    def __init__(
        self,
        message_id: str,
        auto_events: bool = True,
        callbacks: Optional[BaseAICallbackHandler] = None,
        protocol_version: str = "v4"
    ):
        self.message_id = message_id
        self.auto_events = auto_events
        self.callbacks = callbacks
        self.protocol_version = protocol_version
        self.message_builder = MessageBuilder(message_id)
        self._lock = asyncio.Lock()
        self._current_text_id: Optional[str] = None
        self._tool_calls: Dict[str, Dict[str, Any]] = {}
        self._accumulated_text = ""
        self.current_step_id: Optional[str] = None
        self.current_usage: Dict[str, int] = {"promptTokens": 0, "completionTokens": 0}
        
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
        self.current_usage: Dict[str, int] = {"promptTokens": 0, "completionTokens": 0}
        
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
        
        try:
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
        except Exception as e:
            # Re-raise the exception to be handled by the caller
            raise e
    
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
            "type": "start-step",
            "messageId": self.current_step_id or str(uuid.uuid4())
        }
    
    def _create_finish_step_chunk(self) -> UIMessageChunkFinishStep:
        """Create finish step chunk."""
        # Determine finish reason based on whether tools were used
        finish_reason = "tool-calls" if self.tool_completed_in_current_step else "stop"
        
        # Use actual usage values from LangChain events
        usage = self.current_usage.copy()
        
        return {
            "type": "finish-step",
            "finishReason": finish_reason,
            "usage": usage,
            "isContinued": False
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
            "type": "finish",
            "finishReason": "stop",
            "usage": self.current_usage.copy()
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
            async for chunk in self._handle_tool_start(event):
                yield chunk
        elif event_type == "on_tool_end":
            async for chunk in self._handle_tool_end(event):
                yield chunk
    
    async def _handle_chat_model_start(self, data: Dict[str, Any]) -> AsyncGenerator[UIMessageChunk, None]:
        """Handle chat model start event."""
        if not self.current_step_active:
            self.step_count += 1
            # Generate a unique step ID for this step
            self.current_step_id = str(uuid.uuid4())
            self.text_id = f"text-{uuid.uuid4()}"
            self.has_text_started = False
            self._accumulated_text = ""
            yield {
                "type": "start-step",
                "messageId": self.current_step_id
            }
            self.current_step_active = True
    
    async def _handle_chat_model_end(self, data: Dict[str, Any]) -> AsyncGenerator[UIMessageChunk, None]:
        """Handle chat model end event."""
        # Extract usage information from LangChain event data
        # First check if usage_metadata is directly in data
        if "usage_metadata" in data:
            usage_metadata = data["usage_metadata"]
            self.current_usage = {
                "promptTokens": usage_metadata.get("input_tokens", 0),
                "completionTokens": usage_metadata.get("output_tokens", 0)
            }
        else:
            # Check in output field
            output = data.get("output", {})
            if hasattr(output, "usage_metadata"):
                usage_metadata = output.usage_metadata
                # usage_metadata is a dict, not an object
                if isinstance(usage_metadata, dict):
                    self.current_usage = {
                        "promptTokens": usage_metadata.get("input_tokens", 0),
                        "completionTokens": usage_metadata.get("output_tokens", 0)
                    }
                else:
                    self.current_usage = {
                        "promptTokens": getattr(usage_metadata, "input_tokens", 0),
                        "completionTokens": getattr(usage_metadata, "output_tokens", 0)
                    }
            elif isinstance(output, dict) and "usage_metadata" in output:
                usage_metadata = output["usage_metadata"]
                self.current_usage = {
                    "promptTokens": usage_metadata.get("input_tokens", 0),
                    "completionTokens": usage_metadata.get("output_tokens", 0)
                }
        
        # End text if it was started
        if self.has_text_started:
            yield ProtocolGenerator.create_text_end(self.text_id, self._accumulated_text, self.protocol_version)
            self.has_text_started = False
        
        # Mark that LLM generation is complete
        # We don't finish the step here because we need to wait for all tool calls to complete
        # The step will be finished either:
        # 1. In _process_intermediate_steps after all tools are processed
        # 2. In the final cleanup if no tools were called
        self.llm_generation_complete = True
        # after all tool outputs are available
    
    async def _handle_tool_start(self, event: Dict[str, Any]) -> AsyncGenerator[UIMessageChunk, None]:
        """Handle tool start events from LangChain.
        
        Only process LangGraph tool events which have complete information.
        Traditional LangChain tool events are handled via intermediate_steps.
        """
        # Check if this is a LangGraph tool event by looking for langgraph metadata
        metadata = event.get("metadata", {})
        is_langgraph = any(key.startswith("langgraph") for key in metadata.keys())
        
        if not is_langgraph:
            # Skip traditional LangChain tool events - they will be handled via intermediate_steps
            return
            yield  # Make this a generator function
        
        # Process LangGraph tool events with complete information
        data = event.get("data", {})
        tool_name = event.get("name", "")
        run_id = event.get("run_id", "")
        
        # LangGraph tool events have input data in data.input
        tool_input = data.get("input")
        
        if tool_input is not None and tool_name and run_id:
            # This is a LangGraph tool event with complete information
            tool_call_id = f"tool_{run_id}"
            
            # Only process if we haven't seen this tool call before
            if tool_call_id not in self.tool_calls:
                # Store tool call info (without output yet)
                self.tool_calls[tool_call_id] = {
                    "name": tool_name,
                    "inputs": tool_input,
                    "outputs": None,  # Will be filled in _handle_tool_end
                    "state": "running"
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
        
        return
        yield  # Make this a generator function
    
    def _serialize_tool_output(self, output: Any) -> Any:
        """Serialize tool output to ensure JSON compatibility."""
        if hasattr(output, 'content'):
            # Handle LangChain ToolMessage or similar objects
            return output.content
        elif hasattr(output, 'dict'):
            # Handle Pydantic models
            return output.dict()
        elif hasattr(output, '__dict__'):
            # Handle other objects with __dict__
            return {k: v for k, v in output.__dict__.items() if not k.startswith('_')}
        elif isinstance(output, (list, tuple)):
            # Handle lists/tuples recursively
            return [self._serialize_tool_output(item) for item in output]
        elif isinstance(output, dict):
            # Handle dictionaries recursively
            return {k: self._serialize_tool_output(v) for k, v in output.items()}
        else:
            # Return as-is for basic types (str, int, float, bool, None)
            return output
    
    async def _handle_tool_end(self, event: Dict[str, Any]) -> AsyncGenerator[UIMessageChunk, None]:
        """Handle tool end events from LangChain.
        
        Only process LangGraph tool events which have complete information.
        Traditional LangChain tool events are handled via intermediate_steps.
        """
        # Check if this is a LangGraph tool event by looking for langgraph metadata
        metadata = event.get("metadata", {})
        is_langgraph = any(key.startswith("langgraph") for key in metadata.keys())
        
        if not is_langgraph:
            # Skip traditional LangChain tool events - they will be handled via intermediate_steps
            return
            yield  # Make this a generator function
        
        # Process LangGraph tool events with complete information
        data = event.get("data", {})
        tool_name = event.get("name", "")
        run_id = event.get("run_id", "")
        
        # LangGraph tool events have output data in data.output
        tool_output = data.get("output")
        
        if tool_output is not None and tool_name and run_id:
            # This is a LangGraph tool event with complete information
            tool_call_id = f"tool_{run_id}"
            
            # Update tool call info with output
            if tool_call_id in self.tool_calls:
                # Serialize the output to ensure JSON compatibility
                serialized_output = self._serialize_tool_output(tool_output)
                
                self.tool_calls[tool_call_id]["outputs"] = serialized_output
                self.tool_calls[tool_call_id]["state"] = "result"
                
                # Emit tool output available
                yield {
                    "type": "tool-output-available",
                    "toolCallId": tool_call_id,
                    "output": serialized_output
                }
                
                # Mark that we have completed a tool call in this step
                self.tool_completed_in_current_step = True
        
        return
        yield  # Make this a generator function
    
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
        """Process intermediate steps to extract tool calls.
        
        Supports both traditional ReAct agent format (action, result) tuples
        and LangGraph format (messages list).
        """
        if not intermediate_steps:
            return
        
        # Handle LangGraph format: intermediate_steps is a list of messages
        if isinstance(intermediate_steps, list) and intermediate_steps:
            # Check if this looks like LangGraph messages format
            first_item = intermediate_steps[0]
            if hasattr(first_item, 'content') or hasattr(first_item, 'tool_calls'):
                 async for chunk in self._process_langgraph_messages(intermediate_steps):
                     yield chunk
                 return
            
        # Handle traditional ReAct agent format: list of (action, result) tuples
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
                        
                        # Emit complete tool input sequence for v5 compatibility
                        # v4 protocol will filter out unwanted events in protocol_strategy.py
                        
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
        
        # After processing all tool calls, handle step completion
        if self.tool_completed_in_current_step and self.current_step_active and self.llm_generation_complete:
            yield {
                "type": "finish-step",
                "finishReason": "tool-calls",
                "usage": self.current_usage.copy(),
                "isContinued": False
            }
            self.current_step_active = False
            self.tool_completed_in_current_step = False
            self.need_new_step_for_text = True
            self.llm_generation_complete = False
        
        # After processing all intermediate steps (all tools in this LLM cycle),
        # finish the current step if we have tool calls and LLM generation is complete
        if self.tool_completed_in_current_step and self.current_step_active and self.llm_generation_complete:
            yield {
                "type": "finish-step",
                "finishReason": "tool-calls",
                "usage": self.current_usage.copy(),
                "isContinued": False
            }
            self.current_step_active = False
            self.tool_completed_in_current_step = False
            self.need_new_step_for_text = True
            self.llm_generation_complete = False
    
    async def _process_langgraph_messages(self, messages) -> AsyncGenerator[UIMessageChunk, None]:
        """Process LangGraph messages format to extract tool calls.
        
        LangGraph intermediate_steps contains a list of messages including:
        - AIMessage with tool_calls
        - ToolMessage with results
        """
        tool_calls_map = {}  # Map tool_call_id to tool info
        tool_results_map = {}  # Map tool_call_id to results
        
        # First pass: collect tool calls and results
        for message in messages:
            # Handle AIMessage with tool_calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_call_id = tool_call.get('id', f"tool_{abs(hash(str(tool_call)))}")
                    tool_calls_map[tool_call_id] = {
                        'name': tool_call.get('name'),
                        'args': tool_call.get('args', {}),
                        'id': tool_call_id
                    }
            
            # Handle ToolMessage with results
            if hasattr(message, 'tool_call_id') and hasattr(message, 'content'):
                tool_call_id = message.tool_call_id
                tool_results_map[tool_call_id] = message.content
        
        # Second pass: emit tool events for matched pairs
        for tool_call_id, tool_info in tool_calls_map.items():
            if tool_call_id in tool_results_map and tool_call_id not in self.tool_calls:
                tool_name = tool_info['name']
                tool_args = tool_info['args']
                tool_result = tool_results_map[tool_call_id]
                
                # Store tool call info
                self.tool_calls[tool_call_id] = {
                    "name": tool_name,
                    "inputs": tool_args,
                    "outputs": tool_result,
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
                input_json = json.dumps(tool_args if isinstance(tool_args, dict) else {"input": tool_args})
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
                    "input": tool_args
                }
                
                # Emit tool output available
                yield {
                    "type": "tool-output-available",
                    "toolCallId": tool_call_id,
                    "output": tool_result
                }
                
                # Mark that we have completed a tool call in this step
                self.tool_completed_in_current_step = True
    
    async def _handle_incremental_text(self, text: str) -> AsyncGenerator[UIMessageChunk, None]:
        """Handle incremental text content using unified protocol generator."""
        if not text:
            return
        
        # Start text if not already started
        if not self.has_text_started:
            yield ProtocolGenerator.create_text_start(self.text_id, self.protocol_version)
            self.has_text_started = True
        
        # Send text delta
        yield ProtocolGenerator.create_text_delta(self.text_id, text, self.protocol_version)
    
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
            "type": "finish",
            "finishReason": "stop",
            "usage": self.current_usage.copy()
        }
    
    def _create_finish_step_event(self) -> UIMessageChunkFinishStep:
        """Create finish step event."""
        # Determine finish reason based on whether tools were used
        finish_reason = "tool-calls" if self.tool_completed_in_current_step else "stop"
        
        # Use actual usage values from LangChain events
        usage = self.current_usage.copy()
        
        return {
            "type": "finish-step",
            "finishReason": finish_reason,
            "usage": usage,
            "isContinued": False
        }
    
    async def _handle_ai_sdk_callbacks(self, callback_handler: BaseAICallbackHandler) -> None:
        """Handle AI SDK compatible callbacks."""
        # Use MessageBuilder to get the final message with all parts (including manual ones)
        message = self.message_builder.build_message()
        
        # Filter out empty TextUIParts from message parts
        from .callbacks import TextUIPart
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