"""LangChain to AI SDK adapter implementation."""

import asyncio
import json
import uuid
from typing import AsyncGenerator, AsyncIterable, Optional, Union, Dict, Any, List, Callable
from datetime import datetime
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.documents import Document

from .callbacks import (
    CallbacksTransformer, 
    StreamCallbacks, 
    BaseAICallbackHandler,
    Message,
    TextUIPart,
    ToolInvocationUIPart,
    ToolInvocation,
    StepStartUIPart,
    LanguageModelUsage
)
from .models import (
    LangChainAIMessageChunk,
    LangChainMessageContentComplex,
    LangChainStreamEvent,
    LangChainStreamInput,
    UIMessageChunk,
)
from .callbacks import (
    UIPart as UIMessagePart,
    TextUIPart,
    ToolInvocationUIPart,
    StepStartUIPart,
    ToolInvocation,
    BaseAICallbackHandler,
    Message,
    LanguageModelUsage
)


def _extract_text_from_ai_message_chunk(chunk: LangChainAIMessageChunk) -> str:
    """Extract text content from LangChain AI message chunk.
    
    Args:
        chunk: LangChain AI message chunk
        
    Returns:
        Extracted text content
    """
    # Handle both dict and object formats
    if isinstance(chunk, dict):
        content = chunk.get("content", "")
    else:
        # Handle AIMessageChunk object
        content = getattr(chunk, "content", "")
    
    if isinstance(content, str):
        return content
    
    # Handle complex content (list of content items)
    text_parts = []
    if hasattr(content, '__iter__') and not isinstance(content, str):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif hasattr(item, 'type') and getattr(item, 'type', None) == "text":
                text_parts.append(getattr(item, 'text', ""))
    
    return "".join(text_parts)


class StreamProcessor:
    """Enhanced stream processor for AI SDK compatibility."""
    
    def __init__(self, message_id: Optional[str] = None, callbacks: Optional[List[BaseAICallbackHandler]] = None):
        """Initialize the StreamProcessor.
        
        Args:
            message_id: Optional message ID for the stream
            callbacks: Optional list of AI SDK callbacks
        """
        self.message_id = message_id or str(uuid.uuid4())
        self.callbacks = callbacks or []
        self.parts: List[UIMessagePart] = []
        self.pending_tool_inputs: Dict[str, Any] = {}
        self.tool_calls: Dict[str, Dict[str, Any]] = {}
        self.current_step = 0
        self.text_content = ""
        self.usage_stats: Optional[LanguageModelUsage] = None
        self.step_has_tools = False
        self.step_text_content = ""
        self.text_buffer = ""  # Buffer for ReAct pattern matching
        
    async def process_stream(
        self, 
        stream: AsyncIterable[LangChainStreamInput]
    ) -> AsyncGenerator[UIMessageChunk, None]:
        """Process LangChain stream and convert to AI SDK format.
        
        Each step corresponds to one LLM submission. Multiple tool calls can be in the same step.
        """
        # Initialize state
        in_text_phase = False
        text_id = None
        step_started = False
        has_any_content = False
        
        # Start the stream
        yield {"type": "start", "messageId": self.message_id}
        
        # Process stream values
        async for value in stream:
            async for chunk in self._process_stream_value(value):
                chunk_type = chunk.get("type")
                
                # Start step when we have the first content
                if not step_started and chunk_type in ["text-delta", "tool-input-start"]:
                    yield {"type": "start-step"}
                    step_started = True
                
                if chunk_type == "text-delta":
                    if not in_text_phase:
                        # Start text phase
                        text_id = f"text-{uuid.uuid4()}"
                        yield {"type": "text-start", "id": text_id}
                        in_text_phase = True
                    
                    # Add text ID to chunk and yield
                    chunk["id"] = text_id
                    self.text_content += chunk["delta"]
                    has_any_content = True
                    yield chunk
                    
                elif chunk_type == "tool-input-start":
                    # End text phase if active before tool starts
                    if in_text_phase and text_id:
                        yield {"type": "text-end", "id": text_id}
                        in_text_phase = False
                        text_id = None
                    
                    self.step_has_tools = True
                    has_any_content = True
                    yield chunk
                    
                elif chunk_type == "tool-output-available":
                    yield chunk
                    
                else:
                    yield chunk
        
        # End final text phase if active
        if in_text_phase and text_id:
            yield {"type": "text-end", "id": text_id}
        
        # Finish step if it was started
        if step_started:
            yield {"type": "finish-step"}
        
        # End the stream
        yield {"type": "finish"}
    

    
    def _extract_tool_inputs_from_event(self, event: LangChainStreamEvent) -> None:
        """Extract tool inputs from intermediate_steps and create ToolInvocation objects directly."""
        data = event.get("data", {})
        
        # Look for intermediate_steps in various locations
        intermediate_steps = None
        if "input" in data and isinstance(data["input"], dict):
            intermediate_steps = data["input"].get("intermediate_steps")
        elif "chunk" in data and isinstance(data["chunk"], dict):
            intermediate_steps = data["chunk"].get("intermediate_steps")
        
        if intermediate_steps:
            for i, step in enumerate(intermediate_steps):
                if isinstance(step, (list, tuple)) and len(step) >= 2:
                    action, result = step[0], step[1]
                    if hasattr(action, 'tool') and hasattr(action, 'tool_input'):
                        tool_name = action.tool
                        tool_input = action.tool_input
                        
                        # Parse tool_input if it's a JSON string
                        if isinstance(tool_input, str):
                            try:
                                import json
                                tool_input = json.loads(tool_input)
                            except (json.JSONDecodeError, ValueError):
                                # If not JSON, treat as single parameter
                                tool_input = {"input": tool_input}
                        
                        # Create ToolInvocation directly here instead of waiting for tool events
                        tool_call_id = f"call_{i}_{str(uuid.uuid4())[:8]}"
                        tool_invocation = ToolInvocation(
                            state="result",
                            step=i,
                            toolCallId=tool_call_id,
                            toolName=tool_name,
                            args=tool_input if isinstance(tool_input, dict) else {"input": tool_input},
                            result=result
                        )
                        
                        # Add to parts if not already added
                        tool_already_added = any(
                            isinstance(part, ToolInvocationUIPart) and 
                            part.toolInvocation.toolName == tool_name and
                            part.toolInvocation.step == i
                            for part in self.parts
                        )
                        
                        if not tool_already_added:
                            self.parts.append(ToolInvocationUIPart(toolInvocation=tool_invocation))
    
    async def _process_stream_value(
        self, 
        value: LangChainStreamInput
    ) -> AsyncGenerator[UIMessageChunk, None]:
        """Process individual stream value and generate appropriate events."""
        # Handle string stream (e.g., from StringOutputParser)
        if isinstance(value, str):
            if value:
                yield {"type": "text-delta", "delta": value}
            return
        
        # Handle LangChain stream events v2
        if isinstance(value, dict) and "event" in value:
            event: LangChainStreamEvent = value
            
            # Extract tool inputs from intermediate_steps
            self._extract_tool_inputs_from_event(event)
            
            # Handle text streaming
            if event["event"] == "on_chat_model_stream":
                chunk_data = event.get("data", {})
                chunk = chunk_data.get("chunk")
                if chunk:
                    text = _extract_text_from_ai_message_chunk(chunk)
                    if text:
                        # Check for ReAct agent tool calls in text
                        async for tool_chunk in self._check_for_react_tool_calls(text):
                            yield tool_chunk
                        
                        yield {"type": "text-delta", "delta": text}
            
            # Handle tool calls
            elif event["event"] == "on_tool_start":
                async for tool_chunk in self._handle_tool_start(event):
                    yield tool_chunk
            
            elif event["event"] == "on_tool_end":
                async for tool_chunk in self._handle_tool_end(event):
                    yield tool_chunk
                # Don't emit step-complete here, wait for next text generation
            
            return
        
        # Handle AI message chunk stream
        if isinstance(value, dict) and "content" in value:
            chunk: LangChainAIMessageChunk = value
            text = _extract_text_from_ai_message_chunk(chunk)
            if text:
                yield {"type": "text-delta", "delta": text}
    
    async def _handle_tool_start(self, event: LangChainStreamEvent) -> AsyncGenerator[UIMessageChunk, None]:
        """Handle tool start event according to AI SDK protocol."""
        data = event.get("data", {})
        tool_name = event.get("name", "unknown")
        run_id = event.get("run_id", str(uuid.uuid4()))
        
        # Generate tool call ID
        tool_call_id = f"call_{self.current_step}_{run_id[:8]}"
        
        # Get tool inputs
        inputs = data.get("input", {})
        if isinstance(inputs, str):
            try:
                inputs = json.loads(inputs)
            except (json.JSONDecodeError, ValueError):
                inputs = {"input": inputs}
        
        # Emit tool-input-start
        yield {
            "type": "tool-input-start",
            "toolCallId": tool_call_id,
            "toolName": tool_name
        }
        
        # Emit tool-input-delta (simulate streaming input)
        input_json = json.dumps(inputs, ensure_ascii=False)
        for i in range(0, len(input_json), 10):  # Stream in chunks of 10 chars
            chunk = input_json[i:i+10]
            yield {
                "type": "tool-input-delta",
                "toolCallId": tool_call_id,
                "inputTextDelta": chunk
            }
        
        # Emit tool-input-available
        yield {
            "type": "tool-input-available",
            "toolCallId": tool_call_id,
            "toolName": tool_name,
            "input": inputs
        }
        
        # Store tool call info for later use
        self.tool_calls[tool_call_id] = {
            "toolName": tool_name,
            "args": inputs,
            "step": self.current_step
        }
    
    async def _check_for_react_tool_calls(self, text: str) -> AsyncGenerator[UIMessageChunk, None]:
        """Check for ReAct agent tool calls in text and generate appropriate events."""
        # Accumulate text for pattern matching
        self.text_buffer += text
        
        # Pattern for ReAct tool calls: "Action: tool_name\nAction Input: input"
        import re
        action_pattern = r'Action:\s*([^\n]+)\s*\nAction Input:\s*([^\n]+?)(?=\n(?:Observation:|Thought:|Action:|Final Answer:)|$)'
        
        matches = re.findall(action_pattern, self.text_buffer)
        
        for tool_name, tool_input in matches:
            tool_name = tool_name.strip()
            tool_input = tool_input.strip()
            
            # Generate tool call ID
            tool_call_id = f"call_{self.current_step}_{len(self.tool_calls)}"
            
            # Check if we already processed this tool call
            tool_signature = f"{tool_name}:{tool_input}"
            if hasattr(self, 'processed_tool_calls') and tool_signature in self.processed_tool_calls:
                continue
            
            if not hasattr(self, 'processed_tool_calls'):
                self.processed_tool_calls = set()
            self.processed_tool_calls.add(tool_signature)
            
            # Parse tool input
            try:
                # Try to parse as JSON if it looks like JSON
                if tool_input.startswith('{') and tool_input.endswith('}'):
                    inputs = json.loads(tool_input)
                else:
                    # Treat as simple string input
                    inputs = {"input": tool_input}
            except:
                inputs = {"input": tool_input}
            
            # Emit tool-input-start
            yield {
                "type": "tool-input-start",
                "toolCallId": tool_call_id,
                "toolName": tool_name
            }
            
            # Emit tool-input-delta (simulate streaming input)
            input_json = json.dumps(inputs, ensure_ascii=False)
            for i in range(0, len(input_json), 10):  # Stream in chunks of 10 chars
                chunk = input_json[i:i+10]
                yield {
                    "type": "tool-input-delta",
                    "toolCallId": tool_call_id,
                    "inputTextDelta": chunk
                }
            
            # Emit tool-input-available
            yield {
                "type": "tool-input-available",
                "toolCallId": tool_call_id,
                "toolName": tool_name,
                "input": inputs
            }
            
            # Store tool call info
            self.tool_calls[tool_call_id] = {
                "toolName": tool_name,
                "args": inputs,
                "step": self.current_step
            }
        
        # Check for tool output (Observation: or direct output after Action Input)
        observation_pattern = r'Observation:\s*([^\n]+(?:\n(?!Thought:|Action:|Final Answer:)[^\n]*)*)'  
        obs_matches = re.findall(observation_pattern, self.text_buffer, re.MULTILINE)
        
        # If no Observation found, check for immediate tool output after Action Input
        if not obs_matches and self.tool_calls:
            # Look for pattern: Action Input: [input] [immediate tool output]
            # Handle both quoted and unquoted inputs
            immediate_output_pattern = r"Action Input:\s*[^\n]*?\s*([^\n]+?)(?=\s*(?:I now know|Thought:|Action:|Final Answer:|$))"
            immediate_matches = re.findall(immediate_output_pattern, self.text_buffer, re.MULTILINE)
            if immediate_matches:
                obs_matches = [match.strip() for match in immediate_matches if match.strip()]
            
            # If still no matches, check Final Answer as fallback
            if not obs_matches:
                final_answer_pattern = r'Final Answer:\s*(.+?)(?=\n|$)'
                final_matches = re.findall(final_answer_pattern, self.text_buffer, re.MULTILINE | re.DOTALL)
                if final_matches:
                    obs_matches = [match.strip() for match in final_matches if match.strip()]
        
        # Match observations with recent tool calls
        if obs_matches and self.tool_calls:
            # Get the most recent tool call
            recent_tool_calls = sorted(self.tool_calls.items(), key=lambda x: x[0], reverse=True)
            
            for i, observation in enumerate(obs_matches):
                if i < len(recent_tool_calls):
                    tool_call_id, call_info = recent_tool_calls[i]
                    
                    # Check if we already processed this observation
                    if not call_info.get("result_processed"):
                        observation = observation.strip()
                        
                        # Emit tool-output-available
                        yield {
                            "type": "tool-output-available",
                            "toolCallId": tool_call_id,
                            "output": observation
                        }
                        
                        # Create ToolInvocation for parts array
                        tool_invocation = ToolInvocation(
                            state="result",
                            step=call_info["step"],
                            toolCallId=tool_call_id,
                            toolName=call_info["toolName"],
                            args=call_info["args"],
                            result=observation
                        )
                        self.parts.append(ToolInvocationUIPart(toolInvocation=tool_invocation))
                        
                        # Mark as processed
                        self.tool_calls[tool_call_id]["result_processed"] = True
    
    async def _handle_tool_end(self, event: LangChainStreamEvent) -> AsyncGenerator[UIMessageChunk, None]:
        """Handle tool end event according to AI SDK protocol."""
        data = event.get("data", {})
        run_id = event.get("run_id", str(uuid.uuid4()))
        
        # Find matching tool call
        tool_call_id = None
        for call_id, call_info in self.tool_calls.items():
            if call_id.endswith(run_id[:8]):
                tool_call_id = call_id
                break
        
        if not tool_call_id:
            # Fallback: create tool call ID
            tool_call_id = f"call_{self.current_step}_{run_id[:8]}"
        
        # Get tool output
        output = data.get("output", "")
        
        # Emit tool-output-available
        yield {
            "type": "tool-output-available",
            "toolCallId": tool_call_id,
            "output": output
        }
        
        # Create ToolInvocation for parts array
        if tool_call_id in self.tool_calls:
            call_info = self.tool_calls[tool_call_id]
            tool_invocation = ToolInvocation(
                state="result",
                step=call_info["step"],
                toolCallId=tool_call_id,
                toolName=call_info["toolName"],
                args=call_info["args"],
                result=output
            )
            self.parts.append(ToolInvocationUIPart(toolInvocation=tool_invocation))


async def _process_langchain_stream(
    stream: AsyncIterable[LangChainStreamInput],
) -> AsyncGenerator[str, None]:
    """Legacy function for backward compatibility - extract text content only.
    
    Args:
        stream: Input stream of LangChain events/chunks/strings
        
    Yields:
        Text content extracted from the stream
    """
    async for value in stream:
        # Handle string stream (e.g., from StringOutputParser)
        if isinstance(value, str):
            yield value
            continue
        
        # Handle LangChain stream events v2
        if isinstance(value, dict) and "event" in value:
            event: LangChainStreamEvent = value
            if event["event"] == "on_chat_model_stream":
                chunk_data = event.get("data", {})
                chunk = chunk_data.get("chunk")
                if chunk:
                    text = _extract_text_from_ai_message_chunk(chunk)
                    if text:
                        yield text
            continue
        
        # Handle AI message chunk stream
        if isinstance(value, dict) and "content" in value:
            chunk: LangChainAIMessageChunk = value
            text = _extract_text_from_ai_message_chunk(chunk)
            if text:
                yield text


async def _apply_callbacks(
    stream: AsyncIterable[str],
    callbacks: Optional[StreamCallbacks] = None,
) -> AsyncGenerator[str, None]:
    """Apply callbacks to text stream.
    
    Args:
        stream: Input text stream
        callbacks: Optional callbacks configuration
        
    Yields:
        Text chunks with callbacks applied
    """
    transformer = CallbacksTransformer(callbacks)
    
    try:
        async for chunk in stream:
            processed_chunk = await transformer.transform(chunk)
            yield processed_chunk
    finally:
        await transformer.finish()


async def _convert_to_ui_message_chunks(
    stream: AsyncIterable[str],
) -> AsyncGenerator[UIMessageChunk, None]:
    """Convert text stream to UI message chunks.
    
    Args:
        stream: Input text stream
        
    Yields:
        UI message chunks in AI SDK format
    """
    # Generate a unique text ID
    text_id = str(uuid.uuid4())
    
    # Emit text-start chunk
    yield {"type": "text-start", "id": text_id}
    
    # Emit text-delta chunks for each text piece
    async for chunk in stream:
        if chunk:  # Only emit non-empty chunks
            yield {"type": "text-delta", "id": text_id, "delta": chunk}
    
    # Emit text-end chunk
    yield {"type": "text-end", "id": text_id}


# Main API functions following AI SDK LangChainAdapter pattern

async def to_data_stream(
    stream: AsyncIterable[LangChainStreamInput],
    callbacks: Optional[Union[StreamCallbacks, BaseAICallbackHandler]] = None,
) -> AsyncGenerator[UIMessageChunk, None]:
    """Convert LangChain output streams to AI SDK data stream.
    
    Supports:
    - LangChain StringOutputParser streams
    - LangChain AIMessageChunk streams  
    - LangChain StreamEvents v2 streams
    
    Args:
        stream: Input stream from LangChain
        callbacks: Optional callbacks for stream lifecycle events
        
    Yields:
        UI message chunks compatible with AI SDK
        
    Example:
        ```python
        from langchain_aisdk_adapter import to_data_stream
        
        # Convert LangChain stream to AI SDK format
        async for chunk in to_data_stream(langchain_stream):
            print(chunk)
        ```
    """
    # Check if using new AI SDK compatible callback system
    if isinstance(callbacks, BaseAICallbackHandler):
        # Use enhanced stream processor with full AI SDK support
        processor = StreamProcessor(callbacks=[callbacks])
        
        # Process the stream and collect final message
        chunks = []
        async for chunk in processor.process_stream(stream):
            chunks.append(chunk)
            yield chunk
        
        # Build final message and call callbacks
        if processor.text_content or processor.parts:
            # Add text part if we have text content
            if processor.text_content:
                processor.parts.insert(0, TextUIPart(text=processor.text_content))
            
            # Create final message
            final_message = Message(
                id=processor.message_id,
                content=processor.text_content,
                role="assistant",
                parts=processor.parts
            )
            
            # Call on_finish callback
            await callbacks.on_finish(final_message, {})
    else:
        # Use legacy processing for backward compatibility
        # Step 1: Process LangChain stream to extract text
        text_stream = _process_langchain_stream(stream)
        
        # Step 2: Apply legacy callbacks if provided
        if callbacks:
            text_stream = _apply_callbacks(text_stream, callbacks)
        
        # Step 3: Convert to UI message chunks
        async for chunk in _convert_to_ui_message_chunks(text_stream):
            yield chunk


async def to_data_stream_response(
    stream: AsyncIterable[LangChainStreamInput],
    callbacks: Optional[Union[StreamCallbacks, BaseAICallbackHandler]] = None,
    init: Optional[dict] = None,
) -> dict:
    """Convert LangChain output streams to data stream response.
    
    Args:
        stream: Input stream from LangChain
        callbacks: Optional callbacks for stream lifecycle events
        init: Optional response initialization parameters
        
    Returns:
        Response object with stream data
        
    Example:
        ```python
        from langchain_aisdk_adapter import to_data_stream_response
        
        response = await to_data_stream_response(langchain_stream)
        ```
    """
    # Convert stream to list for response
    chunks = []
    async for chunk in to_data_stream(stream, callbacks):
        chunks.append(chunk)
    
    response = {
        "data": chunks,
        "status": "success"
    }
    
    if init:
        response.update(init)
    
    return response


async def merge_into_data_stream(
    stream: AsyncIterable[LangChainStreamInput],
    data_stream: AsyncGenerator[UIMessageChunk, None],
    callbacks: Optional[Union[StreamCallbacks, BaseAICallbackHandler]] = None,
) -> AsyncGenerator[UIMessageChunk, None]:
    """Merge LangChain output streams into an existing data stream.
    
    Args:
        stream: Input stream from LangChain
        dataStream: Existing data stream to merge into
        callbacks: Optional callbacks for stream lifecycle events
        
    Yields:
        Merged UI message chunks
        
    Example:
        ```python
        from langchain_aisdk_adapter import merge_into_data_stream
        
        async for chunk in merge_into_data_stream(langchain_stream, existing_stream):
            print(chunk)
        ```
    """
    # First yield from existing data stream
    async for chunk in data_stream:
        yield chunk
    
    # Then merge LangChain stream
    async for chunk in to_data_stream(stream, callbacks):
        yield chunk


# Legacy function names for backward compatibility - REMOVED
# Use to_data_stream, to_data_stream_response, merge_into_data_stream instead