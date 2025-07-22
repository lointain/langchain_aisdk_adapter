#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility Functions for AI SDK Adapter

Helper functions for stream processing, event handling, and data conversion.
"""

from typing import Any, AsyncGenerator, Dict, List, Optional
from langchain_core.messages import AIMessageChunk
from langchain_core.documents import Document
from datetime import datetime, timezone
import uuid

from .models import AI_SDK_PROTOCOL_PREFIXES
from .factory import (
    create_text_part, create_reasoning_part, create_data_part, create_error_part,
    create_source_part, create_tool_call_streaming_start_part, create_tool_call_part,
    create_tool_result_part, create_start_step_part, create_finish_step_part,
    create_finish_message_part, create_message_annotation_part
)


# State management
class AIStreamState:
    """State management for AI SDK streaming"""
    
    def __init__(self, config=None, callback=None):
        self.text_sent = False
        self.reasoning_sent = False
        self.tool_calls = {}  # tool_call_id -> {name, args}
        self.tool_call_counter = 0
        self.config = config
        self.callback = callback
        
        # Message building state
        self.message_id = str(uuid.uuid4())
        self.message_content = ""
        self.message_parts = []
        self.current_step_id = None
        self.step_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.recent_text_content = ""  # Store recent text for tool argument parsing
        
        # Step tracking for proper e/f protocol timing
        self.active_steps = []  # Stack of active step IDs
        self.step_contexts = {}  # step_id -> context info
        
    def add_text_content(self, text: str):
        """Add text content to the message"""
        self.message_content += text
        # Keep recent text for tool argument parsing (last 1000 chars)
        self.recent_text_content = (self.recent_text_content + text)[-1000:]
        
    def add_part(self, part_type: str, **kwargs):
        """Add a UI part to the message"""
        from .callbacks import (
            TextUIPart, ReasoningUIPart, ToolInvocationUIPart, 
            SourceUIPart, FileUIPart, StepStartUIPart, ErrorUIPart
        )
        
        if part_type == 'text':
            # Check if there's already a TextUIPart to append to
            existing_text_part = None
            for part in reversed(self.message_parts):  # Check from the end for efficiency
                if hasattr(part, 'type') and part.type == 'text':
                    existing_text_part = part
                    break
                # Stop looking if we hit a non-text part
                elif hasattr(part, 'type') and part.type != 'text':
                    break
            
            if existing_text_part:
                # Append to existing TextUIPart
                existing_text_part.text += kwargs['text']
            else:
                # Create new TextUIPart
                part = TextUIPart(text=kwargs['text'])
                self.message_parts.append(part)
        elif part_type == 'reasoning':
            part = ReasoningUIPart(reasoning=kwargs['reasoning'])
            self.message_parts.append(part)
        elif part_type == 'tool-invocation':
            from .callbacks import ToolInvocation
            tool_invocation = ToolInvocation(
                state=kwargs['state'],
                step=kwargs.get('step'),
                toolCallId=kwargs['toolCallId'],
                toolName=kwargs['toolName'],
                args=kwargs['args'],
                result=kwargs.get('result')
            )
            part = ToolInvocationUIPart(toolInvocation=tool_invocation)
            self.message_parts.append(part)
        elif part_type == 'source':
            part = SourceUIPart(source=kwargs['source'])
            self.message_parts.append(part)
        elif part_type == 'file':
            part = FileUIPart(
                name=kwargs['name'],
                mimeType=kwargs['mimeType'],
                data=kwargs['data']
            )
            self.message_parts.append(part)
        elif part_type == 'step-start':
            part = StepStartUIPart()
            self.message_parts.append(part)
        elif part_type == 'error':
            part = ErrorUIPart(error=kwargs['error'])
            self.message_parts.append(part)
        
    def start_step(self, context_type: str = "unknown", context_data: dict = None) -> str:
        """Start a new step and return step ID"""
        step_id = f"step_{self.step_count}_{uuid.uuid4().hex[:8]}"
        self.active_steps.append(step_id)
        self.step_contexts[step_id] = {
            'type': context_type,
            'data': context_data or {},
            'started_at': datetime.now(timezone.utc)
        }
        self.current_step_id = step_id
        self.step_count += 1  # Increment after using current count
        return step_id
        
    def finish_step(self, step_id: str = None):
        """Finish current or specified step"""
        if step_id is None:
            step_id = self.current_step_id
        
        if step_id and step_id in self.active_steps:
            self.active_steps.remove(step_id)
            if step_id in self.step_contexts:
                del self.step_contexts[step_id]
        
        # Update current step to the most recent active step
        self.current_step_id = self.active_steps[-1] if self.active_steps else None
        
    def build_final_message(self):
        """Build the final Message object for callback"""
        from .callbacks import Message, LanguageModelUsage
        
        return Message(
            id=self.message_id,
            createdAt=datetime.now(timezone.utc),
            content=self.message_content,
            role='assistant',
            parts=self.message_parts.copy()
        )


# Stream chunk type detection functions
def _is_ai_sdk_protocol_string(chunk: Any) -> bool:
    """Check if chunk is an AI SDK protocol string"""
    return isinstance(chunk, str) and chunk.startswith(AI_SDK_PROTOCOL_PREFIXES)


def _is_langgraph_event(chunk: Any) -> bool:
    """Check if chunk is a LangGraph event"""
    return isinstance(chunk, dict) and 'event' in chunk


def _has_nested_ai_message_chunk(chunk: Dict[str, Any]) -> bool:
    """Check if LangGraph event contains nested AIMessageChunk"""
    data = chunk.get('data', {})
    return 'chunk' in data and isinstance(data['chunk'], AIMessageChunk)


def _is_agent_executor_output(chunk: Any) -> bool:
    """Check if chunk is AgentExecutor output"""
    return (
        isinstance(chunk, dict) and 
        "output" in chunk and 
        isinstance(chunk["output"], str)
    )


def _ensure_newline_ending(content: str) -> str:
    """Ensure content ends with newline character"""
    return content if content.endswith('\n') else content + '\n'


async def _handle_stream_end(state: AIStreamState) -> AsyncGenerator[str, None]:
    """Handle cleanup work when stream ends and send finish message"""
    if state.tool_calls:
        print(f"Warning: Stream ended with {len(state.tool_calls)} incomplete tool calls")
        state.tool_calls.clear()
    
    # Send finish message if any text was sent during the stream (and if enabled)
    if state.text_sent and state.config and state.config.is_protocol_enabled('d'):
        from .factory import create_finish_message_part
        yield create_finish_message_part(
            finish_reason="stop",
            prompt_tokens=state.total_prompt_tokens,
            completion_tokens=state.total_completion_tokens
        ).ai_sdk_part_content
    
    # Call finish callback if available
    if state.callback:
        try:
            final_message = state.build_final_message()
            usage = None
            if state.total_prompt_tokens > 0 or state.total_completion_tokens > 0:
                from .callbacks import LanguageModelUsage
                usage = LanguageModelUsage(
                    promptTokens=state.total_prompt_tokens,
                    completionTokens=state.total_completion_tokens,
                    totalTokens=state.total_prompt_tokens + state.total_completion_tokens
                )
            
            options = {'usage': usage} if usage else {}
            await state.callback.on_finish(final_message, options)
        except Exception as callback_error:
            print(f"Warning: Callback finish handler failed: {callback_error}")


# AI Message Chunk processing
async def _process_ai_message_chunk(
    chunk: AIMessageChunk, 
    state: AIStreamState
) -> AsyncGenerator[str, None]:
    """Process AIMessageChunk, generating text and tool call chunks
    
    Args:
        chunk: LangChain's AIMessageChunk object
        state: Current stream state
        
    Yields:
        AI SDK protocol compliant strings
    """
    # Process reasoning content (for models like DeepSeek R1) - only if enabled
    if state.config and state.config.is_protocol_enabled('g'):
        if hasattr(chunk, 'response_metadata') and chunk.response_metadata:
            # Check for reasoning_content (DeepSeek R1 format)
            reasoning = chunk.response_metadata.get('reasoning_content')
            if not reasoning:
                # Fallback to reasoning field
                reasoning = chunk.response_metadata.get('reasoning')
            if reasoning and not state.reasoning_sent:
                from .factory import create_reasoning_part
                yield create_reasoning_part(reasoning).ai_sdk_part_content
                state.reasoning_sent = True
        
        # Check for reasoning in additional_kwargs
        if hasattr(chunk, 'additional_kwargs') and chunk.additional_kwargs:
            # Check for reasoning_content (DeepSeek R1 format)
            reasoning = chunk.additional_kwargs.get('reasoning_content')
            if not reasoning:
                # Fallback to reasoning field
                reasoning = chunk.additional_kwargs.get('reasoning')
            if reasoning and not state.reasoning_sent:
                from .factory import create_reasoning_part
                yield create_reasoning_part(reasoning).ai_sdk_part_content
                state.reasoning_sent = True
                state.add_part('reasoning', reasoning=reasoning)
    
    # Process text content - treat all as regular text output (0: protocol)
    # Don't automatically detect reasoning patterns
    if isinstance(chunk.content, str) and chunk.content:
        content = chunk.content
        
        # All text content uses text protocol (0:)
        if not state.config or state.config.is_protocol_enabled('0'):
            yield create_text_part(content).ai_sdk_part_content
        state.text_sent = True
        state.add_part('text', text=content)
        
        # Always add to recent text content for tool argument parsing
        state.add_text_content(content)

    # Process tool calls
    async for tool_part in _process_tool_calls_from_chunk(chunk, state):
        yield tool_part

    # Update token usage if available
    if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
        if 'input_tokens' in chunk.usage_metadata:
            state.total_prompt_tokens = chunk.usage_metadata['input_tokens']
        if 'output_tokens' in chunk.usage_metadata:
            state.total_completion_tokens = chunk.usage_metadata['output_tokens']
    
    # When message is complete and contains tool calls, send complete tool calls
    if chunk.usage_metadata and chunk.usage_metadata.get('finish_reason') == 'tool_calls':
        async for tool_call_part in _emit_completed_tool_calls(state):
            yield tool_call_part
    
    # Also check if this is the final chunk with tool calls (alternative condition)
    elif hasattr(chunk, 'tool_calls') and chunk.tool_calls and state.tool_calls:
        async for tool_call_part in _emit_completed_tool_calls(state):
            yield tool_call_part


async def _process_tool_calls_from_chunk(
    chunk: AIMessageChunk, 
    state: AIStreamState
) -> AsyncGenerator[str, None]:
    """Process tool call information from AIMessageChunk
    
    Args:
        chunk: AIMessageChunk object
        state: Current stream state
        
    Yields:
        Tool call related AI SDK protocol strings
    """
    for tc_chunk in chunk.tool_call_chunks:
        tool_call_id = tc_chunk.get('id')
        tool_name = tc_chunk.get('name')
        
        # Skip if essential fields are missing
        if not tool_call_id or not tool_name:
            continue
            
        if tool_call_id not in state.tool_calls:
            state.tool_calls[tool_call_id] = {'name': tool_name, 'args': ""}
            # Send tool call streaming start part when first encountered (if enabled)
            if not state.config or state.config.is_protocol_enabled('b'):
                yield create_tool_call_streaming_start_part(tool_call_id, tool_name).ai_sdk_part_content

        # Accumulate tool parameter deltas
        if 'args' in tc_chunk and tc_chunk['args'] is not None:
            state.tool_calls[tool_call_id]['args'] += tc_chunk['args']
            # Send tool call delta (if enabled)
            if state.config and state.config.is_protocol_enabled('c'):
                from .factory import create_tool_call_delta_part
                yield create_tool_call_delta_part(tool_call_id, tc_chunk['args']).ai_sdk_part_content
    
    # This function is now a proper async generator that can yield values


async def _emit_completed_tool_calls(
    state: AIStreamState
) -> AsyncGenerator[str, None]:
    """Send completed tool calls
    
    Args:
        state: Current stream state
        
    Yields:
        Complete tool call AI SDK protocol strings
    """
    # Only emit tool calls if enabled
    if not state.config or state.config.is_protocol_enabled('9'):
        for tool_call_id, tool_info in state.tool_calls.items():
            try:
                import json
                args = json.loads(tool_info['args']) if tool_info['args'] else {}
                yield create_tool_call_part(tool_call_id, tool_info['name'], args).ai_sdk_part_content
                
                # Add tool invocation UI part
                state.add_part('tool-invocation', 
                              toolCallId=tool_call_id, 
                              toolName=tool_info['name'], 
                              args=args, 
                              state='call',
                              step=state.step_count)
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse tool call args for {tool_call_id}: {tool_info['args']}")
                yield create_tool_call_part(tool_call_id, tool_info['name'], {}).ai_sdk_part_content
                
                # Add tool invocation UI part with empty args
                state.add_part('tool-invocation', 
                              toolCallId=tool_call_id, 
                              toolName=tool_info['name'], 
                              args={}, 
                              state='call',
                              step=state.step_count)


# LangGraph event processing
async def _process_graph_event(
    chunk: Dict[str, Any], 
    state: AIStreamState
) -> AsyncGenerator[str, None]:
    """Process LangGraph events, converting them to AI SDK protocol parts
    
    Args:
        chunk: LangGraph event dictionary
        state: Current stream state
        
    Yields:
        AI SDK protocol strings
    """
    event = chunk.get('event')
    name = chunk.get('name', '')
    tags = chunk.get('tags', [])
    data = chunk.get('data', {})
    current_id = chunk.get('run_id', '')
    
    # Process LLM events (including chat model events)
    if event in ['on_llm_start', 'on_llm_stream', 'on_llm_end', 'on_chat_model_start', 'on_chat_model_stream', 'on_chat_model_end']:
        async for llm_part in _process_llm_events(event, data, state):
            yield llm_part
    
    # Process tool events with proper step management
    elif event in ['on_tool_start', 'on_tool_end']:
        async for tool_part in _process_tool_events(event, data, current_id, state, name):
            yield tool_part
    
    # Process agent events
    elif event in ['on_agent_action', 'on_agent_finish']:
        async for agent_part in _process_agent_events(event, data, state):
            yield agent_part
    
    # Process custom events
    elif event in ['on_chain_start', 'on_chain_end', 'on_chain_error']:
        async for custom_part in _process_custom_events(event, name, tags, data, current_id, state):
            yield custom_part


async def _process_llm_events(
    event: str, 
    data: Dict[str, Any], 
    state: AIStreamState
) -> AsyncGenerator[str, None]:
    """Process LLM start, stream, and end events
    
    Args:
        event: Event type
        data: Event data
        state: Stream state
        
    Yields:
        AI SDK protocol strings
    """
    if event in ["on_llm_start", "on_chat_model_start"]:
        # Check if there's an unfinished step and end it first
        if state.current_step_id and (not state.config or state.config.is_protocol_enabled('e')):
            yield create_finish_step_part(
                finish_reason="continue",
                prompt_tokens=state.total_prompt_tokens,
                completion_tokens=state.total_completion_tokens,
                is_continued=True
            ).ai_sdk_part_content
            state.finish_step()
        
        # Start new step
        if not state.config or state.config.is_protocol_enabled('f'):
            step_id = state.start_step("llm", {
                "event": event, 
                "data": data
            })
            yield create_start_step_part(step_id).ai_sdk_part_content
            state.add_part('step-start')
        
        # Handle input data - could be dict with prompts or list of messages
        input_data = data.get("input", [])
        if input_data:
            # Note: Removed automatic 8: protocol generation for llm_start
            # 8: protocol should only contain user-defined message annotations
            pass
    
    elif event in ["on_llm_stream", "on_chat_model_stream"]:
        # Handle LLM streaming chunks
        chunk_data = data.get('chunk')
        if isinstance(chunk_data, AIMessageChunk):
            async for ai_sdk_part in _process_ai_message_chunk(chunk_data, state):
                yield ai_sdk_part
    
    elif event in ["on_llm_end", "on_chat_model_end"]:
        # LLM has ended - could emit final response
        response = data.get("output", {})
        if response:
            # Note: Removed automatic 8: protocol generation for llm_end
            # 8: protocol should only contain user-defined message annotations
            pass
        
        # Check if this is the final LLM call (no more tools to execute)
        # If so, finish the current step
        # This will be overridden by agent_finish if there are more steps


async def _process_tool_events(
    event: str, 
    data: Dict[str, Any], 
    current_id: str,
    state: AIStreamState,
    name: str = ""
) -> AsyncGenerator[str, None]:
    """Process tool start and end events
    
    Args:
        event: Event type
        data: Event data
        current_id: Current run ID
        state: Stream state
        name: Tool name from event
        
    Yields:
        AI SDK protocol strings
    """
    if event == "on_tool_start":
        # Extract tool name from various possible locations
        tool_name = "unknown_tool"  # Default fallback
        
        # Try different ways to get tool name, prioritizing the name parameter
        if name and name.strip():
            tool_name = name
        elif "name" in data:
            tool_name = data["name"]
        elif "tool_name" in data:
            tool_name = data["tool_name"]
        elif isinstance(data.get("tool"), dict) and "name" in data["tool"]:
            tool_name = data["tool"]["name"]
        elif hasattr(data.get("serialized"), "get") and data["serialized"].get("name"):
            tool_name = data["serialized"]["name"]
        elif hasattr(data.get("serialized"), "get") and data["serialized"].get("id"):
            # Sometimes the tool name is in the id field
            tool_name = data["serialized"]["id"][-1] if isinstance(data["serialized"]["id"], list) else data["serialized"]["id"]
        
        # Extract tool arguments from various possible locations
        tool_args = {}
        if "input" in data and isinstance(data["input"], dict):
            tool_args = data["input"]
        elif "args" in data:
            tool_args = data["args"]
        elif "arguments" in data:
            tool_args = data["arguments"]
        
        # Try to get tool args from state's recent text content if not found in data
        if not tool_args and hasattr(state, 'recent_text_content'):
            # Parse tool arguments from recent agent text like "Action Input: Beijing" or "Action Input: 15 * 24"
            recent_text = state.recent_text_content or ""
            import re
            # Look for the most recent Action Input before this tool call
            # Use a more specific pattern that stops at the next Action or end of text
            action_input_match = re.search(r'Action Input:\s*([^\n]+?)(?=\n(?:Action|Observation|Thought|Final Answer)|$)', recent_text, re.IGNORECASE | re.DOTALL)
            if action_input_match:
                action_input = action_input_match.group(1).strip()
                # Remove quotes if present
                action_input = action_input.strip('"\'')
                # For simple cases, use the input as a single argument
                if tool_name == 'get_weather':
                    tool_args = {'city': action_input}
                elif tool_name == 'calculate_math':
                    tool_args = {'expression': action_input}
                else:
                    # Generic case: use 'input' as key
                    tool_args = {'input': action_input}
        
        tool_call_id = current_id
        
        # Generate tool call start protocol (b:) FIRST (if enabled)
        if not state.config or state.config.is_protocol_enabled('b'):
            yield create_tool_call_streaming_start_part(tool_call_id, tool_name).ai_sdk_part_content
        
        # Generate tool call protocol (9:) SECOND (if enabled)
        if not state.config or state.config.is_protocol_enabled('9'):
            yield create_tool_call_part(tool_call_id, tool_name, tool_args).ai_sdk_part_content
        
        # Add tool invocation UI part
        state.add_part('tool-invocation', 
                      toolCallId=tool_call_id, 
                      toolName=tool_name, 
                      args=tool_args, 
                      state='call',
                      step=state.step_count)

    elif event == "on_tool_end":
        tool_output = data.get("output", "")
        tool_call_id = current_id
        
        # Generate tool result protocol (a:) (if enabled)
        if not state.config or state.config.is_protocol_enabled('a'):
            if _is_document_list(tool_output):
                async for source_part in _process_document_list(tool_output, state):
                    yield source_part
                yield create_tool_result_part(tool_call_id, "Documents retrieved.").ai_sdk_part_content
            else:
                yield create_tool_result_part(tool_call_id, str(tool_output)).ai_sdk_part_content
        
        # Update tool invocation UI part with result
        for part in state.message_parts:
            if (hasattr(part, 'type') and part.type == 'tool-invocation' and
                hasattr(part, 'toolInvocation') and 
                part.toolInvocation.toolCallId == tool_call_id):
                part.toolInvocation.result = str(tool_output)
                part.toolInvocation.state = 'result'
                break
        
        # Tool execution completed - no need to finish step since tools run within LLM steps


async def _process_agent_events(
    event: str, 
    data: Dict[str, Any], 
    state: AIStreamState
) -> AsyncGenerator[str, None]:
    """Process agent thinking process events (on_agent_action/finish)
    
    Args:
        event: Event type
        data: Event data
        state: Stream state
        
    Yields:
        AI SDK protocol strings
    """
    if event == "on_agent_action":
        # Extract tool information from agent action
        if "tool" in data and "tool_input" in data:
            tool_name = data["tool"]
            tool_args = data["tool_input"]
            tool_call_id = f"call_{current_id[:8]}"
            
            # Generate tool call start protocol (b:) FIRST (if enabled)
            if not state.config or state.config.is_protocol_enabled('b'):
                yield create_tool_call_streaming_start_part(tool_call_id, tool_name).ai_sdk_part_content
            
            # Generate tool call protocol (9:) SECOND (if enabled)
            if not state.config or state.config.is_protocol_enabled('9'):
                yield create_tool_call_part(tool_call_id, tool_name, tool_args).ai_sdk_part_content
            
            # Store tool info for later result matching
            state.add_part('tool-invocation', 
                          toolCallId=tool_call_id, 
                          toolName=tool_name, 
                          args=tool_args, 
                          state='call',
                          step=state.step_count)
        
        # Keep reasoning protocol (g:) available but don't auto-trigger
        # Agent thinking can be accessed via log field if needed
        thought = data.get("log", "")
        if thought:
            # g: protocol is available but not automatically triggered
            # Can be manually enabled if needed
            pass

    elif event == "on_agent_finish":
        final_answer = data["output"]
        if final_answer and isinstance(final_answer, str):
            if not state.config or state.config.is_protocol_enabled('0'):
                yield create_text_part(final_answer).ai_sdk_part_content
            state.text_sent = True
            state.add_text_content(final_answer)
            state.add_part('text', text=final_answer)
        
        # Agent finished - check if there's an unfinished step and end it
        if state.current_step_id and (not state.config or state.config.is_protocol_enabled('e')):
            yield create_finish_step_part(
                finish_reason="stop",
                prompt_tokens=state.total_prompt_tokens,
                completion_tokens=state.total_completion_tokens,
                is_continued=False
            ).ai_sdk_part_content
            state.finish_step()


async def _process_custom_events(
    event: str, 
    name: str, 
    tags: List[str], 
    data: Dict[str, Any], 
    current_id: str,
    state: AIStreamState
) -> AsyncGenerator[str, None]:
    """Process custom data events and chain events
    
    Args:
        event: Event type
        name: Component name
        tags: Event tags
        data: Event data
        current_id: Current run ID
        state: Stream state
        
    Yields:
        AI SDK protocol strings
    """
    # Process LangGraph node lifecycle events
    # Check for graph:step:X or graph:node tags
    is_graph_node = any(tag.startswith('graph:step:') or tag == 'graph:node' for tag in tags)
    if is_graph_node:
        async for custom_part in _process_graph_node_events(event, name, tags, data, current_id, state):
            yield custom_part
    
    # Process AgentExecutor chain events
    elif name == "AgentExecutor":
        async for agent_part in _process_agent_executor_events(event, name, data, current_id, state):
            yield agent_part


async def _process_graph_node_events(
    event: str, 
    name: str, 
    tags: List[str], 
    data: Dict[str, Any], 
    current_id: str,
    state: AIStreamState
) -> AsyncGenerator[str, None]:
    """Process LangGraph node lifecycle events
    
    Args:
        event: Event type
        name: Node name
        tags: Event tags
        data: Event data
        current_id: Current run ID
        state: Stream state
        
    Yields:
        AI SDK protocol strings
    """
    node_type = _extract_node_type(tags)
    
    if event == 'on_chain_start':
        yield create_data_part([{
            'custom_type': 'node-start', 
            'node_id': current_id, 
            'name': name, 
            'node_type': node_type
        }]).ai_sdk_part_content
        
    elif event == 'on_chain_end':
        yield create_data_part([{
            'custom_type': 'node-end', 
            'node_id': current_id
        }]).ai_sdk_part_content
        
    elif event == 'on_chain_error':
        yield create_data_part([{
            'custom_type': 'node-error', 
            'node_id': current_id, 
            'error': str(data)
        }]).ai_sdk_part_content


async def _process_agent_executor_events(
    event: str, 
    name: str, 
    data: Dict[str, Any],
    current_id: str,
    state: AIStreamState
) -> AsyncGenerator[str, None]:
    """Process AgentExecutor chain events
    
    Args:
        event: Event type
        name: Component name
        data: Event data
        current_id: Current run ID
        state: Stream state
        
    Yields:
        AI SDK protocol strings
    """
    if event == 'on_chain_start':
        yield create_data_part([{
            'custom_type': 'agent-executor-start', 
            'name': name, 
            'inputs': data.get('input')
        }]).ai_sdk_part_content
        
    elif event == 'on_chain_end':
        yield create_data_part([{
            'custom_type': 'agent-executor-end', 
            'output': data.get('output')
        }]).ai_sdk_part_content


# Document processing utilities
def _is_document_list(tool_output: Any) -> bool:
    """Check if tool output is a list of Document objects"""
    return (
        isinstance(tool_output, list) and 
        all(isinstance(item, Document) for item in tool_output)
    )


async def _process_document_list(documents: List[Document], state: AIStreamState = None) -> AsyncGenerator[str, None]:
    """Process Document object list, convert to Source Part
    
    Args:
        documents: List of Document objects
        state: Stream state for tracking UI parts
        
    Yields:
        AI SDK protocol strings
    """
    for doc in documents:
        url = doc.metadata.get("source", "")
        title = _extract_document_title(doc)
        yield create_source_part(url, title).ai_sdk_part_content
        
        # Add source UI part if state is available
        if state:
            state.add_part('source', source={
                "url": url,
                "title": title,
                "content": doc.page_content,
                "metadata": doc.metadata
            })


def _extract_node_type(tags: List[str]) -> str:
    """Extract node type from tags"""
    for tag in tags:
        if tag.startswith('graph:node_type:'):
            return tag.split(':', 2)[-1]
    return 'unknown'


def _extract_document_title(document: Document) -> str:
    """Extract title from Document object"""
    title = document.metadata.get("title")
    if title:
        return title
    
    # If no title, use first 50 characters of page content
    content = document.page_content
    if len(content) > 50:
        return content[:50] + "..."
    return content


# Component detection function for backward compatibility
def _is_major_workflow_component(name: str, tags: List[str]) -> bool:
    """Check if a component is a major workflow component
    
    This function identifies LangChain components that represent major workflow
    components like agents, chains, and graphs.
    
    Args:
        name: Component name
        tags: List of component tags
        
    Returns:
        bool: True if component is a major workflow component
    """
    # Check by exact name matches for major components
    major_component_names = {
        'AgentExecutor', 'LangGraph', 'CompiledGraph',
        'ConversationalRetrievalChain', 'RetrievalQA', 'LLMChain',
        'SequentialChain', 'RouterChain'
    }
    
    if name in major_component_names:
        return True
    
    # Check by tags
    major_tags = {'agent', 'chain', 'executor', 'langgraph', 'graph'}
    if any(tag in major_tags for tag in tags):
        return True
    
    # Check for graph step tags
    if any(tag.startswith('graph:step:') for tag in tags):
        return True
    
    # Check for graph node tags
    if 'graph' in tags:
        return True
    
    return False


# Step creation functions removed - f: and e: protocols now only used for actual agent steps


# Removed duplicate function definitions - using factory imports instead