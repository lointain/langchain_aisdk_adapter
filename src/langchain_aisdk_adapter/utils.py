#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility Functions for AI SDK Adapter

Helper functions for stream processing, event handling, and data conversion.
"""

from typing import Any, AsyncGenerator, Dict, List
from langchain_core.messages import AIMessageChunk
from langchain_core.documents import Document

from .models import AI_SDK_PROTOCOL_PREFIXES
from .factory import (
    create_text_part, create_reasoning_part, create_data_part, create_error_part,
    create_source_part, create_tool_call_streaming_start_part, create_tool_call_part,
    create_tool_result_part
)


# State management
class AIStreamState:
    """State management for AI SDK streaming"""
    
    def __init__(self):
        self.text_sent = False
        self.reasoning_sent = False
        self.tool_calls = {}  # tool_call_id -> {name, args}
        self.tool_call_counter = 0
        self.config = None  # Will be set by adapter


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
        yield create_finish_message_part(finish_reason="stop").ai_sdk_part_content


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
    
    # Process text content - only if enabled
    if isinstance(chunk.content, str) and chunk.content:
        if not state.config or state.config.is_protocol_enabled('0'):
            yield create_text_part(chunk.content).ai_sdk_part_content
        state.text_sent = True

    # Process tool calls
    async for tool_part in _process_tool_calls_from_chunk(chunk, state):
        yield tool_part

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
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse tool call args for {tool_call_id}: {tool_info['args']}")
                yield create_tool_call_part(tool_call_id, tool_info['name'], {}).ai_sdk_part_content


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
    
    # Process LLM events
    if event in ['on_llm_start', 'on_llm_stream', 'on_llm_end']:
        async for llm_part in _process_llm_events(event, data, state):
            yield llm_part
    
    # Process tool events with proper step management
    elif event in ['on_tool_start', 'on_tool_end']:
        async for tool_part in _process_tool_events(event, data, current_id, state):
            yield tool_part
    
    # Process agent events
    elif event in ['on_agent_action', 'on_agent_finish']:
        async for agent_part in _process_agent_events(event, data, state):
            yield agent_part
    
    # Process custom events
    elif event in ['on_chain_start', 'on_chain_end', 'on_chain_error']:
        async for custom_part in _process_custom_events(event, name, tags, data, current_id):
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
    if event == "on_llm_start":
        # LLM started, no step protocol needed for basic LLM events
        pass
    
    elif event == "on_llm_stream":
        # Handle LLM streaming chunks
        chunk_data = data.get('chunk')
        if isinstance(chunk_data, AIMessageChunk):
            async for ai_sdk_part in _process_ai_message_chunk(chunk_data, state):
                yield ai_sdk_part
    
    elif event == "on_llm_end":
        # LLM finished, no step protocol needed for basic LLM events
        pass


async def _process_tool_events(
    event: str, 
    data: Dict[str, Any], 
    current_id: str,
    state: AIStreamState
) -> AsyncGenerator[str, None]:
    """Process tool start and end events
    
    Args:
        event: Event type
        data: Event data
        current_id: Current run ID
        state: Stream state
        
    Yields:
        AI SDK protocol strings
    """
    if event == "on_tool_start":
        tool_name = data.get("name", data.get("tool_name", "unknown_tool"))
        tool_args = data.get("input", {})
        tool_call_id = current_id
        
        # Generate tool call protocol (9:) (if enabled)
        if not state.config or state.config.is_protocol_enabled('9'):
            yield create_tool_call_part(tool_call_id, tool_name, tool_args).ai_sdk_part_content
        
        # Generate tool call start protocol (b:) (if enabled)
        if not state.config or state.config.is_protocol_enabled('b'):
            yield create_tool_call_start_part(tool_call_id, tool_name).ai_sdk_part_content

    elif event == "on_tool_end":
        tool_output = data.get("output", "")
        tool_call_id = current_id
        
        # Generate tool result protocol (a:) (if enabled)
        if not state.config or state.config.is_protocol_enabled('a'):
            if _is_document_list(tool_output):
                async for source_part in _process_document_list(tool_output):
                    yield source_part
                yield create_tool_result_part(tool_call_id, "Documents retrieved.").ai_sdk_part_content
            else:
                yield create_tool_result_part(tool_call_id, str(tool_output)).ai_sdk_part_content


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
        thought = data["log"]
        if thought and (not state.config or state.config.is_protocol_enabled('g')):
            yield create_reasoning_part(thought).ai_sdk_part_content

    elif event == "on_agent_finish":
        final_answer = data["output"]
        if final_answer and isinstance(final_answer, str):
            if not state.config or state.config.is_protocol_enabled('0'):
                yield create_text_part(final_answer).ai_sdk_part_content
            state.text_sent = True


async def _process_custom_events(
    event: str, 
    name: str, 
    tags: List[str], 
    data: Dict[str, Any], 
    current_id: str
) -> AsyncGenerator[str, None]:
    """Process custom data events and chain events
    
    Args:
        event: Event type
        name: Component name
        tags: Event tags
        data: Event data
        current_id: Current run ID
        
    Yields:
        AI SDK protocol strings
    """
    # Process LangGraph node lifecycle events
    # Check for graph:step:X or graph:node tags
    is_graph_node = any(tag.startswith('graph:step:') or tag == 'graph:node' for tag in tags)
    if is_graph_node:
        async for custom_part in _process_graph_node_events(event, name, tags, data, current_id):
            yield custom_part
    
    # Process AgentExecutor chain events
    elif name == "AgentExecutor":
        async for agent_part in _process_agent_executor_events(event, name, data):
            yield agent_part


async def _process_graph_node_events(
    event: str, 
    name: str, 
    tags: List[str], 
    data: Dict[str, Any], 
    current_id: str
) -> AsyncGenerator[str, None]:
    """Process LangGraph node lifecycle events
    
    Args:
        event: Event type
        name: Node name
        tags: Event tags
        data: Event data
        current_id: Current run ID
        
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
    data: Dict[str, Any]
) -> AsyncGenerator[str, None]:
    """Process AgentExecutor chain events
    
    Args:
        event: Event type
        name: Component name
        data: Event data
        
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


async def _process_document_list(documents: List[Document]) -> AsyncGenerator[str, None]:
    """Process Document object list, convert to Source Part
    
    Args:
        documents: List of Document objects
        
    Yields:
        AI SDK protocol strings
    """
    for doc in documents:
        url = doc.metadata.get("source", "")
        title = _extract_document_title(doc)
        yield create_source_part(url, title).ai_sdk_part_content


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


def create_tool_call_start_part(tool_call_id: str, tool_name: str = "unknown_tool"):
    """Create tool call start part using proper factory function
    
    Args:
        tool_call_id: Unique identifier for the tool call
        tool_name: Name of the tool being called, defaults to "unknown_tool"
    
    Returns:
        AISDKPartEmitter for tool call start part
    """
    from .factory import factory
    return factory.tool_call_start(tool_call_id, tool_name)


def create_tool_call_streaming_start_part(tool_call_id: str, tool_name: str):
    """Create tool call streaming start part using proper factory function"""
    from .factory import factory
    return factory.tool_call_start(tool_call_id, tool_name)