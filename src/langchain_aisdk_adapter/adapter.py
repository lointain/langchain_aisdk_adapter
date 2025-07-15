#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core AI SDK Adapter

Main adapter class for converting LangChain streams to AI SDK protocol.
"""

from typing import Any, AsyncGenerator, Union
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import Runnable

from .utils import (
    AIStreamState, _is_ai_sdk_protocol_string, _is_langgraph_event,
    _has_nested_ai_message_chunk, _is_agent_executor_output,
    _ensure_newline_ending, _handle_stream_end, _process_ai_message_chunk,
    _process_graph_event
)
from .factory import create_text_part, create_error_part


class AISDKAdapter:
    """
    AI SDK Adapter for LangChain
    
    Converts LangChain streaming outputs to AI SDK protocol compliant format.
    Supports various LangChain components including LLMs, agents, tools, and LangGraph.
    """
    
    @staticmethod
    async def astream(
        langchain_stream: AsyncGenerator[Any, None]
    ) -> AsyncGenerator[str, None]:
        """
        Convert LangChain async stream to AI SDK protocol stream
        
        Args:
            langchain_stream: Async generator from LangChain components
            
        Yields:
            AI SDK protocol compliant strings
            
        Example:
            ```python
            from langchain_aisdk_adapter import AISDKAdapter
            
            # Convert LangChain stream to AI SDK format
            async for ai_sdk_part in AISDKAdapter.astream(langchain_stream):
                print(ai_sdk_part)
            ```
        """
        state = AIStreamState()
        
        try:
            async for chunk in langchain_stream:
                async for ai_sdk_part in _process_stream_chunk(chunk, state):
                    yield ai_sdk_part
        except Exception as e:
            # Send error part when stream processing fails
            yield create_error_part(f"Stream processing error: {str(e)}").ai_sdk_part_content
        finally:
            # Cleanup when stream ends and send finish message
            async for finish_part in _handle_stream_end(state):
                yield finish_part


# Stream chunk processing function
async def _process_stream_chunk(
    chunk: Any, 
    state: AIStreamState
) -> AsyncGenerator[str, None]:
    """
    Process individual stream chunk, determining type and converting to AI SDK format
    
    Args:
        chunk: Individual chunk from LangChain stream
        state: Current stream state
        
    Yields:
        AI SDK protocol strings
    """
    # Handle pre-formatted AI SDK protocol strings
    if _is_ai_sdk_protocol_string(chunk):
        yield chunk
        return
    
    # Handle AIMessageChunk objects
    if isinstance(chunk, AIMessageChunk):
        async for ai_sdk_part in _process_ai_message_chunk(chunk, state):
            yield ai_sdk_part
        return
    
    # Handle LangGraph events
    if _is_langgraph_event(chunk):
        # Check for nested AIMessageChunk in LangGraph events
        if _has_nested_ai_message_chunk(chunk):
            nested_chunk = chunk['data']['chunk']
            async for ai_sdk_part in _process_ai_message_chunk(nested_chunk, state):
                yield ai_sdk_part
        else:
            async for ai_sdk_part in _process_graph_event(chunk, state):
                yield ai_sdk_part
        return
    
    # Handle AgentExecutor output
    if _is_agent_executor_output(chunk):
        # Generate the actual output
        output_text = _ensure_newline_ending(chunk["output"])
        yield create_text_part(output_text).ai_sdk_part_content
        state.text_sent = True
        return
    
    # Handle plain string content
    if isinstance(chunk, str):
        yield create_text_part(chunk).ai_sdk_part_content
        state.text_sent = True
        return
    
    # Handle unknown chunk types
    print(f"Warning: Unknown chunk type: {type(chunk)}, content: {chunk}")