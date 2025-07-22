#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core AI SDK Adapter

Main adapter class for converting LangChain streams to AI SDK protocol.
"""

from typing import Any, AsyncGenerator, Union, Optional, Dict, List
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import Runnable

from .config import AdapterConfig, default_config
from .callbacks import BaseAICallbackHandler, Message, UIPart, LanguageModelUsage
from .utils import (
    AIStreamState, _is_ai_sdk_protocol_string, _is_langgraph_event,
    _has_nested_ai_message_chunk, _is_agent_executor_output,
    _ensure_newline_ending, _handle_stream_end, _process_ai_message_chunk,
    _process_graph_event
)
from .factory import (
    create_text_part, create_error_part, create_reasoning_part, create_source_part,
    create_file_part, create_data_part, create_tool_call_part, create_tool_result_part,
    create_start_step_part, create_finish_step_part, create_finish_message_part,
    create_message_annotation_part
)


class LangChainAdapter:
    """
    AI SDK Adapter for LangChain
    
    Converts LangChain streaming outputs to AI SDK protocol compliant format.
    Supports various LangChain components including LLMs, agents, tools, and LangGraph.
    
    This adapter uses an instance-based approach for thread safety and state management.
    Each adapter instance maintains its own state and can be used for manual UI part creation.
    """
    
    def __init__(self, config: Optional[AdapterConfig] = None, callback: Optional[BaseAICallbackHandler] = None):
        """
        Initialize the adapter with configuration and callback.
        
        Args:
            config: Optional configuration to control which protocols are automatically generated
            callback: Optional callback handler for observing stream lifecycle
        """
        self.config = config or default_config
        self.callback = callback
        self.state = AIStreamState()
        self.state.config = self.config
        self.state.callback = self.callback
    
    async def to_data_stream_response(
        self,
        stream: AsyncGenerator[Any, None]
    ) -> AsyncGenerator[str, None]:
        """
        Convert LangChain async stream to AI SDK protocol stream
        
        Args:
            langchain_stream: Async generator from LangChain components
            
        Yields:
            AI SDK protocol compliant strings
            
        Example:
            ```python
            from langchain_aisdk_adapter import LangChainAdapter, AdapterConfig, BaseAICallbackHandler
            
            # Create adapter instance
            adapter = LangChainAdapter(config=config, callback=callback)
            
            # Convert LangChain stream to AI SDK format
            async for ai_sdk_part in adapter.to_data_stream_response(langchain_stream):
                print(ai_sdk_part)
            ```
        """
        
        # Initialize state for this stream with config and callback
        state = AIStreamState(config=self.config, callback=self.callback)
        
        try:
            async for chunk in stream:
                async for ai_sdk_part in _process_stream_chunk(chunk, state):
                    yield ai_sdk_part
        except Exception as e:
            # Send error part when stream processing fails (if enabled)
            if self.config.is_protocol_enabled('3'):
                yield create_error_part(f"Stream processing error: {str(e)}").ai_sdk_part_content
            
            # Call error callback if available
            if self.callback:
                try:
                    await self.callback.on_error(e)
                except Exception as callback_error:
                    print(f"Warning: Callback error handler failed: {callback_error}")
        finally:
            # Cleanup when stream ends and send finish message
            async for finish_part in _handle_stream_end(state):
                yield finish_part
    
    # Manual UI Part Creation Methods
    
    async def text(self, text: str) -> str:
        """
        Manually create and emit a text part.
        
        TextUIPart is cumulative - if there's already a TextUIPart in the message,
        the new text will be appended to it rather than creating a new part.
        
        Args:
            text: Text content to add
            
        Returns:
            AI SDK protocol string for the text part
        """
        if self.config.is_protocol_enabled('0'):
            part = create_text_part(text)
            self.state.add_text_content(text)
            self.state.add_part('text', text=text)
            return part.ai_sdk_part_content
        return ""
    
    async def reasoning(self, reasoning: str) -> str:
        """
        Manually create and emit a reasoning part.
        
        Args:
            reasoning: Reasoning content to add
            
        Returns:
            AI SDK protocol string for the reasoning part
        """
        if self.config.is_protocol_enabled('g'):
            part = create_reasoning_part(reasoning)
            self.state.add_part('reasoning', reasoning=reasoning)
            return part.ai_sdk_part_content
        return ""
    
    async def source(self, url: str, title: Optional[str] = None) -> str:
        """
        Manually create and emit a source part.
        
        Args:
            url: URL of the source
            title: Optional title of the source
            
        Returns:
            AI SDK protocol string for the source part
        """
        if self.config.is_protocol_enabled('h'):
            part = create_source_part(url, title)
            source_data = {'url': url}
            if title:
                source_data['title'] = title
            self.state.add_part('source', source=source_data)
            return part.ai_sdk_part_content
        return ""
    
    async def file(self, data: str, mime_type: str, name: Optional[str] = None) -> str:
        """
        Manually create and emit a file part.
        
        Args:
            data: Base64 encoded file data
            mime_type: MIME type of the file
            name: Optional file name
            
        Returns:
            AI SDK protocol string for the file part
        """
        if self.config.is_protocol_enabled('k'):
            part = create_file_part(data, mime_type)
            self.state.add_part('file', data=data, mimeType=mime_type, name=name or 'file')
            return part.ai_sdk_part_content
        return ""
    
    async def data(self, data: List[Any]) -> str:
        """
        Manually create and emit a data part.
        
        Args:
            data: Custom JSON data array
            
        Returns:
            AI SDK protocol string for the data part
        """
        if self.config.is_protocol_enabled('2'):
            part = create_data_part(data)
            self.state.add_part('data', data=data)
            return part.ai_sdk_part_content
        return ""
    
    async def error(self, error_message: str) -> str:
        """
        Manually create and emit an error part.
        
        Args:
            error_message: Error message to add
            
        Returns:
            AI SDK protocol string for the error part
        """
        if self.config.is_protocol_enabled('3'):
            part = create_error_part(error_message)
            self.state.add_part('error', error=error_message)
            return part.ai_sdk_part_content
        return ""
    
    async def annotation(self, annotations: List[Any]) -> str:
        """
        Manually create and emit a message annotation part.
        
        Args:
            annotations: List of annotation metadata
            
        Returns:
            AI SDK protocol string for the annotation part
        """
        if self.config.is_protocol_enabled('8'):
            part = create_message_annotation_part(annotations)
            self.state.add_part('annotation', annotations=annotations)
            return part.ai_sdk_part_content
        return ""
    
    async def tool_call(self, tool_name: str, args: Dict[str, Any], tool_call_id: Optional[str] = None) -> str:
        """
        Manually create and emit a tool call part.
        
        Args:
            tool_name: Name of the tool to call
            args: Arguments for the tool call
            tool_call_id: Optional tool call ID (auto-generated if not provided)
            
        Returns:
            AI SDK protocol string for the tool call part
        """
        if self.config.is_protocol_enabled('9'):
            if tool_call_id is None:
                import uuid
                tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
            
            part = create_tool_call_part(tool_call_id, tool_name, args)
            self.state.add_part('tool-invocation', 
                              toolCallId=tool_call_id, 
                              toolName=tool_name, 
                              args=args, 
                              state='call')
            return part.ai_sdk_part_content
        return ""
    
    async def tool_result(self, tool_call_id: str, result: Any) -> str:
        """
        Manually create and emit a tool result part.
        
        This method updates an existing ToolInvocationUIPart rather than creating
        a new one, as per the protocol mapping specification.
        
        Args:
            tool_call_id: ID of the tool call this result belongs to
            result: Result of the tool execution
            
        Returns:
            AI SDK protocol string for the tool result part
        """
        if self.config.is_protocol_enabled('a'):
            part = create_tool_result_part(tool_call_id, result)
            
            # Update existing tool invocation UI part with result
            for ui_part in self.state.message_parts:
                if (hasattr(ui_part, 'toolCallId') and 
                    ui_part.toolCallId == tool_call_id and 
                    ui_part.type == 'tool-invocation'):
                    ui_part.result = result
                    ui_part.state = 'result'
                    break
            
            return part.ai_sdk_part_content
        return ""
    
    async def step_start(self, message_id: Optional[str] = None) -> str:
        """
        Manually create and emit a step start part.
        
        Args:
            message_id: Optional message ID (auto-generated if not provided)
            
        Returns:
            AI SDK protocol string for the step start part
        """
        if self.config.is_protocol_enabled('f'):
            if message_id is None:
                message_id = self.state.start_step()
            
            part = create_start_step_part(message_id)
            self.state.add_part('step-start')
            return part.ai_sdk_part_content
        return ""
    
    async def step_finish(self, finish_reason: str = "stop", 
                         prompt_tokens: int = 0, 
                         completion_tokens: int = 0, 
                         is_continued: bool = False) -> str:
        """
        Manually create and emit a step finish part.
        
        Args:
            finish_reason: Reason for finishing the step
            prompt_tokens: Number of prompt tokens used
            completion_tokens: Number of completion tokens used
            is_continued: Whether the step will be continued
            
        Returns:
            AI SDK protocol string for the step finish part
        """
        if self.config.is_protocol_enabled('e'):
            part = create_finish_step_part(finish_reason, prompt_tokens, completion_tokens, is_continued)
            self.state.finish_step()
            return part.ai_sdk_part_content
        return ""
    
    async def message_finish(self, finish_reason: str = "stop", 
                           prompt_tokens: int = 0, 
                           completion_tokens: int = 0) -> str:
        """
        Manually create and emit a message finish part.
        
        Args:
            finish_reason: Reason for finishing the message
            prompt_tokens: Number of prompt tokens used
            completion_tokens: Number of completion tokens used
            
        Returns:
            AI SDK protocol string for the message finish part
        """
        if self.config.is_protocol_enabled('d'):
            part = create_finish_message_part(finish_reason, prompt_tokens, completion_tokens)
            return part.ai_sdk_part_content
        return ""
    
    def get_current_message(self) -> Message:
        """
        Get the current message being built.
        
        Returns:
            Current Message object with all parts
        """
        return self.state.build_final_message()


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
        # Generate the actual output (if text protocol is enabled)
        output_text = _ensure_newline_ending(chunk["output"])
        if not state.config or state.config.is_protocol_enabled('0'):
            yield create_text_part(output_text).ai_sdk_part_content
        state.text_sent = True
        return
    
    # Handle plain string content
    if isinstance(chunk, str):
        if not state.config or state.config.is_protocol_enabled('0'):
            yield create_text_part(chunk).ai_sdk_part_content
        state.text_sent = True
        return
    
    # Handle unknown chunk types
    print(f"Warning: Unknown chunk type: {type(chunk)}, content: {chunk}")