#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI SDK Protocol Part Creation Factory

Factory class and functions for creating AISDKPartEmitter instances for various AI SDK protocol parts.
"""

import json
from typing import Any, Dict, List, Optional
from pydantic import ValidationError

from .models import AI_SDK_PROTOCOL_MAP, UsageInfo
from .emitter import AISDKPartEmitter


class AISDKFactory:
    """
    Factory class for creating AI SDK protocol parts with simplified method names.
    
    This class provides a clean interface for creating various AI SDK protocol parts
    with better IDE auto-completion and organized method grouping.
    
    Example:
        ```python
        factory = AISDKFactory()
        
        # Create text part
        text_part = factory.text('Hello World')
        
        # Create tool call
        tool_call = factory.tool_call('call_123', 'search', {'query': 'python'})
        
        # Create error
        error_part = factory.error('Something went wrong')
        ```
    """
    
    @staticmethod
    def part(type_id: str, content: Any) -> AISDKPartEmitter:
        """
        Generic factory method for creating any AI SDK protocol part.
        
        Args:
            type_id: AI SDK protocol type identifier (e.g. '0' for text, '9' for tool calls)
            content: Part content, will be validated and serialized according to type
            
        Returns:
            AISDKPartEmitter instance with properly formatted AI SDK protocol string
            
        Raises:
            ValueError: If type_id is not supported or content validation fails
        """
        if type_id not in AI_SDK_PROTOCOL_MAP:
            raise ValueError(f"Unsupported AI SDK protocol type: {type_id}")
        
        model_class = AI_SDK_PROTOCOL_MAP[type_id]
        
        try:
            # Validate content using corresponding Pydantic model
            if hasattr(model_class, 'model_validate'):
                # Handle RootModel (Pydantic v2)
                validated_content = model_class.model_validate(content)
                serialized_content = json.dumps(validated_content.model_dump())
            else:
                # Handle regular models
                validated_content = model_class(**content if isinstance(content, dict) else content)
                serialized_content = validated_content.model_dump_json()
            
            # Format as AI SDK protocol string
            ai_sdk_string = f"{type_id}:{serialized_content}\n"
            return AISDKPartEmitter(ai_sdk_string)
            
        except (ValidationError, TypeError) as e:
            raise ValueError(f"Content validation failed for type {type_id}: {str(e)}")
    
    # Basic content parts
    @staticmethod
    def text(text: str) -> AISDKPartEmitter:
        """Create text part (0:) for appending text content to messages."""
        return AISDKFactory.part('0', text)
    
    @staticmethod
    def data(data: List[Any]) -> AISDKPartEmitter:
        """Create data part (2:) for custom JSON data arrays."""
        return AISDKFactory.part('2', data)
    
    @staticmethod
    def error(error_message: str) -> AISDKPartEmitter:
        """Create error part (3:) for error information."""
        return AISDKFactory.part('3', error_message)
    
    # Reasoning parts
    @staticmethod
    def reasoning(reasoning: str) -> AISDKPartEmitter:
        """Create reasoning part (g:) for AI model's reasoning process."""
        return AISDKFactory.part('g', reasoning)
    
    @staticmethod
    def redacted_reasoning(data: str) -> AISDKPartEmitter:
        """Create redacted reasoning part (i:) for edited reasoning data."""
        return AISDKFactory.part('i', {'data': data})
    
    @staticmethod
    def reasoning_signature(signature: str) -> AISDKPartEmitter:
        """Create reasoning signature part (j:) for signature verification."""
        return AISDKFactory.part('j', {'signature': signature})
    
    # Resource parts
    @staticmethod
    def source(url: str, title: Optional[str] = None) -> AISDKPartEmitter:
        """Create source part (h:) for referencing external resources."""
        content = {'url': url}
        if title is not None:
            content['title'] = title
        return AISDKFactory.part('h', content)
    
    @staticmethod
    def file(data: str, mime_type: str) -> AISDKPartEmitter:
        """Create file part (k:) for Base64 encoded binary files."""
        return AISDKFactory.part('k', {'data': data, 'mimeType': mime_type})
    
    # Annotation parts
    @staticmethod
    def annotation(annotations: List[Any]) -> AISDKPartEmitter:
        """Create message annotation part (8:) for metadata annotations."""
        return AISDKFactory.part('8', annotations)
    
    # Tool-related parts
    @staticmethod
    def tool_call_start(tool_call_id: str, tool_name: str) -> AISDKPartEmitter:
        """Create tool call streaming start part (b:) to mark beginning of tool call."""
        return AISDKFactory.part('b', {'toolCallId': tool_call_id, 'toolName': tool_name})
    
    @staticmethod
    def tool_call_delta(tool_call_id: str, args_text_delta: str) -> AISDKPartEmitter:
        """Create tool call delta part (c:) for incremental tool parameter updates."""
        return AISDKFactory.part('c', {'toolCallId': tool_call_id, 'argsTextDelta': args_text_delta})
    
    @staticmethod
    def tool_call(tool_call_id: str, tool_name: str, args: Dict[str, Any]) -> AISDKPartEmitter:
        """Create tool call part (9:) for complete tool call information."""
        return AISDKFactory.part('9', {
            'toolCallId': tool_call_id,
            'toolName': tool_name,
            'args': args
        })
    
    @staticmethod
    def tool_result(tool_call_id: str, result: Any) -> AISDKPartEmitter:
        """Create tool result part (a:) for tool execution results."""
        return AISDKFactory.part('a', {'toolCallId': tool_call_id, 'result': result})
    
    # Step control parts
    @staticmethod
    def start_step(message_id: str) -> AISDKPartEmitter:
        """Create start step part (f:) to mark beginning of processing step."""
        return AISDKFactory.part('f', {'messageId': message_id})
    
    @staticmethod
    def finish_step(
        finish_reason: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        is_continued: bool = False
    ) -> AISDKPartEmitter:
        """Create finish step part (e:) to mark end of processing step."""
        usage = UsageInfo(promptTokens=prompt_tokens, completionTokens=completion_tokens)
        return AISDKFactory.part('e', {
            'finishReason': finish_reason,
            'usage': usage.model_dump(),
            'isContinued': is_continued
        })
    
    @staticmethod
    def finish_message(
        finish_reason: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0
    ) -> AISDKPartEmitter:
        """Create finish message part (d:) to mark completion of entire message."""
        usage = UsageInfo(promptTokens=prompt_tokens, completionTokens=completion_tokens)
        return AISDKFactory.part('d', {
            'finishReason': finish_reason,
            'usage': usage.model_dump()
        })


# Create a default factory instance for convenience
factory = AISDKFactory()


def create_ai_sdk_part(type_id: str, content: Any) -> AISDKPartEmitter:
    """
    Generic factory function for creating AISDKPartEmitter instances for any AI SDK protocol part.
    Automatically performs Pydantic validation based on type_id.
    
    Args:
        type_id: AI SDK protocol type identifier (e.g. '0' for text, '9' for tool calls)
        content: Part content, will be validated and serialized according to type
        
    Returns:
        AISDKPartEmitter instance with properly formatted AI SDK protocol string
        
    Raises:
        ValueError: If type_id is not supported or content validation fails
        
    Example:
        ```python
        # Create text part
        text_emitter = create_ai_sdk_part('0', 'Hello World')
        
        # Create tool call part
        tool_call_emitter = create_ai_sdk_part('9', {
            'toolCallId': 'call_123',
            'toolName': 'search',
            'args': {'query': 'python'}
        })
        ```
    """
    if type_id not in AI_SDK_PROTOCOL_MAP:
        raise ValueError(f"Unsupported AI SDK protocol type: {type_id}")
    
    model_class = AI_SDK_PROTOCOL_MAP[type_id]
    
    try:
        # Validate content using corresponding Pydantic model
        if hasattr(model_class, 'model_validate'):
            # Handle RootModel (Pydantic v2)
            validated_content = model_class.model_validate(content)
            serialized_content = json.dumps(validated_content.model_dump())
        else:
            # Handle regular models
            validated_content = model_class(**content if isinstance(content, dict) else content)
            serialized_content = validated_content.model_dump_json()
        
        # Format as AI SDK protocol string
        ai_sdk_string = f"{type_id}:{serialized_content}\n"
        return AISDKPartEmitter(ai_sdk_string)
        
    except (ValidationError, TypeError) as e:
        raise ValueError(f"Content validation failed for type {type_id}: {str(e)}")


# Convenience factory functions for common AI SDK protocol parts

def create_text_part(text: str) -> AISDKPartEmitter:
    """
    Create text part (0:) for appending text content to messages
    
    Args:
        text: Text content to append
        
    Returns:
        AISDKPartEmitter for text part
    """
    return create_ai_sdk_part('0', text)


def create_reasoning_part(reasoning: str) -> AISDKPartEmitter:
    """
    Create reasoning part (g:) for AI model's reasoning process
    
    Args:
        reasoning: Reasoning content
        
    Returns:
        AISDKPartEmitter for reasoning part
    """
    return create_ai_sdk_part('g', reasoning)


def create_redacted_reasoning_part(data: str) -> AISDKPartEmitter:
    """
    Create redacted reasoning part (i:) for edited reasoning data
    
    Args:
        data: Redacted reasoning data
        
    Returns:
        AISDKPartEmitter for redacted reasoning part
    """
    return create_ai_sdk_part('i', {'data': data})


def create_reasoning_signature_part(signature: str) -> AISDKPartEmitter:
    """
    Create reasoning signature part (j:) for signature verification
    
    Args:
        signature: Signature for reasoning verification
        
    Returns:
        AISDKPartEmitter for reasoning signature part
    """
    return create_ai_sdk_part('j', {'signature': signature})


def create_source_part(url: str, title: Optional[str] = None) -> AISDKPartEmitter:
    """
    Create source part (h:) for referencing external resources
    
    Args:
        url: URL of the external resource
        title: Optional title of the resource
        
    Returns:
        AISDKPartEmitter for source part
    """
    content = {'url': url}
    if title is not None:
        content['title'] = title
    return create_ai_sdk_part('h', content)


def create_file_part(data: str, mime_type: str) -> AISDKPartEmitter:
    """
    Create file part (k:) for Base64 encoded binary files
    
    Args:
        data: Base64 encoded binary data
        mime_type: MIME type (e.g., 'image/png')
        
    Returns:
        AISDKPartEmitter for file part
    """
    return create_ai_sdk_part('k', {'data': data, 'mimeType': mime_type})


def create_data_part(data: List[Any]) -> AISDKPartEmitter:
    """
    Create data part (2:) for custom JSON data arrays
    
    Args:
        data: Custom JSON data array
        
    Returns:
        AISDKPartEmitter for data part
    """
    return create_ai_sdk_part('2', data)


def create_message_annotation_part(annotations: List[Any]) -> AISDKPartEmitter:
    """
    Create message annotation part (8:) for metadata annotations
    
    Args:
        annotations: List of annotation metadata
        
    Returns:
        AISDKPartEmitter for message annotation part
    """
    return create_ai_sdk_part('8', annotations)


def create_error_part(error_message: str) -> AISDKPartEmitter:
    """
    Create error part (3:) for error information
    
    Args:
        error_message: Error message string
        
    Returns:
        AISDKPartEmitter for error part
    """
    return create_ai_sdk_part('3', error_message)


def create_tool_call_streaming_start_part(tool_call_id: str, tool_name: str) -> AISDKPartEmitter:
    """
    Create tool call streaming start part (b:) to mark beginning of tool call
    
    Args:
        tool_call_id: Unique identifier for the tool call
        tool_name: Name of the tool being called
        
    Returns:
        AISDKPartEmitter for tool call streaming start part
    """
    return create_ai_sdk_part('b', {'toolCallId': tool_call_id, 'toolName': tool_name})


def create_tool_call_delta_part(tool_call_id: str, args_text_delta: str) -> AISDKPartEmitter:
    """
    Create tool call delta part (c:) for incremental tool parameter updates
    
    Args:
        tool_call_id: Unique identifier for the tool call
        args_text_delta: Incremental text delta for tool arguments
        
    Returns:
        AISDKPartEmitter for tool call delta part
    """
    return create_ai_sdk_part('c', {'toolCallId': tool_call_id, 'argsTextDelta': args_text_delta})


def create_tool_call_part(tool_call_id: str, tool_name: str, args: Dict[str, Any]) -> AISDKPartEmitter:
    """
    Create tool call part (9:) for complete tool call information
    
    Args:
        tool_call_id: Unique identifier for the tool call
        tool_name: Name of the tool being called
        args: Complete tool arguments dictionary
        
    Returns:
        AISDKPartEmitter for tool call part
    """
    return create_ai_sdk_part('9', {
        'toolCallId': tool_call_id,
        'toolName': tool_name,
        'args': args
    })


def create_tool_result_part(tool_call_id: str, result: Any) -> AISDKPartEmitter:
    """
    Create tool result part (a:) for tool execution results
    
    Args:
        tool_call_id: Unique identifier for the tool call
        result: Result of tool execution
        
    Returns:
        AISDKPartEmitter for tool result part
    """
    return create_ai_sdk_part('a', {'toolCallId': tool_call_id, 'result': result})


def create_start_step_part(message_id: str) -> AISDKPartEmitter:
    """
    Create start step part (f:) to mark beginning of processing step
    
    Args:
        message_id: Unique identifier for the message
        
    Returns:
        AISDKPartEmitter for start step part
    """
    return create_ai_sdk_part('f', {'messageId': message_id})


def create_finish_step_part(
    finish_reason: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    is_continued: bool = False
) -> AISDKPartEmitter:
    """
    Create finish step part (e:) to mark end of processing step
    
    Args:
        finish_reason: Reason for finishing the step
        prompt_tokens: Number of prompt tokens used
        completion_tokens: Number of completion tokens used
        is_continued: Whether the step will be continued
        
    Returns:
        AISDKPartEmitter for finish step part
    """
    usage = UsageInfo(promptTokens=prompt_tokens, completionTokens=completion_tokens)
    return create_ai_sdk_part('e', {
        'finishReason': finish_reason,
        'usage': usage.model_dump(),
        'isContinued': is_continued
    })


def create_finish_message_part(
    finish_reason: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0
) -> AISDKPartEmitter:
    """
    Create finish message part (d:) to mark completion of entire message
    
    Args:
        finish_reason: Reason for finishing the message
        prompt_tokens: Number of prompt tokens used
        completion_tokens: Number of completion tokens used
        
    Returns:
        AISDKPartEmitter for finish message part
    """
    usage = UsageInfo(promptTokens=prompt_tokens, completionTokens=completion_tokens)
    return create_ai_sdk_part('d', {
        'finishReason': finish_reason,
        'usage': usage.model_dump()
    })