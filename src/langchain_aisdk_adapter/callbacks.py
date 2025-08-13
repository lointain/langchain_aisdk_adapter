#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI SDK Compatible Callback System

Implements callback handlers that are fully compatible with AI SDK standards.
"""

from typing import Dict, Any, List, Optional, Union, Literal, Protocol, Callable, Awaitable
from pydantic import BaseModel, Field
from datetime import datetime
from abc import ABC, abstractmethod

# Basic JSON types
JSONValue = Union[None, str, int, float, bool, Dict[str, Any], List[Any]]

# Usage statistics
class LanguageModelUsage(BaseModel):
    """Language model token usage statistics"""
    promptTokens: int
    completionTokens: int
    totalTokens: int

# UIPart type definitions
class TextUIPart(BaseModel):
    """Text UI part for displaying text content"""
    type: Literal['text'] = 'text'
    text: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization, excluding internal fields"""
        return {
            "type": self.type,
            "text": self.text
        }

class ReasoningUIPart(BaseModel):
    """Reasoning UI part for displaying AI reasoning process"""
    type: Literal['reasoning'] = 'reasoning'
    reasoning: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization, excluding internal fields"""
        return {
            "type": self.type,
            "reasoning": self.reasoning
        }

class ToolInvocation(BaseModel):
    """Tool invocation object containing call/result information"""
    state: Literal['partial-call', 'call', 'result']
    step: Optional[int] = None
    toolCallId: str
    toolName: str
    args: Dict[str, Any]
    result: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization, excluding internal fields"""
        result = {
            "state": self.state,
            "toolCallId": self.toolCallId,
            "toolName": self.toolName,
            "args": self.args
        }
        if self.step is not None:
            result["step"] = self.step
        if self.result is not None:
            result["result"] = self.result
        return result

class ToolInvocationUIPart(BaseModel):
    """Tool invocation UI part for displaying tool calls"""
    type: Literal['tool-invocation'] = 'tool-invocation'
    toolInvocation: ToolInvocation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization, excluding internal fields"""
        return {
            "type": self.type,
            "toolInvocation": self.toolInvocation.to_dict()
        }

class SourceUIPart(BaseModel):
    """Source UI part for displaying referenced sources"""
    type: Literal['source'] = 'source'
    source: Dict[str, Any]  # LanguageModelV1Source
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization, excluding internal fields"""
        return {
            "type": self.type,
            "source": self.source
        }

class FileUIPart(BaseModel):
    """File UI part for displaying file attachments"""
    type: Literal['file'] = 'file'
    url: str
    mediaType: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization, excluding internal fields"""
        return {
            "type": self.type,
            "url": self.url,
            "mediaType": self.mediaType
        }

class StepStartUIPart(BaseModel):
    """Step start UI part for marking workflow step beginning"""
    type: Literal['step-start'] = 'step-start'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization, excluding internal fields"""
        return {
            "type": self.type
        }

class ErrorUIPart(BaseModel):
    """Error UI part for displaying error information"""
    type: Literal['error'] = 'error'
    error: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization, excluding internal fields"""
        return {
            "type": self.type,
            "error": self.error
        }

# UIPart union type
UIPart = Union[
    TextUIPart,
    ReasoningUIPart,
    ToolInvocationUIPart,
    SourceUIPart,
    FileUIPart,
    StepStartUIPart,
    ErrorUIPart,
]


def serialize_ui_part(part: UIPart) -> Dict[str, Any]:
    """Serialize a UIPart to dictionary for JSON serialization.
    
    This function calls the appropriate to_dict method for each UIPart type,
    ensuring clean JSON serialization without internal Pydantic fields.
    
    Args:
        part: The UIPart instance to serialize
        
    Returns:
        Dictionary representation suitable for JSON serialization
    """
    if hasattr(part, 'to_dict'):
        return part.to_dict()
    else:
        # Fallback for any UIPart that doesn't have to_dict method
        return part.model_dump(exclude_unset=True, exclude_none=True)

# Attachment definition
class Attachment(BaseModel):
    """File attachment information"""
    name: Optional[str] = None
    contentType: Optional[str] = None
    url: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization, excluding internal fields"""
        result = {"url": self.url}
        if self.name is not None:
            result["name"] = self.name
        if self.contentType is not None:
            result["contentType"] = self.contentType
        return result

# Message definition
class Message(BaseModel):
    """AI SDK compatible message structure"""
    id: str
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    content: str
    role: Literal['system', 'user', 'assistant', 'data']
    annotations: Optional[List[JSONValue]] = None
    parts: List[UIPart] = Field(default_factory=list)
    experimental_attachments: Optional[List[Attachment]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization, excluding internal fields"""
        result = {
            "id": self.id,
            "createdAt": self.createdAt.isoformat(),
            "content": self.content,
            "role": self.role,
            "parts": [serialize_ui_part(part) for part in self.parts]
        }
        if self.annotations is not None:
            result["annotations"] = self.annotations
        if self.experimental_attachments is not None:
            result["experimental_attachments"] = [att.to_dict() for att in self.experimental_attachments]
        return result
    
    def to_json(self, **kwargs) -> str:
        """Convert to JSON string for easy serialization to database
        
        Args:
            **kwargs: Additional arguments passed to json.dumps (e.g., indent, ensure_ascii)
            
        Returns:
            JSON string representation of the message
        """
        import json
        return json.dumps(self.to_dict(), **kwargs)
    
    def get_serialized_parts(self) -> List[Dict[str, Any]]:
        """Get serialized parts as list of dictionaries for database storage
        
        Returns:
            List of serialized UIPart dictionaries
        """
        return [serialize_ui_part(part) for part in self.parts]

class StreamCallbacks:
    """Configuration options and helper callback methods for stream lifecycle events.
    
    This class provides a simple callback interface for basic stream events,
    compatible with the original adapter functionality.
    """

    def __init__(
        self,
        on_start: Optional[Callable[[], Union[None, Awaitable[None]]]] = None,
        on_final: Optional[Callable[[str], Union[None, Awaitable[None]]]] = None,
        on_token: Optional[Callable[[str], Union[None, Awaitable[None]]]] = None,
        on_text: Optional[Callable[[str], Union[None, Awaitable[None]]]] = None,
    ):
        """Initialize stream callbacks.
        
        Args:
            on_start: Called once when the stream is initialized
            on_final: Called once when the stream is closed with the final completion message
            on_token: Called for each tokenized message
            on_text: Called for each text chunk
        """
        self.on_start = on_start
        self.on_final = on_final
        self.on_token = on_token
        self.on_text = on_text


class CallbacksTransformer:
    """Transform stream that handles callbacks during stream processing.
    
    This class processes a stream of text chunks and invokes optional callback functions
    at different stages of the stream's lifecycle:
    - on_start: Called once when the stream is initialized
    - on_token: Called for each tokenized message
    - on_text: Called for each text chunk
    - on_final: Called once when the stream is closed with the final completion message
    """

    def __init__(self, callbacks: Optional[StreamCallbacks] = None):
        """Initialize the callbacks transformer.
        
        Args:
            callbacks: Optional callbacks configuration
        """
        self.callbacks = callbacks or StreamCallbacks()
        self.aggregated_response = ""
        self._started = False

    async def _call_callback(
        self, 
        callback: Optional[Callable], 
        *args
    ) -> None:
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

    async def start(self) -> None:
        """Initialize the transformer and call on_start callback."""
        if not self._started:
            self._started = True
            await self._call_callback(self.callbacks.on_start)

    async def transform(self, chunk: str) -> str:
        """Transform a text chunk and call relevant callbacks.
        
        Args:
            chunk: Text chunk to process
            
        Returns:
            The same text chunk (pass-through)
        """
        if not self._started:
            await self.start()

        self.aggregated_response += chunk

        await self._call_callback(self.callbacks.on_token, chunk)
        await self._call_callback(self.callbacks.on_text, chunk)

        return chunk

    async def finish(self) -> None:
        """Finalize the transformer and call on_final callback."""
        await self._call_callback(self.callbacks.on_final, self.aggregated_response)


# Step information for multi-step operations
class StepResult(BaseModel):
    """Result of a completed step in multi-step generation"""
    stepType: Literal['initial', 'continue', 'tool-call', 'tool-result']
    text: str
    toolCalls: List[Dict[str, Any]] = Field(default_factory=list)
    toolResults: List[Dict[str, Any]] = Field(default_factory=list)
    finishReason: Optional[Literal['stop', 'length', 'content-filter', 'tool-calls', 'error', 'other', 'unknown']] = None
    usage: Optional[LanguageModelUsage] = None
    warnings: Optional[List[str]] = None
    logprobs: Optional[Dict[str, Any]] = None
    providerMetadata: Optional[Dict[str, Any]] = None

# Chunk information for streaming
class TextChunk(BaseModel):
    """Text chunk from streaming generation"""
    type: Literal['text-delta'] = 'text-delta'
    textDelta: str

class ToolCallChunk(BaseModel):
    """Tool call chunk from streaming generation"""
    type: Literal['tool-call-delta'] = 'tool-call-delta'
    toolCallType: Literal['function']
    toolCallId: str
    toolName: str
    argsTextDelta: str

class ToolResultChunk(BaseModel):
    """Tool result chunk from streaming generation"""
    type: Literal['tool-result'] = 'tool-result'
    toolCallId: str
    toolName: str
    result: Any
    isError: bool = False

class FinishChunk(BaseModel):
    """Finish chunk indicating end of generation"""
    type: Literal['finish'] = 'finish'
    finishReason: Literal['stop', 'length', 'content-filter', 'tool-calls', 'error', 'other', 'unknown']
    usage: Optional[LanguageModelUsage] = None
    logprobs: Optional[Dict[str, Any]] = None

class ErrorChunk(BaseModel):
    """Error chunk indicating an error occurred"""
    type: Literal['error'] = 'error'
    error: str

# Union type for all chunk types
StreamChunk = Union[TextChunk, ToolCallChunk, ToolResultChunk, FinishChunk, ErrorChunk]

class BaseAICallbackHandler(ABC):
    """Abstract base class for AI SDK callbacks with comprehensive event handling.

    This class defines a set of optional, independent hooks that can be implemented
    to observe the lifecycle of an AI SDK stream. Implementations should be
    self-contained and not call super() methods.
    
    Based on AI SDK streamText callback interface:
    https://ai-sdk.dev/docs/reference/ai-sdk-core/stream-text
    """

    async def on_chunk(self, chunk: StreamChunk) -> None:
        """Called for each chunk in the stream.
        
        This callback is invoked for every chunk that is streamed, including:
        - Text deltas (textDelta)
        - Tool call deltas (toolCallDelta) 
        - Tool results (toolResult)
        - Finish events (finish)
        - Error events (error)
        
        Args:
            chunk: The stream chunk containing type-specific data
        """
        pass

    async def on_finish(self, message: Message, options: Dict[str, Any]) -> None:
        """Called when the stream processing is complete.

        This is called when the generation has finished successfully.
        It provides the final message and metadata including usage statistics.

        Args:
            message: The final, complete Message object
            options: Dictionary containing metadata such as:
                - usage: Token usage statistics
                - finishReason: Reason for completion
                - steps: Array of all steps (for multi-step generations)
                - warnings: Any warnings from the provider
                - providerMetadata: Provider-specific metadata
        """
        pass

    async def on_error(self, error: Exception) -> None:
        """Called when an unhandled error occurs during stream processing.

        This is called when an error occurs that prevents the stream from
        completing successfully. The error could be from the model provider,
        network issues, or other unexpected failures.

        Args:
            error: The exception that was raised
        """
        pass
    
    async def on_step_finish(self, step: StepResult) -> None:
        """Called when a step in multi-step generation is finished.
        
        This is called for each completed step in multi-step generations
        (when using tools or continue functionality). It provides access to
        intermediate results including tool calls and their results.
        
        Args:
            step: The completed step containing:
                - stepType: Type of step (initial, continue, tool-call, tool-result)
                - text: Generated text for this step
                - toolCalls: Tool calls made in this step
                - toolResults: Results from tool executions
                - finishReason: Reason this step finished
                - usage: Token usage for this step
        """
        pass
    
    async def on_abort(self, steps: List[StepResult]) -> None:
        """Called when the stream is aborted via AbortSignal.
        
        This is called when the generation is cancelled or aborted before
        completion (e.g., user clicks stop button). It provides access to
        any steps that were completed before the abort.
        
        Note: on_finish is NOT called when on_abort is called.
        
        Args:
            steps: Array of all completed steps before the abort
        """
        pass