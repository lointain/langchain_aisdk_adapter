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

class ReasoningUIPart(BaseModel):
    """Reasoning UI part for displaying AI reasoning process"""
    type: Literal['reasoning'] = 'reasoning'
    reasoning: str

class ToolInvocation(BaseModel):
    """Tool invocation object containing call/result information"""
    state: Literal['partial-call', 'call', 'result']
    step: Optional[int] = None
    toolCallId: str
    toolName: str
    args: Dict[str, Any]
    result: Optional[Any] = None

class ToolInvocationUIPart(BaseModel):
    """Tool invocation UI part for displaying tool calls"""
    type: Literal['tool-invocation'] = 'tool-invocation'
    toolInvocation: ToolInvocation

class SourceUIPart(BaseModel):
    """Source UI part for displaying referenced sources"""
    type: Literal['source'] = 'source'
    source: Dict[str, Any]  # LanguageModelV1Source

class FileUIPart(BaseModel):
    """File UI part for displaying file attachments"""
    type: Literal['file'] = 'file'
    url: str
    mediaType: str

class StepStartUIPart(BaseModel):
    """Step start UI part for marking workflow step beginning"""
    type: Literal['step-start'] = 'step-start'

class StepStartUIPart(BaseModel):
    """Step start UI part for marking workflow step beginning"""
    type: Literal['step-start'] = 'step-start'

class ErrorUIPart(BaseModel):
    """Error UI part for displaying error information"""
    type: Literal['error'] = 'error'
    error: str

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

# Attachment definition
class Attachment(BaseModel):
    """File attachment information"""
    name: Optional[str] = None
    contentType: Optional[str] = None
    url: str

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


class BaseAICallbackHandler(ABC):
    """Abstract base class for AI SDK callbacks with pure hook mode.

    This class defines a set of optional, independent hooks that can be implemented
    to observe the lifecycle of an AI SDK stream. Implementations should be
    self-contained and not call super() methods.
    """

    async def on_finish(self, message: Message, options: Dict[str, Any]) -> None:
        """Called when the stream processing is complete.

        This is a pure hook for observation and should not modify the input.

        Args:
            message: The final, complete Message object.
            options: A dictionary containing metadata, such as 'usage'.
        """
        pass

    async def on_error(self, error: Exception) -> None:
        """Called when an unhandled error occurs during stream processing.

        This is a pure hook for logging or error handling.

        Args:
            error: The exception that was raised.
        """
        pass