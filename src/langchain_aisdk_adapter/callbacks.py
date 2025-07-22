#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI SDK Compatible Callback System

Implements callback handlers that are fully compatible with AI SDK standards.
"""

from typing import Dict, Any, List, Optional, Union, Literal, Protocol
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
    name: str
    mimeType: str
    data: str

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