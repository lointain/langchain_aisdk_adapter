#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI SDK Stream Protocol Pydantic Model Definitions

Based on Vercel AI SDK UI Stream Protocol specification
Reference: https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, RootModel


class TextPartContent(RootModel[str]):
    """Text part (0:) - Text content appended to the message"""
    root: str


class ReasoningPartContent(RootModel[str]):
    """Reasoning part (g:) - AI model's reasoning process"""
    root: str


class RedactedReasoningPartContent(BaseModel):
    """Redacted reasoning part (i:) - Edited reasoning data"""
    data: str


class ReasoningSignaturePartContent(BaseModel):
    """Reasoning signature part (j:) - Signature verification for reasoning"""
    signature: str


class SourcePartContent(BaseModel):
    """Source part (h:) - Referenced external resources"""
    url: str
    title: Optional[str] = None


class FilePartContent(BaseModel):
    """File part (k:) - Base64 encoded binary files"""
    data: str = Field(description="Base64 encoded binary data")
    mimeType: str = Field(description="MIME type, e.g. 'image/png'")


class DataPartContent(RootModel[List[Any]]):
    """Data part (2:) - Custom JSON data array"""
    root: List[Any]


class MessageAnnotationPartContent(RootModel[List[Any]]):
    """Message annotation part (8:) - Metadata annotations for messages"""
    root: List[Any]


class ErrorPartContent(RootModel[str]):
    """Error part (3:) - Error information"""
    root: str


class ToolCallStreamingStartPartContent(BaseModel):
    """Tool call streaming start part (b:) - Marks the beginning of tool call"""
    toolCallId: str
    toolName: str


class ToolCallDeltaPartContent(BaseModel):
    """Tool call delta part (c:) - Incremental updates for tool parameters"""
    toolCallId: str
    argsTextDelta: str


class ToolCallPartContent(BaseModel):
    """Tool call part (9:) - Complete tool call information"""
    toolCallId: str
    toolName: str
    args: Dict[str, Any]


class ToolResultPartContent(BaseModel):
    """Tool result part (a:) - Results of tool execution"""
    toolCallId: str
    result: Any


class StartStepPartContent(BaseModel):
    """Start step part (f:) - Marks the beginning of processing step"""
    messageId: str


class UsageInfo(BaseModel):
    """Token usage statistics information"""
    promptTokens: int = 0
    completionTokens: int = 0


class FinishStepPartContent(BaseModel):
    """Finish step part (e:) - Marks the end of processing step"""
    finishReason: str
    usage: UsageInfo
    isContinued: bool = False


class FinishMessagePartContent(BaseModel):
    """Finish message part (d:) - Marks the completion of entire message"""
    finishReason: str
    usage: UsageInfo


# AI SDK Protocol Type Mapping
# Define AI SDK protocol type IDs and corresponding Pydantic models
AI_SDK_PROTOCOL_MAP = {
    "0": TextPartContent,
    "g": ReasoningPartContent,
    "i": RedactedReasoningPartContent,
    "j": ReasoningSignaturePartContent,
    "h": SourcePartContent,
    "k": FilePartContent,
    "2": DataPartContent,
    "8": MessageAnnotationPartContent,
    "3": ErrorPartContent,
    "b": ToolCallStreamingStartPartContent,
    "c": ToolCallDeltaPartContent,
    "9": ToolCallPartContent,
    "a": ToolResultPartContent,
    "f": StartStepPartContent,
    "e": FinishStepPartContent,
    "d": FinishMessagePartContent,
}


# AI SDK Protocol Prefixes
AI_SDK_PROTOCOL_PREFIXES = tuple(AI_SDK_PROTOCOL_MAP.keys())