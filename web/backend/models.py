from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum

class StreamMode(str, Enum):
    """Stream processing mode"""
    AUTO = "auto"
    MANUAL = "manual"

class ChatMessage(BaseModel):
    """Chat message model"""
    role: str = Field(..., description="Message role: user, assistant, system")
    content: str = Field(..., description="Message content")

class ChatRequest(BaseModel):
    """Chat request model - compatible with @ai-sdk/vue useChat format"""
    messages: List[ChatMessage] = Field(..., description="Messages array from useChat")
    message_id: Optional[str] = Field(None, description="Optional message ID")
    stream_mode: StreamMode = Field(StreamMode.AUTO, description="Stream processing mode")
    agent_config: Optional[Dict[str, Any]] = Field(None, description="Agent configuration")

class ChatResponse(BaseModel):
    """Chat response model"""
    message_id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    message_id: Optional[str] = None