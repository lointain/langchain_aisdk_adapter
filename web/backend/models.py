from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union
from enum import Enum

class StreamMode(str, Enum):
    """Stream processing mode"""
    AUTO = "auto"
    MANUAL = "manual"

class MessagePart(BaseModel):
    """Message part for AI SDK v5 format"""
    type: str = Field(..., description="Part type: text, image, etc.")
    text: Optional[str] = Field(None, description="Text content for text parts")
    image: Optional[str] = Field(None, description="Image URL for image parts")

class ChatMessage(BaseModel):
    """Chat message model - supports both legacy and AI SDK v5 formats"""
    id: Optional[str] = Field(None, description="Message ID")
    role: str = Field(..., description="Message role: user, assistant, system")
    content: Optional[str] = Field(None, description="Message content (legacy format)")
    parts: Optional[List[MessagePart]] = Field(None, description="Message parts (AI SDK v5 format)")
    
    def get_content(self) -> str:
        """Extract content from either format"""
        if self.content:
            return self.content
        elif self.parts:
            # Extract text from parts array
            text_parts = [part.text for part in self.parts if part.type == "text" and part.text]
            return " ".join(text_parts)
        return ""

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