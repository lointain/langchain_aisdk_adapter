#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Protocol Strategy for AI SDK v4 and v5 Support

This module implements the strategy pattern to support both AI SDK v4 and v5 protocols.
"""

import json
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

from .models import UIMessageChunk


class ProtocolStrategy(ABC):
    """Abstract base class for protocol strategies."""
    
    @abstractmethod
    def get_headers(self) -> Dict[str, str]:
        """Get protocol-specific response headers."""
        pass
    
    @abstractmethod
    def format_chunk(self, chunk: UIMessageChunk) -> str:
        """Format message chunk to protocol-specific format."""
        pass
    
    @abstractmethod
    def convert_text_sequence(self, text_chunks: List[str]) -> List[UIMessageChunk]:
        """Convert text sequence to protocol-specific message chunks."""
        pass
    
    @abstractmethod
    def get_termination_marker(self) -> str:
        """Get protocol-specific stream termination marker."""
        pass


class AISDKv4Strategy(ProtocolStrategy):
    """AI SDK v4 protocol strategy implementation."""
    
    def get_headers(self) -> Dict[str, str]:
        """Get AI SDK v4 response headers."""
        return {
            "Content-Type": "text/plain; charset=utf-8",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "x-vercel-ai-data-stream": "v1"
        }
    
    def format_chunk(self, chunk: UIMessageChunk) -> str:
        """Format chunk to AI SDK v4 format."""
        chunk_type = chunk.get("type")
        
        if chunk_type == "text-delta":
            delta = chunk.get("delta", "")
            # Escape quotes and format as v4 text part
            escaped_delta = json.dumps(delta, ensure_ascii=False)
            return f"0:{escaped_delta}\n"
        
        elif chunk_type == "text-start":
            # v4 doesn't have explicit text-start, handled by first text-delta
            return ""
        
        elif chunk_type == "text-end":
            # v4 doesn't have explicit text-end, handled by next non-text chunk
            return ""
        
        elif chunk_type == "start-step":
            message_id = chunk.get("messageId", "")
            return f'f:{{"messageId":"{message_id}"}}\n'
        
        elif chunk_type == "finish-step":
            finish_reason = chunk.get("finishReason", "stop")
            usage = chunk.get("usage", {})
            is_continued = chunk.get("isContinued", False)
            step_data = {
                "finishReason": finish_reason,
                "usage": usage,
                "isContinued": is_continued
            }
            return f'e:{json.dumps(step_data, ensure_ascii=False)}\n'
        
        elif chunk_type == "tool-input-start":
            tool_call_id = chunk.get("toolCallId", "")
            tool_name = chunk.get("toolName", "")
            tool_data = {
                "toolCallId": tool_call_id,
                "toolName": tool_name
            }
            return f'b:{json.dumps(tool_data, ensure_ascii=False)}\n'
        
        elif chunk_type == "tool-input-delta":
            tool_call_id = chunk.get("toolCallId", "")
            delta = chunk.get("delta", "")
            tool_data = {
                "toolCallId": tool_call_id,
                "argsTextDelta": delta
            }
            return f'c:{json.dumps(tool_data, ensure_ascii=False)}\n'
        
        elif chunk_type == "tool-input-available":
            tool_call_id = chunk.get("toolCallId", "")
            tool_name = chunk.get("toolName", "")
            args = chunk.get("input", {})
            tool_data = {
                "toolCallId": tool_call_id,
                "toolName": tool_name,
                "args": args
            }
            return f'9:{json.dumps(tool_data, ensure_ascii=False)}\n'
        
        elif chunk_type == "tool-output-available":
            tool_call_id = chunk.get("toolCallId", "")
            result = chunk.get("output", "")
            tool_data = {
                "toolCallId": tool_call_id,
                "result": result
            }
            return f'a:{json.dumps(tool_data, ensure_ascii=False)}\n'
        
        elif chunk_type == "data":
            data = chunk.get("data", [])
            return f'2:{json.dumps(data, ensure_ascii=False)}\n'
        
        elif chunk_type == "error":
            error_text = chunk.get("errorText", "")
            escaped_error = json.dumps(error_text, ensure_ascii=False)
            return f'3:{escaped_error}\n'
        
        elif chunk_type == "reasoning":
            text = chunk.get("text", "")
            escaped_text = json.dumps(text, ensure_ascii=False)
            return f'g:{escaped_text}\n'
        
        elif chunk_type == "source-url":
            source_data = {
                "sourceType": "url",
                "id": chunk.get("sourceId", ""),
                "url": chunk.get("url", ""),
                "title": chunk.get("title", "")
            }
            return f'h:{json.dumps(source_data, ensure_ascii=False)}\n'
        
        elif chunk_type == "file":
            file_data = {
                "data": chunk.get("data", ""),
                "mimeType": chunk.get("mediaType", "")
            }
            return f'k:{json.dumps(file_data, ensure_ascii=False)}\n'
        
        elif chunk_type == "finish":
            finish_reason = chunk.get("finishReason", "stop")
            usage = chunk.get("usage", {})
            finish_data = {
                "finishReason": finish_reason,
                "usage": usage
            }
            return f'd:{json.dumps(finish_data, ensure_ascii=False)}\n'
        
        # For unsupported chunk types, return empty string
        return ""
    
    def convert_text_sequence(self, text_chunks: List[str]) -> List[UIMessageChunk]:
        """Convert text sequence to v4 format (only text-delta chunks)."""
        return [{
            "type": "text-delta",
            "id": "text-v4",  # v4 doesn't need real IDs
            "delta": chunk
        } for chunk in text_chunks]
    
    def get_termination_marker(self) -> str:
        """Get v4 termination marker."""
        return 'd:{"finishReason":"stop","usage":{}}\n'


class AISDKv5Strategy(ProtocolStrategy):
    """AI SDK v5 protocol strategy implementation."""
    
    def get_headers(self) -> Dict[str, str]:
        """Get AI SDK v5 response headers."""
        return {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "x-vercel-ai-ui-message-stream": "v1"
        }
    
    def format_chunk(self, chunk: UIMessageChunk) -> str:
        """Format chunk to AI SDK v5 SSE format."""
        # Convert chunk to JSON string
        if hasattr(chunk, 'dict'):
            chunk_dict = chunk.dict()
        elif isinstance(chunk, dict):
            chunk_dict = chunk
        else:
            chunk_dict = {"type": "error", "errorText": "Invalid chunk format"}
        
        # Format as SSE: data: CONTENT_JSON\n\n
        json_str = json.dumps(chunk_dict, ensure_ascii=False)
        return f"data: {json_str}\n\n"
    
    def convert_text_sequence(self, text_chunks: List[str]) -> List[UIMessageChunk]:
        """Convert text sequence to v5 format (start/delta/end sequence)."""
        text_id = str(uuid.uuid4())
        chunks = [{"type": "text-start", "id": text_id}]
        
        for chunk in text_chunks:
            chunks.append({
                "type": "text-delta",
                "id": text_id,
                "delta": chunk
            })
        
        chunks.append({"type": "text-end", "id": text_id})
        return chunks
    
    def get_termination_marker(self) -> str:
        """Get v5 termination marker."""
        return "data: [DONE]\n\n"


class ProtocolConfig:
    """Protocol configuration manager."""
    
    def __init__(self, version: str = "v5"):
        """Initialize protocol configuration.
        
        Args:
            version: Protocol version ("v4" or "v5")
        """
        self.version = version
        self.strategy = self._create_strategy()
    
    def _create_strategy(self) -> ProtocolStrategy:
        """Create protocol strategy based on version."""
        if self.version == "v4":
            return AISDKv4Strategy()
        elif self.version == "v5":
            return AISDKv5Strategy()
        else:
            raise ValueError(f"Unsupported protocol version: {self.version}")