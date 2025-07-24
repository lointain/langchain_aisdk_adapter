#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Message Builder for AI SDK Protocol

This module provides the MessageBuilder class for constructing Message objects
from UIMessageChunk events according to AI SDK protocol.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .callbacks import (
    Message,
    UIPart,
    TextUIPart,
    ToolInvocationUIPart,
    ToolInvocation,
    StepStartUIPart,
)
from .models import UIMessageChunk


class MessageBuilder:
    """Builds Message objects from UIMessageChunk events.
    
    This class accumulates UIMessageChunk events and constructs
    a complete Message object with proper parts and content.
    """

    def __init__(self, message_id: Optional[str] = None):
        self.message_id = message_id or str(uuid.uuid4())
        self.parts: List[UIPart] = []
        self.content = ""
        self.created_at = datetime.now()
        self._lock = asyncio.Lock()
        self._new_parts: List[UIPart] = []
        self._current_text_parts = {}  # Track current TextUIPart objects by ID

    async def add_chunk(self, chunk: UIMessageChunk) -> List[UIPart]:
        """Add a UIMessageChunk and update the message state.
        
        This method processes all types of UIMessageChunk and converts them to appropriate UIPart objects.
        It ensures that parts are accumulated regardless of auto_events setting.
        
        Returns:
            List of newly generated UIPart objects for this chunk.
        """
        async with self._lock:
            self._new_parts.clear()  # Clear previous new parts
            chunk_type = chunk.get("type")
            
            # Skip chunks without type
            if not chunk_type:
                return self._new_parts.copy()
            
            # Handle step-related chunks
            if chunk_type == "start-step":
                # Import StepStartUIPart here to avoid circular imports
                from .callbacks import StepStartUIPart
                # Only create step-start part if we don't already have one at the end
                # This prevents duplicate step-start parts
                if not self.parts or not (hasattr(self.parts[-1], 'type') and self.parts[-1].type == "step-start"):
                    part = StepStartUIPart(type="step-start")
                    self.parts.append(part)
                    self._new_parts.append(part)
            
            # Handle text-related chunks - SIMPLIFIED IMPLEMENTATION
            elif chunk_type == "text-start":
                # text-start: 创建新的TextUIPart对象
                text_id = chunk.get("id", "default")
                text_part = TextUIPart(type="text", text="")
                self._current_text_parts[text_id] = text_part
            elif chunk_type == "text-delta":
                # text-delta: 累加text到对应的TextUIPart对象
                text_id = chunk.get("id", "default")
                delta = chunk.get("textDelta", chunk.get("delta", ""))
                if text_id in self._current_text_parts:
                    self._current_text_parts[text_id].text += delta
            elif chunk_type == "text-end":
                # text-end: 将TextUIPart对象放入parts数组
                text_id = chunk.get("id", "default")
                if text_id in self._current_text_parts:
                    text_part = self._current_text_parts[text_id]
                    if text_part.text.strip():  # 只有非空文本才添加
                        self.parts.append(text_part)
                        self._new_parts.append(text_part)
                    # 清理当前文本部分
                    del self._current_text_parts[text_id]
            
            # Handle tool-related chunks
            elif chunk_type == "tool-input-start":
                # Tool input start doesn't create a part immediately
                pass
            elif chunk_type == "tool-input-delta":
                # Tool input delta doesn't create a part immediately
                pass
            elif chunk_type == "tool-input-available":
                # Store tool information but don't create part yet
                # Only create part when tool has result (state="result")
                tool_call_id = chunk.get("toolCallId", "")
                tool_input = chunk.get("input", {})
                args_dict = tool_input if isinstance(tool_input, dict) else {"input": tool_input}
                
                # Store tool info for later use
                if not hasattr(self, '_pending_tools'):
                    self._pending_tools = {}
                self._pending_tools[tool_call_id] = {
                    "toolName": chunk.get("toolName", ""),
                    "args": args_dict
                }
            elif chunk_type == "tool-output-available":
                # Create tool invocation part only when result is available
                tool_call_id = chunk.get("toolCallId", "")
                if hasattr(self, '_pending_tools') and tool_call_id in self._pending_tools:
                    # Check if we already have a tool invocation with this toolCallId
                    existing_tool = None
                    for part in self.parts:
                        if (hasattr(part, 'toolInvocation') and 
                            hasattr(part.toolInvocation, 'toolCallId') and 
                            part.toolInvocation.toolCallId == tool_call_id):
                            existing_tool = part
                            break
                    
                    # Only create new tool invocation if it doesn't exist
                    if existing_tool is None:
                        tool_info = self._pending_tools[tool_call_id]
                        
                        # Calculate step number based on existing tool invocations
                        step = sum(1 for part in self.parts if hasattr(part, 'toolInvocation'))
                        
                        tool_invocation = ToolInvocation(
                            state="result",
                            step=step,
                            toolCallId=tool_call_id,
                            toolName=tool_info["toolName"],
                            args=tool_info["args"],
                            result=chunk.get("output")
                        )
                        part = ToolInvocationUIPart(
                            type="tool-invocation",
                            toolInvocation=tool_invocation
                        )
                        self.parts.append(part)
                        self._new_parts.append(part)
                        
                        # Clean up pending tool
                        del self._pending_tools[tool_call_id]
            elif chunk_type == "tool-output-error":
                # Create tool invocation part with error state
                tool_call_id = chunk.get("toolCallId", "")
                if hasattr(self, '_pending_tools') and tool_call_id in self._pending_tools:
                    # Check if we already have a tool invocation with this toolCallId
                    existing_tool = None
                    for part in self.parts:
                        if (hasattr(part, 'toolInvocation') and 
                            hasattr(part.toolInvocation, 'toolCallId') and 
                            part.toolInvocation.toolCallId == tool_call_id):
                            existing_tool = part
                            break
                    
                    # Only create new tool invocation if it doesn't exist
                    if existing_tool is None:
                        tool_info = self._pending_tools[tool_call_id]
                        
                        # Calculate step number based on existing tool invocations
                        step = sum(1 for part in self.parts if hasattr(part, 'toolInvocation'))
                        
                        tool_invocation = ToolInvocation(
                            state="error",
                            step=step,
                            toolCallId=tool_call_id,
                            toolName=tool_info["toolName"],
                            args=tool_info["args"],
                            result={"error": chunk.get("errorText", "Unknown error")}
                        )
                        part = ToolInvocationUIPart(
                            type="tool-invocation",
                            toolInvocation=tool_invocation
                        )
                        self.parts.append(part)
                        self._new_parts.append(part)
                        
                        # Clean up pending tool
                        del self._pending_tools[tool_call_id]
            
            # Handle reasoning chunks (if supported in the future)
            elif chunk_type in ["reasoning", "reasoning-start", "reasoning-delta", "reasoning-end"]:
                # For now, treat reasoning as text content
                if chunk_type == "reasoning":
                    reasoning_text = chunk.get("text", "")
                    if reasoning_text:
                        self.content += reasoning_text
                        part = TextUIPart(type="text", text=reasoning_text)
                        self.parts.append(part)
                        self._new_parts.append(part)
            
            # Handle file chunks
            elif chunk_type == "file":
                # Import FileUIPart here to avoid circular imports
                from .callbacks import FileUIPart
                # Map from AI SDK file chunk to FileUIPart
                file_part = FileUIPart(
                    type="file",
                    url=chunk.get("url", ""),
                    mediaType=chunk.get("mediaType", "")
                )
                self.parts.append(file_part)
                self._new_parts.append(file_part)
            
            # Handle data chunks (type can be any string starting with 'data-')
            elif chunk_type.startswith("data") or chunk_type == "data":
                # Data chunks don't have a specific UIPart type in the current implementation
                # They are handled as metadata or ignored for now
                pass
            
            # Handle source chunks
            elif chunk_type == "source-url":
                # Import SourceUIPart here to avoid circular imports
                from .callbacks import SourceUIPart
                # Map from AI SDK source chunk to SourceUIPart
                source_data = {
                    "url": chunk.get("url", ""),
                    "description": chunk.get("description")
                }
                source_part = SourceUIPart(
                    type="source",
                    source=source_data
                )
                self.parts.append(source_part)
                self._new_parts.append(source_part)
            elif chunk_type == "source-document":
                # Import SourceUIPart here to avoid circular imports
                from .callbacks import SourceUIPart
                # Map from AI SDK source document chunk to SourceUIPart
                source_data = {
                    "sourceId": chunk.get("sourceId", ""),
                    "mediaType": chunk.get("mediaType", ""),
                    "title": chunk.get("title", ""),
                    "filename": chunk.get("filename")
                }
                source_part = SourceUIPart(
                    type="source",
                    source=source_data
                )
                self.parts.append(source_part)
                self._new_parts.append(source_part)
            
            # Handle error chunks
            elif chunk_type == "error":
                # Import ErrorUIPart here to avoid circular imports
                from .callbacks import ErrorUIPart
                error_part = ErrorUIPart(
                    type="error",
                    error=chunk.get("errorText", "Error occurred")
                )
                self.parts.append(error_part)
                self._new_parts.append(error_part)
                # Also add to content for backward compatibility
                error_text = chunk.get("errorText", "Error occurred")
                self.content += f"\n[Error: {error_text}]"
            
            # Handle other chunk types (start, finish, etc.)
            elif chunk_type in ["start", "finish", "finish-step", "abort"]:
                # These don't create parts but are important for protocol flow
                pass
            
            return self._new_parts.copy()

    def build_message(self) -> Message:
        """Build the final Message object."""
        return Message(
            id=self.message_id,
            createdAt=self.created_at,
            content=self.content,
            role="assistant",
            parts=self.parts.copy()
        )