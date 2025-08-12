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
    
    This class records UIMessageChunk events as stream history and constructs
    a complete Message object with proper parts generated from stream analysis.
    """

    def __init__(self, message_id: Optional[str] = None):
        self.message_id = message_id or str(uuid.uuid4())
        self.content = ""
        self.created_at = datetime.now()
        self._lock = asyncio.Lock()
        self._stream_history: List[UIMessageChunk] = []  # Record all chunks for later analysis
        self._current_text_parts = {}  # Track current TextUIPart objects by ID

    async def add_chunk(self, chunk: UIMessageChunk) -> List[UIPart]:
        """Add a UIMessageChunk to stream history for later analysis.
        
        This method only records chunks in stream history without generating parts.
        Parts will be generated later by analyzing the complete stream history.
        
        Returns:
            Empty list (parts are generated later from stream history analysis).
        """
        async with self._lock:
            chunk_type = chunk.get("type")
            
            # Skip chunks without type
            if not chunk_type:
                return []
            
            # Record chunk in stream history
            self._stream_history.append(chunk)
            
            # Handle text accumulation for content (backward compatibility)
            if chunk_type == "text-start":
                text_id = chunk.get("id", "default")
                text_part = TextUIPart(type="text", text="")
                self._current_text_parts[text_id] = text_part
            elif chunk_type == "text-delta":
                text_id = chunk.get("id", "default")
                delta = chunk.get("textDelta", chunk.get("delta", ""))
                if text_id in self._current_text_parts:
                    self._current_text_parts[text_id].text += delta
                    self.content += delta  # Update content for backward compatibility
            elif chunk_type == "text-end":
                text_id = chunk.get("id", "default")
                if text_id in self._current_text_parts:
                    # Clean up current text parts
                    del self._current_text_parts[text_id]
            
            # Return empty list - parts will be generated from stream history analysis
            return []

    def build_message(self) -> Message:
        """Build the final message by analyzing stream history to generate parts."""
        # Generate parts from stream history analysis
        parts = self._analyze_stream_history_to_generate_parts()
        
        return Message(
            id=self.message_id,
            createdAt=self.created_at,
            content=self.content,
            role="assistant",
            parts=parts
        )
    
    def _analyze_stream_history_to_generate_parts(self) -> List[UIPart]:
        """Analyze complete stream history to generate parts array.
        
        This method processes the entire stream history to identify and create
        appropriate UIPart objects based on the sequence of UIMessageChunk events.
        """
        parts = []
        pending_tools = {}  # Track tool invocations in progress
        
        for chunk in self._stream_history:
            chunk_type = chunk.get("type", "")
            
            # Handle step-start chunks
            if chunk_type == "step-start" or chunk_type == "start-step":
                # Create step start part
                step_part = StepStartUIPart(type="step-start")
                parts.append(step_part)
            
            # Handle text-related chunks
            elif chunk_type == "text-start":
                # Text start doesn't create a part immediately, but we track it
                pass
            elif chunk_type == "text-delta":
                # Text deltas are accumulated in content, no individual parts needed
                pass
            elif chunk_type == "text-end":
                # Text end creates a consolidated text part with accumulated content
                text_id = chunk.get("id", "default")
                # Find all text deltas for this text_id and combine them
                accumulated_text = ""
                for history_chunk in self._stream_history:
                    if (history_chunk.get("type") == "text-delta" and 
                        history_chunk.get("id") == text_id):
                        accumulated_text += history_chunk.get("delta", history_chunk.get("textDelta", ""))
                
                if accumulated_text:
                    text_part = TextUIPart(type="text", text=accumulated_text)
                    parts.append(text_part)
            elif chunk_type == "text":
                # Direct text chunk - create text part
                text_content = chunk.get("text", "")
                if text_content:
                    text_part = TextUIPart(type="text", text=text_content)
                    parts.append(text_part)
            
            # Handle tool-related chunks
            elif chunk_type == "tool-input-start":
                # Tool input start doesn't create a part immediately
                pass
            elif chunk_type == "tool-input-delta":
                # Tool input delta doesn't create a part immediately
                pass
            elif chunk_type == "tool-input-available":
                # Store tool information for later use
                tool_call_id = chunk.get("toolCallId", "")
                tool_input = chunk.get("input", {})
                args_dict = tool_input if isinstance(tool_input, dict) else {"input": tool_input}
                
                pending_tools[tool_call_id] = {
                    "toolName": chunk.get("toolName", ""),
                    "args": args_dict
                }
            elif chunk_type == "tool-output-available":
                # Create tool invocation part when result is available
                tool_call_id = chunk.get("toolCallId", "")
                if tool_call_id in pending_tools:
                    tool_info = pending_tools[tool_call_id]
                    
                    # Calculate step number based on existing tool invocations
                    step = sum(1 for part in parts if hasattr(part, 'toolInvocation'))
                    
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
                    parts.append(part)
                    
                    # Clean up pending tool
                    del pending_tools[tool_call_id]
            elif chunk_type == "tool-output-error":
                # Create tool invocation part with error state
                tool_call_id = chunk.get("toolCallId", "")
                if tool_call_id in pending_tools:
                    tool_info = pending_tools[tool_call_id]
                    
                    # Calculate step number based on existing tool invocations
                    step = sum(1 for part in parts if hasattr(part, 'toolInvocation'))
                    
                    tool_invocation = ToolInvocation(
                        state="result",
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
                    parts.append(part)
                    
                    # Clean up pending tool
                    del pending_tools[tool_call_id]
            
            # Handle reasoning chunks (if supported in the future)
            elif chunk_type in ["reasoning", "reasoning-start", "reasoning-delta", "reasoning-end"]:
                # For now, treat reasoning as text content
                if chunk_type == "reasoning":
                    reasoning_text = chunk.get("text", "")
                    if reasoning_text:
                        part = TextUIPart(type="text", text=reasoning_text)
                        parts.append(part)
            
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
                parts.append(file_part)
            
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
                parts.append(source_part)
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
                parts.append(source_part)
            
            # Handle error chunks
            elif chunk_type == "error":
                # Import ErrorUIPart here to avoid circular imports
                from .callbacks import ErrorUIPart
                error_part = ErrorUIPart(
                    type="error",
                    error=chunk.get("errorText", "Error occurred")
                )
                parts.append(error_part)
            
            # Handle other chunk types (start, finish, etc.)
            elif chunk_type in ["start", "finish", "finish-step", "abort"]:
                # These don't create parts but are important for protocol flow
                pass
        
        return parts