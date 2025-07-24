#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text Processing Adapter for AI SDK v4 and v5

This module handles the differences in text processing between AI SDK v4 and v5 protocols.
"""

import uuid
from typing import List, Optional

from .models import UIMessageChunk


class TextProcessingAdapter:
    """Text processing adapter for handling v4 and v5 text differences."""
    
    def __init__(self, protocol_version: str):
        """Initialize text processing adapter.
        
        Args:
            protocol_version: Protocol version ("v4" or "v5")
        """
        self.protocol_version = protocol_version
        self.current_text_id: Optional[str] = None
        self.text_buffer: List[str] = []
        self._text_started = False
    
    def process_text_chunk(self, text: str) -> List[UIMessageChunk]:
        """Process text chunk and return protocol-specific message chunks.
        
        Args:
            text: Text content to process
            
        Returns:
            List of UIMessageChunk objects
        """
        if self.protocol_version == "v4":
            # v4: Direct text-delta chunks
            return [{
                "type": "text-delta",
                "id": "text-v4",  # v4 doesn't need real IDs
                "delta": text
            }]
        else:
            # v5: Manage start/delta/end sequence
            chunks = []
            
            if not self._text_started:
                # Start new text sequence
                self.current_text_id = str(uuid.uuid4())
                self._text_started = True
                chunks.append({
                    "type": "text-start",
                    "id": self.current_text_id
                })
            
            # Add text delta
            chunks.append({
                "type": "text-delta",
                "id": self.current_text_id,
                "delta": text
            })
            
            return chunks
    
    def finish_text_sequence(self) -> List[UIMessageChunk]:
        """Finish current text sequence.
        
        Returns:
            List of UIMessageChunk objects (empty for v4, text-end for v5)
        """
        if self.protocol_version == "v5" and self._text_started and self.current_text_id:
            chunk = {
                "type": "text-end",
                "id": self.current_text_id
            }
            self._reset_text_state()
            return [chunk]
        
        self._reset_text_state()
        return []
    
    def _reset_text_state(self):
        """Reset text processing state."""
        self.current_text_id = None
        self.text_buffer.clear()
        self._text_started = False
    
    def is_text_active(self) -> bool:
        """Check if text sequence is currently active.
        
        Returns:
            True if text sequence is active
        """
        return self._text_started
    
    def convert_text_sequence(self, text_chunks: List[str]) -> List[UIMessageChunk]:
        """Convert a complete text sequence to protocol-specific chunks.
        
        Args:
            text_chunks: List of text chunks
            
        Returns:
            List of UIMessageChunk objects
        """
        if self.protocol_version == "v4":
            # v4: Only text-delta chunks
            return [{
                "type": "text-delta",
                "id": "text-v4",
                "delta": chunk
            } for chunk in text_chunks]
        else:
            # v5: Complete start/delta/end sequence
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