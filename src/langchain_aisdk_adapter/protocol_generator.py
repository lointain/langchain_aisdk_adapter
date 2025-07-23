#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Protocol Generator for AI SDK

This module provides the ProtocolGenerator class for creating
AI SDK protocol chunks in a unified way.
"""

from .models import (
    UIMessageChunkTextStart,
    UIMessageChunkTextDelta,
    UIMessageChunkTextEnd,
)


class ProtocolGenerator:
    """Unified protocol generator for all chunk types."""
    
    @staticmethod
    def create_text_start(text_id: str) -> UIMessageChunkTextStart:
        """Create text-start protocol chunk."""
        return {
            "type": "text-start",
            "id": text_id
        }
    
    @staticmethod
    def create_text_delta(text_id: str, delta: str) -> UIMessageChunkTextDelta:
        """Create text-delta protocol chunk."""
        return {
            "type": "text-delta",
            "id": text_id,
            "delta": delta
        }
    
    @staticmethod
    def create_text_end(text_id: str, text: str = "") -> UIMessageChunkTextEnd:
        """Create text-end protocol chunk."""
        return {
            "type": "text-end",
            "id": text_id,
            "text": text
        }