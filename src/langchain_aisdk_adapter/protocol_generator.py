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
    def create_text_start(text_id: str, protocol_version: str = "v4") -> UIMessageChunkTextStart:
        """Create text-start protocol chunk."""
        if protocol_version == "v4":
            # v4 protocol uses different structure
            return {
                "type": "text-start",
                "id": text_id
            }
        else:
            # v5 protocol
            return {
                "type": "text-start",
                "id": text_id
            }
    
    @staticmethod
    def create_text_delta(text_id: str, delta: str, protocol_version: str = "v4") -> UIMessageChunkTextDelta:
        """Create text-delta protocol chunk."""
        if protocol_version == "v4":
            # v4 protocol uses different structure
            return {
                "type": "text-delta",
                "id": text_id,
                "delta": delta
            }
        else:
            # v5 protocol
            return {
                "type": "text-delta",
                "id": text_id,
                "delta": delta
            }
    
    @staticmethod
    def create_text_end(text_id: str, text: str = "", protocol_version: str = "v4") -> UIMessageChunkTextEnd:
        """Create text-end protocol chunk."""
        if protocol_version == "v4":
            # v4 protocol uses different structure
            return {
                "type": "text-end",
                "id": text_id,
                "text": text
            }
        else:
            # v5 protocol
            return {
                 "type": "text-end",
                 "id": text_id,
                 "text": text
             }