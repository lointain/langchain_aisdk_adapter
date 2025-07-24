"""Tests for MessageBuilder class."""

import pytest
import uuid
from datetime import datetime
from unittest.mock import AsyncMock

from langchain_aisdk_adapter.message_builder import MessageBuilder
from langchain_aisdk_adapter.models import UIMessageChunk
from langchain_aisdk_adapter.callbacks import (
    Message,
    UIPart,
    TextUIPart,
    ToolInvocationUIPart,
    ToolInvocation,
    StepStartUIPart
)


class TestMessageBuilder:
    """Test cases for MessageBuilder class."""

    def test_init_default_values(self):
        """Test MessageBuilder initialization with default values."""
        builder = MessageBuilder()
        
        assert builder.message_id is not None
        assert len(builder.message_id) > 0
        assert builder.parts == []
        assert builder.content == ""
        assert isinstance(builder.created_at, datetime)
        assert builder._current_text_parts == {}

    def test_init_custom_message_id(self):
        """Test MessageBuilder initialization with custom message ID."""
        custom_id = "custom-message-id"
        builder = MessageBuilder(message_id=custom_id)
        
        assert builder.message_id == custom_id

    @pytest.mark.asyncio
    async def test_add_chunk_start_step(self):
        """Test adding start-step chunk."""
        builder = MessageBuilder()
        
        chunk: UIMessageChunk = {
            "type": "start-step",
            "messageId": "step-123"
        }
        
        new_parts = await builder.add_chunk(chunk)
        
        assert len(new_parts) == 1
        assert new_parts[0].type == "step-start"
        assert len(builder.parts) == 1
        assert isinstance(builder.parts[0], StepStartUIPart)

    @pytest.mark.asyncio
    async def test_add_chunk_text_sequence(self):
        """Test adding text-start, text-delta, text-end sequence."""
        builder = MessageBuilder()
        
        # Text start
        start_chunk: UIMessageChunk = {
            "type": "text-start",
            "id": "text-1"
        }
        new_parts = await builder.add_chunk(start_chunk)
        assert len(new_parts) == 0  # No parts added yet
        assert "text-1" in builder._current_text_parts
        
        # Text delta 1
        delta_chunk1: UIMessageChunk = {
            "type": "text-delta",
            "id": "text-1",
            "textDelta": "Hello "
        }
        new_parts = await builder.add_chunk(delta_chunk1)
        assert len(new_parts) == 0  # Still accumulating
        assert builder._current_text_parts["text-1"].text == "Hello "
        
        # Text delta 2
        delta_chunk2: UIMessageChunk = {
            "type": "text-delta",
            "id": "text-1",
            "textDelta": "world!"
        }
        new_parts = await builder.add_chunk(delta_chunk2)
        assert len(new_parts) == 0  # Still accumulating
        assert builder._current_text_parts["text-1"].text == "Hello world!"
        
        # Text end
        end_chunk: UIMessageChunk = {
            "type": "text-end",
            "id": "text-1"
        }
        new_parts = await builder.add_chunk(end_chunk)
        assert len(new_parts) == 1  # Now part is added
        assert isinstance(new_parts[0], TextUIPart)
        assert new_parts[0].text == "Hello world!"
        assert len(builder.parts) == 1
        assert "text-1" not in builder._current_text_parts  # Cleaned up

    @pytest.mark.asyncio
    async def test_add_chunk_text_empty_content(self):
        """Test that empty text content is not added to parts."""
        builder = MessageBuilder()
        
        # Text start
        start_chunk: UIMessageChunk = {
            "type": "text-start",
            "id": "text-empty"
        }
        await builder.add_chunk(start_chunk)
        
        # Text end without any delta (empty content)
        end_chunk: UIMessageChunk = {
            "type": "text-end",
            "id": "text-empty"
        }
        new_parts = await builder.add_chunk(end_chunk)
        
        assert len(new_parts) == 0  # Empty text should not be added
        assert len(builder.parts) == 0

    @pytest.mark.asyncio
    async def test_add_chunk_tool_sequence(self):
        """Test adding tool-related chunks."""
        builder = MessageBuilder()
        
        # Tool input available
        input_chunk: UIMessageChunk = {
            "type": "tool-input-available",
            "toolCallId": "tool-123",
            "toolName": "search_tool",
            "input": {"query": "test query"}
        }
        new_parts = await builder.add_chunk(input_chunk)
        assert len(new_parts) == 0  # No part created yet
        assert hasattr(builder, '_pending_tools')
        assert "tool-123" in builder._pending_tools
        
        # Tool output available
        output_chunk: UIMessageChunk = {
            "type": "tool-output-available",
            "toolCallId": "tool-123",
            "output": "Search results"
        }
        new_parts = await builder.add_chunk(output_chunk)
        assert len(new_parts) == 1  # Tool invocation part created
        assert isinstance(new_parts[0], ToolInvocationUIPart)
        assert new_parts[0].toolInvocation.toolCallId == "tool-123"
        assert new_parts[0].toolInvocation.toolName == "search_tool"
        assert new_parts[0].toolInvocation.state == "result"
        assert new_parts[0].toolInvocation.result == "Search results"
        assert len(builder.parts) == 1

    @pytest.mark.asyncio
    async def test_add_chunk_tool_error(self):
        """Test adding tool error chunk."""
        builder = MessageBuilder()
        
        # Tool input available
        input_chunk: UIMessageChunk = {
            "type": "tool-input-available",
            "toolCallId": "tool-error",
            "toolName": "failing_tool",
            "input": {"param": "value"}
        }
        await builder.add_chunk(input_chunk)
        
        # Tool output error
        error_chunk: UIMessageChunk = {
            "type": "tool-output-error",
            "toolCallId": "tool-error",
            "errorText": "Tool execution failed"
        }
        new_parts = await builder.add_chunk(error_chunk)
        
        assert len(new_parts) == 1
        assert isinstance(new_parts[0], ToolInvocationUIPart)
        assert new_parts[0].toolInvocation.state == "result"
        assert new_parts[0].toolInvocation.result == {"error": "Tool execution failed"}

    @pytest.mark.asyncio
    async def test_add_chunk_duplicate_tool_calls(self):
        """Test that duplicate tool calls are not added."""
        builder = MessageBuilder()
        
        # First tool sequence
        input_chunk: UIMessageChunk = {
            "type": "tool-input-available",
            "toolCallId": "tool-dup",
            "toolName": "test_tool",
            "input": {"param": "value"}
        }
        await builder.add_chunk(input_chunk)
        
        output_chunk: UIMessageChunk = {
            "type": "tool-output-available",
            "toolCallId": "tool-dup",
            "output": "First result"
        }
        new_parts1 = await builder.add_chunk(output_chunk)
        assert len(new_parts1) == 1
        assert len(builder.parts) == 1
        
        # Try to add the same tool call again
        new_parts2 = await builder.add_chunk(output_chunk)
        assert len(new_parts2) == 0  # No new parts
        assert len(builder.parts) == 1  # Still only one part

    @pytest.mark.asyncio
    async def test_add_chunk_reasoning(self):
        """Test adding reasoning chunk."""
        builder = MessageBuilder()
        
        reasoning_chunk: UIMessageChunk = {
            "type": "reasoning",
            "text": "Let me think about this..."
        }
        
        new_parts = await builder.add_chunk(reasoning_chunk)
        
        assert len(new_parts) == 1
        assert isinstance(new_parts[0], TextUIPart)
        assert new_parts[0].text == "Let me think about this..."
        assert builder.content == "Let me think about this..."

    @pytest.mark.asyncio
    async def test_add_chunk_unknown_type(self):
        """Test adding chunk with unknown type."""
        builder = MessageBuilder()
        
        unknown_chunk: UIMessageChunk = {
            "type": "unknown-type",
            "data": "some data"
        }
        
        new_parts = await builder.add_chunk(unknown_chunk)
        
        assert len(new_parts) == 0
        assert len(builder.parts) == 0

    @pytest.mark.asyncio
    async def test_add_chunk_no_type(self):
        """Test adding chunk without type field."""
        builder = MessageBuilder()
        
        no_type_chunk: UIMessageChunk = {
            "data": "some data"
        }
        
        new_parts = await builder.add_chunk(no_type_chunk)
        
        assert len(new_parts) == 0
        assert len(builder.parts) == 0

    @pytest.mark.asyncio
    async def test_multiple_text_parts(self):
        """Test handling multiple text parts with different IDs."""
        builder = MessageBuilder()
        
        # First text part
        await builder.add_chunk({"type": "text-start", "id": "text-1"})
        await builder.add_chunk({"type": "text-delta", "id": "text-1", "textDelta": "First "})
        
        # Second text part (overlapping)
        await builder.add_chunk({"type": "text-start", "id": "text-2"})
        await builder.add_chunk({"type": "text-delta", "id": "text-2", "textDelta": "Second "})
        
        # Complete first text part
        await builder.add_chunk({"type": "text-delta", "id": "text-1", "textDelta": "part"})
        new_parts1 = await builder.add_chunk({"type": "text-end", "id": "text-1"})
        
        # Complete second text part
        await builder.add_chunk({"type": "text-delta", "id": "text-2", "textDelta": "part"})
        new_parts2 = await builder.add_chunk({"type": "text-end", "id": "text-2"})
        
        assert len(new_parts1) == 1
        assert len(new_parts2) == 1
        assert len(builder.parts) == 2
        assert builder.parts[0].text == "First part"
        assert builder.parts[1].text == "Second part"

    @pytest.mark.asyncio
    async def test_tool_step_calculation(self):
        """Test that tool step numbers are calculated correctly."""
        builder = MessageBuilder()
        
        # First tool
        await builder.add_chunk({
            "type": "tool-input-available",
            "toolCallId": "tool-1",
            "toolName": "tool1",
            "input": {}
        })
        new_parts1 = await builder.add_chunk({
            "type": "tool-output-available",
            "toolCallId": "tool-1",
            "output": "result1"
        })
        
        # Second tool
        await builder.add_chunk({
            "type": "tool-input-available",
            "toolCallId": "tool-2",
            "toolName": "tool2",
            "input": {}
        })
        new_parts2 = await builder.add_chunk({
            "type": "tool-output-available",
            "toolCallId": "tool-2",
            "output": "result2"
        })
        
        assert new_parts1[0].toolInvocation.step == 0
        assert new_parts2[0].toolInvocation.step == 1

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """Test thread-safe access to MessageBuilder."""
        builder = MessageBuilder()
        
        # Simulate concurrent access
        import asyncio
        
        async def add_text_chunk(text_id: str, content: str):
            await builder.add_chunk({"type": "text-start", "id": text_id})
            await builder.add_chunk({"type": "text-delta", "id": text_id, "textDelta": content})
            return await builder.add_chunk({"type": "text-end", "id": text_id})
        
        # Run multiple text additions concurrently
        results = await asyncio.gather(
            add_text_chunk("text-1", "Content 1"),
            add_text_chunk("text-2", "Content 2"),
            add_text_chunk("text-3", "Content 3")
        )
        
        # All should succeed
        assert all(len(result) == 1 for result in results)
        assert len(builder.parts) == 3
        
        # Check that all content is present
        texts = [part.text for part in builder.parts if hasattr(part, 'text')]
        assert "Content 1" in texts
        assert "Content 2" in texts
        assert "Content 3" in texts

    def test_message_id_generation(self):
        """Test that auto-generated message IDs are valid UUIDs."""
        builder = MessageBuilder()
        
        # Should be able to parse as UUID
        uuid.UUID(builder.message_id)

    @pytest.mark.asyncio
    async def test_complex_message_building(self):
        """Test building a complex message with multiple part types."""
        builder = MessageBuilder()
        
        # Add step start
        await builder.add_chunk({"type": "start-step", "messageId": "step-1"})
        
        # Add text content
        await builder.add_chunk({"type": "text-start", "id": "text-1"})
        await builder.add_chunk({"type": "text-delta", "id": "text-1", "textDelta": "I'll help you with that. "})
        await builder.add_chunk({"type": "text-end", "id": "text-1"})
        
        # Add tool call
        await builder.add_chunk({
            "type": "tool-input-available",
            "toolCallId": "tool-1",
            "toolName": "search",
            "input": {"query": "test"}
        })
        await builder.add_chunk({
            "type": "tool-output-available",
            "toolCallId": "tool-1",
            "output": "Search results"
        })
        
        # Add more text
        await builder.add_chunk({"type": "text-start", "id": "text-2"})
        await builder.add_chunk({"type": "text-delta", "id": "text-2", "textDelta": "Based on the results..."})
        await builder.add_chunk({"type": "text-end", "id": "text-2"})
        
        # Verify the complete message structure
        assert len(builder.parts) == 4  # step-start, text, tool-invocation, text
        
        part_types = [getattr(part, 'type', None) for part in builder.parts]
        assert "step-start" in part_types
        assert "text" in part_types
        assert "tool-invocation" in part_types
        
        # Verify text content
        text_parts = [part for part in builder.parts if hasattr(part, 'text')]
        assert len(text_parts) == 2
        assert any("I'll help you" in part.text for part in text_parts)
        assert any("Based on the results" in part.text for part in text_parts)
        
        # Verify tool invocation
        tool_parts = [part for part in builder.parts if hasattr(part, 'toolInvocation')]
        assert len(tool_parts) == 1
        assert tool_parts[0].toolInvocation.toolName == "search"
        assert tool_parts[0].toolInvocation.result == "Search results"