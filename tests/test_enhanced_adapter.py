#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the enhanced LangChain to AI SDK adapter.
"""

import asyncio
import pytest
from typing import Any, Dict, List, AsyncGenerator

from langchain_aisdk_adapter import (
    to_data_stream,
    BaseAICallbackHandler,
    Message,
    TextUIPart,
    ToolInvocationUIPart,
)


class TestCallbackHandler(BaseAICallbackHandler):
    """Test callback handler for capturing events."""
    
    def __init__(self):
        self.messages: List[Message] = []
        self.errors: List[Exception] = []
        self.options_list: List[Dict[str, Any]] = []
    
    async def on_finish(self, message: Message, options: Dict[str, Any]) -> None:
        self.messages.append(message)
        self.options_list.append(options)
    
    async def on_error(self, error: Exception) -> None:
        self.errors.append(error)


async def create_simple_text_stream() -> AsyncGenerator[str, None]:
    """Create a simple text stream."""
    yield "Hello "
    yield "world!"


async def create_langchain_event_stream() -> AsyncGenerator[Dict[str, Any], None]:
    """Create a mock LangChain event stream."""
    
    # Chat model start
    yield {
        "event": "on_chat_model_start",
        "data": {"input": {"messages": [["human", "Test message"]]}}
    }
    
    # Text content
    yield {
        "event": "on_chat_model_stream",
        "data": {"chunk": {"content": "Hello "}}
    }
    
    yield {
        "event": "on_chat_model_stream",
        "data": {"chunk": {"content": "world!"}}
    }
    
    # Chat model end
    yield {
        "event": "on_chat_model_end",
        "data": {"output": {"content": "Hello world!"}}
    }


async def create_tool_event_stream() -> AsyncGenerator[Dict[str, Any], None]:
    """Create a mock stream with tool events."""
    
    # Chat model start
    yield {
        "event": "on_chat_model_start",
        "data": {}
    }
    
    # Initial text
    yield {
        "event": "on_chat_model_stream",
        "data": {"chunk": {"content": "Let me check that for you."}}
    }
    
    # Tool start
    yield {
        "event": "on_tool_start",
        "data": {
            "run_id": "test_tool_1",
            "name": "test_tool",
            "inputs": {"query": "test"}
        }
    }
    
    # Tool end
    yield {
        "event": "on_tool_end",
        "data": {
            "run_id": "test_tool_1",
            "outputs": "Tool result"
        }
    }
    
    # Final text
    yield {
        "event": "on_chat_model_stream",
        "data": {"chunk": {"content": " The result is: Tool result"}}
    }
    
    # Chat model end
    yield {
        "event": "on_chat_model_end",
        "data": {}
    }


@pytest.mark.asyncio
async def test_simple_text_stream():
    """Test processing a simple text stream."""
    callback_handler = TestCallbackHandler()
    
    chunks = []
    async for chunk in to_data_stream(create_simple_text_stream(), callbacks=callback_handler):
        chunks.append(chunk)
    
    # Check basic structure
    assert len(chunks) > 0
    
    # Should have start and finish events
    start_chunks = [c for c in chunks if c.get("type") == "start"]
    finish_chunks = [c for c in chunks if c.get("type") == "finish"]
    
    assert len(start_chunks) == 1
    assert len(finish_chunks) == 1
    assert "messageId" in start_chunks[0]
    
    # Should have text events
    text_deltas = [c for c in chunks if c.get("type") == "text-delta"]
    assert len(text_deltas) >= 2
    
    # Check callback was called
    assert len(callback_handler.messages) == 1
    message = callback_handler.messages[0]
    assert "Hello world!" in message.content
    assert message.role == "assistant"


@pytest.mark.asyncio
async def test_langchain_event_stream():
    """Test processing LangChain event stream."""
    callback_handler = TestCallbackHandler()
    
    chunks = []
    async for chunk in to_data_stream(create_langchain_event_stream(), callbacks=callback_handler):
        chunks.append(chunk)
    
    # Check event types
    event_types = [c.get("type") for c in chunks]
    
    assert "start" in event_types
    assert "finish" in event_types
    assert "start-step" in event_types
    assert "finish-step" in event_types
    assert "text-start" in event_types
    assert "text-delta" in event_types
    assert "text-end" in event_types
    
    # Check callback
    assert len(callback_handler.messages) == 1
    message = callback_handler.messages[0]
    assert "Hello world!" in message.content


@pytest.mark.asyncio
async def test_tool_event_stream():
    """Test processing stream with tool events."""
    callback_handler = TestCallbackHandler()
    
    chunks = []
    async for chunk in to_data_stream(create_tool_event_stream(), callbacks=callback_handler):
        chunks.append(chunk)
    
    # Check tool events
    event_types = [c.get("type") for c in chunks]
    
    assert "tool-input-start" in event_types
    assert "tool-input-available" in event_types
    assert "tool-output-available" in event_types
    
    # Find tool events
    tool_input_start = [c for c in chunks if c.get("type") == "tool-input-start"]
    tool_input_available = [c for c in chunks if c.get("type") == "tool-input-available"]
    tool_output_available = [c for c in chunks if c.get("type") == "tool-output-available"]
    
    assert len(tool_input_start) == 1
    assert len(tool_input_available) == 1
    assert len(tool_output_available) == 1
    
    # Check tool details
    assert tool_input_start[0]["toolName"] == "test_tool"
    assert tool_input_start[0]["toolCallId"] == "test_tool_1"
    
    assert tool_input_available[0]["toolName"] == "test_tool"
    assert tool_input_available[0]["input"] == {"query": "test"}
    
    assert tool_output_available[0]["output"] == "Tool result"
    
    # Check callback message
    assert len(callback_handler.messages) == 1
    message = callback_handler.messages[0]
    
    # Should have tool parts
    tool_parts = [p for p in message.parts if isinstance(p, ToolInvocationUIPart)]
    assert len(tool_parts) == 1
    
    tool_inv = tool_parts[0].toolInvocation
    assert tool_inv.toolName == "test_tool"
    assert tool_inv.state == "result"
    assert tool_inv.args == {"query": "test"}
    assert tool_inv.result == "Tool result"


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in the enhanced adapter."""
    
    async def error_stream():
        yield "Some text"
        raise ValueError("Test error")
    
    callback_handler = TestCallbackHandler()
    
    with pytest.raises(ValueError, match="Test error"):
        chunks = []
        async for chunk in to_data_stream(error_stream(), callbacks=callback_handler):
            chunks.append(chunk)
    
    # Error callback should have been called
    assert len(callback_handler.errors) == 1
    assert isinstance(callback_handler.errors[0], ValueError)
    assert str(callback_handler.errors[0]) == "Test error"


@pytest.mark.asyncio
async def test_message_id_consistency():
    """Test that message ID is consistent across events."""
    callback_handler = TestCallbackHandler()
    
    chunks = []
    async for chunk in to_data_stream(create_simple_text_stream(), callbacks=callback_handler):
        chunks.append(chunk)
    
    # Get message ID from start event
    start_chunk = next(c for c in chunks if c.get("type") == "start")
    message_id = start_chunk["messageId"]
    
    # Check callback message has same ID
    assert len(callback_handler.messages) == 1
    assert callback_handler.messages[0].id == message_id


@pytest.mark.asyncio
async def test_text_aggregation():
    """Test that text is properly aggregated in the final message."""
    callback_handler = TestCallbackHandler()
    
    async def multi_text_stream():
        yield "First "
        yield "second "
        yield "third."
    
    chunks = []
    async for chunk in to_data_stream(multi_text_stream(), callbacks=callback_handler):
        chunks.append(chunk)
    
    # Check text deltas
    text_deltas = [c for c in chunks if c.get("type") == "text-delta"]
    assert len(text_deltas) == 3
    
    # Check aggregated content
    assert len(callback_handler.messages) == 1
    message = callback_handler.messages[0]
    assert message.content == "First second third."
    
    # Check text part
    text_parts = [p for p in message.parts if isinstance(p, TextUIPart)]
    assert len(text_parts) == 1
    assert text_parts[0].text == "First second third."


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])