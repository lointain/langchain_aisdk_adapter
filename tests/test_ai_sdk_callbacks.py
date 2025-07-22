#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for AI SDK compatible callback system.
"""

import pytest
from typing import Dict, Any, List
from datetime import datetime

from langchain_aisdk_adapter import (
    to_data_stream,
    BaseAICallbackHandler,
    Message,
    TextUIPart,
    ToolInvocationUIPart,
    StepStartUIPart,
    ToolInvocation,
)


class MockCallback(BaseAICallbackHandler):
    """Test callback handler for capturing events."""
    
    def __init__(self):
        self.messages: List[Message] = []
        self.options: List[Dict[str, Any]] = []
        self.errors: List[Exception] = []
    
    async def on_finish(self, message: Message, options: Dict[str, Any]) -> None:
        self.messages.append(message)
        self.options.append(options)
    
    async def on_error(self, error: Exception) -> None:
        self.errors.append(error)


@pytest.fixture
def test_callback():
    """Fixture providing a test callback instance."""
    return MockCallback()


@pytest.fixture
def simple_text_stream():
    """Fixture providing a simple text stream."""
    async def _stream():
        yield "Hello, "
        yield "world!"
        yield " How are you?"
    return _stream()


@pytest.fixture
def tool_call_stream():
    """Fixture providing a stream with tool calls."""
    async def _stream():
        yield "I'll check the weather for you."
        
        # Tool start event
        yield {
            "event": "on_tool_start",
            "run_id": "test_run_123",
            "name": "weather_tool",
            "data": {
                "input": {
                    "location": "New York",
                    "unit": "fahrenheit"
                }
            }
        }
        
        # Tool end event
        yield {
            "event": "on_tool_end",
            "run_id": "test_run_123",
            "data": {
                "output": {
                    "temperature": 72,
                    "condition": "sunny",
                    "humidity": 45
                }
            }
        }
        
        yield "The weather in New York is 72°F and sunny."
    
    return _stream()


class TestAISDKCallbacks:
    """Test suite for AI SDK compatible callbacks."""
    
    @pytest.mark.asyncio
    async def test_basic_text_stream_with_callback(self, simple_text_stream, test_callback):
        """Test basic text streaming with AI SDK callback."""
        chunks = []
        async for chunk in to_data_stream(simple_text_stream, callbacks=test_callback):
            chunks.append(chunk)
        
        # Verify chunks
        assert len(chunks) >= 3  # At least start, deltas, end
        assert chunks[0]["type"] == "start"
        
        # Find specific chunk types
        chunk_types = [c["type"] for c in chunks]
        assert "start-step" in chunk_types
        assert "text-start" in chunk_types
        assert "text-end" in chunk_types
        assert "finish-step" in chunk_types
        assert chunks[-1]["type"] == "finish"
        
        # Verify callback was called
        assert len(test_callback.messages) == 1
        message = test_callback.messages[0]
        assert message.content == "Hello, world! How are you?"
        assert message.role == "assistant"
        assert len(message.parts) >= 1  # Should have at least one text part
        
        # Check for text part
        text_parts = [p for p in message.parts if isinstance(p, TextUIPart)]
        assert len(text_parts) == 1
        assert text_parts[0].text == "Hello, world! How are you?"
    
    @pytest.mark.asyncio
    async def test_tool_call_stream_with_callback(self, tool_call_stream, test_callback):
        """Test tool call streaming with AI SDK callback."""
        chunks = []
        async for chunk in to_data_stream(tool_call_stream, callbacks=test_callback):
            chunks.append(chunk)
        
        # Verify tool-related chunks exist
        tool_chunks = [c for c in chunks if c["type"].startswith("tool-")]
        assert len(tool_chunks) > 0
        
        # Verify callback was called
        assert len(test_callback.messages) == 1
        message = test_callback.messages[0]
        
        # Check for tool invocation parts
        tool_parts = [p for p in message.parts if isinstance(p, ToolInvocationUIPart)]
        assert len(tool_parts) == 1
        
        tool_invocation = tool_parts[0].toolInvocation
        assert tool_invocation.toolName == "weather_tool"
        assert tool_invocation.state == "result"
        assert tool_invocation.args["location"] == "New York"
        assert tool_invocation.result["temperature"] == 72
    
    @pytest.mark.asyncio
    async def test_step_management(self, tool_call_stream, test_callback):
        """Test step management in enhanced stream."""
        chunks = []
        async for chunk in to_data_stream(tool_call_stream, callbacks=test_callback):
            chunks.append(chunk)
        
        # Count step events
        start_steps = [c for c in chunks if c["type"] == "start-step"]
        finish_steps = [c for c in chunks if c["type"] == "finish-step"]
        
        # Should have multiple steps due to tool call
        assert len(start_steps) >= 2
        assert len(finish_steps) >= 1
        
        # Verify step parts in message
        message = test_callback.messages[0]
        step_parts = [p for p in message.parts if isinstance(p, StepStartUIPart)]
        assert len(step_parts) >= 2
    
    @pytest.mark.asyncio
    async def test_backward_compatibility(self, simple_text_stream):
        """Test that legacy callback system still works."""
        from langchain_aisdk_adapter import StreamCallbacks
        
        final_text = None
        
        def on_final(text):
            nonlocal final_text
            final_text = text
        
        callbacks = StreamCallbacks(on_final=on_final)
        
        chunks = []
        async for chunk in to_data_stream(simple_text_stream, callbacks=callbacks):
            chunks.append(chunk)
        
        # Should still work with legacy callbacks
        assert len(chunks) >= 3
        assert final_text == "Hello, world! How are you?"
    
    @pytest.mark.asyncio
    async def test_message_structure(self, simple_text_stream, test_callback):
        """Test the structure of the Message object."""
        async for chunk in to_data_stream(simple_text_stream, callbacks=test_callback):
            pass
        
        message = test_callback.messages[0]
        
        # Verify message structure
        assert isinstance(message.id, str)
        assert len(message.id) > 0
        assert isinstance(message.createdAt, datetime)
        assert message.role == "assistant"
        assert isinstance(message.content, str)
        assert isinstance(message.parts, list)
        assert message.annotations is None or isinstance(message.annotations, list)
        assert message.experimental_attachments is None or isinstance(message.experimental_attachments, list)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, test_callback):
        """Test error handling in callback system."""
        async def error_stream():
            yield "Some text"
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            async for chunk in to_data_stream(error_stream(), callbacks=test_callback):
                pass
        
        # Verify error callback was called
        assert len(test_callback.errors) == 1
        assert isinstance(test_callback.errors[0], ValueError)
        assert str(test_callback.errors[0]) == "Test error"


class TestDataStructures:
    """Test suite for AI SDK data structures."""
    
    def test_text_ui_part(self):
        """Test TextUIPart structure."""
        part = TextUIPart(text="Hello world")
        assert part.type == "text"
        assert part.text == "Hello world"
    
    def test_tool_invocation(self):
        """Test ToolInvocation structure."""
        tool = ToolInvocation(
            state="result",
            step=1,
            toolCallId="call_123",
            toolName="test_tool",
            args={"param": "value"},
            result={"output": "success"}
        )
        assert tool.state == "result"
        assert tool.step == 1
        assert tool.toolCallId == "call_123"
        assert tool.toolName == "test_tool"
        assert tool.args == {"param": "value"}
        assert tool.result == {"output": "success"}
    
    def test_tool_invocation_ui_part(self):
        """Test ToolInvocationUIPart structure."""
        tool = ToolInvocation(
            state="call",
            toolCallId="call_456",
            toolName="another_tool",
            args={"input": "test"}
        )
        part = ToolInvocationUIPart(toolInvocation=tool)
        assert part.type == "tool-invocation"
        assert part.toolInvocation == tool
    
    def test_step_start_ui_part(self):
        """Test StepStartUIPart structure."""
        part = StepStartUIPart()
        assert part.type == "step-start"
    
    def test_message_structure(self):
        """Test Message structure."""
        message = Message(
            id="msg_123",
            content="Test message",
            role="assistant",
            parts=[TextUIPart(text="Hello")]
        )
        assert message.id == "msg_123"
        assert message.content == "Test message"
        assert message.role == "assistant"
        assert len(message.parts) == 1
        assert isinstance(message.parts[0], TextUIPart)
        assert isinstance(message.createdAt, datetime)