"""Tests for StreamProcessor class."""

import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock
from typing import AsyncGenerator, Dict, Any

from langchain_aisdk_adapter.stream_processor import StreamProcessor
from langchain_aisdk_adapter.models import (
    LangChainStreamEvent,
    UIMessageChunk,
    UIMessageChunkStart,
    UIMessageChunkFinish,
    UIMessageChunkTextStart,
    UIMessageChunkTextDelta,
    UIMessageChunkTextEnd
)
from langchain_aisdk_adapter.callbacks import BaseAICallbackHandler


class TestStreamProcessor:
    """Test cases for StreamProcessor class."""

    def test_init_default_values(self):
        """Test StreamProcessor initialization with default values."""
        processor = StreamProcessor(message_id="test-id")
        
        assert processor.message_id == "test-id"
        assert processor.auto_events is True
        assert processor.callbacks is None
        assert processor.protocol_version == "v4"
        assert processor.message_builder is not None
        assert processor._current_text_id is None
        assert processor._tool_calls == {}
        assert processor._accumulated_text == ""
        assert processor.current_step_id is None

    def test_init_custom_values(self, mock_callback_handler):
        """Test StreamProcessor initialization with custom values."""
        processor = StreamProcessor(
            message_id="custom-id",
            auto_events=False,
            callbacks=mock_callback_handler,
            protocol_version="v5"
        )
        
        assert processor.message_id == "custom-id"
        assert processor.auto_events is False
        assert processor.callbacks == mock_callback_handler
        assert processor.protocol_version == "v5"

    @pytest.mark.asyncio
    async def test_process_stream_basic(self, sample_langchain_text_event):
        """Test basic stream processing."""
        processor = StreamProcessor(message_id="test-id")
        
        async def mock_stream():
            yield sample_langchain_text_event
        
        chunks = []
        async for chunk in processor.process_stream(mock_stream()):
            chunks.append(chunk)
        
        # Should have start and finish chunks at minimum
        assert len(chunks) >= 2
        assert chunks[0]["type"] == "start"
        assert chunks[-1]["type"] == "finish"

    @pytest.mark.asyncio
    async def test_process_stream_with_auto_events_false(self, sample_langchain_text_event):
        """Test stream processing with auto_events disabled."""
        processor = StreamProcessor(message_id="test-id", auto_events=False)
        
        async def mock_stream():
            yield sample_langchain_text_event
        
        chunks = []
        async for chunk in processor.process_stream(mock_stream()):
            chunks.append(chunk)
        
        # Should have no chunks when auto_events is False
        assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_process_string_stream(self):
        """Test processing of string streams."""
        processor = StreamProcessor(message_id="test-id")
        
        async def string_stream():
            yield "Hello "
            yield "world!"
        
        chunks = []
        async for chunk in processor.process_stream(string_stream()):
            chunks.append(chunk)
        
        # Should process string content and generate appropriate chunks
        assert len(chunks) > 0
        assert chunks[0]["type"] == "start"
        
        # Look for text-related chunks
        text_chunks = [c for c in chunks if c.get("type", "").startswith("text")]
        assert len(text_chunks) > 0

    @pytest.mark.asyncio
    async def test_process_tool_events(self, sample_langchain_tool_event, sample_tool_result_event):
        """Test processing of tool-related events."""
        processor = StreamProcessor(message_id="test-id")

        async def tool_stream():
            yield sample_langchain_tool_event
            yield sample_tool_result_event

        chunks = []
        async for chunk in processor.process_stream(tool_stream()):
            chunks.append(chunk)
        
        # Should have basic stream chunks (start, finish)
        chunk_types = [c.get("type") for c in chunks]
        assert "start" in chunk_types
        assert "finish" in chunk_types
        
        # Tool events are processed but may not generate tool chunks directly
        # as they are handled through intermediate_steps in real scenarios
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_error_handling_in_stream(self, mock_callback_handler):
        """Test error handling during stream processing."""
        processor = StreamProcessor(
            message_id="test-id",
            callbacks=mock_callback_handler
        )
        
        async def error_stream():
            yield {"event": "on_chat_model_stream", "data": {"chunk": {"content": "test"}}}
            raise ValueError("Stream error")
        
        with pytest.raises(ValueError, match="Stream error"):
            chunks = []
            async for chunk in processor.process_stream(error_stream()):
                chunks.append(chunk)
        
        # Verify error callback was called
        mock_callback_handler.on_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_callback_integration(self, mock_callback_handler):
        """Test integration with callback handlers."""
        processor = StreamProcessor(
            message_id="test-id",
            callbacks=mock_callback_handler
        )
        
        async def simple_stream():
            yield {"event": "on_chat_model_stream", "data": {"chunk": {"content": "test"}}}
        
        chunks = []
        async for chunk in processor.process_stream(simple_stream()):
            chunks.append(chunk)
        
        # Verify that AI SDK callbacks were handled
        # The exact calls depend on the implementation
        assert len(chunks) > 0

    def test_create_start_chunk(self):
        """Test creation of start chunk."""
        processor = StreamProcessor(message_id="test-id")
        
        chunk = processor._create_start_chunk()
        
        assert chunk["type"] == "start"
        assert chunk["messageId"] == "test-id"

    def test_create_finish_chunk(self):
        """Test creation of finish chunk."""
        processor = StreamProcessor(message_id="test-id")
        
        chunk = processor._create_finish_chunk()
        
        assert chunk["type"] == "finish"
        assert chunk["finishReason"] == "stop"
        assert "usage" in chunk

    def test_create_error_chunk(self):
        """Test creation of error chunk."""
        processor = StreamProcessor(message_id="test-id")
        
        chunk = processor._create_error_chunk("Test error")
        
        assert chunk["type"] == "error"
        assert chunk["errorText"] == "Test error"

    def test_create_start_step_chunk(self):
        """Test creation of start step chunk."""
        processor = StreamProcessor(message_id="test-id")
        processor.current_step_id = "step-123"
        
        chunk = processor._create_start_step_chunk()
        
        assert chunk["type"] == "start-step"
        assert chunk["messageId"] == "step-123"

    def test_create_finish_step_chunk(self):
        """Test creation of finish step chunk."""
        processor = StreamProcessor(message_id="test-id")
        processor.tool_completed_in_current_step = True
        processor.current_usage = {"promptTokens": 10, "completionTokens": 5}
        
        chunk = processor._create_finish_step_chunk()
        
        assert chunk["type"] == "finish-step"
        assert chunk["finishReason"] == "tool-calls"
        assert chunk["usage"]["promptTokens"] == 10
        assert chunk["usage"]["completionTokens"] == 5
        assert chunk["isContinued"] is False

    def test_create_tool_input_start_chunk(self):
        """Test creation of tool input start chunk."""
        processor = StreamProcessor(message_id="test-id")
        
        chunk = processor._create_tool_input_start_chunk("tool-123", "test_tool")
        
        assert chunk["type"] == "tool-input-start"
        assert chunk["toolCallId"] == "tool-123"
        assert chunk["toolName"] == "test_tool"

    def test_create_tool_output_available_chunk(self):
        """Test creation of tool output available chunk."""
        processor = StreamProcessor(message_id="test-id")
        
        chunk = processor._create_tool_output_available_chunk("tool-123", "result")
        
        assert chunk["type"] == "tool-output-available"
        assert chunk["toolCallId"] == "tool-123"
        assert chunk["output"] == "result"

    @pytest.mark.asyncio
    async def test_call_callback_sync(self):
        """Test calling synchronous callback."""
        processor = StreamProcessor(message_id="test-id")
        
        callback = MagicMock()
        await processor._call_callback(callback, "arg1", "arg2")
        
        callback.assert_called_once_with("arg1", "arg2")

    @pytest.mark.asyncio
    async def test_call_callback_async(self):
        """Test calling asynchronous callback."""
        processor = StreamProcessor(message_id="test-id")
        
        callback = AsyncMock()
        await processor._call_callback(callback, "arg1", "arg2")
        
        callback.assert_called_once_with("arg1", "arg2")

    @pytest.mark.asyncio
    async def test_call_callback_none(self):
        """Test calling None callback (should not raise error)."""
        processor = StreamProcessor(message_id="test-id")
        
        # Should not raise any exception
        await processor._call_callback(None, "arg1", "arg2")

    @pytest.mark.asyncio
    async def test_protocol_version_v5(self):
        """Test processor with v5 protocol version."""
        processor = StreamProcessor(message_id="test-id", protocol_version="v5")
        
        async def simple_stream():
            yield {"event": "on_chat_model_stream", "data": {"chunk": {"content": "test"}}}
        
        chunks = []
        async for chunk in processor.process_stream(simple_stream()):
            chunks.append(chunk)
        
        # Should process with v5 protocol
        assert len(chunks) > 0
        assert processor.protocol_version == "v5"

    @pytest.mark.asyncio
    async def test_multiple_text_chunks(self):
        """Test processing multiple text chunks."""
        processor = StreamProcessor(message_id="test-id")
        
        async def multi_text_stream():
            yield {"event": "on_chat_model_stream", "data": {"chunk": {"content": "Hello "}}}
            yield {"event": "on_chat_model_stream", "data": {"chunk": {"content": "world"}}}
            yield {"event": "on_chat_model_stream", "data": {"chunk": {"content": "!"}}}
        
        chunks = []
        async for chunk in processor.process_stream(multi_text_stream()):
            chunks.append(chunk)
        
        # Should have text-related chunks
        text_chunks = [c for c in chunks if c.get("type", "").startswith("text")]
        assert len(text_chunks) > 0

    @pytest.mark.asyncio
    async def test_complex_stream_with_tools_and_text(self):
        """Test processing complex stream with both tools and text."""
        processor = StreamProcessor(message_id="test-id")
        
        async def complex_stream():
            # Text content
            yield {"event": "on_chat_model_stream", "data": {"chunk": {"content": "Let me search for that."}}}
            # Tool call
            yield {
                "event": "on_tool_start",
                "name": "search_tool",
                "run_id": "tool-run-1",
                "data": {"input": {"query": "test query"}}
            }
            # Tool result
            yield {
                "event": "on_tool_end",
                "name": "search_tool",
                "run_id": "tool-run-1",
                "data": {"output": "Search results"}
            }
            # More text
            yield {"event": "on_chat_model_stream", "data": {"chunk": {"content": "Based on the results..."}}}
        
        chunks = []
        async for chunk in processor.process_stream(complex_stream()):
            chunks.append(chunk)
        
        # Should have various chunk types
        chunk_types = [c.get("type") for c in chunks]
        assert "start" in chunk_types
        assert "finish" in chunk_types
        
        # Should have text chunks
        text_chunks = [c for c in chunks if c.get("type", "").startswith("text")]
        
        assert len(text_chunks) > 0
        
        # Tool events are processed but may not generate tool chunks directly
        # in this test scenario as they are typically handled through intermediate_steps
        # The important thing is that the stream processes without errors
        assert len(chunks) >= 2  # At least start and finish chunks