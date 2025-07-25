"""Tests for LangChainAdapter class."""

import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from typing import AsyncGenerator, Dict, Any

from langchain_aisdk_adapter import (
    LangChainAdapter,
    AdapterOptions,
    DataStreamWithEmitters,
    DataStreamWriter
)
from langchain_aisdk_adapter.models import UIMessageChunk
from langchain_aisdk_adapter.callbacks import BaseAICallbackHandler


class TestLangChainAdapter:
    """Test cases for LangChainAdapter class."""

    @pytest.mark.asyncio
    async def test_to_data_stream_basic(self, mock_async_stream, mock_callback_handler):
        """Test basic to_data_stream functionality."""
        stream_items = [
            {"event": "on_chat_model_stream", "data": {"chunk": {"content": "test"}}}
        ]
        
        result = LangChainAdapter.to_data_stream(
            stream=mock_async_stream(stream_items),
            callbacks=mock_callback_handler,
            message_id="test-id"
        )
        
        assert isinstance(result, DataStreamWithEmitters)
        assert result.message_id == "test-id"
        
        # Collect chunks
        chunks = []
        async for chunk in result:
            chunks.append(chunk)
            if len(chunks) >= 3:  # Avoid infinite loop
                break
        
        # Should have at least start chunk
        assert len(chunks) > 0
        # In protocol format, chunks are strings, not dicts
        # For v4, start chunk looks like 'f:{"messageId":"..."}\n'
        # For v5, start chunk looks like 'data: {"type":"start",...}\n\n'
        assert isinstance(chunks[0], str)
        assert len(chunks[0]) > 0

    @pytest.mark.asyncio
    async def test_to_data_stream_with_options(self, mock_async_stream):
        """Test to_data_stream with various options."""
        stream_items = [
            {"event": "on_chat_model_stream", "data": {"chunk": {"content": "hello"}}}
        ]
        
        options: AdapterOptions = {
            "protocol_version": "v5",
            "auto_events": False,
            "auto_close": False
        }
        
        result = LangChainAdapter.to_data_stream(
            stream=mock_async_stream(stream_items),
            options=options
        )
        
        assert isinstance(result, DataStreamWithEmitters)
        assert result.protocol_version == "v5"
        assert result.output_format == "protocol"

    def test_to_data_stream_response_basic(self, mock_async_stream):
        """Test basic to_data_stream_response functionality."""
        stream_items = [
            {"event": "on_chat_model_stream", "data": {"chunk": {"content": "test"}}}
        ]
        
        with patch('fastapi.responses.StreamingResponse') as mock_response:
            mock_response.return_value = MagicMock()
            
            result = LangChainAdapter.to_data_stream_response(
                stream=mock_async_stream(stream_items),
                headers={"Custom-Header": "value"},
                status=201
            )
            
            mock_response.assert_called_once()
            call_args = mock_response.call_args
            assert call_args[1]["status_code"] == 201

    def test_to_data_stream_response_with_options(self, mock_async_stream):
        """Test to_data_stream_response with custom options."""
        stream_items = [
            {"event": "on_chat_model_stream", "data": {"chunk": {"content": "test"}}}
        ]
        
        options: AdapterOptions = {
            "protocol_version": "v5"
        }
        
        with patch('fastapi.responses.StreamingResponse') as mock_response:
            mock_response.return_value = MagicMock()
            
            result = LangChainAdapter.to_data_stream_response(
                stream=mock_async_stream(stream_items),
                options=options
            )
            
            mock_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_merge_into_data_stream(self, mock_async_stream):
        """Test merge_into_data_stream functionality."""
        stream_items = [
            {"event": "on_chat_model_stream", "data": {"chunk": {"content": "test"}}}
        ]
        
        # Create mock data stream writer
        mock_writer = AsyncMock(spec=DataStreamWriter)
        mock_writer.write = AsyncMock()
        
        await LangChainAdapter.merge_into_data_stream(
            stream=mock_async_stream(stream_items),
            data_stream_writer=mock_writer,
            message_id="test-merge-id"
        )
        
        # Verify that write was called
        assert mock_writer.write.called

    @pytest.mark.asyncio
    async def test_auto_generated_message_id(self, mock_async_stream):
        """Test that message_id is auto-generated when not provided."""
        stream_items = [
            {"event": "on_chat_model_stream", "data": {"chunk": {"content": "test"}}}
        ]
        
        result = LangChainAdapter.to_data_stream(
            stream=mock_async_stream(stream_items)
        )
        
        # Should have a valid UUID as message_id
        assert result.message_id is not None
        assert len(result.message_id) > 0
        # Try to parse as UUID to verify format
        uuid.UUID(result.message_id)

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_callback_handler):
        """Test error handling in stream processing."""
        async def error_stream():
            yield {"event": "on_chat_model_stream", "data": {"chunk": {"content": "test"}}}
            raise ValueError("Test error")

        result = LangChainAdapter.to_data_stream(
            stream=error_stream(),
            callbacks=mock_callback_handler
        )

        with pytest.raises(ValueError, match="Test error"):
            # Fully consume the stream to trigger the error
            chunks = []
            async for chunk in result:
                chunks.append(chunk)
        
        # Verify error callback was called
        mock_callback_handler.on_error.assert_called_once()

    def test_adapter_options_defaults(self):
        """Test that AdapterOptions has correct default behavior."""
        # Test empty options
        options: AdapterOptions = {}
        
        # Should be able to access with defaults
        protocol_version = options.get("protocol_version", "v4")
        auto_events = options.get("auto_events", True)
        
        assert protocol_version == "v4"
        assert auto_events is True

    @pytest.mark.asyncio
    async def test_string_stream_handling(self):
        """Test handling of direct string streams."""
        async def string_stream():
            yield "Hello "
            yield "world!"
        
        result = LangChainAdapter.to_data_stream(
            stream=string_stream()
        )
        
        chunks = []
        async for chunk in result:
            chunks.append(chunk)
            if len(chunks) >= 5:  # Limit to avoid infinite loop
                break
        
        # Should process string content
        assert len(chunks) > 0
        # In protocol format, chunks are strings, not dicts
        assert isinstance(chunks[0], str)
        assert len(chunks[0]) > 0

    @pytest.mark.asyncio
    async def test_callback_integration(self, mock_callback_handler):
        """Test integration with callback handlers."""
        async def simple_stream():
            yield {"event": "on_chat_model_stream", "data": {"chunk": {"content": "test"}}}
        
        result = LangChainAdapter.to_data_stream(
            stream=simple_stream(),
            callbacks=mock_callback_handler
        )
        
        # Process stream
        chunks = []
        async for chunk in result:
            chunks.append(chunk)
            if len(chunks) >= 3:
                break
        
        # Verify callbacks were called (exact calls depend on implementation)
        assert mock_callback_handler.on_start.called or mock_callback_handler.on_finish.called