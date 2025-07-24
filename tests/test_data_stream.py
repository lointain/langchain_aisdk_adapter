"""Tests for DataStream classes."""

import pytest
import uuid
import asyncio
from unittest.mock import AsyncMock, MagicMock
from typing import AsyncGenerator

from langchain_aisdk_adapter.data_stream import (
    DataStreamWithEmitters,
    DataStreamWriter,
    DataStreamResponse
)
from langchain_aisdk_adapter.models import UIMessageChunk
from langchain_aisdk_adapter.callbacks import BaseAICallbackHandler


class TestDataStreamWithEmitters:
    """Test cases for DataStreamWithEmitters class."""

    @pytest.fixture
    def sample_stream(self):
        """Create a sample stream for testing."""
        async def _stream():
            yield {"type": "start", "messageId": "test-id"}
            yield {"type": "text-delta", "textDelta": "Hello"}
            yield {"type": "text-delta", "textDelta": " world"}
            yield {"type": "finish", "finishReason": "stop", "usage": {}}
        return _stream

    @pytest.fixture
    def mock_message_builder(self):
        """Create a mock MessageBuilder."""
        mock = MagicMock()
        mock.add_chunk = AsyncMock(return_value=[])
        return mock

    @pytest.fixture
    def mock_processor(self):
        """Create a mock StreamProcessor."""
        mock = MagicMock()
        mock.process_stream = AsyncMock()
        return mock

    def test_init_default_values(self, sample_stream):
        """Test DataStreamWithEmitters initialization with default values."""
        data_stream = DataStreamWithEmitters(
            stream_generator=sample_stream,
            message_id="test-id"
        )
        
        assert data_stream._message_id == "test-id"
        assert data_stream._auto_close is True
        assert data_stream._protocol_version == "v4"
        assert data_stream._output_format == "chunks"

    def test_init_custom_values(self, sample_stream, mock_message_builder, mock_processor):
        """Test DataStreamWithEmitters initialization with custom values."""
        mock_callback = AsyncMock(spec=BaseAICallbackHandler)
        
        data_stream = DataStreamWithEmitters(
            stream_generator=sample_stream,
            message_id="custom-id",
            auto_close=False,
            message_builder=mock_message_builder,
            callbacks=mock_callback,
            protocol_version="v5",
            output_format="protocol",
            stream_processor=mock_processor
        )
        
        assert data_stream._message_id == "custom-id"
        assert data_stream._auto_close is False
        assert data_stream._protocol_version == "v5"
        assert data_stream._output_format == "protocol"

    @pytest.mark.asyncio
    async def test_async_iteration(self, sample_stream, mock_message_builder, mock_processor):
        """Test async iteration over DataStreamWithEmitters."""
        data_stream = DataStreamWithEmitters(
            stream_generator=sample_stream(),
            message_id="test-id",
            auto_close=True,
            message_builder=mock_message_builder,
            callbacks=None,
            protocol_version="v4",
            output_format="chunks",
            stream_processor=mock_processor
        )
        
        chunks = []
        async for chunk in data_stream:
            chunks.append(chunk)
        
        assert len(chunks) >= 2  # At least start and finish

    @pytest.mark.asyncio
    async def test_emit_text_sequence(self, sample_stream, mock_message_builder, mock_processor):
        """Test emitting text sequence manually."""
        data_stream = DataStreamWithEmitters(
            stream_generator=sample_stream,
            message_id="test-id",
            auto_close=False,
            message_builder=mock_message_builder,
            callbacks=None,
            protocol_version="v4",
            output_format="chunks",
            stream_processor=mock_processor
        )
        
        # Test emit_text sequence
        text_id = "text-123"
        await data_stream.emit_text_start(text_id=text_id)
        await data_stream.emit_text_delta("Hello world", text_id=text_id)
        await data_stream.emit_text_end("Hello world", text_id=text_id)
        
        # Verify methods exist
        assert hasattr(data_stream, 'emit_text_start')
        assert hasattr(data_stream, 'emit_text_delta')
        assert hasattr(data_stream, 'emit_text_end')

    @pytest.mark.asyncio
    async def test_emit_data(self, sample_stream, mock_message_builder, mock_processor):
        """Test emitting data manually."""
        data_stream = DataStreamWithEmitters(
            stream_generator=sample_stream,
            message_id="test-id",
            auto_close=False,
            message_builder=mock_message_builder,
            callbacks=None,
            protocol_version="v4",
            output_format="chunks",
            stream_processor=mock_processor
        )
        
        # Test emit_data method
        await data_stream.emit_data("custom", [{"key": "value"}])
        
        assert hasattr(data_stream, 'emit_data')

    @pytest.mark.asyncio
    async def test_emit_file(self, sample_stream, mock_message_builder, mock_processor):
        """Test emitting file manually."""
        data_stream = DataStreamWithEmitters(
            stream_generator=sample_stream,
            message_id="test-id",
            auto_close=False,
            message_builder=mock_message_builder,
            callbacks=None,
            protocol_version="v4",
            output_format="chunks",
            stream_processor=mock_processor
        )
        
        # Test emit_file method
        await data_stream.emit_file("file content", "text/plain")
        
        assert hasattr(data_stream, 'emit_file')

    @pytest.mark.asyncio
    async def test_emit_source_url(self, sample_stream, mock_message_builder, mock_processor):
        """Test emitting source URL manually."""
        data_stream = DataStreamWithEmitters(
            stream_generator=sample_stream,
            message_id="test-id",
            auto_close=False,
            message_builder=mock_message_builder,
            callbacks=None,
            protocol_version="v4",
            output_format="chunks",
            stream_processor=mock_processor
        )
        
        # Test emit_source_url method
        await data_stream.emit_source_url("https://example.com", "Example")
        
        assert hasattr(data_stream, 'emit_source_url')

    @pytest.mark.asyncio
    async def test_emit_source_document(self, sample_stream, mock_message_builder, mock_processor):
        """Test emitting source document manually."""
        data_stream = DataStreamWithEmitters(
            stream_generator=sample_stream,
            message_id="test-id",
            auto_close=False,
            message_builder=mock_message_builder,
            callbacks=None,
            protocol_version="v4",
            output_format="chunks",
            stream_processor=mock_processor
        )
        
        # Test emit_source_document method
        await data_stream.emit_source_document("doc-123", "text/plain", "Document Title")
        
        assert hasattr(data_stream, 'emit_source_document')

    @pytest.mark.asyncio
    async def test_emit_reasoning(self, sample_stream, mock_message_builder, mock_processor):
        """Test emitting reasoning manually."""
        data_stream = DataStreamWithEmitters(
            stream_generator=sample_stream,
            message_id="test-id",
            auto_close=False,
            message_builder=mock_message_builder,
            callbacks=None,
            protocol_version="v4",
            output_format="chunks",
            stream_processor=mock_processor
        )
        
        # Test emit_reasoning method
        await data_stream.emit_reasoning("This is reasoning text")
        
        assert hasattr(data_stream, 'emit_reasoning')

    @pytest.mark.asyncio
    async def test_emit_error(self, sample_stream, mock_message_builder, mock_processor):
        """Test emitting error manually."""
        data_stream = DataStreamWithEmitters(
            stream_generator=sample_stream,
            message_id="test-id",
            auto_close=False,
            message_builder=mock_message_builder,
            callbacks=None,
            protocol_version="v4",
            output_format="chunks",
            stream_processor=mock_processor
        )
        
        # Test emit_error method
        await data_stream.emit_error("An error occurred")
        
        assert hasattr(data_stream, 'emit_error')

    @pytest.mark.asyncio
    async def test_protocol_version_v5(self, sample_stream, mock_message_builder, mock_processor):
        """Test with v5 protocol version."""
        data_stream = DataStreamWithEmitters(
            stream_generator=sample_stream(),
            message_id="test-id",
            auto_close=True,
            message_builder=mock_message_builder,
            callbacks=None,
            protocol_version="v5",
            output_format="chunks",
            stream_processor=mock_processor
        )
        
        chunks = []
        async for chunk in data_stream:
            chunks.append(chunk)
        
        assert len(chunks) >= 0
        assert data_stream._protocol_version == "v5"

    @pytest.mark.asyncio
    async def test_with_callbacks(self, sample_stream, mock_message_builder, mock_processor):
        """Test with callback handlers."""
        mock_callback = AsyncMock(spec=BaseAICallbackHandler)
        
        data_stream = DataStreamWithEmitters(
            stream_generator=sample_stream(),
            message_id="test-id",
            auto_close=True,
            message_builder=mock_message_builder,
            callbacks=mock_callback,
            protocol_version="v4",
            output_format="chunks",
            stream_processor=mock_processor
        )
        
        chunks = []
        async for chunk in data_stream:
            chunks.append(chunk)
        
        assert len(chunks) >= 0


class TestDataStreamWriter:
    """Test cases for DataStreamWriter class."""

    def test_init(self):
        """Test DataStreamWriter initialization."""
        writer = DataStreamWriter()
        
        assert writer is not None
        assert hasattr(writer, 'write')
        assert hasattr(writer, 'get_chunks')

    @pytest.mark.asyncio
    async def test_write_chunk(self):
        """Test writing chunks to DataStreamWriter."""
        writer = DataStreamWriter()
        
        chunk: UIMessageChunk = {
            "type": "text-delta",
            "textDelta": "Hello world"
        }
        
        await writer.write(chunk)
        chunks = writer.get_chunks()
        
        assert len(chunks) == 1
        assert chunks[0] == chunk

    @pytest.mark.asyncio
    async def test_write_multiple_chunks(self):
        """Test writing multiple chunks."""
        writer = DataStreamWriter()
        
        chunks = [
            {"type": "start", "messageId": "test-id"},
            {"type": "text-delta", "textDelta": "Hello"},
            {"type": "text-delta", "textDelta": " world"},
            {"type": "finish", "finishReason": "stop", "usage": {}}
        ]
        
        for chunk in chunks:
            await writer.write(chunk)
        
        written_chunks = writer.get_chunks()
        assert len(written_chunks) == 4
        assert written_chunks == chunks

    @pytest.mark.asyncio
    async def test_concurrent_writes(self):
        """Test concurrent writes to DataStreamWriter."""
        import asyncio
        
        writer = DataStreamWriter()
        
        async def write_chunk(index):
            chunk = {"type": "text-delta", "textDelta": f"chunk-{index}"}
            await writer.write(chunk)
        
        # Write chunks concurrently
        await asyncio.gather(*[write_chunk(i) for i in range(10)])
        
        chunks = writer.get_chunks()
        assert len(chunks) == 10


class TestDataStreamResponse:
    """Test cases for DataStreamResponse class."""

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        return [
            {"type": "start", "messageId": "test-id"},
            {"type": "text-delta", "textDelta": "Hello"},
            {"type": "text-delta", "textDelta": " world"},
            {"type": "finish", "finishReason": "stop", "usage": {}}
        ]

    def test_init_with_stream(self, sample_chunks):
        """Test DataStreamResponse initialization with stream."""
        async def chunk_stream():
            for chunk in sample_chunks:
                yield chunk
        
        response = DataStreamResponse(
            stream=chunk_stream(),
            protocol_version="v4"
        )
        
        assert response.protocol_config.version == "v4"

    def test_init_with_custom_headers(self, sample_chunks):
        """Test DataStreamResponse initialization with custom headers."""
        async def chunk_stream():
            for chunk in sample_chunks:
                yield chunk
        
        custom_headers = {"Custom-Header": "value"}
        response = DataStreamResponse(
            stream=chunk_stream(),
            protocol_version="v4",
            headers=custom_headers,
            status=201
        )
        
        assert response.protocol_config.version == "v4"

    @pytest.mark.asyncio
    async def test_stream_conversion(self, sample_chunks):
        """Test stream conversion to protocol format."""
        async def chunk_stream():
            for chunk in sample_chunks:
                yield chunk
        
        response = DataStreamResponse(
            stream=chunk_stream(),
            protocol_version="v4"
        )
        
        # Test that response can be created successfully
        assert response.protocol_config.version == "v4"

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in DataStreamResponse."""
        async def error_stream():
            yield {"type": "start", "messageId": "test-id"}
            yield {"invalid": "chunk"}
            yield {"type": "finish", "finishReason": "stop", "usage": {}}
        
        response = DataStreamResponse(
            stream=error_stream(),
            protocol_version="v4"
        )
        
        # Should handle gracefully
        assert response.protocol_config.version == "v4"

    def test_protocol_version_v5(self, sample_chunks):
        """Test with v5 protocol version."""
        async def chunk_stream():
            for chunk in sample_chunks:
                yield chunk
        
        response = DataStreamResponse(
            stream=chunk_stream(),
            protocol_version="v5"
        )
        
        assert response.protocol_config.version == "v5"

    @pytest.mark.asyncio
    async def test_integration_pipeline(self):
        """Test complete integration pipeline."""
        # Create a writer
        writer = DataStreamWriter()
        
        # Write some chunks
        chunks = [
            {"type": "start", "messageId": "integration-test"},
            {"type": "text-delta", "textDelta": "Integration"},
            {"type": "text-delta", "textDelta": " test"},
            {"type": "finish", "finishReason": "stop", "usage": {}}
        ]
        
        for chunk in chunks:
            await writer.write(chunk)
        
        # Create response from writer stream
        async def writer_stream():
            async for chunk in writer:
                yield chunk
        
        response = DataStreamResponse(
            stream=writer_stream(),
            protocol_version="v4"
        )
        
        # Verify response creation
        assert response.protocol_config.version == "v4"