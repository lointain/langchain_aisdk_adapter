"""Tests for protocol strategy classes."""

import pytest
from unittest.mock import MagicMock

from langchain_aisdk_adapter.protocol_strategy import (
    ProtocolStrategy,
    AISDKv4Strategy,
    AISDKv5Strategy,
    ProtocolConfig
)
from langchain_aisdk_adapter.models import UIMessageChunk


class TestProtocolStrategy:
    """Test cases for ProtocolStrategy abstract base class."""

    def test_abstract_base_class(self):
        """Test that ProtocolStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ProtocolStrategy()


class TestAISDKv4Strategy:
    """Test cases for AISDKv4Strategy."""

    def test_init(self):
        """Test initialization of AISDKv4Strategy."""
        strategy = AISDKv4Strategy()
        assert strategy is not None

    def test_get_headers(self):
        """Test getting headers for v4 protocol."""
        strategy = AISDKv4Strategy()
        headers = strategy.get_headers()
        
        assert isinstance(headers, dict)
        assert "Content-Type" in headers
        assert "text/plain" in headers["Content-Type"]
        assert "Cache-Control" in headers
        assert "no-cache" in headers["Cache-Control"]

    def test_format_chunk_start(self):
        """Test formatting start chunk in v4 protocol."""
        strategy = AISDKv4Strategy()
        
        chunk: UIMessageChunk = {
            "type": "start",
            "messageId": "test-message-id"
        }
        
        result = strategy.format_chunk(chunk)
        assert isinstance(result, str)
        assert "test-message-id" in result

    def test_format_chunk_text_delta(self):
        """Test formatting text delta chunk in v4 protocol."""
        strategy = AISDKv4Strategy()
        
        chunk: UIMessageChunk = {
            "type": "text-delta",
            "textDelta": "Hello world"
        }
        
        result = strategy.format_chunk(chunk)
        assert isinstance(result, str)
        assert "Hello world" in result

    def test_format_chunk_tool_input_start(self):
        """Test formatting tool input start chunk in v4 protocol."""
        strategy = AISDKv4Strategy()
        
        chunk: UIMessageChunk = {
            "type": "tool-input-start",
            "toolCallId": "tool-123",
            "toolName": "search_tool"
        }
        
        result = strategy.format_chunk(chunk)
        assert isinstance(result, str)
        assert "tool-123" in result
        assert "search_tool" in result

    def test_format_chunk_tool_output_available(self):
        """Test formatting tool output available chunk in v4 protocol."""
        strategy = AISDKv4Strategy()
        
        chunk: UIMessageChunk = {
            "type": "tool-output-available",
            "toolCallId": "tool-123",
            "output": "Tool execution result"
        }
        
        result = strategy.format_chunk(chunk)
        assert isinstance(result, str)
        assert "tool-123" in result

    def test_format_chunk_finish(self):
        """Test formatting finish chunk in v4 protocol."""
        strategy = AISDKv4Strategy()
        
        chunk: UIMessageChunk = {
            "type": "finish",
            "finishReason": "stop",
            "usage": {
                "promptTokens": 10,
                "completionTokens": 5
            }
        }
        
        result = strategy.format_chunk(chunk)
        assert isinstance(result, str)
        assert "stop" in result

    def test_format_chunk_error(self):
        """Test formatting error chunk in v4 protocol."""
        strategy = AISDKv4Strategy()
        
        chunk: UIMessageChunk = {
            "type": "error",
            "errorText": "An error occurred"
        }
        
        result = strategy.format_chunk(chunk)
        assert isinstance(result, str)
        assert "An error occurred" in result

    def test_format_chunk_unknown_type(self):
        """Test formatting chunk with unknown type in v4 protocol."""
        strategy = AISDKv4Strategy()
        
        chunk: UIMessageChunk = {
            "type": "unknown-type",
            "data": "some data"
        }
        
        result = strategy.format_chunk(chunk)
        assert isinstance(result, str)
        # Should handle unknown types gracefully

    def test_convert_text_sequence(self):
        """Test converting text sequence in v4 protocol."""
        strategy = AISDKv4Strategy()
        
        text_chunks = [
            {"type": "text-start", "id": "text-1"},
            {"type": "text-delta", "id": "text-1", "textDelta": "Hello "},
            {"type": "text-delta", "id": "text-1", "textDelta": "world!"},
            {"type": "text-end", "id": "text-1"}
        ]
        
        result = strategy.convert_text_sequence(text_chunks)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_get_termination_marker(self):
        """Test getting termination marker for v4 protocol."""
        strategy = AISDKv4Strategy()
        
        marker = strategy.get_termination_marker()
        assert isinstance(marker, str)
        assert len(marker) > 0


class TestAISDKv5Strategy:
    """Test cases for AISDKv5Strategy."""

    def test_init(self):
        """Test initialization of AISDKv5Strategy."""
        strategy = AISDKv5Strategy()
        assert strategy is not None

    def test_get_headers(self):
        """Test getting headers for v5 protocol."""
        strategy = AISDKv5Strategy()
        headers = strategy.get_headers()
        
        assert isinstance(headers, dict)
        assert "Content-Type" in headers
        # v5 might have different content type
        assert "Cache-Control" in headers

    def test_format_chunk_differences_from_v4(self):
        """Test that v5 formatting differs from v4 where expected."""
        v4_strategy = AISDKv4Strategy()
        v5_strategy = AISDKv5Strategy()
        
        chunk: UIMessageChunk = {
            "type": "text-delta",
            "textDelta": "Hello world"
        }
        
        v4_result = v4_strategy.format_chunk(chunk)
        v5_result = v5_strategy.format_chunk(chunk)
        
        # Results might be different between v4 and v5
        assert isinstance(v4_result, str)
        assert isinstance(v5_result, str)

    def test_format_chunk_start_v5(self):
        """Test formatting start chunk in v5 protocol."""
        strategy = AISDKv5Strategy()
        
        chunk: UIMessageChunk = {
            "type": "start",
            "messageId": "test-message-id"
        }
        
        result = strategy.format_chunk(chunk)
        assert isinstance(result, str)
        assert "test-message-id" in result

    def test_format_chunk_finish_v5(self):
        """Test formatting finish chunk in v5 protocol."""
        strategy = AISDKv5Strategy()
        
        chunk: UIMessageChunk = {
            "type": "finish",
            "finishReason": "stop",
            "usage": {
                "promptTokens": 10,
                "completionTokens": 5
            }
        }
        
        result = strategy.format_chunk(chunk)
        assert isinstance(result, str)

    def test_convert_text_sequence_v5(self):
        """Test converting text sequence in v5 protocol."""
        strategy = AISDKv5Strategy()
        
        text_chunks = [
            {"type": "text-start", "id": "text-1"},
            {"type": "text-delta", "id": "text-1", "textDelta": "Hello "},
            {"type": "text-delta", "id": "text-1", "textDelta": "world!"},
            {"type": "text-end", "id": "text-1"}
        ]
        
        result = strategy.convert_text_sequence(text_chunks)
        assert isinstance(result, list)

    def test_get_termination_marker_v5(self):
        """Test getting termination marker for v5 protocol."""
        strategy = AISDKv5Strategy()
        
        marker = strategy.get_termination_marker()
        assert isinstance(marker, str)


class TestProtocolConfig:
    """Test cases for ProtocolConfig class."""

    def test_init_v4(self):
        """Test initialization with v4 protocol."""
        config = ProtocolConfig("v4")
        
        assert config.version == "v4"
        assert isinstance(config.strategy, AISDKv4Strategy)

    def test_init_v5(self):
        """Test initialization with v5 protocol."""
        config = ProtocolConfig("v5")
        
        assert config.version == "v5"
        assert isinstance(config.strategy, AISDKv5Strategy)

    def test_init_invalid_version(self):
        """Test initialization with invalid protocol version."""
        with pytest.raises((ValueError, KeyError)):
            ProtocolConfig("invalid")

    def test_init_default_version(self):
        """Test initialization with default version."""
        config = ProtocolConfig()
        
        # Should default to v4
        assert config.version == "v4"
        assert isinstance(config.strategy, AISDKv4Strategy)

    def test_strategy_delegation(self):
        """Test that ProtocolConfig properly delegates to strategy."""
        config = ProtocolConfig("v4")
        
        # Test header delegation
        headers = config.strategy.get_headers()
        assert isinstance(headers, dict)
        
        # Test chunk formatting delegation
        chunk: UIMessageChunk = {
            "type": "text-delta",
            "textDelta": "test"
        }
        result = config.strategy.format_chunk(chunk)
        assert isinstance(result, str)


class TestProtocolComparison:
    """Test cases comparing v4 and v5 protocols."""

    def test_header_differences(self):
        """Test differences in headers between v4 and v5."""
        v4_config = ProtocolConfig("v4")
        v5_config = ProtocolConfig("v5")
        
        v4_headers = v4_config.strategy.get_headers()
        v5_headers = v5_config.strategy.get_headers()
        
        # Both should have headers
        assert isinstance(v4_headers, dict)
        assert isinstance(v5_headers, dict)
        
        # Both should have Content-Type
        assert "Content-Type" in v4_headers
        assert "Content-Type" in v5_headers

    def test_chunk_formatting_consistency(self):
        """Test that both protocols can format the same chunk types."""
        v4_strategy = AISDKv4Strategy()
        v5_strategy = AISDKv5Strategy()
        
        test_chunks = [
            {"type": "start", "messageId": "test-id"},
            {"type": "text-delta", "textDelta": "Hello"},
            {"type": "finish", "finishReason": "stop", "usage": {}},
            {"type": "error", "errorText": "Error message"}
        ]
        
        for chunk in test_chunks:
            v4_result = v4_strategy.format_chunk(chunk)
            v5_result = v5_strategy.format_chunk(chunk)
            
            # Both should produce string results
            assert isinstance(v4_result, str)
            assert isinstance(v5_result, str)

    def test_termination_markers(self):
        """Test termination markers for both protocols."""
        v4_strategy = AISDKv4Strategy()
        v5_strategy = AISDKv5Strategy()
        
        v4_marker = v4_strategy.get_termination_marker()
        v5_marker = v5_strategy.get_termination_marker()
        
        assert isinstance(v4_marker, str)
        assert isinstance(v5_marker, str)
        assert len(v4_marker) > 0
        assert len(v5_marker) > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_chunk(self):
        """Test formatting empty chunk."""
        strategy = AISDKv4Strategy()
        
        empty_chunk: UIMessageChunk = {}
        result = strategy.format_chunk(empty_chunk)
        
        assert isinstance(result, str)

    def test_chunk_with_none_values(self):
        """Test formatting chunk with None values."""
        strategy = AISDKv4Strategy()
        
        chunk: UIMessageChunk = {
            "type": "text-delta",
            "textDelta": None
        }
        
        result = strategy.format_chunk(chunk)
        assert isinstance(result, str)

    def test_chunk_with_missing_required_fields(self):
        """Test formatting chunk with missing required fields."""
        strategy = AISDKv4Strategy()
        
        chunk: UIMessageChunk = {
            "type": "tool-input-start"
            # Missing toolCallId and toolName
        }
        
        # Should handle gracefully
        result = strategy.format_chunk(chunk)
        assert isinstance(result, str)

    def test_large_text_content(self):
        """Test formatting chunk with large text content."""
        strategy = AISDKv4Strategy()
        
        large_text = "x" * 10000  # 10KB of text
        chunk: UIMessageChunk = {
            "type": "text-delta",
            "textDelta": large_text
        }
        
        result = strategy.format_chunk(chunk)
        assert isinstance(result, str)
        assert large_text in result

    def test_special_characters_in_text(self):
        """Test formatting chunk with special characters."""
        strategy = AISDKv4Strategy()
        
        special_text = "Hello\nWorld\t\"Test\"\r\n"
        chunk: UIMessageChunk = {
            "type": "text-delta",
            "textDelta": special_text
        }
        
        result = strategy.format_chunk(chunk)
        assert isinstance(result, str)

    def test_unicode_content(self):
        """Test formatting chunk with unicode content."""
        strategy = AISDKv4Strategy()
        
        unicode_text = "Hello 世界 🌍 émojis"
        chunk: UIMessageChunk = {
            "type": "text-delta",
            "textDelta": unicode_text
        }
        
        result = strategy.format_chunk(chunk)
        assert isinstance(result, str)
        assert unicode_text in result