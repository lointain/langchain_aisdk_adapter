"""Test models

Test creation, validation and serialization of all AI SDK protocol related Pydantic models.
"""

import pytest
from typing import Any, Dict

from langchain_aisdk_adapter.models import (
    TextPartContent,
    ReasoningPartContent,
    SourcePartContent,
    FilePartContent,
    DataPartContent,
    MessageAnnotationPartContent,
    ErrorPartContent,
    ToolCallStreamingStartPartContent,
    ToolCallDeltaPartContent,
    ToolCallPartContent,
    ToolResultPartContent,
    StartStepPartContent,
    FinishStepPartContent,
    FinishMessagePartContent,
    UsageInfo,
)


class TestAISDKPartContent:
    """Test basic AI SDK part content models"""
    
    def test_text_part_creation(self):
        """Test text part creation"""
        text_part = TextPartContent(root="Hello, world!")
        assert text_part.root == "Hello, world!"
        assert text_part.model_dump() == "Hello, world!"
    
    def test_text_part_with_special_characters(self):
        """Test text part with special characters"""
        text_part = TextPartContent(root='Hello "world" with \n newline')
        assert text_part.root == 'Hello "world" with \n newline'
        assert text_part.model_dump() == 'Hello "world" with \n newline'
    
    def test_reasoning_part_creation(self):
        """Test reasoning part creation"""
        reasoning_part = ReasoningPartContent(root="Let me think...")
        assert reasoning_part.root == "Let me think..."
        assert reasoning_part.model_dump() == "Let me think..."
    
    def test_source_part_creation(self):
        """Test source part creation"""
        source_part = SourcePartContent(
            url="https://wikipedia.org",
            title="Wikipedia"
        )
        assert source_part.url == "https://wikipedia.org"
        assert source_part.title == "Wikipedia"
        expected = {"url": "https://wikipedia.org", "title": "Wikipedia"}
        assert source_part.model_dump() == expected
    
    def test_file_part_creation(self):
        """Test file part creation"""
        file_part = FilePartContent(
            data="dGVzdCBkYXRh",  # base64 encoded "test data"
            mimeType="text/plain"
        )
        assert file_part.data == "dGVzdCBkYXRh"
        assert file_part.mimeType == "text/plain"
        expected = {"data": "dGVzdCBkYXRh", "mimeType": "text/plain"}
        assert file_part.model_dump() == expected
    
    def test_data_part_creation(self):
        """Test data part creation"""
        data = ["key", "value", 42]
        data_part = DataPartContent(root=data)
        assert data_part.root == data
        assert data_part.model_dump() == data
    
    def test_error_part_creation(self):
        """Test error part creation"""
        error_part = ErrorPartContent(root="Something went wrong")
        assert error_part.root == "Something went wrong"
        assert error_part.model_dump() == "Something went wrong"
    
    def test_tool_call_part_creation(self):
        """Test tool call part creation"""
        tool_call_part = ToolCallPartContent(
            toolCallId="call_123",
            toolName="search",
            args={"query": "Python"}
        )
        assert tool_call_part.toolCallId == "call_123"
        assert tool_call_part.toolName == "search"
        assert tool_call_part.args == {"query": "Python"}
        expected = {"toolCallId": "call_123", "toolName": "search", "args": {"query": "Python"}}
        assert tool_call_part.model_dump() == expected
    
    def test_tool_result_part_creation(self):
        """Test tool result part creation"""
        tool_result_part = ToolResultPartContent(
            toolCallId="call_123",
            result="Search completed"
        )
        assert tool_result_part.toolCallId == "call_123"
        assert tool_result_part.result == "Search completed"
        expected = {"toolCallId": "call_123", "result": "Search completed"}
        assert tool_result_part.model_dump() == expected
    
    def test_finish_message_part_creation(self):
        """Test finish message part creation"""
        usage = UsageInfo(promptTokens=10, completionTokens=20)
        finish_part = FinishMessagePartContent(
            finishReason="stop",
            usage=usage
        )
        assert finish_part.finishReason == "stop"
        assert finish_part.usage.promptTokens == 10
        assert finish_part.usage.completionTokens == 20
        expected = {"finishReason": "stop", "usage": {"promptTokens": 10, "completionTokens": 20}}
        assert finish_part.model_dump() == expected


class TestModelValidation:
    """Test model validation"""
    
    def test_text_part_empty_text(self):
        """Test empty text validation"""
        text_part = TextPartContent(root="")
        assert text_part.root == ""
        assert text_part.model_dump() == ""
    
    def test_tool_call_part_validation(self):
        """Test tool call part validation"""
        # Test that empty toolCallId is allowed
        tool_call_part = ToolCallPartContent(toolCallId="", toolName="search", args={})
        assert tool_call_part.toolCallId == ""
        assert tool_call_part.toolName == "search"
        assert tool_call_part.args == {}
    
    def test_source_part_validation(self):
        """Test source part validation"""
        # Test required fields
        source_part = SourcePartContent(url="https://test.com", title="Test")
        assert source_part.url == "https://test.com"
        assert source_part.title == "Test"


class TestJSONSerialization:
    """Test JSON serialization"""
    
    def test_complex_data_serialization(self):
        """Test complex data serialization"""
        complex_data = [
            {
                "nested": {
                    "array": [1, 2, 3],
                    "string": "test",
                    "boolean": True,
                    "null": None
                }
            }
        ]
        data_part = DataPartContent(root=complex_data)
        content = data_part.model_dump()
        assert "nested" in str(content)
        assert "array" in str(content)
        assert content == complex_data
    
    def test_unicode_handling(self):
        """Test Unicode character handling"""
        text_part = TextPartContent(root="你好世界 🌍")
        content = text_part.model_dump()
        assert content == "你好世界 🌍"
        assert "你好世界 🌍" in content