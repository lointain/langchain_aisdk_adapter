"""Test adapter module

Test the AISDKAdapter class and its streaming functionality.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage, HumanMessage, AIMessageChunk
from langchain_core.outputs import LLMResult, Generation

from langchain_aisdk_adapter.adapter import AISDKAdapter


class TestAISDKAdapter:
    """Test AISDKAdapter class"""
    
    async def mock_langchain_stream(self, chunks):
        """Create a mock LangChain stream"""
        for chunk in chunks:
            yield chunk
    
    def test_adapter_class_exists(self):
        """Test that AISDKAdapter class exists"""
        assert AISDKAdapter is not None
        assert hasattr(AISDKAdapter, 'astream')
    
    @pytest.mark.asyncio
    async def test_astream_basic(self):
        """Test basic streaming functionality"""
        # Mock AIMessageChunk objects with proper attributes
        mock_chunks = [
            AIMessageChunk(content="Hello", tool_call_chunks=[]),
            AIMessageChunk(content=" world", tool_call_chunks=[]),
            AIMessageChunk(content="!", tool_call_chunks=[])
        ]
        
        # Create mock stream
        mock_stream = self.mock_langchain_stream(mock_chunks)
        
        # Collect streamed parts
        parts = []
        async for part in AISDKAdapter.astream(mock_stream):
            parts.append(part)
        
        # Verify we got AI SDK protocol strings
        assert len(parts) >= 3
        
        # Check that parts are strings with AI SDK protocol format
        for part in parts:
            assert isinstance(part, str)
            assert part.startswith('0:')  # Text parts start with '0:'
    
    @pytest.mark.asyncio
    async def test_astream_with_finish_reason(self):
        """Test streaming with finish reason"""
        # Mock chunk with finish reason
        mock_chunk = AIMessageChunk(
            content="Done",
            tool_call_chunks=[],
            usage_metadata={
                "finish_reason": "stop",
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15
            }
        )
        
        mock_stream = self.mock_langchain_stream([mock_chunk])
        
        parts = []
        async for part in AISDKAdapter.astream(mock_stream):
            parts.append(part)
        
        # Should have at least text parts
        text_parts = [p for p in parts if p.startswith('0:')]
        
        assert len(text_parts) > 0
        assert any('Done' in p for p in text_parts)
    
    @pytest.mark.asyncio
    async def test_astream_empty_content(self):
        """Test streaming with empty content chunks"""
        mock_chunks = [
            AIMessageChunk(content="", tool_call_chunks=[]),
            AIMessageChunk(content="Hello", tool_call_chunks=[]),
            AIMessageChunk(content="", tool_call_chunks=[])
        ]
        
        mock_stream = self.mock_langchain_stream(mock_chunks)
        
        parts = []
        async for part in AISDKAdapter.astream(mock_stream):
            parts.append(part)
        
        # Should only get parts for non-empty content
        text_parts = [p for p in parts if p.startswith('0:')]
        assert len(text_parts) >= 1
        assert any('Hello' in p for p in text_parts)
    
    @pytest.mark.asyncio
    async def test_astream_with_string_chunks(self):
        """Test streaming with string chunks"""
        mock_chunks = ["Hello", " world", "!"]
        
        mock_stream = self.mock_langchain_stream(mock_chunks)
        
        parts = []
        async for part in AISDKAdapter.astream(mock_stream):
            parts.append(part)
        
        # Verify we got AI SDK protocol strings
        assert len(parts) == 3
        for part in parts:
            assert isinstance(part, str)
            assert part.startswith('0:')  # Text parts start with '0:'
    
    @pytest.mark.asyncio
    async def test_astream_with_agent_executor_output(self):
        """Test streaming with AgentExecutor output format"""
        mock_chunks = [{"output": "Agent response"}]
        
        mock_stream = self.mock_langchain_stream(mock_chunks)
        
        parts = []
        async for part in AISDKAdapter.astream(mock_stream):
            parts.append(part)
        
        # Should get text part for agent output
        assert len(parts) >= 1
        assert any('Agent response' in part for part in parts)
    
    @pytest.mark.asyncio
    async def test_astream_with_pre_formatted_protocol(self):
        """Test streaming with pre-formatted AI SDK protocol strings"""
        mock_chunks = ['0:"Hello"\n', '0:" world"\n']
        
        mock_stream = self.mock_langchain_stream(mock_chunks)
        
        parts = []
        async for part in AISDKAdapter.astream(mock_stream):
            parts.append(part)
        
        # Should pass through pre-formatted strings unchanged
        assert len(parts) == 2
        assert parts[0] == '0:"Hello"\n'
        assert parts[1] == '0:" world"\n'
    
    @pytest.mark.asyncio
    async def test_astream_with_langgraph_events(self):
        """Test streaming with LangGraph events"""
        # Mock LangGraph event with nested AIMessageChunk
        langgraph_event = {
            "event": "on_llm_stream",
            "name": "llm",
            "tags": [],
            "data": {
                "chunk": AIMessageChunk(content="LangGraph response", tool_call_chunks=[])
            },
            "run_id": "run_123"
        }
        
        mock_stream = self.mock_langchain_stream([langgraph_event])
        
        parts = []
        async for part in AISDKAdapter.astream(mock_stream):
            parts.append(part)
        
        # Should process the nested AIMessageChunk
        assert len(parts) >= 1
        assert any('LangGraph response' in part for part in parts)
    
    @pytest.mark.asyncio
    async def test_astream_with_langgraph_tool_events(self):
        """Test streaming with LangGraph tool events"""
        # Mock LangGraph tool start event
        tool_start_event = {
            "event": "on_tool_start",
            "name": "search_tool",
            "tags": [],
            "data": {
                "tool_name": "search",
                "input": {"query": "test"}
            },
            "run_id": "tool_123"
        }
        
        # Mock LangGraph tool end event
        tool_end_event = {
            "event": "on_tool_end",
            "name": "search_tool",
            "tags": [],
            "data": {
                "output": "Search results"
            },
            "run_id": "tool_123"
        }
        
        mock_stream = self.mock_langchain_stream([tool_start_event, tool_end_event])
        
        parts = []
        async for part in AISDKAdapter.astream(mock_stream):
            parts.append(part)
        
        # Should have tool call and tool result parts
        assert len(parts) >= 2
        
        # Check for tool call parts (type 9)
        tool_call_parts = [p for p in parts if p.startswith('9:')]
        assert len(tool_call_parts) >= 1
        
        # Check for tool result parts (type a)
        tool_result_parts = [p for p in parts if p.startswith('a:')]
        assert len(tool_result_parts) >= 1
    
    @pytest.mark.asyncio
    async def test_astream_with_unknown_chunk_type(self):
        """Test streaming with unknown chunk types"""
        # Mock unknown chunk type
        unknown_chunk = {"unknown": "data", "type": "mystery"}
        
        mock_stream = self.mock_langchain_stream([unknown_chunk])
        
        parts = []
        async for part in AISDKAdapter.astream(mock_stream):
            parts.append(part)
        
        # Should handle unknown chunks gracefully (may not produce output)
        # This test mainly ensures no exceptions are raised
        assert isinstance(parts, list)
    
    @pytest.mark.asyncio
    async def test_astream_error_handling(self):
        """Test error handling in streaming"""
        async def error_stream():
            yield AIMessageChunk(
                content="Hello", 
                tool_call_chunks=[],
                usage_metadata={
                    "input_tokens": 5,
                    "output_tokens": 3,
                    "total_tokens": 8
                }
            )
            raise ValueError("Test error")
        
        parts = []
        async for part in AISDKAdapter.astream(error_stream()):
            parts.append(part)
        
        # Should have at least the first chunk and an error part
        assert len(parts) >= 2
        
        # Should have text part for "Hello"
        text_parts = [p for p in parts if p.startswith('0:') and 'Hello' in p]
        assert len(text_parts) >= 1
        
        # Should have error part (type 3)
        error_parts = [p for p in parts if p.startswith('3:')]
        assert len(error_parts) >= 1
        assert any('Test error' in p for p in error_parts)
    
    @pytest.mark.asyncio
    async def test_process_stream_chunk_directly(self):
        """Test _process_stream_chunk function directly"""
        from langchain_aisdk_adapter.adapter import _process_stream_chunk
        from langchain_aisdk_adapter.utils import AIStreamState
        
        state = AIStreamState()
        
        # Test with AIMessageChunk
        chunk = AIMessageChunk(
            content="Direct test", 
            tool_call_chunks=[],
            usage_metadata={
                "input_tokens": 5,
                "output_tokens": 3,
                "total_tokens": 8
            }
        )
        
        results = []
        async for result in _process_stream_chunk(chunk, state):
            results.append(result)
        
        assert len(results) >= 1
        assert state.text_sent is True
        assert any('Direct test' in r for r in results)
        
        # Test with string chunk
        state = AIStreamState()
        results = []
        async for result in _process_stream_chunk("String chunk", state):
            results.append(result)
        
        assert len(results) >= 1
        assert state.text_sent is True
        assert any('String chunk' in r for r in results)
        
        # Test with pre-formatted AI SDK string
        state = AIStreamState()
        results = []
        async for result in _process_stream_chunk('0:"Pre-formatted"\n', state):
            results.append(result)
        
        assert len(results) == 1
        assert results[0] == '0:"Pre-formatted"\n'
        
        # Test with AgentExecutor output
        state = AIStreamState()
        results = []
        async for result in _process_stream_chunk({"output": "Agent output"}, state):
            results.append(result)
        
        assert len(results) >= 1
        assert state.text_sent is True
        assert any('Agent output' in r for r in results)
    
    @pytest.mark.asyncio
    async def test_astream_with_tool_call_chunks(self):
        """Test streaming with tool call chunks"""
        # Mock AIMessageChunk with tool calls
        tool_chunk = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"id": "call_123", "name": "search", "args": '{"query": "test"}'}
            ],
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "finish_reason": "tool_calls"
            }
        )
        
        mock_stream = self.mock_langchain_stream([tool_chunk])
        
        parts = []
        async for part in AISDKAdapter.astream(mock_stream):
            parts.append(part)
        
        # Should have tool call parts
        assert len(parts) >= 1
        
        # Check for tool call parts (type 9)
        tool_call_parts = [p for p in parts if p.startswith('9:')]
        assert len(tool_call_parts) >= 1
    
    @pytest.mark.asyncio
    async def test_astream_with_document_output(self):
        """Test streaming with Document output from tools"""
        from langchain_core.documents import Document
        
        # Mock LangGraph tool end event with Document output
        doc = Document(
            page_content="Document content",
            metadata={"source": "http://example.com", "title": "Test Doc"}
        )
        
        tool_end_event = {
            "event": "on_tool_end",
            "name": "retriever",
            "tags": [],
            "data": {
                "output": [doc]
            },
            "run_id": "tool_123"
        }
        
        mock_stream = self.mock_langchain_stream([tool_end_event])
        
        parts = []
        async for part in AISDKAdapter.astream(mock_stream):
            parts.append(part)
        
        # Should have source parts (type h) and tool result parts (type a)
        assert len(parts) >= 2
        
        # Check for source parts
        source_parts = [p for p in parts if p.startswith('h:')]
        assert len(source_parts) >= 1
        
        # Check for tool result parts
        tool_result_parts = [p for p in parts if p.startswith('a:')]
        assert len(tool_result_parts) >= 1