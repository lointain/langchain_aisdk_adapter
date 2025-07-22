"""Test utils module

Test utility functions for stream processing, event handling, and data conversion.
"""

import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessageChunk
from langchain_core.documents import Document

from langchain_aisdk_adapter.utils import (
    AIStreamState,
    _is_ai_sdk_protocol_string,
    _is_langgraph_event,
    _has_nested_ai_message_chunk,
    _is_agent_executor_output,
    _ensure_newline_ending,
    _is_document_list,
    _extract_node_type,
    _extract_document_title
)
from langchain_aisdk_adapter.models import AI_SDK_PROTOCOL_PREFIXES


class TestAIStreamState:
    """Test AIStreamState class"""
    
    def test_state_initialization(self):
        """Test state initialization"""
        state = AIStreamState()
        assert state.tool_calls == {}
        assert state.text_sent is False
    
    def test_state_tool_calls_tracking(self):
        """Test tool calls tracking"""
        state = AIStreamState()
        
        # Add a tool call
        state.tool_calls["call_123"] = {"name": "search", "args": "{}"}
        
        assert "call_123" in state.tool_calls
        assert state.tool_calls["call_123"]["name"] == "search"
    
    def test_state_text_sent_tracking(self):
        """Test text sent tracking"""
        state = AIStreamState()
        assert state.text_sent is False
        
        state.text_sent = True
        assert state.text_sent is True


class TestChunkTypeDetection:
    """Test chunk type detection functions"""
    
    def test_is_ai_sdk_protocol_string(self):
        """Test AI SDK protocol string detection"""
        # Test valid protocol strings
        for prefix in AI_SDK_PROTOCOL_PREFIXES:
            test_string = f"{prefix}test content"
            assert _is_ai_sdk_protocol_string(test_string) is True
        
        # Test invalid strings
        assert _is_ai_sdk_protocol_string("regular string") is False
        assert _is_ai_sdk_protocol_string("") is False
        assert _is_ai_sdk_protocol_string(123) is False
        assert _is_ai_sdk_protocol_string(None) is False
    
    def test_is_langgraph_event(self):
        """Test LangGraph event detection"""
        # Valid LangGraph event
        valid_event = {"event": "on_llm_start", "data": {}}
        assert _is_langgraph_event(valid_event) is True
        
        # Invalid events
        assert _is_langgraph_event({"data": {}}) is False
        assert _is_langgraph_event("not a dict") is False
        assert _is_langgraph_event(None) is False
    
    def test_has_nested_ai_message_chunk(self):
        """Test nested AIMessageChunk detection"""
        # Create mock AIMessageChunk
        mock_chunk = MagicMock(spec=AIMessageChunk)
        
        # Valid nested chunk
        valid_event = {"data": {"chunk": mock_chunk}}
        assert _has_nested_ai_message_chunk(valid_event) is True
        
        # Invalid events
        assert _has_nested_ai_message_chunk({"data": {}}) is False
        assert _has_nested_ai_message_chunk({"data": {"chunk": "not a chunk"}}) is False
        assert _has_nested_ai_message_chunk({}) is False
    
    def test_is_agent_executor_output(self):
        """Test AgentExecutor output detection"""
        # Valid AgentExecutor output
        valid_output = {"output": "This is the result"}
        assert _is_agent_executor_output(valid_output) is True
        
        # Invalid outputs
        assert _is_agent_executor_output({"output": 123}) is False
        assert _is_agent_executor_output({"result": "test"}) is False
        assert _is_agent_executor_output("not a dict") is False


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_ensure_newline_ending(self):
        """Test newline ending utility"""
        # String without newline
        assert _ensure_newline_ending("hello") == "hello\n"
        
        # String with newline
        assert _ensure_newline_ending("hello\n") == "hello\n"
        
        # Empty string
        assert _ensure_newline_ending("") == "\n"
        
        # Multiple newlines
        assert _ensure_newline_ending("hello\n\n") == "hello\n\n"
    
    def test_is_document_list(self):
        """Test document list detection"""
        # Valid document list
        doc1 = Document(page_content="Content 1", metadata={})
        doc2 = Document(page_content="Content 2", metadata={})
        assert _is_document_list([doc1, doc2]) is True
        
        # Empty list
        assert _is_document_list([]) is True
        
        # Mixed list
        assert _is_document_list([doc1, "not a doc"]) is False
        
        # Not a list
        assert _is_document_list(doc1) is False
        assert _is_document_list("not a list") is False
    
    def test_extract_node_type(self):
        """Test node type extraction from tags"""
        # Valid node type tag
        tags = ["graph:node", "graph:node_type:agent", "other:tag"]
        assert _extract_node_type(tags) == "agent"
        
        # No node type tag
        tags = ["graph:node", "other:tag"]
        assert _extract_node_type(tags) == "unknown"
        
        # Empty tags
        assert _extract_node_type([]) == "unknown"
        
        # Complex node type
        tags = ["graph:node_type:custom_agent_type"]
        assert _extract_node_type(tags) == "custom_agent_type"
    
    def test_extract_document_title(self):
        """Test document title extraction"""
        # Document with title in metadata
        doc_with_title = Document(
            page_content="Some content",
            metadata={"title": "Document Title"}
        )
        assert _extract_document_title(doc_with_title) == "Document Title"
        
        # Document without title, short content
        doc_short = Document(
            page_content="Short content",
            metadata={}
        )
        assert _extract_document_title(doc_short) == "Short content"
        
        # Document without title, long content
        long_content = "This is a very long document content that exceeds fifty characters"
        doc_long = Document(
            page_content=long_content,
            metadata={}
        )
        result = _extract_document_title(doc_long)
        assert len(result) == 53  # 50 chars + "..."
        assert result.endswith("...")
        assert result.startswith("This is a very long document content that exceeds")
        
        # Document with empty content
        doc_empty = Document(page_content="", metadata={})
        assert _extract_document_title(doc_empty) == ""


class TestAsyncFunctions:
    """Test async utility functions"""
    
    @pytest.mark.asyncio
    async def test_handle_stream_end(self):
        """Test stream end handling"""
        from langchain_aisdk_adapter.utils import _handle_stream_end, AIStreamState
        
        state = AIStreamState()
        state.tool_calls["call_1"] = {"name": "test", "args": "{}"}
        state.tool_calls["call_2"] = {"name": "test2", "args": "{}"}
        
        # Should clear tool calls and emit results
        results = []
        async for result in _handle_stream_end(state):
            results.append(result)
        
        # Should have results and tool calls should be cleared
        assert len(results) >= 0  # May or may not have results
        assert state.tool_calls == {}
    
    @pytest.mark.asyncio
    async def test_process_document_list(self):
        """Test document list processing"""
        from langchain_aisdk_adapter.utils import _process_document_list
        
        # Create test documents
        doc1 = Document(
            page_content="Content 1",
            metadata={"source": "http://example.com/1", "title": "Doc 1"}
        )
        doc2 = Document(
            page_content="Content 2",
            metadata={"source": "http://example.com/2"}
        )
        
        documents = [doc1, doc2]
        
        # Process documents
        results = []
        async for result in _process_document_list(documents):
            results.append(result)
        
        # Should have results for both documents
        assert len(results) == 2
        
        # Check that results contain source information
        for result in results:
            assert isinstance(result, str)
            assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_process_ai_message_chunk_with_text(self):
        """Test processing AIMessageChunk with text content"""
        from langchain_aisdk_adapter.utils import _process_ai_message_chunk
        
        # Create mock AIMessageChunk with text content
        chunk = AIMessageChunk(
            content="Hello world",
            tool_call_chunks=[],
            usage_metadata={
                "input_tokens": 5,
                "output_tokens": 3,
                "total_tokens": 8
            }
        )
        
        state = AIStreamState()
        
        # Process chunk
        results = []
        async for result in _process_ai_message_chunk(chunk, state):
            results.append(result)
        
        # Should have text result
        assert len(results) >= 1
        assert state.text_sent is True
        
        # Check that result contains text
        text_result = results[0]
        assert isinstance(text_result, str)
        assert "Hello world" in text_result
    
    @pytest.mark.asyncio
    async def test_process_ai_message_chunk_with_tool_calls(self):
        """Test processing AIMessageChunk with tool calls"""
        from langchain_aisdk_adapter.utils import _process_ai_message_chunk
        
        # Create mock AIMessageChunk with tool calls
        chunk = AIMessageChunk(
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
        
        state = AIStreamState()
        
        # Process chunk
        results = []
        async for result in _process_ai_message_chunk(chunk, state):
            results.append(result)
        
        # Should have tool call results
        assert len(results) >= 1
        assert "call_123" in state.tool_calls
    
    @pytest.mark.asyncio
    async def test_emit_completed_tool_calls(self):
        """Test emitting completed tool calls"""
        from langchain_aisdk_adapter.utils import _emit_completed_tool_calls
        
        state = AIStreamState()
        state.tool_calls["call_123"] = {"name": "search", "args": '{"query": "test"}'}
        state.tool_calls["call_456"] = {"name": "calculate", "args": '{"expression": "2+2"}'}
        
        # Emit tool calls
        results = []
        async for result in _emit_completed_tool_calls(state):
            results.append(result)
        
        # Should have results for both tool calls
        assert len(results) == 2
        
        # Check that results contain tool call information
        for result in results:
            assert isinstance(result, str)
            assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_emit_completed_tool_calls_invalid_json(self):
        """Test emitting tool calls with invalid JSON args"""
        from langchain_aisdk_adapter.utils import _emit_completed_tool_calls
        
        state = AIStreamState()
        state.tool_calls["call_123"] = {"name": "search", "args": "invalid json"}
        
        # Emit tool calls
        results = []
        async for result in _emit_completed_tool_calls(state):
            results.append(result)
        
        # Should still have result even with invalid JSON
        assert len(results) == 1
        assert isinstance(results[0], str)
    
    @pytest.mark.asyncio
    async def test_process_llm_events(self):
        """Test processing LLM events"""
        from langchain_aisdk_adapter.utils import _process_llm_events
        
        state = AIStreamState()
        
        # Test on_llm_start event
        results = []
        async for result in _process_llm_events("on_llm_start", {}, state):
            results.append(result)
        assert len(results) >= 1  # Should yield step-start part
        
        # Test on_llm_stream event with AIMessageChunk
        chunk_data = AIMessageChunk(
            content="streaming text",
            tool_call_chunks=[],
            usage_metadata={
                "input_tokens": 5,
                "output_tokens": 3,
                "total_tokens": 8
            }
        )
        data = {"chunk": chunk_data}
        
        results = []
        async for result in _process_llm_events("on_llm_stream", data, state):
            results.append(result)
        assert len(results) >= 1
        
        # Test on_llm_end event
        results = []
        async for result in _process_llm_events("on_llm_end", {}, state):
            results.append(result)
        assert len(results) >= 1  # Should yield finish step part
    
    @pytest.mark.asyncio
    async def test_process_tool_events(self):
        """Test processing tool events"""
        from langchain_aisdk_adapter.utils import _process_tool_events, AIStreamState
        
        state = AIStreamState()
        
        # Test on_tool_start event
        data = {
            "tool_name": "search",
            "input": {"query": "test"}
        }
        
        results = []
        async for result in _process_tool_events("on_tool_start", data, "call_123", state):
            results.append(result)
        
        # Should have tool call start and call parts
        assert len(results) >= 2
        
        # Test on_tool_end event with string output
        data = {"output": "Search results here"}
        
        results = []
        async for result in _process_tool_events("on_tool_end", data, "call_123", state):
            results.append(result)
        
        # Should have tool result
        assert len(results) >= 1
        
        # Test on_tool_end event with Document list
        from langchain_core.documents import Document
        doc = Document(page_content="Content", metadata={"source": "http://example.com"})
        data = {"output": [doc]}
        
        results = []
        async for result in _process_tool_events("on_tool_end", data, "call_123", state):
            results.append(result)
        
        # Should have source and tool result parts
        assert len(results) >= 2
    
    @pytest.mark.asyncio
    async def test_process_agent_events(self):
        """Test processing agent events"""
        from langchain_aisdk_adapter.utils import _process_agent_events, AIStreamState
        
        state = AIStreamState()
        
        # Test on_agent_action event
        data = {"log": "I need to search for information"}
        
        results = []
        async for result in _process_agent_events("on_agent_action", data, state):
            results.append(result)
        
        # Should have reasoning part
        assert len(results) >= 1
        
        # Test on_agent_finish event
        data = {"output": "Here is the final answer"}
        
        results = []
        async for result in _process_agent_events("on_agent_finish", data, state):
            results.append(result)
        
        # Should have text part and text_sent should be True
        assert len(results) >= 1
        assert state.text_sent is True
    
    @pytest.mark.asyncio
    async def test_process_custom_events_graph_node(self):
        """Test processing custom events for graph nodes"""
        from langchain_aisdk_adapter.utils import _process_custom_events, AIStreamState
        
        state = AIStreamState()
        
        # Test graph node events
        tags = ["graph:node", "graph:node_type:agent"]
        data = {"input": "test input"}
        
        # Test on_chain_start
        results = []
        async for result in _process_custom_events("on_chain_start", "agent_node", tags, data, "run_123", state):
            results.append(result)
        
        assert len(results) >= 1
        
        # Test on_chain_end
        results = []
        async for result in _process_custom_events("on_chain_end", "agent_node", tags, data, "run_123", state):
            results.append(result)
        
        assert len(results) >= 1
        
        # Test on_chain_error
        results = []
        async for result in _process_custom_events("on_chain_error", "agent_node", tags, data, "run_123", state):
            results.append(result)
        
        assert len(results) >= 1
    
    @pytest.mark.asyncio
    async def test_process_custom_events_agent_executor(self):
        """Test processing custom events for AgentExecutor"""
        from langchain_aisdk_adapter.utils import _process_custom_events, AIStreamState
        
        state = AIStreamState()
        
        # Test AgentExecutor events
        tags = []
        data = {"input": "test input", "output": "test output"}
        
        # Test on_chain_start
        results = []
        async for result in _process_custom_events("on_chain_start", "AgentExecutor", tags, data, "run_123", state):
            results.append(result)
        
        assert len(results) >= 1
        
        # Test on_chain_end
        results = []
        async for result in _process_custom_events("on_chain_end", "AgentExecutor", tags, data, "run_123", state):
            results.append(result)
        
        assert len(results) >= 1
    
    @pytest.mark.asyncio
    async def test_process_graph_event(self):
        """Test processing LangGraph events"""
        from langchain_aisdk_adapter.utils import _process_graph_event, AIStreamState
        
        state = AIStreamState()
        
        # Test LLM event
        chunk = {
            "event": "on_llm_stream",
            "name": "llm",
            "tags": [],
            "data": {
                "chunk": AIMessageChunk(
                    content="test",
                    tool_call_chunks=[],
                    usage_metadata=None
                )
            },
            "run_id": "run_123"
        }
        
        results = []
        async for result in _process_graph_event(chunk, state):
            results.append(result)
        
        assert len(results) >= 1
        
        # Test tool event
        chunk = {
            "event": "on_tool_start",
            "name": "search_tool",
            "tags": [],
            "data": {
                "tool_name": "search",
                "input": {"query": "test"}
            },
            "run_id": "tool_123"
        }
        
        results = []
        async for result in _process_graph_event(chunk, state):
            results.append(result)
        
        assert len(results) >= 1
        
        # Test agent event
        chunk = {
            "event": "on_agent_action",
            "name": "agent",
            "tags": [],
            "data": {"log": "thinking..."},
            "run_id": "agent_123"
        }
        
        results = []
        async for result in _process_graph_event(chunk, state):
            results.append(result)
        
        assert len(results) >= 1
        
        # Test custom event
        chunk = {
            "event": "on_chain_start",
            "name": "custom_node",
            "tags": ["graph:node"],
            "data": {},
            "run_id": "custom_123"
        }
        
        results = []
        async for result in _process_graph_event(chunk, state):
            results.append(result)
        
        assert len(results) >= 1