"""Pytest configuration and fixtures for langchain_aisdk_adapter tests."""

import asyncio
import pytest
from typing import AsyncGenerator, Dict, Any, List
from unittest.mock import AsyncMock, MagicMock

from langchain_aisdk_adapter.models import (
    LangChainStreamEvent,
    LangChainAIMessageChunk,
    UIMessageChunk
)
from langchain_aisdk_adapter.callbacks import BaseAICallbackHandler


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_langchain_text_event() -> LangChainStreamEvent:
    """Sample LangChain text streaming event."""
    return {
        "event": "on_chat_model_stream",
        "name": "ChatOpenAI",
        "run_id": "test-run-id",
        "data": {
            "chunk": {
                "content": "Hello world",
                "type": "ai",
                "id": "test-chunk-id"
            }
        },
        "metadata": {}
    }


@pytest.fixture
def sample_langchain_tool_event() -> LangChainStreamEvent:
    """Sample LangChain tool call event."""
    return {
        "event": "on_tool_start",
        "name": "test_tool",
        "run_id": "tool-run-id",
        "data": {
            "input": {"query": "test query"}
        },
        "metadata": {}
    }


@pytest.fixture
def sample_tool_result_event() -> LangChainStreamEvent:
    """Sample tool result event."""
    return {
        "event": "on_tool_end",
        "name": "test_tool",
        "run_id": "tool-run-id",
        "data": {
            "output": "Tool execution result"
        },
        "metadata": {}
    }


@pytest.fixture
def mock_callback_handler() -> BaseAICallbackHandler:
    """Mock callback handler for testing."""
    handler = AsyncMock(spec=BaseAICallbackHandler)
    handler.on_start = AsyncMock()
    handler.on_finish = AsyncMock()
    handler.on_error = AsyncMock()
    handler.on_step_start = AsyncMock()
    handler.on_step_finish = AsyncMock()
    handler.on_text = AsyncMock()
    handler.on_tool_call = AsyncMock()
    handler.on_tool_result = AsyncMock()
    return handler


@pytest.fixture
async def sample_stream() -> AsyncGenerator[LangChainStreamEvent, None]:
    """Sample async stream of LangChain events."""
    events = [
        {
            "event": "on_chat_model_stream",
            "name": "ChatOpenAI",
            "run_id": "test-run-id",
            "data": {
                "chunk": {
                    "content": "Hello ",
                    "type": "ai",
                    "id": "chunk-1"
                }
            },
            "metadata": {}
        },
        {
            "event": "on_chat_model_stream",
            "name": "ChatOpenAI",
            "run_id": "test-run-id",
            "data": {
                "chunk": {
                    "content": "world!",
                    "type": "ai",
                    "id": "chunk-2"
                }
            },
            "metadata": {}
        }
    ]
    
    for event in events:
        yield event


@pytest.fixture
def sample_ui_chunks() -> List[UIMessageChunk]:
    """Sample UI message chunks for testing."""
    return [
        {"type": "start", "messageId": "test-message-id"},
        {"type": "text-start", "id": "text-1"},
        {"type": "text-delta", "id": "text-1", "textDelta": "Hello "},
        {"type": "text-delta", "id": "text-1", "textDelta": "world!"},
        {"type": "text-end", "id": "text-1"},
        {"type": "finish", "finishReason": "stop", "usage": {"promptTokens": 10, "completionTokens": 5}}
    ]


class MockAsyncIterable:
    """Mock async iterable for testing."""
    
    def __init__(self, items):
        self.items = items
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if not self.items:
            raise StopAsyncIteration
        return self.items.pop(0)


@pytest.fixture
def mock_async_stream():
    """Factory for creating mock async streams."""
    def _create_stream(items):
        return MockAsyncIterable(items)
    return _create_stream