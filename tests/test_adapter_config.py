#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for LangChainAdapter with AdapterConfig
"""

import pytest
from langchain_core.messages import AIMessageChunk
from langchain_aisdk_adapter import LangChainAdapter, AdapterConfig


class TestAdapterWithConfig:
    """Test LangChainAdapter with different configurations"""
    
    @pytest.mark.asyncio
    async def test_default_config_enables_all(self):
        """Test that default config enables all protocols"""
        async def mock_stream():
            yield "Hello World"
        
        # Create adapter instance with default config
        adapter = LangChainAdapter()
        
        parts = []
        async for part in adapter.to_data_stream_response(mock_stream()):
            parts.append(part)
        
        # Should have text part and finish message
        assert len(parts) == 2
        assert parts[0].startswith('0:"Hello World"')
        assert parts[1].startswith('d:')
    
    @pytest.mark.asyncio
    async def test_disabled_text_protocol(self):
        """Test disabling text protocol"""
        config = AdapterConfig(disabled_protocols={'0'})
        
        async def mock_stream():
            yield "Hello World"
        
        # Create adapter instance with custom config
        adapter = LangChainAdapter(config=config)
        
        parts = []
        async for part in adapter.to_data_stream_response(mock_stream()):
            parts.append(part)
        
        # Should only have finish message, no text
        assert len(parts) == 1
        assert parts[0].startswith('d:')
    
    @pytest.mark.asyncio
    async def test_disabled_finish_message(self):
        """Test disabling finish message protocol"""
        config = AdapterConfig(disabled_protocols={'d'})
        
        async def mock_stream():
            yield "Hello World"
        
        # Create adapter instance with custom config
        adapter = LangChainAdapter(config=config)
        
        parts = []
        async for part in adapter.to_data_stream_response(mock_stream()):
            parts.append(part)
        
        # Should only have text part, no finish message
        assert len(parts) == 1
        assert parts[0].startswith('0:"Hello World"')
    
    @pytest.mark.asyncio
    async def test_minimal_config(self):
        """Test minimal configuration"""
        config = AdapterConfig.minimal()
        
        async def mock_stream():
            yield "Hello World"
        
        # Create adapter instance with minimal config
        adapter = LangChainAdapter(config=config)
        
        parts = []
        async for part in adapter.to_data_stream_response(mock_stream()):
            parts.append(part)
        
        # Should only have text part (finish message disabled in minimal)
        assert len(parts) == 1
        assert parts[0].startswith('0:"Hello World"')
    
    @pytest.mark.asyncio
    async def test_ai_message_chunk_with_disabled_text(self):
        """Test AIMessageChunk processing with disabled text protocol"""
        config = AdapterConfig(disabled_protocols={'0'})
        
        async def mock_stream():
            chunk = AIMessageChunk(content="Hello from AI")
            yield chunk
        
        # Create adapter instance with custom config
        adapter = LangChainAdapter(config=config)
        
        parts = []
        async for part in adapter.to_data_stream_response(mock_stream()):
            parts.append(part)
        
        # Should only have finish message, no text
        assert len(parts) == 1
        assert parts[0].startswith('d:')
    
    @pytest.mark.asyncio
    async def test_error_handling_with_disabled_error_protocol(self):
        """Test error handling when error protocol is disabled"""
        config = AdapterConfig(disabled_protocols={'3'})
        
        async def mock_stream():
            raise ValueError("Test error")
            yield "This won't be reached"
        
        # Create adapter instance with custom config
        adapter = LangChainAdapter(config=config)
        
        parts = []
        async for part in adapter.to_data_stream_response(mock_stream()):
            parts.append(part)
        
        # Should have no parts since error protocol is disabled
        assert len(parts) == 0
    
    @pytest.mark.asyncio
    async def test_convenience_methods(self):
        """Test convenience configuration methods"""
        # Test disable_steps
        config = AdapterConfig()
        config.disable_steps()
        
        assert not config.is_protocol_enabled('e')
        assert not config.is_protocol_enabled('f')
        
        # Test disable_tool_streaming
        config = AdapterConfig()
        config.disable_tool_streaming()
        
        assert not config.is_protocol_enabled('b')
        assert not config.is_protocol_enabled('c')
        
        # Test disable_finish_messages
        config = AdapterConfig()
        config.disable_finish_messages()
        
        assert not config.is_protocol_enabled('d')
    
    @pytest.mark.asyncio
    async def test_pre_formatted_ai_sdk_strings_bypass_config(self):
        """Test that pre-formatted AI SDK strings bypass configuration"""
        config = AdapterConfig(disabled_protocols={'0'})  # Disable text
        
        async def mock_stream():
            yield '0:"Pre-formatted text"\n'  # Pre-formatted AI SDK string
        
        # Create adapter instance with custom config
        adapter = LangChainAdapter(config=config)
        
        parts = []
        async for part in adapter.to_data_stream_response(mock_stream()):
            parts.append(part)
        
        # Pre-formatted strings should bypass config and be yielded as-is
        assert len(parts) == 1
        assert parts[0] == '0:"Pre-formatted text"\n'