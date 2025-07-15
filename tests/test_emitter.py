"""Test emitter module

Test the AISDKPartEmitter class and its streaming functionality.
"""

import pytest
from langchain_aisdk_adapter.emitter import AISDKPartEmitter


class TestAISDKPartEmitter:
    """Test AISDKPartEmitter class"""
    
    def test_emitter_initialization_with_content(self):
        """Test emitter initialization with AI SDK content"""
        content = '0:"Hello, world!"'
        emitter = AISDKPartEmitter(ai_sdk_part_content=content)
        
        assert emitter.ai_sdk_part_content == '0:"Hello, world!"\n'
    
    def test_emitter_initialization_with_newline(self):
        """Test emitter initialization with content that already has newline"""
        content = '0:"Hello, world!"\n'
        emitter = AISDKPartEmitter(ai_sdk_part_content=content)
        
        assert emitter.ai_sdk_part_content == '0:"Hello, world!"\n'
    
    def test_emitter_str_representation(self):
        """Test string representation of emitter"""
        content = '0:"Hello"'
        emitter = AISDKPartEmitter(ai_sdk_part_content=content)
        
        str_repr = str(emitter)
        # Check that the string representation contains the class name
        assert "AISDKPartEmitter" in str_repr or "emitter" in str_repr.lower()
    
    def test_emitter_with_different_content_types(self):
        """Test emitter with different AI SDK protocol types"""
        # Test with text content
        text_emitter = AISDKPartEmitter(ai_sdk_part_content='0:"Hello"')
        assert text_emitter.ai_sdk_part_content == '0:"Hello"\n'
        
        # Test with data content
        data_emitter = AISDKPartEmitter(ai_sdk_part_content='2:["key", "value"]')
        assert data_emitter.ai_sdk_part_content == '2:["key", "value"]\n'
        
        # Test with error content
        error_emitter = AISDKPartEmitter(ai_sdk_part_content='3:"Error message"')
        assert error_emitter.ai_sdk_part_content == '3:"Error message"\n'

    
    @pytest.mark.asyncio
    async def test_emitter_invoke(self):
        """Test synchronous invoke method"""
        emitter = AISDKPartEmitter(ai_sdk_part_content='0:"Hello"')
        input_data = "test input"
        
        result = emitter.invoke(input_data)
        assert result == input_data
    
    @pytest.mark.asyncio
    async def test_emitter_ainvoke(self):
        """Test asynchronous invoke method"""
        emitter = AISDKPartEmitter(ai_sdk_part_content='0:"Hello"')
        input_data = "test input"
        
        result = await emitter.ainvoke(input_data)
        assert result == input_data
    
    def test_emitter_stream(self):
        """Test synchronous streaming"""
        content = '0:"Hello"'
        emitter = AISDKPartEmitter(ai_sdk_part_content=content)
        input_data = "test input"
        
        results = list(emitter.stream(input_data))
        
        assert len(results) == 2
        assert results[0] == '0:"Hello"\n'  # AI SDK content first
        assert results[1] == input_data     # Then input data
    
    @pytest.mark.asyncio
    async def test_emitter_astream(self):
        """Test asynchronous streaming"""
        content = '0:"Hello"'
        emitter = AISDKPartEmitter(ai_sdk_part_content=content)
        input_data = "test input"
        
        results = []
        async for item in emitter.astream(input_data):
            results.append(item)
        
        assert len(results) == 2
        assert results[0] == '0:"Hello"\n'  # AI SDK content first
        assert results[1] == input_data     # Then input data
    
    def test_emitter_content_modification(self):
        """Test modifying emitter content after initialization"""
        emitter = AISDKPartEmitter(ai_sdk_part_content='0:"Original"')
        
        # Modify the content
        emitter.ai_sdk_part_content = '0:"Modified"\n'
        
        assert emitter.ai_sdk_part_content == '0:"Modified"\n'