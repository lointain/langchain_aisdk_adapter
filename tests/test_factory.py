"""Test factory functions

Test all AI SDK part factory functions to ensure they correctly create corresponding model instances.
"""

import pytest
from typing import Any, Dict

from langchain_aisdk_adapter.factory import (
    create_ai_sdk_part,
    create_text_part,
    create_reasoning_part,
    create_redacted_reasoning_part,
    create_reasoning_signature_part,
    create_source_part,
    create_file_part,
    create_data_part,
    create_message_annotation_part,
    create_error_part,
    create_tool_call_streaming_start_part,
    create_tool_call_delta_part,
    create_tool_call_part,
    create_tool_result_part,
    create_start_step_part,
    create_finish_step_part,
    create_finish_message_part,
    AISDKFactory,
)
# Import AISDKPartEmitter for type checking
from langchain_aisdk_adapter.emitter import AISDKPartEmitter


class TestBasicFactoryFunctions:
    """Test basic factory functions"""
    
    def test_create_text_part(self):
        """Test creating text part"""
        part = create_text_part("Hello, world!")
        assert isinstance(part, AISDKPartEmitter)
        assert part.ai_sdk_part_content == '0:"Hello, world!"\n'
    
    def test_create_reasoning_part(self):
        """Test creating reasoning part"""
        part = create_reasoning_part("Let me think...")
        assert isinstance(part, AISDKPartEmitter)
        assert part.ai_sdk_part_content == 'g:"Let me think..."\n'
    
    def test_create_source_part(self):
        """Test creating source part"""
        part = create_source_part(
            url="https://wikipedia.org",
            title="Wikipedia"
        )
        assert isinstance(part, AISDKPartEmitter)
        assert 'h:' in part.ai_sdk_part_content
        assert 'Wikipedia' in part.ai_sdk_part_content
        assert 'wikipedia.org' in part.ai_sdk_part_content
    
    def test_create_file_part(self):
        """Test creating file part"""
        part = create_file_part(
            data="dGVzdCBkYXRh",  # base64 encoded "test data"
            mime_type="text/plain"
        )
        assert isinstance(part, AISDKPartEmitter)
        assert 'k:' in part.ai_sdk_part_content
        assert 'dGVzdCBkYXRh' in part.ai_sdk_part_content
    
    def test_create_data_part(self):
        """Test creating data part"""
        data = ["key", "value", 42]
        part = create_data_part(data)
        assert isinstance(part, AISDKPartEmitter)
        assert '2:' in part.ai_sdk_part_content
    
    def test_create_error_part(self):
        """Test creating error part"""
        part = create_error_part("Something went wrong")
        assert isinstance(part, AISDKPartEmitter)
        assert part.ai_sdk_part_content == '3:"Something went wrong"\n'


class TestToolFactoryFunctions:
    """Test tool-related factory functions"""
    
    def test_create_tool_call_part(self):
        """Test creating tool call part"""
        part = create_tool_call_part(
            tool_call_id="call_123",
            tool_name="search",
            args={"query": "Python"}
        )
        assert isinstance(part, AISDKPartEmitter)
        assert '9:' in part.ai_sdk_part_content
        assert 'call_123' in part.ai_sdk_part_content
        assert 'search' in part.ai_sdk_part_content
    
    def test_create_tool_result_part(self):
        """Test creating tool result part"""
        part = create_tool_result_part(
            tool_call_id="call_123",
            result="Search completed"
        )
        assert isinstance(part, AISDKPartEmitter)
        assert 'a:' in part.ai_sdk_part_content
        assert 'call_123' in part.ai_sdk_part_content
        assert 'Search completed' in part.ai_sdk_part_content
    
    def test_create_tool_call_streaming_start_part(self):
        """Test creating tool call streaming start part"""
        part = create_tool_call_streaming_start_part(
            tool_call_id="call_123",
            tool_name="search"
        )
        assert isinstance(part, AISDKPartEmitter)
        assert 'b:' in part.ai_sdk_part_content
        assert 'call_123' in part.ai_sdk_part_content
    
    def test_create_tool_call_delta_part(self):
        """Test creating tool call delta part"""
        part = create_tool_call_delta_part(
            tool_call_id="call_123",
            args_text_delta="Py"
        )
        assert isinstance(part, AISDKPartEmitter)
        assert 'c:' in part.ai_sdk_part_content
        assert 'call_123' in part.ai_sdk_part_content


class TestAdvancedFactoryFunctions:
    """Test advanced factory functions"""
    
    def test_create_finish_message_part(self):
        """Test creating finish message part"""
        part = create_finish_message_part(
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=20
        )
        assert isinstance(part, AISDKPartEmitter)
        assert 'd:' in part.ai_sdk_part_content
        assert 'stop' in part.ai_sdk_part_content
    
    def test_create_start_step_part(self):
        """Test creating start step part"""
        part = create_start_step_part(
            message_id="step_123"
        )
        assert isinstance(part, AISDKPartEmitter)
        assert 'f:' in part.ai_sdk_part_content
        assert 'step_123' in part.ai_sdk_part_content
    
    def test_create_finish_step_part(self):
        """Test creating finish step part"""
        part = create_finish_step_part(
            finish_reason="completed",
            prompt_tokens=5,
            completion_tokens=10
        )
        assert isinstance(part, AISDKPartEmitter)
        assert 'e:' in part.ai_sdk_part_content
        assert 'completed' in part.ai_sdk_part_content
    
    def test_create_message_annotation_part(self):
        """Test creating message annotation part"""
        part = create_message_annotation_part(
            annotations=[{"type": "warning", "message": "This is a warning"}]
        )
        assert isinstance(part, AISDKPartEmitter)
        assert '8:' in part.ai_sdk_part_content


class TestGenericFactoryFunction:
    """Test generic factory function"""
    
    def test_create_ai_sdk_part_text(self):
        """Test generic factory function creating text part"""
        part = create_ai_sdk_part("0", "Hello, world!")
        assert isinstance(part, AISDKPartEmitter)
        assert part.ai_sdk_part_content == '0:"Hello, world!"\n'
    
    def test_create_ai_sdk_part_data(self):
        """Test generic factory function creating data part"""
        data = ["key", "value"]
        part = create_ai_sdk_part("2", data)
        assert isinstance(part, AISDKPartEmitter)
        assert '2:' in part.ai_sdk_part_content
    
    def test_create_ai_sdk_part_error(self):
        """Test generic factory function creating error part"""
        part = create_ai_sdk_part("3", "Something went wrong")
        assert isinstance(part, AISDKPartEmitter)
        assert part.ai_sdk_part_content == '3:"Something went wrong"\n'
    
    def test_create_ai_sdk_part_invalid_type(self):
        """Test generic factory function handling invalid type"""
        with pytest.raises(ValueError):
            create_ai_sdk_part("invalid_type", "Hello")
    
    def test_create_ai_sdk_part_tool_call(self):
          """Test generic factory with tool call type"""
          content = {
              'toolCallId': 'call_123',
              'toolName': 'search',
              'args': {'query': 'python'}
          }
          result = create_ai_sdk_part('9', content)
          assert isinstance(result, AISDKPartEmitter)
          assert '9:' in result.ai_sdk_part_content
          assert 'call_123' in result.ai_sdk_part_content
    
    def test_create_ai_sdk_part_validation_error(self):
        """Test generic factory with invalid content"""
        with pytest.raises(ValueError):
            create_ai_sdk_part('9', 'invalid_tool_call_content')


class TestReasoningFactoryFunctions:
    """Test reasoning-related factory functions"""
    
    def test_create_redacted_reasoning_part(self):
        """Test redacted reasoning part creation"""
        result = create_redacted_reasoning_part('sensitive data')
        assert isinstance(result, AISDKPartEmitter)
        assert 'i:' in result.ai_sdk_part_content
        assert 'sensitive data' in result.ai_sdk_part_content
    
    def test_create_reasoning_signature_part(self):
        """Test reasoning signature part creation"""
        result = create_reasoning_signature_part('signature123')
        assert isinstance(result, AISDKPartEmitter)
        assert 'j:' in result.ai_sdk_part_content
        assert 'signature123' in result.ai_sdk_part_content


class TestFactoryFunctionEdgeCases:
    """Test factory function edge cases"""
    
    def test_create_text_part_empty_string(self):
        """Test creating text part with empty string"""
        part = create_text_part("")
        assert isinstance(part, AISDKPartEmitter)
        assert part.ai_sdk_part_content == '0:""\n'
    
    def test_create_data_part_empty_list(self):
        """Test creating data part with empty list"""
        part = create_data_part([])
        assert isinstance(part, AISDKPartEmitter)
        assert part.ai_sdk_part_content == '2:[]\n'
    
    def test_create_tool_call_part_complex_args(self):
        """Test creating tool call part with complex arguments"""
        complex_args = {
            "nested": {
                "array": [1, 2, 3],
                "string": "test",
                "boolean": True
            }
        }
        part = create_tool_call_part(
            tool_call_id="call_123",
            tool_name="complex_tool",
            args=complex_args
        )
        assert isinstance(part, AISDKPartEmitter)
        assert '9:' in part.ai_sdk_part_content
        assert 'call_123' in part.ai_sdk_part_content
        assert 'complex_tool' in part.ai_sdk_part_content
    
    def test_create_source_part_optional_fields(self):
        """Test creating source part with optional fields"""
        part = create_source_part(
            url="",
            title="Test Source"
        )
        assert isinstance(part, AISDKPartEmitter)
        assert 'h:' in part.ai_sdk_part_content
        assert 'Test Source' in part.ai_sdk_part_content
    
    def test_invalid_tool_call_content(self):
        """Test tool call with invalid content structure"""
        with pytest.raises(ValueError):
            create_tool_call_part('call_123', 'search', 'invalid_args')
    
    def test_invalid_finish_step_content(self):
        """Test finish step with invalid parameters"""
        # This should work fine with default values
        result = create_finish_step_part('stop')
        assert 'e:' in result.ai_sdk_part_content
    
    def test_invalid_source_url(self):
        """Test source part with various URL formats"""
        # Should work with any string as URL
        result = create_source_part('not-a-url')
        assert 'h:' in result.ai_sdk_part_content
        assert 'not-a-url' in result.ai_sdk_part_content


class TestAISDKFactoryClass:
    """Test the AISDKFactory class methods"""
    
    def test_factory_text(self):
        """Test factory text method"""
        result = AISDKFactory.text('Hello')
        assert isinstance(result, AISDKPartEmitter)
        assert result.ai_sdk_part_content == '0:"Hello"\n'
    
    def test_factory_data(self):
        """Test factory data method"""
        result = AISDKFactory.data([1, 2, 3])
        assert isinstance(result, AISDKPartEmitter)
        assert '2:' in result.ai_sdk_part_content
    
    def test_factory_error(self):
        """Test factory error method"""
        result = AISDKFactory.error('Error message')
        assert isinstance(result, AISDKPartEmitter)
        assert '3:' in result.ai_sdk_part_content
        assert 'Error message' in result.ai_sdk_part_content
    
    def test_factory_reasoning(self):
        """Test factory reasoning method"""
        result = AISDKFactory.reasoning('Thinking...')
        assert isinstance(result, AISDKPartEmitter)
        assert 'g:' in result.ai_sdk_part_content
        assert 'Thinking...' in result.ai_sdk_part_content
    
    def test_factory_redacted_reasoning(self):
        """Test factory redacted reasoning method"""
        result = AISDKFactory.redacted_reasoning('redacted')
        assert isinstance(result, AISDKPartEmitter)
        assert 'i:' in result.ai_sdk_part_content
        assert 'redacted' in result.ai_sdk_part_content
    
    def test_factory_reasoning_signature(self):
        """Test factory reasoning signature method"""
        result = AISDKFactory.reasoning_signature('sig123')
        assert isinstance(result, AISDKPartEmitter)
        assert 'j:' in result.ai_sdk_part_content
        assert 'sig123' in result.ai_sdk_part_content
    
    def test_factory_source(self):
        """Test factory source method"""
        result = AISDKFactory.source('https://example.com', 'Title')
        assert isinstance(result, AISDKPartEmitter)
        assert 'h:' in result.ai_sdk_part_content
        assert 'https://example.com' in result.ai_sdk_part_content
        assert 'Title' in result.ai_sdk_part_content
    
    def test_factory_file(self):
        """Test factory file method"""
        result = AISDKFactory.file('base64data', 'image/png')
        assert isinstance(result, AISDKPartEmitter)
        assert 'k:' in result.ai_sdk_part_content
        assert 'base64data' in result.ai_sdk_part_content
        assert 'image/png' in result.ai_sdk_part_content
    
    def test_factory_annotation(self):
        """Test factory annotation method"""
        result = AISDKFactory.annotation([{'type': 'highlight'}])
        assert isinstance(result, AISDKPartEmitter)
        assert '8:' in result.ai_sdk_part_content
    
    def test_factory_tool_call_start(self):
        """Test factory tool call start method"""
        result = AISDKFactory.tool_call_start('call_123', 'search')
        assert isinstance(result, AISDKPartEmitter)
        assert 'b:' in result.ai_sdk_part_content
        assert 'call_123' in result.ai_sdk_part_content
        assert 'search' in result.ai_sdk_part_content
    
    def test_factory_tool_call_delta(self):
        """Test factory tool call delta method"""
        result = AISDKFactory.tool_call_delta('call_123', 'delta')
        assert isinstance(result, AISDKPartEmitter)
        assert 'c:' in result.ai_sdk_part_content
        assert 'call_123' in result.ai_sdk_part_content
        assert 'delta' in result.ai_sdk_part_content
    
    def test_factory_tool_call(self):
        """Test factory tool call method"""
        result = AISDKFactory.tool_call('call_123', 'search', {'query': 'test'})
        assert isinstance(result, AISDKPartEmitter)
        assert '9:' in result.ai_sdk_part_content
        assert 'call_123' in result.ai_sdk_part_content
        assert 'search' in result.ai_sdk_part_content
    
    def test_factory_tool_result(self):
        """Test factory tool result method"""
        result = AISDKFactory.tool_result('call_123', 'result data')
        assert isinstance(result, AISDKPartEmitter)
        assert 'a:' in result.ai_sdk_part_content
        assert 'call_123' in result.ai_sdk_part_content
        assert 'result data' in result.ai_sdk_part_content
    
    def test_factory_start_step(self):
        """Test factory start step method"""
        result = AISDKFactory.start_step('msg_123')
        assert isinstance(result, AISDKPartEmitter)
        assert 'f:' in result.ai_sdk_part_content
        assert 'msg_123' in result.ai_sdk_part_content
    
    def test_factory_finish_step(self):
        """Test factory finish step method"""
        result = AISDKFactory.finish_step('stop', 10, 20, True)
        assert isinstance(result, AISDKPartEmitter)
        assert 'e:' in result.ai_sdk_part_content
        assert 'stop' in result.ai_sdk_part_content
        assert '10' in result.ai_sdk_part_content
        assert '20' in result.ai_sdk_part_content
        assert 'true' in result.ai_sdk_part_content.lower()
    
    def test_factory_finish_message(self):
        """Test factory finish message method"""
        result = AISDKFactory.finish_message('stop', 15, 25)
        assert isinstance(result, AISDKPartEmitter)
        assert 'd:' in result.ai_sdk_part_content
        assert 'stop' in result.ai_sdk_part_content
        assert '15' in result.ai_sdk_part_content
        assert '25' in result.ai_sdk_part_content
    
    def test_factory_part_invalid_type(self):
        """Test factory part method with invalid type"""
        with pytest.raises(ValueError, match="Unsupported AI SDK protocol type"):
            AISDKFactory.part('invalid', 'content')
    
    def test_factory_part_validation_error(self):
        """Test factory part method with validation error"""
        with pytest.raises(ValueError, match="Content validation failed"):
            AISDKFactory.part('9', 'invalid_content')