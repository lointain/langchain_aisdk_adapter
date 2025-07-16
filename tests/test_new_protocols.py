"""Test new protocol types h:, i:, j:, k:

Test the newly supported AI SDK protocol types for sources, reasoning edits, 
reasoning signatures, and files.
"""

import pytest
from langchain_aisdk_adapter.factory import (
    create_source_part,
    create_redacted_reasoning_part,
    create_reasoning_signature_part,
    create_file_part,
    AISDKFactory
)
from langchain_aisdk_adapter.models import (
    SourcePartContent,
    RedactedReasoningPartContent,
    ReasoningSignaturePartContent,
    FilePartContent
)


class TestNewProtocols:
    """Test new AI SDK protocol types h:, i:, j:, k:"""
    
    def test_source_part_h_protocol(self):
        """Test h: protocol for source information"""
        # Test basic source creation
        url = "https://example.com/doc"
        title = "Example Document"
        
        emitter = create_source_part(url, title)
        assert emitter.ai_sdk_part_content.startswith('h:')
        assert '"url": "https://example.com/doc"' in emitter.ai_sdk_part_content
        assert '"title": "Example Document"' in emitter.ai_sdk_part_content
        
        # Test with factory
        factory_emitter = AISDKFactory.source(url, title)
        assert factory_emitter.ai_sdk_part_content == emitter.ai_sdk_part_content
    
    def test_redacted_reasoning_part_i_protocol(self):
        """Test i: protocol for redacted reasoning content"""
        reasoning_text = "This reasoning has been redacted for privacy"
        
        emitter = create_redacted_reasoning_part(reasoning_text)
        assert emitter.ai_sdk_part_content.startswith('i:')
        assert reasoning_text in emitter.ai_sdk_part_content
        
        # Test with factory
        factory_emitter = AISDKFactory.redacted_reasoning(reasoning_text)
        assert factory_emitter.ai_sdk_part_content == emitter.ai_sdk_part_content
    
    def test_reasoning_signature_part_j_protocol(self):
        """Test j: protocol for reasoning signatures"""
        signature = "signature_abc123"
        
        emitter = create_reasoning_signature_part(signature)
        assert emitter.ai_sdk_part_content.startswith('j:')
        assert '"signature": "signature_abc123"' in emitter.ai_sdk_part_content
        
        # Test with factory
        factory_emitter = AISDKFactory.reasoning_signature(signature)
        assert factory_emitter.ai_sdk_part_content == emitter.ai_sdk_part_content
    
    def test_file_part_k_protocol(self):
        """Test k: protocol for file attachments"""
        data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        mime_type = "image/png"
        
        emitter = create_file_part(data, mime_type)
        assert emitter.ai_sdk_part_content.startswith('k:')
        assert '"data":' in emitter.ai_sdk_part_content
        assert '"mimeType": "image/png"' in emitter.ai_sdk_part_content
        
        # Test with factory
        factory_emitter = AISDKFactory.file(data, mime_type)
        assert factory_emitter.ai_sdk_part_content == emitter.ai_sdk_part_content
    
    def test_source_part_validation(self):
        """Test source part content validation"""
        # Test valid source data
        content = SourcePartContent(url="https://example.com", title="Test")
        assert content.url == "https://example.com"
        assert content.title == "Test"
        
        # Test without title
        content_no_title = SourcePartContent(url="https://example.com")
        assert content_no_title.url == "https://example.com"
        assert content_no_title.title is None
    
    def test_redacted_reasoning_validation(self):
        """Test redacted reasoning content validation"""
        reasoning_text = "[REDACTED] reasoning content"
        content = RedactedReasoningPartContent(data=reasoning_text)
        assert content.data == reasoning_text
        
        # Test empty string
        empty_content = RedactedReasoningPartContent(data="")
        assert empty_content.data == ""
    
    def test_reasoning_signature_validation(self):
        """Test reasoning signature content validation"""
        signature = "test_signature_123"
        content = ReasoningSignaturePartContent(signature=signature)
        assert content.signature == signature
    
    def test_file_part_validation(self):
        """Test file part content validation"""
        data = "dGVzdCBkYXRh"  # base64 for "test data"
        mime_type = "text/plain"
        content = FilePartContent(data=data, mimeType=mime_type)
        assert content.data == data
        assert content.mimeType == mime_type
        
        # Test with different mime type
        pdf_data = "JVBERi0xLjQK"  # base64 for PDF header
        pdf_content = FilePartContent(data=pdf_data, mimeType="application/pdf")
        assert pdf_content.data == pdf_data
        assert pdf_content.mimeType == "application/pdf"
    
    def test_factory_generic_part_method(self):
        """Test AISDKFactory.part() method with new protocol types"""
        factory = AISDKFactory()
        
        # Test h: protocol
        source_data = {"url": "https://test.com", "title": "Test"}
        h_emitter = factory.part('h', source_data)
        assert h_emitter.ai_sdk_part_content.startswith('h:')
        
        # Test i: protocol
        reasoning_data = {"data": "redacted reasoning"}
        i_emitter = factory.part('i', reasoning_data)
        assert i_emitter.ai_sdk_part_content.startswith('i:')
        
        # Test j: protocol
        signature_data = {"signature": "test_signature"}
        j_emitter = factory.part('j', signature_data)
        assert j_emitter.ai_sdk_part_content.startswith('j:')
        
        # Test k: protocol
        file_data = {"data": "dGVzdA==", "mimeType": "text/plain"}
        k_emitter = factory.part('k', file_data)
        assert k_emitter.ai_sdk_part_content.startswith('k:')
    
    def test_protocol_content_formatting(self):
        """Test that protocol content is properly formatted"""
        # Test that all new protocols end with newline
        source_emitter = create_source_part("https://test.com")
        assert source_emitter.ai_sdk_part_content.endswith('\n')
        
        reasoning_emitter = create_redacted_reasoning_part("test")
        assert reasoning_emitter.ai_sdk_part_content.endswith('\n')
        
        signature_emitter = create_reasoning_signature_part("test_signature")
        assert signature_emitter.ai_sdk_part_content.endswith('\n')
        
        file_emitter = create_file_part("dGVzdA==", "text/plain")
        assert file_emitter.ai_sdk_part_content.endswith('\n')
    
    def test_protocol_json_serialization(self):
        """Test that data structures are properly JSON serialized"""
        # Test source with title
        emitter = create_source_part("https://example.com", "Test Title")
        content = emitter.ai_sdk_part_content
        assert '"url": "https://example.com"' in content
        assert '"title": "Test Title"' in content
        
        # Test file with base64 data
        file_emitter = create_file_part("dGVzdCBkYXRh", "text/plain")
        file_content = file_emitter.ai_sdk_part_content
        assert '"data": "dGVzdCBkYXRh"' in file_content
        assert '"mimeType": "text/plain"' in file_content