"""Test extended step protocol support

Test that e: and f: protocols work with more LangChain agent types beyond LangGraph.
"""

import pytest
from unittest.mock import MagicMock
from langchain_aisdk_adapter.utils import _is_major_workflow_component


class TestExtendedStepProtocols:
    """Test extended step protocol support for various LangChain components"""
    
    def test_is_major_workflow_component_langgraph(self):
        """Test LangGraph components are recognized as major workflow components"""
        # Test LangGraph specific components
        assert _is_major_workflow_component('LangGraph', ['langgraph', 'graph'])
        
        # Test CompiledGraph
        assert _is_major_workflow_component('CompiledGraph', ['graph'])
        
        # Test with graph:step: tags
        assert _is_major_workflow_component('SomeNode', ['graph:step:1', 'custom'])
    
    def test_is_major_workflow_component_langchain_agents(self):
        """Test various LangChain agent types are recognized"""
        # Test AgentExecutor
        assert _is_major_workflow_component('AgentExecutor', ['agent'])
        
        # Test ConversationalRetrievalChain
        assert _is_major_workflow_component('ConversationalRetrievalChain', ['chain'])
        
        # Test RetrievalQA
        assert _is_major_workflow_component('RetrievalQA', ['qa'])
        
        # Test LLMChain
        assert _is_major_workflow_component('LLMChain', ['chain'])
        
        # Test SequentialChain
        assert _is_major_workflow_component('SequentialChain', ['chain'])
        
        # Test RouterChain
        assert _is_major_workflow_component('RouterChain', ['chain'])
    
    def test_is_major_workflow_component_by_tags(self):
        """Test components recognized by tags"""
        # Test agent tag
        assert _is_major_workflow_component('CustomAgent', ['agent', 'custom'])
        
        # Test chain tag
        assert _is_major_workflow_component('CustomChain', ['chain', 'custom'])
        
        # Test executor tag
        assert _is_major_workflow_component('CustomExecutor', ['executor', 'custom'])
    
    def test_is_major_workflow_component_negative_cases(self):
        """Test components that should NOT be recognized as major workflow components"""
        # Test simple LLM
        assert not _is_major_workflow_component('OpenAI', ['llm'])
        
        # Test retriever
        assert not _is_major_workflow_component('VectorStoreRetriever', ['retriever'])
        
        # Test tool
        assert not _is_major_workflow_component('SearchTool', ['tool'])
        
        # Test parser
        assert not _is_major_workflow_component('OutputParser', ['parser'])
        
        # Test memory
        assert not _is_major_workflow_component('ConversationBufferMemory', ['memory'])
    
    def test_is_major_workflow_component_edge_cases(self):
        """Test edge cases for component recognition"""
        # Test with empty name and tags
        assert not _is_major_workflow_component('', [])
        
        # Test with empty name but valid tags
        assert _is_major_workflow_component('', ['agent'])
        
        # Test with valid name but empty tags
        assert _is_major_workflow_component('AgentExecutor', [])
        
        # Test with empty tags
        assert not _is_major_workflow_component('SomeComponent', [])
        
        # Test case sensitivity
        assert not _is_major_workflow_component('agentexecutor', ['AGENT'])  # Should be case sensitive
    
    def test_is_major_workflow_component_partial_matches(self):
        """Test that exact name matches work correctly"""
        # Test exact matches for major components
        assert _is_major_workflow_component('AgentExecutor', ['custom'])
        
        # Test names that don't exactly match
        assert not _is_major_workflow_component('MyAgentExecutor', ['custom'])
        assert not _is_major_workflow_component('ChainWrapper', ['wrapper'])
        
        # Test exact LLMChain match
        assert _is_major_workflow_component('LLMChain', ['custom'])