#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for AdapterConfig functionality
"""

import pytest
from langchain_aisdk_adapter.config import AdapterConfig


class TestAdapterConfig:
    """Test AdapterConfig class functionality"""
    
    def test_default_config(self):
        """Test default configuration enables all protocols"""
        config = AdapterConfig()
        
        # All protocols should be enabled by default
        assert config.is_protocol_enabled('0')  # text
        assert config.is_protocol_enabled('9')  # tool call
        assert config.is_protocol_enabled('a')  # tool result
        assert config.is_protocol_enabled('b')  # tool call start
        assert config.is_protocol_enabled('d')  # finish message
        assert config.is_protocol_enabled('e')  # finish step
        assert config.is_protocol_enabled('f')  # start step
        assert config.is_protocol_enabled('g')  # reasoning
    
    def test_disable_protocol(self):
        """Test disabling specific protocols"""
        config = AdapterConfig()
        
        # Disable text protocol
        config.disable_protocol('0')
        assert not config.is_protocol_enabled('0')
        assert config.is_protocol_enabled('9')  # Others should still be enabled
        
        # Disable multiple protocols
        config.disable_protocol('e')
        config.disable_protocol('f')
        assert not config.is_protocol_enabled('e')
        assert not config.is_protocol_enabled('f')
    
    def test_enable_protocol(self):
        """Test re-enabling disabled protocols"""
        config = AdapterConfig(disabled_protocols={'0', 'e', 'f'})
        
        # Initially disabled
        assert not config.is_protocol_enabled('0')
        assert not config.is_protocol_enabled('e')
        
        # Re-enable
        config.enable_protocol('0')
        assert config.is_protocol_enabled('0')
        assert not config.is_protocol_enabled('e')  # Still disabled
    
    def test_disable_steps(self):
        """Test convenience method for disabling steps"""
        config = AdapterConfig()
        config.disable_steps()
        
        assert not config.is_protocol_enabled('e')  # finish step
        assert not config.is_protocol_enabled('f')  # start step
        assert config.is_protocol_enabled('0')     # text should still be enabled
    
    def test_disable_tool_streaming(self):
        """Test convenience method for disabling tool streaming"""
        config = AdapterConfig()
        config.disable_tool_streaming()
        
        assert not config.is_protocol_enabled('b')  # tool call start
        assert not config.is_protocol_enabled('c')  # tool call delta
        assert config.is_protocol_enabled('9')     # tool call should still be enabled
        assert config.is_protocol_enabled('a')     # tool result should still be enabled
    
    def test_disable_finish_messages(self):
        """Test convenience method for disabling finish messages"""
        config = AdapterConfig()
        config.disable_finish_messages()
        
        assert not config.is_protocol_enabled('d')  # finish message
        assert config.is_protocol_enabled('e')     # finish step should still be enabled
    
    def test_minimal_config(self):
        """Test minimal configuration preset"""
        config = AdapterConfig.minimal()
        
        # Basic protocols should be enabled
        assert config.is_protocol_enabled('0')  # text
        assert config.is_protocol_enabled('9')  # tool call
        assert config.is_protocol_enabled('a')  # tool result
        assert config.is_protocol_enabled('3')  # error
        
        # Advanced protocols should be disabled
        assert not config.is_protocol_enabled('b')  # tool call start
        assert not config.is_protocol_enabled('c')  # tool call delta
        assert not config.is_protocol_enabled('d')  # finish message
        assert not config.is_protocol_enabled('e')  # finish step
        assert not config.is_protocol_enabled('f')  # start step
        assert not config.is_protocol_enabled('g')  # reasoning
        assert not config.is_protocol_enabled('h')  # source
        assert not config.is_protocol_enabled('8')  # annotations
    
    def test_tools_only_config(self):
        """Test tools-only configuration preset"""
        config = AdapterConfig.tools_only()
        
        # Tool-related protocols should be enabled
        assert config.is_protocol_enabled('0')  # text
        assert config.is_protocol_enabled('9')  # tool call
        assert config.is_protocol_enabled('a')  # tool result
        assert config.is_protocol_enabled('b')  # tool call start
        assert config.is_protocol_enabled('c')  # tool call delta
        assert config.is_protocol_enabled('3')  # error
        
        # Step and message protocols should be disabled
        assert not config.is_protocol_enabled('d')  # finish message
        assert not config.is_protocol_enabled('e')  # finish step
        assert not config.is_protocol_enabled('f')  # start step
        assert not config.is_protocol_enabled('g')  # reasoning
    
    def test_custom_disabled_protocols(self):
        """Test custom set of disabled protocols"""
        disabled = {'0', '9', 'g'}
        config = AdapterConfig(disabled_protocols=disabled)
        
        assert not config.is_protocol_enabled('0')
        assert not config.is_protocol_enabled('9')
        assert not config.is_protocol_enabled('g')
        assert config.is_protocol_enabled('a')
        assert config.is_protocol_enabled('e')
    
    def test_pause_protocols_context_manager(self):
        """Test pause_protocols context manager"""
        config = AdapterConfig()
        
        # Initially all protocols enabled
        assert config.is_protocol_enabled('0')
        assert config.is_protocol_enabled('g')
        assert config.is_protocol_enabled('9')
        
        # Pause specific protocols
        with config.pause_protocols(['0', 'g']):
            assert not config.is_protocol_enabled('0')  # paused
            assert not config.is_protocol_enabled('g')  # paused
            assert config.is_protocol_enabled('9')     # not paused
        
        # After context, protocols should be restored
        assert config.is_protocol_enabled('0')
        assert config.is_protocol_enabled('g')
        assert config.is_protocol_enabled('9')
    
    def test_pause_protocols_with_existing_disabled(self):
        """Test pause_protocols with already disabled protocols"""
        config = AdapterConfig(disabled_protocols={'e', 'f'})
        
        # Initially e and f are disabled
        assert not config.is_protocol_enabled('e')
        assert not config.is_protocol_enabled('f')
        assert config.is_protocol_enabled('0')
        assert config.is_protocol_enabled('g')
        
        # Pause additional protocols
        with config.pause_protocols(['0', 'g']):
            assert not config.is_protocol_enabled('0')  # paused
            assert not config.is_protocol_enabled('g')  # paused
            assert not config.is_protocol_enabled('e')  # still disabled
            assert not config.is_protocol_enabled('f')  # still disabled
            assert config.is_protocol_enabled('9')     # not affected
        
        # After context, only original disabled protocols remain
        assert config.is_protocol_enabled('0')      # restored
        assert config.is_protocol_enabled('g')      # restored
        assert not config.is_protocol_enabled('e')  # original state
        assert not config.is_protocol_enabled('f')  # original state
    
    def test_protocols_context_manager(self):
        """Test protocols context manager (opposite of pause_protocols)"""
        config = AdapterConfig()
        
        # Initially all protocols enabled
        assert config.is_protocol_enabled('0')
        assert config.is_protocol_enabled('g')
        assert config.is_protocol_enabled('9')
        assert config.is_protocol_enabled('a')
        
        # Enable only specific protocols
        with config.protocols(['0', 'g']):
            assert config.is_protocol_enabled('0')      # enabled
            assert config.is_protocol_enabled('g')      # enabled
            assert not config.is_protocol_enabled('9')  # disabled
            assert not config.is_protocol_enabled('a')  # disabled
            assert not config.is_protocol_enabled('e')  # disabled
            assert not config.is_protocol_enabled('f')  # disabled
        
        # After context, all protocols should be restored
        assert config.is_protocol_enabled('0')
        assert config.is_protocol_enabled('g')
        assert config.is_protocol_enabled('9')
        assert config.is_protocol_enabled('a')
    
    def test_protocols_with_existing_disabled(self):
        """Test protocols context manager with already disabled protocols"""
        config = AdapterConfig(disabled_protocols={'e', 'f'})
        
        # Initially e and f are disabled
        assert not config.is_protocol_enabled('e')
        assert not config.is_protocol_enabled('f')
        assert config.is_protocol_enabled('0')
        assert config.is_protocol_enabled('g')
        assert config.is_protocol_enabled('9')
        
        # Enable only specific protocols
        with config.protocols(['0', 'g']):
            assert config.is_protocol_enabled('0')      # enabled
            assert config.is_protocol_enabled('g')      # enabled
            assert not config.is_protocol_enabled('9')  # disabled by context
            assert not config.is_protocol_enabled('a')  # disabled by context
            assert not config.is_protocol_enabled('e')  # disabled by context
            assert not config.is_protocol_enabled('f')  # disabled by context
        
        # After context, original state should be restored
        assert config.is_protocol_enabled('0')      # restored
        assert config.is_protocol_enabled('g')      # restored
        assert config.is_protocol_enabled('9')      # restored
        assert not config.is_protocol_enabled('e')  # original state
        assert not config.is_protocol_enabled('f')  # original state
    
    def test_nested_context_managers(self):
        """Test nested pause_protocols and protocols context managers"""
        config = AdapterConfig()
        
        # Initially all enabled
        assert config.is_protocol_enabled('0')
        assert config.is_protocol_enabled('g')
        assert config.is_protocol_enabled('9')
        
        with config.pause_protocols(['g']):
            assert config.is_protocol_enabled('0')
            assert not config.is_protocol_enabled('g')  # paused
            assert config.is_protocol_enabled('9')
            
            with config.protocols(['0', '9']):
                assert config.is_protocol_enabled('0')      # enabled by inner
                assert not config.is_protocol_enabled('g')  # disabled by inner
                assert config.is_protocol_enabled('9')      # enabled by inner
                assert not config.is_protocol_enabled('a')  # disabled by inner
            
            # Back to outer context
            assert config.is_protocol_enabled('0')
            assert not config.is_protocol_enabled('g')  # still paused
            assert config.is_protocol_enabled('9')
            assert config.is_protocol_enabled('a')      # restored
        
        # Back to original state
        assert config.is_protocol_enabled('0')
        assert config.is_protocol_enabled('g')      # restored
        assert config.is_protocol_enabled('9')