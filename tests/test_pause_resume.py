"""Test pause/resume functionality

Test the pause_protocols context manager functionality in AdapterConfig.
"""

import pytest
from langchain_aisdk_adapter.config import AdapterConfig


class TestPauseResume:
    """Test pause/resume protocol functionality"""
    
    def test_pause_protocols_context_manager(self):
        """Test pause_protocols context manager basic functionality"""
        config = AdapterConfig()
        
        # Initially no protocols are disabled
        assert config.disabled_protocols == set()
        assert config.is_protocol_enabled('0')
        assert config.is_protocol_enabled('e')
        assert config.is_protocol_enabled('f')
        
        # Test pausing specific protocols
        with config.pause_protocols(['0', 'e']):
            assert '0' in config.disabled_protocols
            assert 'e' in config.disabled_protocols
            assert 'f' not in config.disabled_protocols
            assert not config.is_protocol_enabled('0')
            assert not config.is_protocol_enabled('e')
            assert config.is_protocol_enabled('f')
        
        # After exiting context, protocols should be restored
        assert config.disabled_protocols == set()
        assert config.is_protocol_enabled('0')
        assert config.is_protocol_enabled('e')
        assert config.is_protocol_enabled('f')
    
    def test_pause_protocols_with_existing_disabled(self):
        """Test pause_protocols when some protocols are already disabled"""
        config = AdapterConfig(disabled_protocols={'3', 'd'})
        
        # Initially some protocols are disabled
        assert config.disabled_protocols == {'3', 'd'}
        assert not config.is_protocol_enabled('3')
        assert not config.is_protocol_enabled('d')
        assert config.is_protocol_enabled('0')
        assert config.is_protocol_enabled('e')
        
        # Test pausing additional protocols
        with config.pause_protocols(['0', 'e']):
            assert config.disabled_protocols == {'3', 'd', '0', 'e'}
            assert not config.is_protocol_enabled('0')
            assert not config.is_protocol_enabled('e')
            assert not config.is_protocol_enabled('3')  # Still disabled
            assert not config.is_protocol_enabled('d')  # Still disabled
            assert config.is_protocol_enabled('f')
        
        # After exiting context, only original disabled protocols remain
        assert config.disabled_protocols == {'3', 'd'}
        assert not config.is_protocol_enabled('3')
        assert not config.is_protocol_enabled('d')
        assert config.is_protocol_enabled('0')
        assert config.is_protocol_enabled('e')
    
    def test_pause_protocols_nested_context(self):
        """Test nested pause_protocols context managers"""
        config = AdapterConfig()
        
        with config.pause_protocols(['0']):
            assert config.disabled_protocols == {'0'}
            assert not config.is_protocol_enabled('0')
            assert config.is_protocol_enabled('e')
            
            with config.pause_protocols(['e', 'f']):
                assert config.disabled_protocols == {'0', 'e', 'f'}
                assert not config.is_protocol_enabled('0')
                assert not config.is_protocol_enabled('e')
                assert not config.is_protocol_enabled('f')
            
            # After inner context, only outer context protocols are disabled
            assert config.disabled_protocols == {'0'}
            assert not config.is_protocol_enabled('0')
            assert config.is_protocol_enabled('e')
            assert config.is_protocol_enabled('f')
        
        # After all contexts, no protocols are disabled
        assert config.disabled_protocols == set()
        assert config.is_protocol_enabled('0')
        assert config.is_protocol_enabled('e')
        assert config.is_protocol_enabled('f')
    
    def test_pause_protocols_empty_list(self):
        """Test pause_protocols with empty protocol list"""
        config = AdapterConfig(disabled_protocols={'3'})
        original_disabled = config.disabled_protocols.copy()
        
        with config.pause_protocols([]):
            # Should not change anything
            assert config.disabled_protocols == original_disabled
        
        # Should still be the same after context
        assert config.disabled_protocols == original_disabled
    
    def test_pause_protocols_duplicate_protocols(self):
        """Test pause_protocols with duplicate protocol names"""
        config = AdapterConfig()
        
        with config.pause_protocols(['0', '0', 'e', 'e']):
            # Should only add each protocol once
            assert config.disabled_protocols == {'0', 'e'}
            assert not config.is_protocol_enabled('0')
            assert not config.is_protocol_enabled('e')
        
        assert config.disabled_protocols == set()
    
    def test_pause_protocols_exception_handling(self):
        """Test that protocols are restored even if exception occurs in context"""
        config = AdapterConfig(disabled_protocols={'3'})
        
        try:
            with config.pause_protocols(['0', 'e']):
                assert config.disabled_protocols == {'3', '0', 'e'}
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Protocols should be restored even after exception
        assert config.disabled_protocols == {'3'}
        assert not config.is_protocol_enabled('3')
        assert config.is_protocol_enabled('0')
        assert config.is_protocol_enabled('e')