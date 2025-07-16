#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the ThreadSafeAdapterConfig functionality.

This module tests the thread-safe configuration wrapper that uses contextvars
for request isolation in multi-user environments like FastAPI.
"""

import asyncio
import pytest
from concurrent.futures import ThreadPoolExecutor
from langchain_aisdk_adapter.config import ThreadSafeAdapterConfig, AdapterConfig


class TestThreadSafeAdapterConfig:
    """Test ThreadSafeAdapterConfig functionality"""
    
    def test_basic_functionality(self):
        """Test basic ThreadSafeAdapterConfig functionality"""
        config = ThreadSafeAdapterConfig()
        
        # Test default behavior
        assert config.is_protocol_enabled('0') is True
        assert config.is_protocol_enabled('g') is True
        
        # Test pause_protocols
        with config.pause_protocols(['0', 'g']):
            assert config.is_protocol_enabled('0') is False
            assert config.is_protocol_enabled('g') is False
            assert config.is_protocol_enabled('9') is True  # Not paused
        
        # Test restoration after context
        assert config.is_protocol_enabled('0') is True
        assert config.is_protocol_enabled('g') is True
    
    def test_protocols_context_manager(self):
        """Test protocols context manager"""
        config = ThreadSafeAdapterConfig()
        
        # Test protocols (enable only specific ones)
        with config.protocols(['0', 'g']):
            assert config.is_protocol_enabled('0') is True
            assert config.is_protocol_enabled('g') is True
            assert config.is_protocol_enabled('9') is False  # Not in allowed list
            assert config.is_protocol_enabled('a') is False  # Not in allowed list
        
        # Test restoration after context
        assert config.is_protocol_enabled('0') is True
        assert config.is_protocol_enabled('9') is True
    
    def test_with_base_config(self):
        """Test ThreadSafeAdapterConfig with custom base config"""
        base_config = AdapterConfig(disabled_protocols={'e', 'f'})
        config = ThreadSafeAdapterConfig(base_config)
        
        # Test base config is respected
        assert config.is_protocol_enabled('0') is True
        assert config.is_protocol_enabled('e') is False  # Disabled in base
        assert config.is_protocol_enabled('f') is False  # Disabled in base
        
        # Test context overrides work with base config
        with config.pause_protocols(['0']):
            assert config.is_protocol_enabled('0') is False  # Paused
            assert config.is_protocol_enabled('e') is False  # Still disabled from base
            assert config.is_protocol_enabled('g') is True   # Not affected
    
    def test_for_request_factory(self):
        """Test for_request factory method"""
        config1 = ThreadSafeAdapterConfig.for_request()
        config2 = ThreadSafeAdapterConfig.for_request()
        
        # Should be different instances
        assert config1 is not config2
        
        # But should behave the same
        assert config1.is_protocol_enabled('0') == config2.is_protocol_enabled('0')
    
    def test_attribute_delegation(self):
        """Test that other attributes are delegated to base config"""
        base_config = AdapterConfig()
        config = ThreadSafeAdapterConfig(base_config)
        
        # Test delegation of methods
        config.disable_protocol('test')
        assert 'test' in base_config.disabled_protocols
        
        config.enable_protocol('test')
        assert 'test' not in base_config.disabled_protocols
    
    @pytest.mark.asyncio
    async def test_async_isolation(self):
        """Test that async contexts are properly isolated"""
        config = ThreadSafeAdapterConfig()
        results = []
        
        async def task1():
            with config.protocols(['0']):
                # Small delay to allow context switching
                await asyncio.sleep(0.01)
                results.append(('task1', config.is_protocol_enabled('0'), config.is_protocol_enabled('g')))
        
        async def task2():
            with config.protocols(['g']):
                # Small delay to allow context switching
                await asyncio.sleep(0.01)
                results.append(('task2', config.is_protocol_enabled('0'), config.is_protocol_enabled('g')))
        
        async def task3():
            # No context manager
            await asyncio.sleep(0.01)
            results.append(('task3', config.is_protocol_enabled('0'), config.is_protocol_enabled('g')))
        
        # Run tasks concurrently
        await asyncio.gather(task1(), task2(), task3())
        
        # Check results
        task1_result = next(r for r in results if r[0] == 'task1')
        task2_result = next(r for r in results if r[0] == 'task2')
        task3_result = next(r for r in results if r[0] == 'task3')
        
        # task1: only '0' enabled
        assert task1_result[1] is True   # '0' enabled
        assert task1_result[2] is False  # 'g' disabled
        
        # task2: only 'g' enabled
        assert task2_result[1] is False  # '0' disabled
        assert task2_result[2] is True   # 'g' enabled
        
        # task3: no context, all enabled
        assert task3_result[1] is True   # '0' enabled
        assert task3_result[2] is True   # 'g' enabled
    
    def test_thread_isolation(self):
        """Test that different threads are properly isolated"""
        config = ThreadSafeAdapterConfig()
        results = []
        
        def thread1():
            with config.protocols(['0']):
                # Small delay to allow context switching
                import time
                time.sleep(0.01)
                results.append(('thread1', config.is_protocol_enabled('0'), config.is_protocol_enabled('g')))
        
        def thread2():
            with config.protocols(['g']):
                # Small delay to allow context switching
                import time
                time.sleep(0.01)
                results.append(('thread2', config.is_protocol_enabled('0'), config.is_protocol_enabled('g')))
        
        def thread3():
            # No context manager
            import time
            time.sleep(0.01)
            results.append(('thread3', config.is_protocol_enabled('0'), config.is_protocol_enabled('g')))
        
        # Run in separate threads
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(thread1),
                executor.submit(thread2),
                executor.submit(thread3)
            ]
            # Wait for all to complete
            for future in futures:
                future.result()
        
        # Check results
        thread1_result = next(r for r in results if r[0] == 'thread1')
        thread2_result = next(r for r in results if r[0] == 'thread2')
        thread3_result = next(r for r in results if r[0] == 'thread3')
        
        # thread1: only '0' enabled
        assert thread1_result[1] is True   # '0' enabled
        assert thread1_result[2] is False  # 'g' disabled
        
        # thread2: only 'g' enabled
        assert thread2_result[1] is False  # '0' disabled
        assert thread2_result[2] is True   # 'g' enabled
        
        # thread3: no context, all enabled
        assert thread3_result[1] is True   # '0' enabled
        assert thread3_result[2] is True   # 'g' enabled
    
    def test_nested_contexts(self):
        """Test nested context managers"""
        config = ThreadSafeAdapterConfig()
        
        with config.protocols(['0', 'g', '9']):
            assert config.is_protocol_enabled('0') is True
            assert config.is_protocol_enabled('g') is True
            assert config.is_protocol_enabled('9') is True
            assert config.is_protocol_enabled('a') is False
            
            with config.pause_protocols(['g']):
                assert config.is_protocol_enabled('0') is True   # Still enabled
                assert config.is_protocol_enabled('g') is False  # Paused
                assert config.is_protocol_enabled('9') is True   # Still enabled
                assert config.is_protocol_enabled('a') is False  # Still disabled
            
            # After inner context
            assert config.is_protocol_enabled('0') is True
            assert config.is_protocol_enabled('g') is True   # Restored
            assert config.is_protocol_enabled('9') is True
            assert config.is_protocol_enabled('a') is False
        
        # After outer context
        assert config.is_protocol_enabled('0') is True
        assert config.is_protocol_enabled('g') is True
        assert config.is_protocol_enabled('9') is True
        assert config.is_protocol_enabled('a') is True  # All restored