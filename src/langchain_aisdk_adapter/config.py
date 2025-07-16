#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration for AI SDK Adapter

Provides configuration options to control which AI SDK protocol parts are automatically generated.
"""

from typing import Set, List, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager
from contextvars import ContextVar
import copy


@dataclass
class AdapterConfig:
    """
    Configuration for LangChain AI SDK Adapter
    
    Controls which AI SDK protocol parts are automatically generated during stream processing.
    Users can disable specific protocol types and manually send them using factory methods.
    
    Example:
        ```python
        from langchain_aisdk_adapter import LangChainAdapter, AdapterConfig
        
        # Disable automatic step generation
        config = AdapterConfig(disabled_protocols={'e', 'f'})
        
        # Use adapter with custom config
        async for part in LangChainAdapter.to_data_stream_response(
            langchain_stream, config=config
        ):
            print(part)
        ```
    """
    
    # Set of protocol type IDs to disable automatic generation
    disabled_protocols: Set[str] = field(default_factory=set)
    
    def is_protocol_enabled(self, protocol_type: str) -> bool:
        """
        Check if a protocol type is enabled for automatic generation
        
        Args:
            protocol_type: AI SDK protocol type identifier (e.g. '0', '9', 'e', 'f')
            
        Returns:
            True if protocol should be automatically generated, False otherwise
        """
        return protocol_type not in self.disabled_protocols
    
    def disable_protocol(self, protocol_type: str) -> None:
        """
        Disable automatic generation of a specific protocol type
        
        Args:
            protocol_type: AI SDK protocol type identifier to disable
        """
        self.disabled_protocols.add(protocol_type)
    
    def enable_protocol(self, protocol_type: str) -> None:
        """
        Enable automatic generation of a specific protocol type
        
        Args:
            protocol_type: AI SDK protocol type identifier to enable
        """
        self.disabled_protocols.discard(protocol_type)
    
    def disable_steps(self) -> None:
        """
        Convenience method to disable automatic step generation (f: start, e: finish)
        """
        self.disabled_protocols.update({'e', 'f'})
    
    def disable_tool_streaming(self) -> None:
        """
        Convenience method to disable automatic tool call streaming (b: start, c: delta)
        """
        self.disabled_protocols.update({'b', 'c'})
    
    def disable_finish_messages(self) -> None:
        """
        Convenience method to disable automatic finish message generation (d:)
        """
        self.disabled_protocols.add('d')
    
    @classmethod
    def minimal(cls) -> 'AdapterConfig':
        """
        Create a minimal configuration that only enables basic text and tool calls
        
        Returns:
            AdapterConfig with most automatic protocols disabled
        """
        return cls(disabled_protocols={'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', '8'})
    
    @classmethod
    def tools_only(cls) -> 'AdapterConfig':
        """
        Create a configuration for tool-focused usage
        
        Returns:
            AdapterConfig with only tool-related protocols enabled
        """
        return cls(disabled_protocols={'d', 'e', 'f', 'g', 'h', 'i', 'j', 'k', '8'})
    
    @contextmanager
    def pause_protocols(self, protocol_types: List[str]):
        """
        Context manager to temporarily pause specific protocol types
        
        Args:
            protocol_types: List of protocol type IDs to pause
            
        Example:
            ```python
            config = AdapterConfig()
            with config.pause_protocols(['0', 'g']):
                # Text and reasoning protocols are paused in this block
                # Intermediate processing won't output to user
                pass
            # Protocols are automatically restored after the block
            ```
        """
        # Store original state
        original_disabled = self.disabled_protocols.copy()
        
        # Add protocols to disabled set
        self.disabled_protocols.update(protocol_types)
        
        try:
            yield self
        finally:
            # Restore original state
            self.disabled_protocols = original_disabled
    
    @contextmanager
    def protocols(self, protocol_types: List[str]):
        """
        Context manager to enable only specific protocol types (opposite of pause_protocols)
        
        This disables all protocols by default and only enables the specified ones.
        Useful when you want to track only specific protocols in a workflow.
        
        Args:
            protocol_types: List of protocol type IDs to enable (all others will be disabled)
            
        Example:
            ```python
            config = AdapterConfig()
            with config.protocols(['0', 'g']):
                # Only text and reasoning protocols are enabled in this block
                # All other protocols are disabled
                pass
            # Original protocol state is restored after the block
            ```
        """
        # Store original state
        original_disabled = self.disabled_protocols.copy()
        
        # Define all known protocol types
        all_protocols = {'0', '1', '2', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'}
        
        # Disable all protocols except the specified ones
        self.disabled_protocols = all_protocols - set(protocol_types)
        
        try:
            yield self
        finally:
            # Restore original state
            self.disabled_protocols = original_disabled


# Context variable for request-scoped protocol overrides
_protocol_overrides: ContextVar[Optional[Set[str]]] = ContextVar('protocol_overrides', default=None)


class ThreadSafeAdapterConfig:
    """
    Thread-safe wrapper for AdapterConfig that uses contextvars for request isolation.
    
    This class provides a more elegant solution for FastAPI multi-user environments
    by ensuring each request has isolated protocol state without requiring
    separate config instances.
    
    Example:
        ```python
        # FastAPI usage - thread-safe by default
        safe_config = ThreadSafeAdapterConfig()
        
        @app.post("/chat")
        async def chat(message: str):
            with safe_config.protocols(['0', 'g']):
                stream = llm.astream([HumanMessage(content=message)])
                return StreamingResponse(
                    LangChainAdapter.to_data_stream_response(stream, config=safe_config),
                    media_type="text/plain"
                )
        ```
    """
    
    def __init__(self, base_config: Optional[AdapterConfig] = None):
        """
        Initialize thread-safe config wrapper
        
        Args:
            base_config: Base configuration to use. If None, creates default config.
        """
        self._base_config = base_config or AdapterConfig()
    
    def is_protocol_enabled(self, protocol_type: str) -> bool:
        """
        Check if a protocol type is enabled, considering context overrides
        
        Args:
            protocol_type: AI SDK protocol type identifier
            
        Returns:
            True if protocol should be automatically generated
        """
        overrides = _protocol_overrides.get()
        if overrides is not None:
            return protocol_type not in overrides
        return self._base_config.is_protocol_enabled(protocol_type)
    
    @contextmanager
    def pause_protocols(self, protocol_types: List[str]):
        """
        Thread-safe context manager to temporarily pause specific protocol types
        
        Args:
            protocol_types: List of protocol type IDs to pause
        """
        # Get current overrides or base disabled protocols
        current_overrides = _protocol_overrides.get()
        if current_overrides is not None:
            new_overrides = current_overrides | set(protocol_types)
        else:
            new_overrides = self._base_config.disabled_protocols | set(protocol_types)
        
        # Set context variable
        token = _protocol_overrides.set(new_overrides)
        try:
            yield self
        finally:
            _protocol_overrides.reset(token)
    
    @contextmanager
    def protocols(self, protocol_types: List[str]):
        """
        Thread-safe context manager to enable only specific protocol types
        
        Args:
            protocol_types: List of protocol type IDs to enable
        """
        # Define all known protocol types
        all_protocols = {'0', '1', '2', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'}
        
        # Disable all protocols except the specified ones
        new_overrides = all_protocols - set(protocol_types)
        
        # Set context variable
        token = _protocol_overrides.set(new_overrides)
        try:
            yield self
        finally:
            _protocol_overrides.reset(token)
    
    @classmethod
    def for_request(cls, base_config: Optional[AdapterConfig] = None) -> 'ThreadSafeAdapterConfig':
        """
        Factory method to create a request-scoped configuration
        
        Args:
            base_config: Base configuration to use
            
        Returns:
            New ThreadSafeAdapterConfig instance for this request
        """
        return cls(base_config)
    
    def __getattr__(self, name):
        """
        Delegate other attributes to base config for compatibility
        """
        return getattr(self._base_config, name)


# Default configuration instances
default_config = AdapterConfig()
safe_config = ThreadSafeAdapterConfig()