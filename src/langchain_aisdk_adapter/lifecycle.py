"""Context lifecycle manager.

Provides automatic context setup and cleanup functionality.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Callable, Any

from .context import DataStreamContext


class ContextLifecycleManager:
    """Context lifecycle manager.
    
    Provides automatic data stream context setup and cleanup functionality.
    """
    
    @staticmethod
    @asynccontextmanager
    async def managed_context(
        data_stream: 'DataStreamWithEmitters'
    ) -> AsyncGenerator['DataStreamWithEmitters', None]:
        """Manage data stream context lifecycle.
        
        Args:
            data_stream: Data stream instance
            
        Yields:
            Data stream instance
        """
        try:
            # Set context
            DataStreamContext.set_current_stream(data_stream)
            yield data_stream
        finally:
            # Clean up context
            DataStreamContext.clear_current_stream()
    
    @staticmethod
    async def with_auto_context(
        data_stream: 'DataStreamWithEmitters',
        operation: Callable[[], Any]
    ) -> Any:
        """Execute operation in automatic context.
        
        Args:
            data_stream: Data stream instance
            operation: Async operation to execute
            
        Returns:
            Operation result
        """
        async with ContextLifecycleManager.managed_context(data_stream):
            if asyncio.iscoroutinefunction(operation):
                return await operation()
            else:
                return operation()
    
    @staticmethod
    def create_context_wrapper(
        data_stream: 'DataStreamWithEmitters'
    ) -> Callable[[Callable], Callable]:
        """Create a context wrapper decorator.
        
        Args:
            data_stream: Data stream instance
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            async def wrapper(*args, **kwargs):
                async with ContextLifecycleManager.managed_context(data_stream):
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
            return wrapper
        return decorator