#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI SDK Protocol Part Emitter

Custom LangChain Runnable for injecting pre-formatted AI SDK protocol parts into streams.
"""

from typing import Any, AsyncIterator, Iterator, Optional, Union
from pydantic import Field
from langchain_core.runnables import Runnable, RunnableConfig


class AISDKPartEmitter(Runnable):
    """
    Custom LangChain Runnable for injecting pre-formatted AI SDK protocol parts into streams
    
    Allows insertion of AI SDK protocol-compliant data parts into LangChain stream processing pipelines
    without modifying existing LangChain components.
    """
    ai_sdk_part_content: str = Field(description="Pre-formatted AI SDK protocol string, e.g. '0:\"hello\"\\n'")

    def __init__(self, ai_sdk_part_content: str, **kwargs: Any) -> None:
        """Initialize AI SDK part emitter
        
        Args:
            ai_sdk_part_content: Pre-formatted AI SDK protocol string
            **kwargs: Other Runnable parameters
        """
        super().__init__(**kwargs)
        # Ensure content ends with newline, which is required by AI SDK protocol
        if not ai_sdk_part_content.endswith('\n'):
            ai_sdk_part_content += '\n'
        self.ai_sdk_part_content = ai_sdk_part_content

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        """Synchronous invocation, directly returns input (non-streaming)
        
        Args:
            input: Input data
            config: Runtime configuration (optional)
            
        Returns:
            Original input data
        """
        return input

    async def ainvoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        """Asynchronous invocation, directly returns input (non-streaming)
        
        Args:
            input: Input data
            config: Runtime configuration (optional)
            
        Returns:
            Original input data
        """
        return input

    def stream(self, input: Any, config: Optional[RunnableConfig] = None) -> Iterator[Union[Any, str]]:
        """Synchronous streaming, sends AI SDK part first, then passes input
        
        Args:
            input: Input data
            config: Runtime configuration (optional)
            
        Yields:
            AI SDK protocol string and input data
        """
        yield self.ai_sdk_part_content  # Send pre-formatted AI SDK protocol string
        yield input  # Pass input to next component in chain

    async def astream(self, input: Any, config: Optional[RunnableConfig] = None) -> AsyncIterator[Union[Any, str]]:
        """Asynchronous streaming, sends AI SDK part first, then passes input
        
        Args:
            input: Input data
            config: Runtime configuration (optional)
            
        Yields:
            AI SDK protocol string and input data
        """
        yield self.ai_sdk_part_content  # Send pre-formatted AI SDK protocol string
        yield input  # Pass input to next component in chain