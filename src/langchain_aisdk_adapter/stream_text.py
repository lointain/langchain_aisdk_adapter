"""Stream Text Module - AI SDK Compatible streamText functionality.

This module provides a simplified streaming interface similar to AI SDK's streamText,
offering an alternative to the core LangChainAdapter functionality.
"""

import asyncio
import uuid
from typing import AsyncIterable, Optional, Dict, Any, Callable, Awaitable, Union, List

from .models import LangChainStreamInput
from .data_stream import DataStreamResponse
from .callbacks import BaseAICallbackHandler
from .context import DataStreamContext
from .langchain_adapter import LangChainAdapter

try:
    from langchain_core.language_models import BaseLanguageModel
    from langchain_core.messages import BaseMessage
    from langchain_core.tools import BaseTool
    from langchain_core.runnables import Runnable
except ImportError:
    BaseLanguageModel = None
    BaseMessage = None
    BaseTool = None
    Runnable = None


class StreamTextResult:
    """Result object for stream_text function."""
    
    def __init__(self, data_stream_with_emitters):
        self._data_stream_with_emitters = data_stream_with_emitters
    
    def __aiter__(self):
        """Make StreamTextResult async iterable."""
        return self._data_stream_with_emitters.__aiter__()
    
    @property
    def data_stream(self):
        """Get the data stream for iteration."""
        return self._data_stream_with_emitters.data_stream
    
    def to_data_stream_response(self) -> 'DataStreamResponse':
        """Convert to DataStreamResponse for FastAPI streaming."""
        return DataStreamResponse(
            stream=self._data_stream_with_emitters,
            protocol_version="v4"
        )


class StreamTextCallbackHandler(BaseAICallbackHandler):
    """Concrete callback handler for stream_text functionality."""
    
    def __init__(self, chunk_cb=None, step_finish_cb=None, abort_cb=None, finish_cb=None, error_cb=None):
        self.chunk_cb = chunk_cb
        self.step_finish_cb = step_finish_cb
        self.abort_cb = abort_cb
        self.finish_cb = finish_cb
        self.error_cb = error_cb
    
    async def on_chunk(self, chunk):
        if self.chunk_cb:
            # Extract text content from chunk for simple callback
            text_content = ""
            if hasattr(chunk, 'textDelta') and chunk.textDelta:
                text_content = chunk.textDelta
            elif hasattr(chunk, 'text') and chunk.text:
                text_content = chunk.text
            elif isinstance(chunk, str):
                text_content = chunk
            
            if text_content:
                await self.chunk_cb(text_content)
    
    async def on_step_finish(self, step):
        if self.step_finish_cb:
            await self.step_finish_cb(step)
    
    async def on_abort(self, steps):
        if self.abort_cb:
            await self.abort_cb("Stream aborted")
    
    async def on_finish(self, message, options):
        if self.finish_cb:
            await self.finish_cb(message, options)
    
    async def on_error(self, error):
        if self.error_cb:
            await self.error_cb(error)


def stream_text(
    model: Optional[BaseLanguageModel] = None,
    system: Optional[str] = None,
    messages: Optional[List[BaseMessage]] = None,
    max_steps: Optional[int] = None,
    tools: Optional[List[BaseTool]] = None,
    on_chunk: Optional[Callable[[str], Awaitable[None]]] = None,
    on_step_finish: Optional[Callable[[Any], Awaitable[None]]] = None,
    on_abort: Optional[Callable[[str], Awaitable[None]]] = None,
    on_finish: Optional[Callable[[Any], Awaitable[None]]] = None,
    on_error: Optional[Callable[[Exception], Awaitable[None]]] = None,
    experimental_active_tools: Optional[List[str]] = None,
    experimental_transform: Optional[Callable[[AsyncIterable[str]], AsyncIterable[str]]] = None,
    experimental_generateMessageId: Optional[Callable[[], str]] = None,
    message_id: Optional[str] = None,
    runnable_factory: Optional[Callable[..., Runnable]] = None,
    protocol_version: str = "v4",
    **options
) -> 'StreamTextResult':
    """Stream text from a language model with AI SDK compatible interface.
    
    This function provides a simplified streaming interface similar to AI SDK's streamText,
    offering callbacks and experimental features for enhanced control over the streaming process.
    
    Args:
        model: The LangChain language model to use
        system: System prompt/message
        messages: List of messages for the conversation
        max_steps: Maximum number of steps for multi-step reasoning
        tools: List of tools available to the model
        on_chunk: Callback for each text chunk (receives string)
        on_step_finish: Callback when a step finishes
        on_abort: Callback when stream is aborted
        on_finish: Callback when stream finishes
        on_error: Callback for errors
        experimental_active_tools: List of tool names to activate
        experimental_transform: Transform function for the stream
        message_id: Optional message ID for tracking
        runnable_factory: Factory function to create custom Runnable objects
        protocol_version: Protocol version to use ("v4" or "v5"), defaults to "v4"
        **options: Additional options passed to the adapter
    
    Returns:
        DataStreamResponse: Streaming response with data_stream attribute
    
    Example:
        ```python
        from langchain_openai import ChatOpenAI
        from langchain_aisdk_adapter.stream_text import stream_text
        
        model = ChatOpenAI(model="gpt-3.5-turbo")
        
        async def handle_chunk(text: str):
            print(f"Chunk: {text}")
        
        response = stream_text(
            model=model,
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": "Hello!"}],
            on_chunk=handle_chunk
        )
        
        async for chunk in response.data_stream:
            # Process chunks
            pass
        ```
    """
    # Validate input: either model or runnable_factory must be provided
    if model is None and runnable_factory is None:
        raise ValueError("Either 'model' or 'runnable_factory' must be provided")
    
    # Create callback handler with provided callbacks
    callback_handler = StreamTextCallbackHandler(
        chunk_cb=on_chunk,
        step_finish_cb=on_step_finish,
        abort_cb=on_abort,
        finish_cb=on_finish,
        error_cb=on_error
    )
    
    # Create context for manual protocol sending
    context = DataStreamContext()
    
    # If runnable_factory is provided, use it to create the runnable
    if runnable_factory is not None:
        # Prepare parameters for the factory function
        factory_params = {
            'model': model,
            'system': system,
            'messages': messages or [],
            'max_steps': max_steps,
            'tools': tools or [],
            'experimental_active_tools': experimental_active_tools or [],
            'context': context
        }
        
        # Create runnable using the factory
        runnable = runnable_factory(**factory_params)
        
        if not hasattr(runnable, 'astream_events'):
            raise ValueError(
                "Runnable created by runnable_factory must have astream_events method"
            )
        
        # Create input for the runnable
        runnable_input = {}
        if messages:
            runnable_input['messages'] = messages
        if system:
            runnable_input['system'] = system
        
        # Stream events from the runnable
        stream = runnable.astream_events(runnable_input, version="v2")
    else:
        # Default behavior: create a simple chain with the model
        if messages is None:
            messages = []
        
        # Filter tools based on experimental_active_tools if provided
        active_tools = tools or []
        if experimental_active_tools and tools:
            active_tools = [
                tool for tool in tools 
                if tool.name in experimental_active_tools
            ]
        
        # Bind tools to model if available
        if active_tools and hasattr(model, 'bind_tools'):
            bound_model = model.bind_tools(active_tools)
        else:
            bound_model = model
        
        # Create input for streaming
        model_input = messages
        if system and messages:
            # Insert system message at the beginning if not already present
            from langchain_core.messages import SystemMessage
            if not any(isinstance(msg, SystemMessage) for msg in messages):
                model_input = [SystemMessage(content=system)] + messages
        
        # Stream events from the model
        stream = bound_model.astream_events(model_input, version="v2")
    
    # Prepare options for the adapter with experimental features
    adapter_options = {
        'experimental_transform': experimental_transform,
        'experimental_generateMessageId': experimental_generateMessageId,
        'protocol_version': protocol_version,
        **options
    }
    
    # Convert to data stream using the adapter's method
    from .langchain_adapter import LangChainAdapter
    
    # Use the adapter's to_data_stream method which returns DataStreamWithEmitters
    data_stream_with_emitters = LangChainAdapter.to_data_stream(
        stream=stream,
        callbacks=callback_handler,
        message_id=message_id,
        options=adapter_options
    )
    
    return StreamTextResult(data_stream_with_emitters)


def stream_text_response(
    model: Optional[BaseLanguageModel] = None,
    system: Optional[str] = None,
    messages: Optional[List[BaseMessage]] = None,
    max_steps: Optional[int] = None,
    tools: Optional[List[BaseTool]] = None,
    on_chunk: Optional[Callable[[str], Awaitable[None]]] = None,
    on_step_finish: Optional[Callable[[Any], Awaitable[None]]] = None,
    on_abort: Optional[Callable[[str], Awaitable[None]]] = None,
    on_finish: Optional[Callable[[Any], Awaitable[None]]] = None,
    on_error: Optional[Callable[[Exception], Awaitable[None]]] = None,
    experimental_active_tools: Optional[List[str]] = None,
    experimental_transform: Optional[Callable[[AsyncIterable[str]], AsyncIterable[str]]] = None,
    experimental_generateMessageId: Optional[Callable[[], str]] = None,
    message_id: Optional[str] = None,
    runnable_factory: Optional[Callable[..., Runnable]] = None,
    protocol_version: str = "v4",
    **options
) -> DataStreamResponse:
    """Stream text from a language model with FastAPI compatible response.
    
    This function provides the same functionality as stream_text but returns
    a DataStreamResponse object suitable for FastAPI streaming responses.
    
    Args:
        model: The language model to use for streaming
        system: System message to include in the conversation
        messages: List of messages for the conversation
        max_steps: Maximum number of steps for multi-step conversations
        tools: List of tools available to the model
        on_chunk: Callback for each text chunk
        on_step_finish: Callback when a step finishes
        on_abort: Callback when streaming is aborted
        on_finish: Callback when streaming finishes
        on_error: Callback when an error occurs
        experimental_active_tools: List of tool names to activate
        experimental_transform: Transform function for the stream
        experimental_generateMessageId: Function to generate message IDs
        message_id: Optional message ID
        runnable_factory: Factory function to create custom runnable
        protocol_version: Protocol version to use ("v4" or "v5"), defaults to "v4"
        **options: Additional options
    
    Returns:
        DataStreamResponse: FastAPI compatible streaming response
    
    Example:
        ```python
        from fastapi import FastAPI
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        
        app = FastAPI()
        model = ChatOpenAI()
        
        @app.post("/stream")
        async def stream_endpoint():
            return stream_text_response(
                model=model,
                messages=[HumanMessage(content="Hello!")]
            )
        ```
    """
    # Use stream_text and convert to response
    result = stream_text(
        model=model,
        system=system,
        messages=messages,
        max_steps=max_steps,
        tools=tools,
        on_chunk=on_chunk,
        on_step_finish=on_step_finish,
        on_abort=on_abort,
        on_finish=on_finish,
        on_error=on_error,
        experimental_active_tools=experimental_active_tools,
        experimental_transform=experimental_transform,
        experimental_generateMessageId=experimental_generateMessageId,
        message_id=message_id,
        runnable_factory=runnable_factory,
        protocol_version=protocol_version,
        **options
    )
    
    return result.to_data_stream_response()