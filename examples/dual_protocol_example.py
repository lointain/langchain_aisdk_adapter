#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual Protocol Support Example

This example demonstrates how to use the LangChain AI SDK Adapter
with both AI SDK v4 and v5 protocols.
"""

import asyncio
from typing import AsyncGenerator

from langchain_aisdk_adapter import (
    LangChainAdapter,
    DataStreamResponse,
    AdapterOptions,
    UIMessageChunk
)


async def create_mock_langchain_stream() -> AsyncGenerator[dict, None]:
    """Create a mock LangChain stream for demonstration."""
    # Simulate LangChain stream events
    events = [
        {
            "event": "on_chat_model_stream",
            "data": {
                "chunk": {
                    "content": "Hello",
                    "type": "AIMessageChunk"
                }
            }
        },
        {
            "event": "on_chat_model_stream",
            "data": {
                "chunk": {
                    "content": " world!",
                    "type": "AIMessageChunk"
                }
            }
        },
        {
            "event": "on_chat_model_end",
            "data": {
                "output": {
                    "content": "Hello world!",
                    "type": "AIMessage"
                }
            }
        }
    ]
    
    for event in events:
        yield event
        await asyncio.sleep(0.1)  # Simulate streaming delay


async def example_v4_protocol():
    """Example using AI SDK v4 protocol."""
    print("\n=== AI SDK v4 Protocol Example ===")
    
    # Create mock stream
    stream = create_mock_langchain_stream()
    
    # Configure for v4 protocol
    options: AdapterOptions = {
        "protocol_version": "v4",
        "auto_events": True,
        "auto_close": True
    }
    
    # Convert to DataStreamResponse
    response = await LangChainAdapter.to_data_stream_response(
        stream=stream,
        options=options
    )
    
    print(f"Response headers: {response.headers}")
    print("Stream content:")
    
    # Iterate through the response content
    async for chunk in response.body_iterator:
        print(f"  {chunk.strip()}")


async def example_v5_protocol():
    """Example using AI SDK v5 protocol."""
    print("\n=== AI SDK v5 Protocol Example ===")
    
    # Create mock stream
    stream = create_mock_langchain_stream()
    
    # Configure for v5 protocol
    options: AdapterOptions = {
        "protocol_version": "v5",
        "auto_events": True,
        "auto_close": True
    }
    
    # Convert to DataStreamResponse
    response = await LangChainAdapter.to_data_stream_response(
        stream=stream,
        options=options
    )
    
    print(f"Response headers: {response.headers}")
    print("Stream content:")
    
    # Iterate through the response content
    async for chunk in response.body_iterator:
        print(f"  {chunk.strip()}")


async def example_manual_protocol_switching():
    """Example of manually switching between protocols."""
    print("\n=== Manual Protocol Switching Example ===")
    
    # Create mock stream
    stream = create_mock_langchain_stream()
    
    # Convert to data stream first
    data_stream = LangChainAdapter.to_data_stream(
        stream=stream,
        options={"protocol_version": "v5"}
    )
    
    print("Creating responses with different protocols:")
    
    # Create v4 response
    v4_response = DataStreamResponse(
        data_stream,
        protocol_version="v4"
    )
    print(f"v4 headers: {v4_response.headers}")
    
    # Create v5 response (need new stream since the first one is consumed)
    stream2 = create_mock_langchain_stream()
    data_stream2 = LangChainAdapter.to_data_stream(
        stream=stream2,
        options={"protocol_version": "v5"}
    )
    
    v5_response = DataStreamResponse(
        data_stream2,
        protocol_version="v5"
    )
    print(f"v5 headers: {v5_response.headers}")


async def example_with_custom_headers():
    """Example with custom headers for different protocols."""
    print("\n=== Custom Headers Example ===")
    
    # Create mock stream
    stream = create_mock_langchain_stream()
    
    # Custom headers
    custom_headers = {
        "X-Custom-Header": "MyValue",
        "Access-Control-Allow-Origin": "*"
    }
    
    # Configure for v4 protocol (default) with custom headers
    options: AdapterOptions = {
        "protocol_version": "v4"
    }
    
    response = await LangChainAdapter.to_data_stream_response(
        stream=stream,
        options=options,
        headers=custom_headers
    )
    
    print(f"Final headers: {response.headers}")
    print("Note: Protocol-specific headers are merged with custom headers")


async def main():
    """Run all examples."""
    print("LangChain AI SDK Adapter - Dual Protocol Support Examples")
    print("==========================================================")
    
    try:
        await example_v4_protocol()
        await example_v5_protocol()
        await example_manual_protocol_switching()
        await example_with_custom_headers()
        
        print("\n=== Summary ===")
        print("✓ AI SDK v4 protocol: Uses custom data stream format with specific headers")
        print("✓ AI SDK v5 protocol: Uses Server-Sent Events (SSE) format")
        print("✓ Protocol switching: Can be configured via options or DataStreamResponse")
        print("✓ Header merging: Custom headers are merged with protocol-specific headers")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())