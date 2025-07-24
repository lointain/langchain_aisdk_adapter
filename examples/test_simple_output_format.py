#!/usr/bin/env python3
"""Simple test for DataStreamWithEmitters basic functionality."""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from langchain_aisdk_adapter.data_stream import DataStreamWithEmitters
from langchain_aisdk_adapter.langchain_adapter import LangChainAdapter, AdapterOptions

async def test_data_stream():
    """Test basic data stream functionality."""
    
    # Create a simple async generator to simulate LangChain stream
    async def simple_stream():
        yield {"type": "start"}
        yield {"type": "text", "content": "Hello"}
        yield {"type": "text", "content": " World"}
        yield {"type": "end"}
    
    print("=== Testing basic data stream ===")
    data_stream = LangChainAdapter.to_data_stream(
        simple_stream(),
        options=AdapterOptions(protocol_version="v4")
    )
    
    chunks = []
    async for chunk in data_stream:
        chunks.append(chunk)
        print(f"Chunk: {chunk}")
    
    print(f"\nTotal chunks: {len(chunks)}")
    
    print("\n=== Testing to_data_stream_response (v4) ===")
    response = LangChainAdapter.to_data_stream_response(
        simple_stream(),
        options=AdapterOptions(protocol_version="v4")
    )
    
    print(f"Response type: {type(response)}")
    print(f"Response headers: {response.headers}")
    print(f"Response status: {response.status_code}")
    
    # Test the response body
    response_chunks = []
    async for chunk in response.body_iterator:
        if chunk.strip():
            response_chunks.append(chunk)
            print(f"Response chunk: {repr(chunk)}")
    
    print(f"\nTotal response chunks: {len(response_chunks)}")
    
    print("\n=== Testing to_data_stream_response (v5) ===")
    response_v5 = LangChainAdapter.to_data_stream_response(
        simple_stream(),
        options=AdapterOptions(protocol_version="v5")
    )
    
    print(f"Response v5 type: {type(response_v5)}")
    print(f"Response v5 headers: {response_v5.headers}")
    print(f"Response v5 status: {response_v5.status_code}")
    
    print("\n=== All tests completed successfully! ===")

if __name__ == "__main__":
    asyncio.run(test_data_stream())