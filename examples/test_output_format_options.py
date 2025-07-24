"""Test output_format options for both to_data_stream and to_data_stream_response."""

import asyncio
from typing import AsyncGenerator

from langchain_aisdk_adapter import LangChainAdapter
from langchain_aisdk_adapter.models import LangChainStreamInput


async def create_mock_stream() -> AsyncGenerator[LangChainStreamInput, None]:
    """Create a mock LangChain stream for testing."""
    # Simulate a simple text generation stream
    yield {
        "event": "on_chat_model_stream",
        "data": {
            "chunk": {
                "content": "Hello",
                "type": "AIMessageChunk"
            }
        },
        "run_id": "test-run-1",
        "name": "ChatOpenAI",
        "tags": [],
        "metadata": {}
    }
    
    yield {
        "event": "on_chat_model_stream",
        "data": {
            "chunk": {
                "content": " world!",
                "type": "AIMessageChunk"
            }
        },
        "run_id": "test-run-1",
        "name": "ChatOpenAI",
        "tags": [],
        "metadata": {}
    }


async def test_to_data_stream_chunks_format():
    """Test to_data_stream with chunks output format."""
    print("=== Testing to_data_stream with chunks format ===")
    
    stream = create_mock_stream()
    data_stream = LangChainAdapter.to_data_stream(
        stream=stream,
        options={"output_format": "chunks", "protocol_version": "v4"}
    )
    
    chunks = []
    async for chunk in data_stream:
        print(f"Chunk: {chunk}")
        chunks.append(chunk)
    
    print(f"Total chunks: {len(chunks)}")
    print()
    return chunks


async def test_to_data_stream_protocol_format():
    """Test to_data_stream with protocol output format."""
    print("=== Testing to_data_stream with protocol format ===")
    
    stream = create_mock_stream()
    data_stream = LangChainAdapter.to_data_stream(
        stream=stream,
        options={"output_format": "protocol", "protocol_version": "v4"}
    )
    
    protocol_chunks = []
    async for chunk in data_stream:
        print(f"Protocol chunk: {chunk}")
        protocol_chunks.append(chunk)
    
    print(f"Total protocol chunks: {len(protocol_chunks)}")
    print()
    return protocol_chunks


async def test_to_data_stream_response_chunks():
    """Test to_data_stream_response with chunks format (should convert to protocol)."""
    print("=== Testing to_data_stream_response with chunks format ===")
    
    stream = create_mock_stream()
    response = LangChainAdapter.to_data_stream_response(
        stream=stream,
        options={"output_format": "chunks", "protocol_version": "v4"}
    )
    
    print(f"Response type: {type(response)}")
    print(f"Response headers: {response.headers}")
    print(f"Response status: {response.status_code}")
    
    # Read response content
    response_chunks = []
    async for chunk in response.body_iterator:
        chunk_str = chunk.decode('utf-8') if isinstance(chunk, bytes) else str(chunk)
        print(f"Response chunk: {repr(chunk_str)}")
        response_chunks.append(chunk_str)
    
    print(f"Total response chunks: {len(response_chunks)}")
    print()
    return response_chunks


async def test_to_data_stream_response_protocol():
    """Test to_data_stream_response with protocol format."""
    print("=== Testing to_data_stream_response with protocol format ===")
    
    stream = create_mock_stream()
    response = LangChainAdapter.to_data_stream_response(
        stream=stream,
        options={"output_format": "protocol", "protocol_version": "v4"}
    )
    
    print(f"Response type: {type(response)}")
    print(f"Response headers: {response.headers}")
    print(f"Response status: {response.status_code}")
    
    # Read response content
    response_chunks = []
    async for chunk in response.body_iterator:
        chunk_str = chunk.decode('utf-8') if isinstance(chunk, bytes) else str(chunk)
        print(f"Response chunk: {repr(chunk_str)}")
        response_chunks.append(chunk_str)
    
    print(f"Total response chunks: {len(response_chunks)}")
    print()
    return response_chunks


async def test_emit_functionality():
    """Test emit functionality with both output formats."""
    print("=== Testing emit functionality ===")
    
    # Test with chunks format
    print("--- Chunks format with emit ---")
    stream = create_mock_stream()
    data_stream = LangChainAdapter.to_data_stream(
        stream=stream,
        options={"output_format": "chunks", "protocol_version": "v4"}
    )
    
    # Emit some manual chunks
    await data_stream.emit_text_delta("Manual text chunk")
    await data_stream.emit_data({"custom": "data"})
    
    chunks = []
    async for chunk in data_stream:
        print(f"Chunk with emit: {chunk}")
        chunks.append(chunk)
        if len(chunks) >= 5:  # Limit output
            break
    
    print(f"Total chunks with emit: {len(chunks)}")
    
    # Test with protocol format
    print("--- Protocol format with emit ---")
    stream2 = create_mock_stream()
    data_stream2 = LangChainAdapter.to_data_stream(
        stream=stream2,
        options={"output_format": "protocol", "protocol_version": "v4"}
    )
    
    # Emit some manual chunks
    await data_stream2.emit_text_delta("Manual protocol text")
    await data_stream2.emit_data({"custom": "protocol_data"})
    
    protocol_chunks = []
    async for chunk in data_stream2:
        print(f"Protocol chunk with emit: {chunk}")
        protocol_chunks.append(chunk)
        if len(protocol_chunks) >= 5:  # Limit output
            break
    
    print(f"Total protocol chunks with emit: {len(protocol_chunks)}")
    print()


async def main():
    """Run all tests."""
    print("Testing output_format options for LangChainAdapter\n")
    
    # Test to_data_stream with different formats
    await test_to_data_stream_chunks_format()
    await test_to_data_stream_protocol_format()
    
    # Test to_data_stream_response with different formats
    await test_to_data_stream_response_chunks()
    await test_to_data_stream_response_protocol()
    
    # Test emit functionality
    await test_emit_functionality()
    
    print("=== All output_format tests completed successfully! ===")


if __name__ == "__main__":
    asyncio.run(main())