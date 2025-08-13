#!/usr/bin/env python3
"""Example usage of the separated stream_text and smooth_stream modules.

This example demonstrates how to use the optional streaming modules
that have been separated from the core LangChainAdapter functionality.
"""

import asyncio
from typing import Dict, Any

# Import the separated streaming functions
from src.langchain_aisdk_adapter.stream_text import stream_text, stream_text_response
from src.langchain_aisdk_adapter.smooth_stream import smooth_stream, create_smooth_text_stream

# Mock model for demonstration (replace with real LangChain model)
class MockChatModel:
    """Mock chat model for demonstration purposes."""
    
    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name
    
    async def astream_events(self, messages, version="v1"):
        """Mock streaming events."""
        # Simulate streaming response
        mock_events = [
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": {"content": "Hello"}}
            },
            {
                "event": "on_chat_model_stream", 
                "data": {"chunk": {"content": " there!"}}
            },
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": {"content": " How can I help you today?"}}
            }
        ]
        
        for event in mock_events:
            await asyncio.sleep(0.1)  # Simulate network delay
            yield event


async def example_basic_stream_text():
    """Example of basic stream_text usage."""
    print("\n=== Basic Stream Text Example ===")
    
    model = MockChatModel()
    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]
    
    # Use stream_text function
    result = stream_text(
        model=model,
        messages=messages
    )
    
    print(f"Result type: {type(result)}")
    print("Streaming response:")
    
    chunk_count = 0
    async for chunk in result.data_stream:
        chunk_count += 1
        print(f"  Chunk {chunk_count}: {chunk}")
    
    print(f"Total chunks: {chunk_count}")


async def example_smooth_stream():
    """Example of smooth_stream usage."""
    print("\n=== Smooth Stream Example ===")
    
    model = MockChatModel()
    messages = [
        {"role": "user", "content": "Tell me a story"}
    ]
    
    # Use stream_text with smooth_stream transformation
    result = stream_text(
        model=model,
        messages=messages,
        experimental_transform=lambda stream: smooth_stream(
            stream, 
            chunking="word",    # Split by words
            delay_in_ms=50      # 50ms delay between chunks
        )
    )
    
    print("Smooth streaming response (word by word):")
    
    chunk_count = 0
    async for chunk in result.data_stream:
        chunk_count += 1
        print(f"  Smooth chunk {chunk_count}: {chunk}")
    
    print(f"Total smooth chunks: {chunk_count}")


async def example_create_smooth_text_stream():
    """Example of create_smooth_text_stream usage."""
    print("\n=== Create Smooth Text Stream Example ===")
    
    # Create a smooth text stream directly
    text = "This is a demonstration of smooth text streaming functionality."
    
    smooth_stream_gen = create_smooth_text_stream(
        text=text,
        chunking="word",
        delay_in_ms=100
    )
    
    print("Direct smooth text streaming:")
    
    chunk_count = 0
    async for chunk in smooth_stream_gen:
        chunk_count += 1
        print(f"  Text chunk {chunk_count}: {chunk}")
    
    print(f"Total text chunks: {chunk_count}")


async def example_stream_text_response():
    """Example of using stream_text_response for FastAPI compatibility."""
    print("\n=== Example: Stream Text Response (FastAPI Compatible) ===")
    
    model = MockChatModel()
    messages = [
        {"role": "user", "content": "Tell me about streaming responses"}
    ]
    
    # Get FastAPI-compatible streaming response
    response = stream_text_response(
        model=model,
        messages=messages,
        system="You are a helpful assistant.",
        temperature=0.7
    )
    
    print(f"Response type: {type(response)}")
    print(f"Status code: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")
    
    # Stream the response (as FastAPI would do)
    print("Streaming response chunks:")
    chunk_count = 0
    async for chunk in response.body_iterator:
        chunk_count += 1
        chunk_str = chunk.decode() if isinstance(chunk, bytes) else str(chunk)
        print(f"  Chunk {chunk_count}: {chunk_str.strip()}")
        if chunk_count >= 5:  # Limit output
            break
    
    print(f"✅ FastAPI-compatible response with {chunk_count} chunks")


async def main():
    """Run all examples."""
    print("🚀 LangChain AI SDK Adapter - Separated Modules Examples")
    print("==========================================================")
    
    try:
        await example_basic_stream_text()
        await example_smooth_stream()
        await example_create_smooth_text_stream()
        await example_stream_text_response()
        
        print("\n✅ All examples completed successfully!")
        print("\n📝 Key Features:")
        print("  • stream_text: AI SDK compatible text streaming")
        print("  • smooth_stream: Configurable chunk smoothing")
        print("  • create_smooth_text_stream: Direct text streaming")
        print("  • stream_text_response: FastAPI-compatible responses")
        print("  • Modular design: Optional streaming functionality")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())