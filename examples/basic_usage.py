"""Basic usage example of LangChain to AI SDK adapter."""

import asyncio
from typing import AsyncGenerator

from langchain_aisdk_adapter import to_data_stream, BaseAICallbackHandler


# Example 1: Convert string stream (e.g., from StringOutputParser)
async def example_string_stream():
    """Example of converting a string stream."""
    print("=== Example 1: String Stream ===")
    
    async def mock_string_stream() -> AsyncGenerator[str, None]:
        """Mock string stream from LangChain StringOutputParser."""
        chunks = ["Hello", " ", "world", "!", " How", " are", " you", "?"]
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0.1)  # Simulate streaming delay
    
    # Convert to AI SDK format
    async for ui_chunk in to_data_stream(mock_string_stream()):
        print(f"UI Chunk: {ui_chunk}")


# Example 2: Convert AI message chunks
async def example_ai_message_chunks():
    """Example of converting LangChain AI message chunks."""
    print("\n=== Example 2: AI Message Chunks ===")
    
    async def mock_ai_message_stream():
        """Mock AI message chunk stream from LangChain model.stream."""
        chunks = [
            {"content": "Hello"},
            {"content": [{"type": "text", "text": " world"}]},
            {"content": "!"},
        ]
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0.1)
    
    async for ui_chunk in to_data_stream(mock_ai_message_stream()):
        print(f"UI Chunk: {ui_chunk}")


# Example 3: Convert stream events (v2 format)
async def example_stream_events():
    """Example of converting LangChain stream events."""
    print("\n=== Example 3: Stream Events ===")
    
    async def mock_stream_events():
        """Mock stream events from LangChain."""
        events = [
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": {"content": "Hello"}}
            },
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": {"content": " from"}}
            },
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": {"content": " LangChain!"}}
            },
        ]
        for event in events:
            yield event
            await asyncio.sleep(0.1)
    
    async for ui_chunk in to_data_stream(mock_stream_events()):
        print(f"UI Chunk: {ui_chunk}")


# Example 4: Using callbacks
async def example_with_callbacks():
    """Example of using callbacks with the adapter."""
    print("\n=== Example 4: With Callbacks ===")
    
    class TestCallbackHandler(BaseAICallbackHandler):
        async def on_start(self):
            print("🚀 Stream started!")
        
        async def on_finish(self, message, options):
            print(f"✅ Final message: '{message.content}'")
            print(f"✅ Message parts: {len(message.parts)} parts")
    
    async def mock_stream():
        chunks = ["Hello", " ", "callback", " ", "world!"]
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0.1)
    
    callbacks = TestCallbackHandler()
    
    print("\nUI Chunks:")
    async for ui_chunk in to_data_stream(mock_stream(), callbacks=callbacks):
        print(f"  {ui_chunk}")


# Example 5: Custom message ID
async def example_custom_message_id():
    """Example of using custom message ID."""
    print("\n=== Example 5: Custom Message ID ===")
    
    async def mock_stream():
        yield "Custom ID example"
    
    async for ui_chunk in to_data_stream(
        mock_stream(), 
        message_id="custom-message-123"
    ):
        print(f"UI Chunk: {ui_chunk}")


# Example 6: Async callbacks
async def example_async_callbacks():
    """Example of using async callbacks."""
    print("\n=== Example 6: Async Callbacks ===")
    
    class AsyncCallbackHandler(BaseAICallbackHandler):
        async def on_start(self):
            await asyncio.sleep(0.01)
            print("🚀 Async stream started!")
        
        async def on_finish(self, message, options):
            await asyncio.sleep(0.01)
            print(f"✅ Async final: '{message.content}'")
    
    async def mock_stream():
        yield "Async"
        yield " callbacks"
        yield " work!"
    
    callbacks = AsyncCallbackHandler()
    
    async for ui_chunk in to_data_stream(mock_stream(), callbacks=callbacks):
        print(f"  UI Chunk: {ui_chunk}")


async def main():
    """Run all examples."""
    await example_string_stream()
    await example_ai_message_chunks()
    await example_stream_events()
    await example_with_callbacks()
    await example_custom_message_id()
    await example_async_callbacks()


if __name__ == "__main__":
    asyncio.run(main())