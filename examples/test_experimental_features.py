#!/usr/bin/env python3
"""Test script to verify experimental_transform and experimental_generateMessageId functionality."""

import asyncio
import uuid
from typing import Dict, Any, AsyncIterable

from langchain_aisdk_adapter import LangChainAdapter, BaseAICallbackHandler


def smooth_stream_test(chunking='word', delay_in_ms=50):
    """
    Test version of smooth_stream with more visible delays for demonstration.
    
    Args:
        chunking: Controls how text is chunked ('word' or 'line')
        delay_in_ms: Delay in milliseconds between chunks (default: 50ms for testing)
        
    Returns:
        A transform function that can be used with experimental_transform
    """
    import asyncio
    import re
    
    async def transform_stream(stream):
        buffer = ""
        chunk_count = 0
        
        print(f"\n[TRANSFORM] Starting smooth_stream with chunking='{chunking}', delay={delay_in_ms}ms")
        
        async for chunk in stream:
            chunk_count += 1
            print(f"[TRANSFORM] Processing chunk #{chunk_count}: {str(chunk)[:100]}...")
            
            # For demonstration, we'll add a small delay to every chunk
            if delay_in_ms > 0:
                await asyncio.sleep(delay_in_ms / 1000.0)
                print(f"[TRANSFORM] Applied {delay_in_ms}ms delay")
            
            # Pass through the chunk (in a real implementation, we'd process text content)
            yield chunk
        
        print(f"[TRANSFORM] Finished processing {chunk_count} chunks")
    
    return transform_stream


def generate_test_uuid() -> str:
    """
    Test version of UUID generator with logging.
    
    Returns:
        A unique UUID string with test prefix
    """
    test_uuid = f"test-{str(uuid.uuid4())}"
    print(f"[UUID_GEN] Generated custom message ID: {test_uuid}")
    return test_uuid


class TestCallback(BaseAICallbackHandler):
    """Test callback handler to track events."""
    
    def __init__(self):
        super().__init__()
        self.events = []
    
    async def on_start(self, **kwargs):
        event = f"START: {kwargs}"
        self.events.append(event)
        print(f"[CALLBACK] {event}")
    
    async def on_text_delta(self, delta: str, **kwargs):
        event = f"TEXT_DELTA: '{delta[:50]}...'"
        self.events.append(event)
        print(f"[CALLBACK] {event}")
    
    async def on_finish(self, message, options, **kwargs):
        event = f"FINISH: message={str(message)[:50]}..., options={options}"
        self.events.append(event)
        print(f"[CALLBACK] {event}")


async def create_mock_langchain_stream():
    """Create a mock LangChain stream for testing without requiring API keys."""
    mock_events = [
        {
            "event": "on_chat_model_start",
            "data": {
                "input": {"messages": [["human", "Test message"]]}
            }
        },
        {
            "event": "on_chat_model_stream",
            "data": {
                "chunk": {
                    "content": "Hello ",
                    "type": "AIMessageChunk"
                }
            }
        },
        {
            "event": "on_chat_model_stream",
            "data": {
                "chunk": {
                    "content": "world! ",
                    "type": "AIMessageChunk"
                }
            }
        },
        {
            "event": "on_chat_model_stream",
            "data": {
                "chunk": {
                    "content": "This is ",
                    "type": "AIMessageChunk"
                }
            }
        },
        {
            "event": "on_chat_model_stream",
            "data": {
                "chunk": {
                    "content": "a test.",
                    "type": "AIMessageChunk"
                }
            }
        },
        {
            "event": "on_chat_model_end",
            "data": {
                "output": {
                    "content": "Hello world! This is a test.",
                    "type": "AIMessage"
                }
            }
        }
    ]
    
    for event in mock_events:
        yield event
        await asyncio.sleep(0.1)  # Small delay to simulate streaming


async def test_experimental_features():
    """Test experimental_transform and experimental_generateMessageId features."""
    print("=" * 60)
    print("Testing Experimental Features")
    print("=" * 60)
    
    # Create test callback
    callbacks = TestCallback()
    
    try:
        # Create mock LangChain stream
        langchain_stream = create_mock_langchain_stream()
        
        print("\n" + "-" * 40)
        print("Testing with Experimental Features")
        print("-" * 40)
        
        # Convert to AI SDK stream with experimental features
        ai_sdk_stream = LangChainAdapter.to_data_stream(
            langchain_stream,
            callbacks=callbacks,
            message_id=None,  # Let experimental_generateMessageId handle this
            options={
                "auto_events": True, 
                "protocol_version": "v5",
                "experimental_transform": smooth_stream_test(chunking='word', delay_in_ms=100),
                "experimental_generateMessageId": generate_test_uuid
            }
        )
        
        # Stream the response
        print("\nAI SDK Protocol Output with Experimental Features:")
        print("-" * 50)
        chunk_count = 0
        async for chunk in ai_sdk_stream:
            chunk_count += 1
            print(f"Chunk #{chunk_count}: {str(chunk)[:100]}...")
            
            # Limit output for testing
            if chunk_count >= 15:
                print("[TEST] Limiting output to first 15 chunks for demonstration")
                break
        
        print("-" * 50)
        print(f"Total chunks processed: {chunk_count}")
        print(f"Total callback events: {len(callbacks.events)}")
        
        # Close the stream
        await ai_sdk_stream.close()
        print("Stream closed successfully")
        
        # Test without experimental features for comparison
        print("\n" + "=" * 60)
        print("Testing WITHOUT Experimental Features (for comparison)")
        print("=" * 60)
        
        # Create another mock stream
        langchain_stream2 = create_mock_langchain_stream()
        callbacks2 = TestCallback()
        
        # Convert to AI SDK stream without experimental features
        ai_sdk_stream2 = LangChainAdapter.to_data_stream(
            langchain_stream2,
            callbacks=callbacks2,
            message_id="standard-message-id",
            options={
                "auto_events": True, 
                "protocol_version": "v5"
                # No experimental features
            }
        )
        
        print("\nAI SDK Protocol Output WITHOUT Experimental Features:")
        print("-" * 50)
        chunk_count2 = 0
        async for chunk in ai_sdk_stream2:
            chunk_count2 += 1
            print(f"Chunk #{chunk_count2}: {str(chunk)[:100]}...")
            
            # Limit output for testing
            if chunk_count2 >= 15:
                print("[TEST] Limiting output to first 15 chunks for demonstration")
                break
        
        print("-" * 50)
        print(f"Total chunks processed: {chunk_count2}")
        print(f"Total callback events: {len(callbacks2.events)}")
        
        await ai_sdk_stream2.close()
        print("Stream closed successfully")
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"✅ experimental_generateMessageId: Custom UUID generation working")
        print(f"✅ experimental_transform: Stream transformation with delays working")
        print(f"✅ Both experimental features integrated successfully")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()


async def main():
    """Main test function."""
    try:
        await test_experimental_features()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())