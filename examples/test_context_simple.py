#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test for DataStreamContext functionality without requiring API keys
"""

import asyncio
from langchain_aisdk_adapter import LangChainAdapter, DataStreamContext
from langchain_aisdk_adapter.callbacks import BaseAICallbackHandler, Message


class TestCallbackHandler(BaseAICallbackHandler):
    """Test callback handler for demonstrating stream processing"""
    
    async def on_start(self) -> None:
        """Called when the stream starts"""
        print("\n" + "="*50)
        print("STREAM STARTED")
        print("="*50)
    
    async def on_finish(self, message: Message, options: dict) -> None:
        """Called when the stream finishes"""
        print("\n" + "="*50)
        print("STREAM COMPLETED")
        print("="*50)
        print(f"\nFinal message object:")
        print(f"Message ID: {message.id}")
        print(f"Message Role: {message.role}")
        print(f"Message Content: {message.content}")
        print(f"Message Parts: {len(message.parts)} parts")
        print(f"Options: {options}")
        print("\n--- End of Stream ---")


async def test_context_functionality():
    """Test DataStreamContext functionality with mock stream"""
    
    # Create a simple mock stream
    async def mock_stream():
        yield {"event": "on_chat_model_stream", "data": {"chunk": {"content": "Hello"}}}
        yield {"event": "on_chat_model_stream", "data": {"chunk": {"content": " world!"}}}
    
    # Initialize callback handler
    callbacks = TestCallbackHandler()
    
    try:
        # Convert to AI SDK stream using LangChainAdapter.to_data_stream
        ai_sdk_stream = LangChainAdapter.to_data_stream(
            mock_stream(),
            callbacks=callbacks,
            message_id="test-message-001",
            options={"auto_context": True}  # Enable auto context
        )
        
        # Test DataStreamContext emit methods
        print("\nTesting DataStreamContext emit methods:")
        
        # Get current context
        context = DataStreamContext.get_current_stream()
        if context:
            print("✓ DataStreamContext is available")
            
            # Test various emit methods
            await context.emit_start()
            print("✓ Emitted start")
            
            # Text sequence
            text_id = "text-001"
            await context.emit_text_start(text_id=text_id)
            print("✓ Emitted text-start")
            
            await context.emit_text_delta(delta="Hello ", text_id=text_id)
            await context.emit_text_delta(delta="world!", text_id=text_id)
            print("✓ Emitted text-delta chunks")
            
            await context.emit_text_end(text="Hello world!", text_id=text_id)
            print("✓ Emitted text-end")
            
            # Custom data
            await context.emit_data(data={"key": "value", "number": 42})
            print("✓ Emitted data chunk")
            
            # Finish
            await context.emit_finish()
            print("✓ Emitted finish")
            
            print("All DataStreamContext emit tests completed.\n")
        else:
            print("❌ DataStreamContext is not available")
        
        # Stream the response
        print("\nAI SDK Protocol Output:")
        print("-" * 40)
        chunk_count = 0
        async for chunk in ai_sdk_stream:
            print(f"Protocol: {chunk}")
            chunk_count += 1
            if chunk_count >= 10:  # Limit output
                break
        print("-" * 40)
        
        # Explicitly close the stream
        await ai_sdk_stream.close()
        print("Stream closed explicitly")
        
    except Exception as e:
        print(f"Error during streaming: {str(e)}")
        import traceback
        traceback.print_exc()


async def main():
    try:
        await test_context_functionality()
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())