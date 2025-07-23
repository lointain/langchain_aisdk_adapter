#!/usr/bin/env python3
"""Test auto_close configuration in DataStreamWithEmitters."""

import asyncio
from langchain_aisdk_adapter import LangChainAdapter
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


async def test_auto_close_true():
    """Test with auto_close=True (default behavior)."""
    print("Testing auto_close=True (default)...")
    
    # Create a simple LangChain model
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Create stream with auto_close=True (default)
    stream = model.astream([HumanMessage(content="Say hello")])
    
    # Convert to data stream with auto_close=True
    data_stream = LangChainAdapter.to_data_stream(
        stream, 
        options={"auto_close": True}
    )
    
    chunk_count = 0
    async for chunk in data_stream:
        chunk_count += 1
        chunk_type = chunk.get('type') if isinstance(chunk, dict) else getattr(chunk, 'type', 'unknown')
        print(f"Chunk {chunk_count}: {chunk_type}")
        
        # Try to emit manual event during stream
        if chunk_count == 2:
            await data_stream.emit_data({"test": "manual_event"})
            print("Manual event emitted")
    
    print(f"Stream ended automatically. Total chunks: {chunk_count}")
    print("✅ auto_close=True test completed\n")


async def test_auto_close_false():
    """Test with auto_close=False (manual close required)."""
    print("Testing auto_close=False...")
    
    # Create a simple LangChain model
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Create stream with auto_close=False
    stream = model.astream([HumanMessage(content="Say hello")])
    
    # Convert to data stream with auto_close=False
    data_stream = LangChainAdapter.to_data_stream(
        stream, 
        options={"auto_close": False}
    )
    
    chunk_count = 0
    manual_close_task = None
    
    async def manual_close_after_delay():
        """Close stream manually after a delay."""
        await asyncio.sleep(2)  # Wait 2 seconds
        await data_stream.close()
        print("Stream closed manually")
    
    # Start manual close task
    manual_close_task = asyncio.create_task(manual_close_after_delay())
    
    try:
        async for chunk in data_stream:
            chunk_count += 1
            chunk_type = chunk.get('type') if isinstance(chunk, dict) else getattr(chunk, 'type', 'unknown')
            print(f"Chunk {chunk_count}: {chunk_type}")
            
            # Try to emit manual event during stream
            if chunk_count == 2:
                await data_stream.emit_data({"test": "manual_event"})
                print("Manual event emitted")
    except Exception as e:
        print(f"Stream iteration ended: {e}")
    
    # Wait for manual close task to complete
    if manual_close_task:
        await manual_close_task
    
    print(f"Stream ended with manual close. Total chunks: {chunk_count}")
    print("✅ auto_close=False test completed\n")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing DataStreamWithEmitters auto_close configuration")
    print("=" * 60)
    
    try:
        await test_auto_close_true()
        await test_auto_close_false()
        
        print("=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())