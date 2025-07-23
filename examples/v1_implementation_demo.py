#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V1 Implementation Demo

This example demonstrates the new V1 implementation features:
1. LangChainAdapter class with three core methods
2. DataStreamResponse for FastAPI integration
3. ManualStreamController for manual stream control
4. Complete UIMessageChunk protocol support
"""

import asyncio
import uuid
from typing import Dict, Any

from langchain_aisdk_adapter import (
    LangChainAdapter,
    DataStreamResponse,
    DataStreamWriter,
    ManualStreamController
)
from langchain_aisdk_adapter.callbacks import StreamCallbacks


async def demo_langchain_adapter():
    """Demonstrate LangChainAdapter usage."""
    print("\n=== LangChainAdapter Demo ===")
    
    # Mock LangChain stream
    async def mock_langchain_stream():
        events = [
            {"event": "on_chain_start", "data": {"input": "Hello"}},
            {"event": "on_llm_stream", "data": {"chunk": {"content": "Hello"}}},
            {"event": "on_llm_stream", "data": {"chunk": {"content": " world"}}},
            {"event": "on_llm_end", "data": {"output": {"content": "Hello world"}}},
            {"event": "on_chain_end", "data": {"output": "Hello world"}}
        ]
        for event in events:
            yield event
            await asyncio.sleep(0.1)
    
    # Create callbacks
    callbacks = StreamCallbacks(
        on_start=lambda: print("Stream started"),
        on_final=lambda text: print(f"Final text: {text}")
    )
    
    # Method 1: to_data_stream
    print("\n1. Using to_data_stream:")
    stream = LangChainAdapter.to_data_stream(
        mock_langchain_stream(),
        callbacks=callbacks,
        message_id="demo-001",
        options={"auto_events": True}
    )
    
    async for chunk in stream:
        print(f"  Chunk: {chunk}")
    
    # Method 2: to_data_stream_response
    print("\n2. Using to_data_stream_response:")
    response = await LangChainAdapter.to_data_stream_response(
        mock_langchain_stream(),
        callbacks=callbacks,
        message_id="demo-002"
    )
    print(f"  Response type: {type(response)}")
    print(f"  Response headers: {response.headers}")


async def demo_manual_stream_controller():
    """Demonstrate ManualStreamController usage."""
    print("\n=== ManualStreamController Demo ===")
    
    controller = ManualStreamController()
    
    # Start a background task to emit events
    async def emit_events():
        await controller.emit_start("manual-demo-001")
        await controller.emit_text_start("text-001", "manual-demo-001")
        
        # Simulate streaming text
        words = ["Hello", " ", "from", " ", "manual", " ", "controller!"]
        for word in words:
            await controller.emit_text_delta(word, "text-001", "manual-demo-001")
            await asyncio.sleep(0.1)
        
        await controller.emit_text_end("Hello from manual controller!", "text-001", "manual-demo-001")
        await controller.emit_finish("manual-demo-001")
        await controller.close()
    
    # Start emitting events
    emit_task = asyncio.create_task(emit_events())
    
    # Consume the stream
    async for chunk in controller:
        print(f"  Manual chunk: {chunk}")
    
    await emit_task


async def demo_data_stream_writer():
    """Demonstrate DataStreamWriter usage."""
    print("\n=== DataStreamWriter Demo ===")
    
    writer = DataStreamWriter()
    
    # Write some chunks
    async def write_chunks():
        await writer.write({"type": "start", "messageId": "writer-demo-001"})
        await writer.write({"type": "text-start", "id": "text-001", "messageId": "writer-demo-001"})
        await writer.write({"type": "text-delta", "id": "text-001", "delta": "Hello", "messageId": "writer-demo-001"})
        await writer.write({"type": "text-delta", "id": "text-001", "delta": " writer!", "messageId": "writer-demo-001"})
        await writer.write({"type": "text-end", "id": "text-001", "messageId": "writer-demo-001"})
        await writer.write({"type": "finish", "messageId": "writer-demo-001"})
        await writer.close()
    
    # Start writing
    write_task = asyncio.create_task(write_chunks())
    
    # Read from writer
    async for chunk in writer:
        print(f"  Writer chunk: {chunk}")
    
    await write_task


async def demo_complete_workflow():
    """Demonstrate a complete workflow combining all features."""
    print("\n=== Complete Workflow Demo ===")
    
    # Create a manual controller for custom events
    manual_controller = ManualStreamController()
    
    # Create a data stream writer for merging
    writer = DataStreamWriter()
    
    async def emit_custom_events():
        """Emit custom events using manual controller."""
        await manual_controller.emit_start("workflow-001")
        await manual_controller.emit_reasoning_start("reasoning-001", "workflow-001")
        await manual_controller.emit_reasoning_delta("Thinking about the problem...", "reasoning-001", "workflow-001")
        await manual_controller.emit_reasoning_end("reasoning-001", "workflow-001")
        await manual_controller.emit_text_start("text-001", "workflow-001")
        await manual_controller.emit_text_delta("Based on my reasoning, ", "text-001", "workflow-001")
        await manual_controller.emit_text_delta("the answer is 42.", "text-001", "workflow-001")
        await manual_controller.emit_text_end("Based on my reasoning, the answer is 42.", "text-001", "workflow-001")
        await manual_controller.emit_finish("workflow-001")
        await manual_controller.close()
    
    async def merge_streams():
        """Merge manual controller stream into writer."""
        async for chunk in manual_controller:
            await writer.write(chunk)
        await writer.close()
    
    # Start both tasks
    emit_task = asyncio.create_task(emit_custom_events())
    merge_task = asyncio.create_task(merge_streams())
    
    # Consume the merged stream
    async for chunk in writer:
        print(f"  Workflow chunk: {chunk}")
    
    await emit_task
    await merge_task


async def main():
    """Run all demos."""
    print("LangChain AI SDK Adapter - V1 Implementation Demo")
    print("=" * 50)
    
    try:
        await demo_langchain_adapter()
        await demo_manual_stream_controller()
        await demo_data_stream_writer()
        await demo_complete_workflow()
        
        print("\n=== Demo Complete ===")
        print("V1 Implementation features:")
        print("✓ LangChainAdapter with three core methods")
        print("✓ DataStreamResponse for FastAPI integration")
        print("✓ ManualStreamController for manual control")
        print("✓ DataStreamWriter for stream merging")
        print("✓ Complete UIMessageChunk protocol support")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())