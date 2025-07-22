#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Async vs Sync Comparison Example

This example demonstrates why the adapter methods need to be async
and what problems would occur with sync methods.
"""

import asyncio
import time
from typing import AsyncGenerator, Generator
from langchain_aisdk_adapter import LangChainAdapter


class SyncAdapter:
    """Hypothetical sync version of the adapter"""
    
    def __init__(self):
        self.state = {"parts": []}
    
    def text(self, text: str) -> str:
        """Sync version - blocks the thread"""
        # Simulate some processing time
        time.sleep(0.01)  # 10ms delay
        self.state["parts"].append({"type": "text", "content": text})
        return f'0:"{text}"'
    
    def source(self, url: str, title: str = None) -> str:
        """Sync version - blocks the thread"""
        time.sleep(0.01)  # 10ms delay
        self.state["parts"].append({"type": "source", "url": url, "title": title})
        return f'h:{{"url":"{url}","title":"{title}"}}'


class AsyncAdapter:
    """Async version of the adapter"""
    
    def __init__(self):
        self.state = {"parts": []}
    
    async def text(self, text: str) -> str:
        """Async version - doesn't block the event loop"""
        # Simulate some async processing (e.g., callback, validation)
        await asyncio.sleep(0.01)  # 10ms delay
        self.state["parts"].append({"type": "text", "content": text})
        return f'0:"{text}"'
    
    async def source(self, url: str, title: str = None) -> str:
        """Async version - doesn't block the event loop"""
        await asyncio.sleep(0.01)  # 10ms delay
        self.state["parts"].append({"type": "source", "url": url, "title": title})
        return f'h:{{"url":"{url}","title":"{title}"}}'


def sync_stream_simulation() -> Generator[str, None, None]:
    """Simulate a sync stream that would block"""
    adapter = SyncAdapter()
    
    print("\n=== Sync Stream Simulation ===")
    start_time = time.time()
    
    # Each call blocks for 10ms
    yield adapter.text("Hello ")
    yield adapter.text("world! ")
    yield adapter.source("https://example.com", "Example")
    yield adapter.text("Done.")
    
    end_time = time.time()
    print(f"Sync stream took: {(end_time - start_time)*1000:.1f}ms")
    print(f"State: {adapter.state}")


async def async_stream_simulation() -> AsyncGenerator[str, None]:
    """Simulate an async stream that doesn't block"""
    adapter = AsyncAdapter()
    
    print("\n=== Async Stream Simulation ===")
    start_time = time.time()
    
    # Each call is async and can be interleaved
    yield await adapter.text("Hello ")
    yield await adapter.text("world! ")
    yield await adapter.source("https://example.com", "Example")
    yield await adapter.text("Done.")
    
    end_time = time.time()
    print(f"Async stream took: {(end_time - start_time)*1000:.1f}ms")
    print(f"State: {adapter.state}")


async def concurrent_streams_demo():
    """Demonstrate why async is important for concurrent streams"""
    print("\n=== Concurrent Streams Demo ===")
    print("Processing 3 streams concurrently...")
    
    async def create_stream(stream_id: int) -> AsyncGenerator[str, None]:
        adapter = AsyncAdapter()
        yield await adapter.text(f"Stream {stream_id}: Start ")
        yield await adapter.text(f"processing ")
        yield await adapter.text(f"complete.\n")
    
    start_time = time.time()
    
    # Process 3 streams concurrently
    tasks = []
    for i in range(3):
        async def process_stream(stream_id=i):
            async for chunk in create_stream(stream_id):
                print(f"Stream {stream_id}: {chunk.strip()}", flush=True)
        
        tasks.append(process_stream())
    
    await asyncio.gather(*tasks)
    
    end_time = time.time()
    print(f"\nConcurrent processing took: {(end_time - start_time)*1000:.1f}ms")
    print("(If this was sync, it would take ~90ms instead of ~30ms)")


async def order_preservation_demo():
    """Demonstrate that async doesn't affect order within a single stream"""
    print("\n=== Order Preservation Demo ===")
    
    adapter = AsyncAdapter()
    
    # Even though these are async, they're awaited in sequence
    # so order is preserved within the same stream
    results = []
    results.append(await adapter.text("First "))
    results.append(await adapter.text("Second "))
    results.append(await adapter.text("Third"))
    
    print("Results in order:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result}")
    
    print("\nState parts in order:")
    for i, part in enumerate(adapter.state["parts"], 1):
        print(f"  {i}. {part}")


async def callback_simulation_demo():
    """Demonstrate why callbacks need async"""
    print("\n=== Callback Simulation Demo ===")
    
    class AsyncCallback:
        async def on_text_added(self, text: str):
            # Simulate async callback (e.g., logging to database, sending webhook)
            await asyncio.sleep(0.005)  # 5ms async operation
            print(f"  📝 Callback: Text '{text}' was added")
        
        async def on_source_added(self, url: str):
            await asyncio.sleep(0.005)  # 5ms async operation
            print(f"  🔗 Callback: Source '{url}' was added")
    
    class AsyncAdapterWithCallbacks(AsyncAdapter):
        def __init__(self, callback):
            super().__init__()
            self.callback = callback
        
        async def text(self, text: str) -> str:
            result = await super().text(text)
            # Call async callback
            await self.callback.on_text_added(text)
            return result
        
        async def source(self, url: str, title: str = None) -> str:
            result = await super().source(url, title)
            # Call async callback
            await self.callback.on_source_added(url)
            return result
    
    callback = AsyncCallback()
    adapter = AsyncAdapterWithCallbacks(callback)
    
    print("Processing with async callbacks:")
    
    result1 = await adapter.text("Hello world!")
    print(f"Result: {result1}")
    
    result2 = await adapter.source("https://example.com", "Example")
    print(f"Result: {result2}")
    
    print("\nIf callbacks were sync, they would block the entire stream!")


async def main():
    """Run all demonstrations"""
    print("=== Why Adapter Methods Need to be Async ===")
    print("\nThis demo shows why the LangChainAdapter methods are designed as async:")
    print("1. Non-blocking operation in event loops")
    print("2. Support for concurrent stream processing")
    print("3. Async callback support")
    print("4. Order preservation within streams")
    
    # Run sync simulation (this would block in real async context)
    print("\n" + "="*60)
    for chunk in sync_stream_simulation():
        print(f"Sync chunk: {chunk.strip()}")
    
    # Run async simulation
    print("\n" + "="*60)
    async for chunk in async_stream_simulation():
        print(f"Async chunk: {chunk.strip()}")
    
    # Demonstrate concurrent processing
    print("\n" + "="*60)
    await concurrent_streams_demo()
    
    # Demonstrate order preservation
    print("\n" + "="*60)
    await order_preservation_demo()
    
    # Demonstrate callback requirements
    print("\n" + "="*60)
    await callback_simulation_demo()
    
    print("\n" + "="*60)
    print("\n🎯 Key Takeaways:")
    print("1. Async methods don't block the event loop")
    print("2. Multiple streams can be processed concurrently")
    print("3. Order is preserved within each individual stream")
    print("4. Async callbacks can perform I/O without blocking")
    print("5. The 'async' keyword is about concurrency, not speed")
    print("\n💡 The methods are async not because they're slow,")
    print("   but because they need to work in async environments!")


if __name__ == "__main__":
    asyncio.run(main())