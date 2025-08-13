#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for smooth_stream functionality

This script demonstrates and tests the LangChainAdapter.smooth_stream method
with different chunking strategies and delay settings.
"""

import asyncio
import re
import time
from typing import AsyncIterable

from langchain_aisdk_adapter import LangChainAdapter


async def create_test_stream(text: str, chunk_size: int = 5) -> AsyncIterable[str]:
    """Create a test stream that yields text in chunks."""
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        yield chunk
        await asyncio.sleep(0.01)  # Small delay to simulate real streaming


async def test_word_chunking():
    """Test smooth_stream with word-based chunking."""
    print("\n=== Testing Word Chunking ===")
    
    text = "Hello world! This is a test of smooth streaming with word-based chunking."
    stream = create_test_stream(text, chunk_size=3)
    
    smooth_stream = LangChainAdapter.smooth_stream(
        stream,
        delay_in_ms=50,  # 50ms delay for visible effect
        chunking='word'
    )
    
    print("Original text:", repr(text))
    print("Smooth stream output:")
    
    start_time = time.time()
    result = ""
    
    async for chunk in smooth_stream:
        result += chunk
        print(f"[{time.time() - start_time:.2f}s] Chunk: {repr(chunk)}")
    
    print(f"\nFinal result: {repr(result)}")
    print(f"Match original: {result == text}")


async def test_line_chunking():
    """Test smooth_stream with line-based chunking."""
    print("\n=== Testing Line Chunking ===")
    
    text = "Line 1\nLine 2\nLine 3\nFinal line"
    stream = create_test_stream(text, chunk_size=4)
    
    smooth_stream = LangChainAdapter.smooth_stream(
        stream,
        delay_in_ms=30,
        chunking='line'
    )
    
    print("Original text:", repr(text))
    print("Smooth stream output:")
    
    start_time = time.time()
    result = ""
    
    async for chunk in smooth_stream:
        result += chunk
        print(f"[{time.time() - start_time:.2f}s] Chunk: {repr(chunk)}")
    
    print(f"\nFinal result: {repr(result)}")
    print(f"Match original: {result == text}")


async def test_regex_chunking():
    """Test smooth_stream with regex-based chunking."""
    print("\n=== Testing Regex Chunking (Chinese) ===")
    
    text = "Hello 你好 world 世界! This is 这是 a test 测试."
    stream = create_test_stream(text, chunk_size=6)
    
    # Pattern to separate Chinese and non-Chinese characters
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+|[a-zA-Z]+\s*|\S\s*')
    
    smooth_stream = LangChainAdapter.smooth_stream(
        stream,
        delay_in_ms=40,
        chunking=chinese_pattern
    )
    
    print("Original text:", repr(text))
    print("Smooth stream output:")
    
    start_time = time.time()
    result = ""
    
    async for chunk in smooth_stream:
        result += chunk
        print(f"[{time.time() - start_time:.2f}s] Chunk: {repr(chunk)}")
    
    print(f"\nFinal result: {repr(result)}")
    print(f"Match original: {result == text}")


async def test_custom_chunking():
    """Test smooth_stream with custom chunking function."""
    print("\n=== Testing Custom Chunking ===")
    
    text = "apple,banana,cherry,date,elderberry"
    stream = create_test_stream(text, chunk_size=7)
    
    def comma_chunker(text: str) -> list[str]:
        """Custom chunker that splits by commas."""
        parts = text.split(',')
        result = []
        for i, part in enumerate(parts[:-1]):
            result.append(part + ',')
        if parts[-1]:  # Add last part if not empty
            result.append(parts[-1])
        return result
    
    smooth_stream = LangChainAdapter.smooth_stream(
        stream,
        delay_in_ms=60,
        chunking=comma_chunker
    )
    
    print("Original text:", repr(text))
    print("Smooth stream output:")
    
    start_time = time.time()
    result = ""
    
    async for chunk in smooth_stream:
        result += chunk
        print(f"[{time.time() - start_time:.2f}s] Chunk: {repr(chunk)}")
    
    print(f"\nFinal result: {repr(result)}")
    print(f"Match original: {result == text}")


async def test_no_delay():
    """Test smooth_stream with no delay."""
    print("\n=== Testing No Delay ===")
    
    text = "Fast streaming without delays!"
    stream = create_test_stream(text, chunk_size=4)
    
    smooth_stream = LangChainAdapter.smooth_stream(
        stream,
        delay_in_ms=0,  # No delay
        chunking='word'
    )
    
    print("Original text:", repr(text))
    print("Smooth stream output:")
    
    start_time = time.time()
    result = ""
    
    async for chunk in smooth_stream:
        result += chunk
        print(f"[{time.time() - start_time:.3f}s] Chunk: {repr(chunk)}")
    
    print(f"\nFinal result: {repr(result)}")
    print(f"Match original: {result == text}")
    print(f"Total time: {time.time() - start_time:.3f}s")


async def test_empty_and_edge_cases():
    """Test smooth_stream with edge cases."""
    print("\n=== Testing Edge Cases ===")
    
    # Test empty stream
    async def empty_stream():
        return
        yield  # This line is never reached
    
    print("Testing empty stream...")
    smooth_stream = LangChainAdapter.smooth_stream(empty_stream(), delay_in_ms=10)
    result = ""
    async for chunk in smooth_stream:
        result += chunk
    print(f"Empty stream result: {repr(result)}")
    
    # Test stream with empty chunks
    async def sparse_stream():
        yield "Hello"
        yield ""
        yield " "
        yield "world"
        yield ""
        yield "!"
    
    print("\nTesting sparse stream...")
    smooth_stream = LangChainAdapter.smooth_stream(sparse_stream(), delay_in_ms=20)
    result = ""
    async for chunk in smooth_stream:
        result += chunk
        print(f"Chunk: {repr(chunk)}")
    print(f"Sparse stream result: {repr(result)}")


async def main():
    """Run all smooth_stream tests."""
    print("Starting LangChainAdapter.smooth_stream tests...")
    
    try:
        await test_word_chunking()
        await test_line_chunking()
        await test_regex_chunking()
        await test_custom_chunking()
        await test_no_delay()
        await test_empty_and_edge_cases()
        
        print("\n=== All Tests Completed Successfully! ===")
        
    except Exception as e:
        print(f"\n=== Test Failed with Error: {e} ===")
        raise


if __name__ == "__main__":
    asyncio.run(main())