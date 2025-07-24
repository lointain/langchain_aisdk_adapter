#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify web API protocol version support
"""

import asyncio
import json
import aiohttp

async def test_api_protocol_versions():
    """Test API with different protocol versions"""
    base_url = "http://localhost:8000"
    
    # Test data
    test_request = {
        "messages": [
            {
                "role": "user",
                "content": "Hello, how are you?"
            }
        ],
        "message_id": "test-api-001",
        "stream_mode": "auto"
    }
    
    async with aiohttp.ClientSession() as session:
        # Test v4 protocol
        print("\n=== Testing v4 Protocol ===")
        v4_request = {**test_request, "protocol_version": "v4"}
        
        try:
            async with session.post(
                f"{base_url}/api/chat/auto",
                json=v4_request,
                headers={"Content-Type": "application/json"}
            ) as response:
                print(f"Status: {response.status}")
                print("Response chunks:")
                
                chunk_count = 0
                async for chunk in response.content.iter_chunked(1024):
                    chunk_text = chunk.decode('utf-8').strip()
                    if chunk_text:
                        print(f"Chunk {chunk_count}: {chunk_text}")
                        chunk_count += 1
                        if chunk_count > 10:  # Limit output
                            print("... (truncated)")
                            break
                            
        except Exception as e:
            print(f"Error testing v4: {e}")
        
        # Test v5 protocol
        print("\n=== Testing v5 Protocol ===")
        v5_request = {**test_request, "protocol_version": "v5"}
        
        try:
            async with session.post(
                f"{base_url}/api/chat/auto",
                json=v5_request,
                headers={"Content-Type": "application/json"}
            ) as response:
                print(f"Status: {response.status}")
                print("Response chunks:")
                
                chunk_count = 0
                async for chunk in response.content.iter_chunked(1024):
                    chunk_text = chunk.decode('utf-8').strip()
                    if chunk_text:
                        print(f"Chunk {chunk_count}: {chunk_text}")
                        chunk_count += 1
                        if chunk_count > 10:  # Limit output
                            print("... (truncated)")
                            break
                            
        except Exception as e:
            print(f"Error testing v5: {e}")

if __name__ == "__main__":
    asyncio.run(test_api_protocol_versions())