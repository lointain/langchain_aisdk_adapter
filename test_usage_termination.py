#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify usage information in termination marker
"""

import asyncio
import os
from langchain_openai import ChatOpenAI
from langchain_aisdk_adapter import LangChainAdapter

async def test_usage_termination():
    """Test usage information in termination marker"""
    # Temporarily disable LangSmith tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    # Check for API key
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("Error: DEEPSEEK_API_KEY environment variable not set")
        return
    
    # Initialize the language model
    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com",
        temperature=0.1,
        streaming=True
    )
    
    # Simple query
    test_query = "Hello, how are you?"
    
    try:
        # Create stream using astream_events to get usage information
        langchain_stream = llm.astream_events([{"role": "user", "content": test_query}], version="v2")
        
        # Convert to AI SDK stream with v4 protocol
        ai_sdk_stream = LangChainAdapter.to_data_stream(
            langchain_stream,
            message_id="test-usage-001",
            options={"protocol_version": "v4", "output_format": "protocol"}
        )
        
        print("\nAI SDK v4 Protocol Output:")
        print("-" * 50)
        
        # Stream the response
        async for chunk in ai_sdk_stream:
            print(f"Protocol: {chunk.strip()}")
        
        print("-" * 50)
        
    except Exception as e:
        print(f"Error during streaming: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_usage_termination())