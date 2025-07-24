#!/usr/bin/env python3
"""Test protocol formatting functionality."""

import asyncio
import os
from typing import AsyncGenerator

# Add the src directory to the path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_aisdk_adapter import LangChainAdapter

# Mock tools for testing
@tool
def get_weather(input: str) -> str:
    """Get weather information for a location."""
    return f"The weather in {input} is sunny with 22°C temperature."

@tool
def calculate_math(input: str) -> str:
    """Calculate mathematical expressions."""
    try:
        result = eval(input)
        return f"The result of {input} is {result}"
    except Exception as e:
        return f"Error calculating {input}: {str(e)}"

async def test_protocol_formatting():
    """Test protocol formatting with different versions."""
    
    # Initialize the model
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        api_key="fake-key",  # Using fake key for testing
        base_url="https://api.openai.com/v1"
    )
    
    # Create agent with tools
    tools = [get_weather, calculate_math]
    agent = create_react_agent(model, tools)
    
    # Test message
    message = HumanMessage(content="What's the weather in Beijing and what is 15*24?")
    
    print("=== Testing V4 Protocol Formatting ===")
    
    try:
        # Test V4 protocol formatting
        stream_v4 = agent.astream_events(
            {"messages": [message]},
            version="v2"
        )
        
        # Convert to protocol-formatted text stream (V4)
        text_stream_v4 = LangChainAdapter.to_data_stream_text(
            stream_v4,
            options={"protocol_version": "v4"}
        )
        
        print("V4 Protocol Output:")
        async for chunk in text_stream_v4:
            print(repr(chunk))
            
    except Exception as e:
        print(f"V4 test failed: {e}")
    
    print("\n=== Testing V5 Protocol Formatting ===")
    
    try:
        # Test V5 protocol formatting
        stream_v5 = agent.astream_events(
            {"messages": [message]},
            version="v2"
        )
        
        # Convert to protocol-formatted text stream (V5)
        text_stream_v5 = LangChainAdapter.to_data_stream_text(
            stream_v5,
            options={"protocol_version": "v5"}
        )
        
        print("V5 Protocol Output:")
        async for chunk in text_stream_v5:
            print(repr(chunk))
            
    except Exception as e:
        print(f"V5 test failed: {e}")
    
    print("\n=== Testing DataStreamWithEmitters (Raw Chunks) ===")
    
    try:
        # Test raw DataStreamWithEmitters output
        stream_raw = agent.astream_events(
            {"messages": [message]},
            version="v2"
        )
        
        # Convert to DataStreamWithEmitters (raw chunks)
        data_stream = LangChainAdapter.to_data_stream(
            stream_raw,
            options={"protocol_version": "v4"}
        )
        
        print("Raw DataStreamWithEmitters Output:")
        chunk_count = 0
        async for chunk in data_stream:
            chunk_count += 1
            if chunk_count <= 5:  # Only show first 5 chunks
                print(f"Chunk {chunk_count}: {chunk}")
            elif chunk_count == 6:
                print("... (more chunks)")
        
        print(f"Total chunks: {chunk_count}")
        
    except Exception as e:
        print(f"Raw chunks test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_protocol_formatting())