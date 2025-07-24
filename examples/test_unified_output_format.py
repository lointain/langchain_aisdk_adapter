#!/usr/bin/env python3
"""Test unified output format functionality."""

import asyncio
import os

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

async def test_unified_output_format():
    """Test unified output format functionality."""
    
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
    message = HumanMessage(content="What's the weather in Beijing?")
    
    print("=== Testing DataStreamWithEmitters with chunks output (default) ===")
    
    try:
        # Test default chunks output
        stream1 = agent.astream_events(
            {"messages": [message]},
            version="v2"
        )
        
        # Convert to DataStreamWithEmitters with default output_format="chunks"
        data_stream_chunks = LangChainAdapter.to_data_stream(
            stream1,
            options={"protocol_version": "v4"}
        )
        
        print("Testing emit methods with chunks output:")
        
        # Test emit methods
        await data_stream_chunks.emit_start()
        await data_stream_chunks.emit_text_start()
        await data_stream_chunks.emit_text_delta("Hello from emit!")
        await data_stream_chunks.emit_text_end("Complete text")
        await data_stream_chunks.emit_finish()
        
        print("✓ All emit methods work with chunks output")
        
        print("\nProcessing chunks:")
        chunk_count = 0
        async for chunk in data_stream_chunks:
            chunk_count += 1
            if chunk_count <= 5:  # Show first 5 chunks
                print(f"Chunk {chunk_count}: {type(chunk)} - {chunk}")
            elif chunk_count == 6:
                print("... (more chunks)")
        
        print(f"Total chunks: {chunk_count}")
        
    except Exception as e:
        print(f"Chunks output test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Testing DataStreamWithEmitters with protocol output ===")
    
    try:
        # Test protocol output
        stream2 = agent.astream_events(
            {"messages": [message]},
            version="v2"
        )
        
        # Convert to DataStreamWithEmitters with output_format="protocol"
        data_stream_protocol = LangChainAdapter.to_data_stream(
            stream2,
            options={
                "protocol_version": "v4",
                "output_format": "protocol"
            }
        )
        
        print("Testing emit methods with protocol output:")
        
        # Test emit methods
        await data_stream_protocol.emit_start()
        await data_stream_protocol.emit_text_start()
        await data_stream_protocol.emit_text_delta("Hello from protocol emit!")
        await data_stream_protocol.emit_text_end("Complete protocol text")
        await data_stream_protocol.emit_finish()
        
        print("✓ All emit methods work with protocol output")
        
        print("\nProcessing protocol text:")
        text_count = 0
        async for text_chunk in data_stream_protocol:
            text_count += 1
            if text_count <= 10:  # Show first 10 text chunks
                print(f"Text {text_count}: {repr(text_chunk)}")
            elif text_count == 11:
                print("... (more text chunks)")
        
        print(f"Total text chunks: {text_count}")
        
    except Exception as e:
        print(f"Protocol output test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Testing V5 Protocol Output ===")
    
    try:
        # Test V5 protocol output
        stream3 = agent.astream_events(
            {"messages": [message]},
            version="v2"
        )
        
        # Convert to DataStreamWithEmitters with V5 protocol output
        data_stream_v5 = LangChainAdapter.to_data_stream(
            stream3,
            options={
                "protocol_version": "v5",
                "output_format": "protocol"
            }
        )
        
        print("V5 Protocol Output:")
        v5_count = 0
        async for text_chunk in data_stream_v5:
            v5_count += 1
            if v5_count <= 5:  # Show first 5 text chunks
                print(f"V5 Text {v5_count}: {repr(text_chunk)}")
            elif v5_count == 6:
                print("... (more V5 text chunks)")
        
        print(f"Total V5 text chunks: {v5_count}")
        
    except Exception as e:
        print(f"V5 protocol test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Testing to_data_stream_response with protocol output ===")
    
    try:
        # Test to_data_stream_response
        stream4 = agent.astream_events(
            {"messages": [message]},
            version="v2"
        )
        
        # Convert to StreamingResponse
        response = LangChainAdapter.to_data_stream_response(
            stream4,
            options={"protocol_version": "v4"}
        )
        
        print(f"Response type: {type(response)}")
        print(f"Response headers: {response.headers}")
        print("✓ to_data_stream_response works correctly")
        
    except Exception as e:
        print(f"to_data_stream_response test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_unified_output_format())