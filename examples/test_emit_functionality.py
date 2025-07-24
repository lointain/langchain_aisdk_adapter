#!/usr/bin/env python3
"""Test emit functionality with protocol formatting."""

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

async def test_emit_functionality():
    """Test emit functionality with DataStreamWithEmitters."""
    
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
    
    print("=== Testing DataStreamWithEmitters with emit methods ===")
    
    try:
        # Create stream
        stream = agent.astream_events(
            {"messages": [message]},
            version="v2"
        )
        
        # Convert to DataStreamWithEmitters
        data_stream = LangChainAdapter.to_data_stream(
            stream,
            options={"protocol_version": "v4"}
        )
        
        print("Testing emit methods:")
        
        # Test emit_start
        await data_stream.emit_start()
        print("✓ emit_start() called")
        
        # Test emit_text_start
        await data_stream.emit_text_start()
        print("✓ emit_text_start() called")
        
        # Test emit_text_delta
        await data_stream.emit_text_delta("Hello from emit_text_delta!")
        print("✓ emit_text_delta() called")
        
        # Test emit_data
        await data_stream.emit_data({"custom": "data", "test": True})
        print("✓ emit_data() called")
        
        # Test emit_file
        await data_stream.emit_file("test.txt", "text/plain", b"Test file content")
        print("✓ emit_file() called")
        
        # Test emit_tool_input_start
        await data_stream.emit_tool_input_start("tool_123", "test_tool")
        print("✓ emit_tool_input_start() called")
        
        # Test emit_tool_input_available
        await data_stream.emit_tool_input_available("tool_123", "test_tool", {"param": "value"})
        print("✓ emit_tool_input_available() called")
        
        # Test emit_tool_output_available
        await data_stream.emit_tool_output_available("tool_123", "Tool execution result")
        print("✓ emit_tool_output_available() called")
        
        # Test emit_text_end
        await data_stream.emit_text_end("Complete text content")
        print("✓ emit_text_end() called")
        
        # Test emit_finish
        await data_stream.emit_finish()
        print("✓ emit_finish() called")
        
        print("\nProcessing stream chunks:")
        chunk_count = 0
        async for chunk in data_stream:
            chunk_count += 1
            if chunk_count <= 10:  # Show first 10 chunks
                print(f"Chunk {chunk_count}: {chunk}")
            elif chunk_count == 11:
                print("... (more chunks)")
        
        print(f"\nTotal chunks processed: {chunk_count}")
        
    except Exception as e:
        print(f"Emit functionality test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Testing protocol-formatted output with emit ===")
    
    try:
        # Create another stream
        stream2 = agent.astream_events(
            {"messages": [message]},
            version="v2"
        )
        
        # Convert to protocol-formatted text stream
        text_stream = LangChainAdapter.to_data_stream_text(
            stream2,
            options={"protocol_version": "v4"}
        )
        
        print("Protocol-formatted output:")
        text_count = 0
        async for text_chunk in text_stream:
            text_count += 1
            if text_count <= 5:  # Show first 5 text chunks
                print(f"Text {text_count}: {repr(text_chunk)}")
            elif text_count == 6:
                print("... (more text chunks)")
        
        print(f"\nTotal text chunks: {text_count}")
        
    except Exception as e:
        print(f"Protocol formatting test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_emit_functionality())