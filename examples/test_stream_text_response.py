#!/usr/bin/env python3
"""
Test Stream Text Response Example

This example demonstrates the stream_text_response function which returns
a DataStreamResponse suitable for FastAPI streaming responses.
"""

import asyncio
import os
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

# Import the stream_text_response function
from langchain_aisdk_adapter import stream_text_response

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Please install it with: uv add python-dotenv")
    print("Falling back to manual environment variable reading.")

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    print("Please install langchain-openai: pip install langchain-openai")
    exit(1)


# Define some test tools
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location.
    
    Args:
        location: The city or location to get weather for
    
    Returns:
        Weather information as a string
    """
    return f"The weather in {location} is sunny and 25°C"


@tool
def calculate_tip(bill_amount: float, tip_percentage: float = 15.0) -> str:
    """Calculate tip amount for a bill.
    
    Args:
        bill_amount: The total bill amount
        tip_percentage: The tip percentage (default 15%)
    
    Returns:
        Tip calculation details
    """
    tip_amount = bill_amount * (tip_percentage / 100)
    total = bill_amount + tip_amount
    return f"Bill: ${bill_amount:.2f}, Tip ({tip_percentage}%): ${tip_amount:.2f}, Total: ${total:.2f}"


async def test_basic_stream_text_response():
    """Test basic stream_text_response functionality."""
    print("=== Testing Basic Stream Text Response ===")
    
    # Check for API key
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("⚠️  Warning: DEEPSEEK_API_KEY not set. Skipping test.")
        return
    
    # Initialize the model
    model = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com",
        temperature=0.7,
        streaming=True
    )
    
    # Test messages
    messages = [
        HumanMessage(content="Hello! Please tell me a short joke.")
    ]
    
    # Callback functions for testing
    async def on_chunk(text: str):
        print(f"Chunk: {text}", end="", flush=True)
    
    async def on_finish(message, options):
        print(f"\n\n=== Stream Finished ===")
        print(f"Final message: {message.content}")
        print(f"Message parts: {len(message.parts)}")
        print(f"Message ID: {message.id}")
    
    try:
        # Call stream_text_response
        response = stream_text_response(
            model=model,
            system="You are a helpful and funny assistant.",
            messages=messages,
            on_chunk=on_chunk,
            on_finish=on_finish,
            message_id="test-basic-response"
        )
        
        print(f"Response type: {type(response)}")
        print(f"Response protocol version: {response.protocol_config.version}")
        print(f"Response headers: {response.headers}")
        
        # Test streaming the response
        print("\n=== Streaming Response Content ===")
        chunk_count = 0
        async for chunk in response.body_iterator:
            chunk_count += 1
            chunk_str = chunk.decode('utf-8') if isinstance(chunk, bytes) else str(chunk)
            print(f"Protocol chunk {chunk_count}: {chunk_str[:100]}..." if len(chunk_str) > 100 else f"Protocol chunk {chunk_count}: {chunk_str}")
            
            # Limit output for testing
            if chunk_count > 10:
                print("... (truncated for testing)")
                break
        
        print(f"\nTotal protocol chunks received: {chunk_count}")
        
    except Exception as e:
        print(f"Error in basic test: {e}")
        import traceback
        traceback.print_exc()


async def test_stream_text_response_with_tools():
    """Test stream_text_response with tools."""
    print("\n\n=== Testing Stream Text Response with Tools ===")
    
    # Check for API key
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("⚠️  Warning: DEEPSEEK_API_KEY not set. Skipping test.")
        return
    
    # Initialize the model
    model = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com",
        temperature=0.3,
        streaming=True
    )
    
    # Test messages
    messages = [
        HumanMessage(content="What's the weather like in New York? Also calculate a 20% tip for a $50 bill.")
    ]
    
    # Tools
    tools = [get_weather, calculate_tip]
    
    # Callback functions
    async def on_chunk(text: str):
        if text.strip():
            print(f"Text: {text}", end="", flush=True)
    
    async def on_step_finish(step):
        print(f"\n[Step finished: {step}]")
    
    async def on_finish(message, options):
        print(f"\n\n=== Tools Test Finished ===")
        print(f"Final message: {message.content}")
        print(f"Message parts: {len(message.parts)}")
        
        # Show parts details
        for i, part in enumerate(message.parts):
            part_type = getattr(part, 'type', 'unknown')
            print(f"Part {i+1}: {part_type}")
    
    try:
        # Call stream_text_response with tools
        response = stream_text_response(
            model=model,
            system="You are a helpful assistant. Use the available tools to answer questions.",
            messages=messages,
            tools=tools,
            on_chunk=on_chunk,
            on_step_finish=on_step_finish,
            on_finish=on_finish,
            message_id="test-tools-response"
        )
        
        print(f"\nTools test response type: {type(response)}")
        print(f"Protocol version: {response.protocol_config.version}")
        
        # Test streaming (limited output)
        print("\n=== Streaming Tools Response ===")
        chunk_count = 0
        async for chunk in response.body_iterator:
            chunk_count += 1
            chunk_str = chunk.decode('utf-8') if isinstance(chunk, bytes) else str(chunk)
            if chunk_count <= 5:  # Show first few chunks
                print(f"Chunk {chunk_count}: {chunk_str[:80]}..." if len(chunk_str) > 80 else f"Chunk {chunk_count}: {chunk_str}")
            elif chunk_count == 6:
                print("... (remaining chunks hidden for brevity)")
        
        print(f"Total chunks in tools test: {chunk_count}")
        
    except Exception as e:
        print(f"Error in tools test: {e}")
        import traceback
        traceback.print_exc()


async def test_fastapi_integration_simulation():
    """Simulate FastAPI integration with stream_text_response."""
    print("\n\n=== Simulating FastAPI Integration ===")
    
    # Check for API key
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("⚠️  Warning: DEEPSEEK_API_KEY not set. Skipping test.")
        return
    
    # This simulates how you would use stream_text_response in a FastAPI endpoint
    model = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com",
        temperature=0.5,
        streaming=True
    )
    
    messages = [
        HumanMessage(content="Write a very short poem about programming.")
    ]
    
    try:
        # This is what you would return from a FastAPI endpoint
        response = stream_text_response(
            model=model,
            system="You are a creative assistant.",
            messages=messages,
            message_id="fastapi-simulation"
        )
        
        print(f"FastAPI response type: {type(response)}")
        print(f"Status code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        print(f"Media type: {response.media_type}")
        
        # In FastAPI, this would be returned directly and handled by the framework
        print("\n=== Response Content (first few chunks) ===")
        chunk_count = 0
        async for chunk in response.body_iterator:
            chunk_count += 1
            chunk_str = chunk.decode('utf-8') if isinstance(chunk, bytes) else str(chunk)
            if chunk_count <= 3:
                print(f"FastAPI chunk {chunk_count}: {repr(chunk_str)}")
            elif chunk_count == 4:
                print("... (FastAPI would handle the rest)")
                break
        
        print("\n✓ FastAPI integration simulation successful!")
        
    except Exception as e:
        print(f"Error in FastAPI simulation: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all tests."""
    print("Stream Text Response Test Suite")
    print("=" * 50)
    
    # Check if DeepSeek API key is set
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("⚠️  Warning: DEEPSEEK_API_KEY not set. Tests may fail.")
        print("Please set your DeepSeek API key in the environment or in .env file.")
        print("You can get your API key from: https://platform.deepseek.com/api_keys")
        print()
    
    # Run tests
    await test_basic_stream_text_response()
    await test_stream_text_response_with_tools()
    await test_fastapi_integration_simulation()
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("\nTo use in FastAPI:")
    print("""
from fastapi import FastAPI
from langchain_aisdk_adapter import stream_text_response

app = FastAPI()

@app.post("/stream")
async def stream_endpoint():
    return stream_text_response(
        model=your_model,
        messages=your_messages
    )
""")


if __name__ == "__main__":
    asyncio.run(main())