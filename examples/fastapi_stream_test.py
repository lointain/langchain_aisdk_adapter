#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI Stream Test with Real DeepSeek API

This example tests the stream handlers used in FastAPI with:
- Real DeepSeek API calls
- AutoStreamHandler and ManualStreamHandler
- AI SDK compatible callbacks
- Error handling and debugging
"""

import asyncio
import os
import sys
import json
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate

# Add the src directory to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import the stream handlers from web backend
web_backend_path = os.path.join(project_root, 'web', 'backend')
if web_backend_path not in sys.path:
    sys.path.insert(0, web_backend_path)

from stream_handlers import AutoStreamHandler, ManualStreamHandler


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.
    
    Args:
        city: The name of the city to get weather for
        
    Returns:
        Weather information for the city
    """
    return f"The weather in {city} is sunny with 22°C temperature."


@tool
def calculate_math(expression: str) -> str:
    """Calculate a mathematical expression safely.
    
    Args:
        expression: Mathematical expression to calculate (e.g., "2+2", "10*5")
        
    Returns:
        The result of the calculation
    """
    try:
        # Simple safe evaluation for basic math
        allowed_chars = set('0123456789+-*/.() ')
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"The result of {expression} is {result}"
        else:
            return "Invalid mathematical expression"
    except Exception as e:
        return f"Error calculating: {str(e)}"


def create_agent_executor():
    """Create agent executor with DeepSeek LLM"""
    # Check for API key
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable not set")
    
    # Initialize the language model
    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com",
        temperature=0.1,
        streaming=True
    )
    
    # Create tools list
    tools = [get_weather, calculate_math]
    
    # Create ReAct prompt template
    prompt = PromptTemplate.from_template(
        """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
    )
    
    # Create agent
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor


async def test_auto_stream_handler():
    """Test AutoStreamHandler"""
    print("\n" + "="*60)
    print("TESTING AUTO STREAM HANDLER")
    print("="*60)
    
    try:
        agent_executor = create_agent_executor()
        
        # Test query
        test_query = "Please check the weather in Beijing and calculate 15 * 24"
        message_id = f"test_auto_{asyncio.get_event_loop().time()}"
        
        print(f"Query: {test_query}")
        print(f"Message ID: {message_id}")
        print("\nStream output:")
        print("-" * 40)
        
        # Create auto stream
        stream = AutoStreamHandler.create_auto_stream(
            agent_executor=agent_executor,
            user_input=test_query,
            chat_history=[],
            message_id=message_id
        )
        
        # Consume stream
        chunk_count = 0
        async for chunk in stream:
            chunk_count += 1
            print(f"Chunk {chunk_count}: {chunk.strip()}")
            
            # Parse and check for errors
            if chunk.startswith("data: "):
                try:
                    data = json.loads(chunk[6:])
                    if data.get("type") == "error":
                        print(f"ERROR DETECTED: {data.get('errorText')}")
                        break
                except json.JSONDecodeError:
                    pass
        
        print(f"\nTotal chunks received: {chunk_count}")
        
    except Exception as e:
        print(f"Error in auto stream test: {str(e)}")
        import traceback
        traceback.print_exc()


async def test_manual_stream_handler():
    """Test ManualStreamHandler"""
    print("\n" + "="*60)
    print("TESTING MANUAL STREAM HANDLER")
    print("="*60)
    
    try:
        agent_executor = create_agent_executor()
        
        # Test query
        test_query = "Please check the weather in Shanghai and calculate 20 + 30"
        message_id = f"test_manual_{asyncio.get_event_loop().time()}"
        
        print(f"Query: {test_query}")
        print(f"Message ID: {message_id}")
        print("\nStream output:")
        print("-" * 40)
        
        # Create manual stream
        stream = ManualStreamHandler.create_manual_stream(
            agent_executor=agent_executor,
            user_input=test_query,
            chat_history=[],
            message_id=message_id
        )
        
        # Consume stream
        chunk_count = 0
        async for chunk in stream:
            chunk_count += 1
            print(f"Chunk {chunk_count}: {chunk.strip()}")
            
            # Parse and check for errors
            if chunk.startswith("data: "):
                try:
                    data = json.loads(chunk[6:])
                    if data.get("type") == "error":
                        print(f"ERROR DETECTED: {data.get('errorText')}")
                        break
                except json.JSONDecodeError:
                    pass
        
        print(f"\nTotal chunks received: {chunk_count}")
        
    except Exception as e:
        print(f"Error in manual stream test: {str(e)}")
        import traceback
        traceback.print_exc()


async def test_message_id_behavior():
    """Test behavior with and without message_id"""
    print("\n" + "="*60)
    print("TESTING MESSAGE ID BEHAVIOR")
    print("="*60)
    
    try:
        agent_executor = create_agent_executor()
        test_query = "What is 5 + 5?"
        
        # Test without message_id
        print("\n1. Testing WITHOUT message_id:")
        print("-" * 30)
        
        stream = AutoStreamHandler.create_auto_stream(
            agent_executor=agent_executor,
            user_input=test_query,
            chat_history=[],
            message_id=None  # No message ID
        )
        
        chunk_count = 0
        async for chunk in stream:
            chunk_count += 1
            if chunk_count <= 3:  # Only show first few chunks
                print(f"Chunk {chunk_count}: {chunk.strip()}")
            if chunk_count >= 10:  # Limit output
                break
        
        print(f"Completed without message_id (showed {min(chunk_count, 3)} of {chunk_count} chunks)")
        
        # Test with message_id
        print("\n2. Testing WITH message_id:")
        print("-" * 30)
        
        stream = AutoStreamHandler.create_auto_stream(
            agent_executor=agent_executor,
            user_input=test_query,
            chat_history=[],
            message_id="test_with_id_123"
        )
        
        chunk_count = 0
        async for chunk in stream:
            chunk_count += 1
            if chunk_count <= 3:  # Only show first few chunks
                print(f"Chunk {chunk_count}: {chunk.strip()}")
            if chunk_count >= 10:  # Limit output
                break
        
        print(f"Completed with message_id (showed {min(chunk_count, 3)} of {chunk_count} chunks)")
        
    except Exception as e:
        print(f"Error in message_id test: {str(e)}")
        import traceback
        traceback.print_exc()


async def main():
    """Main test function"""
    # Temporarily disable LangSmith tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    
    print("FastAPI Stream Handlers Test")
    print("============================")
    
    try:
        # Test message_id behavior first (simpler)
        await test_message_id_behavior()
        
        # Test auto stream handler
        await test_auto_stream_handler()
        
        # Test manual stream handler
        await test_manual_stream_handler()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())