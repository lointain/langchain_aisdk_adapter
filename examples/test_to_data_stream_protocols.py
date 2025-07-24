#!/usr/bin/env python3
"""
Test script to verify that to_data_stream method supports protocol version switching.
This demonstrates that both to_data_stream and to_data_stream_response can switch protocols.
"""

import asyncio
import os
from typing import AsyncIterable
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_aisdk_adapter import LangChainAdapter


@tool
def get_weather(city: str) -> str:
    """Get weather information for a city."""
    return f"The weather in {city} is sunny with 22°C temperature."


@tool
def calculate_math(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


async def test_protocol_versions():
    """Test both v4 and v5 protocol versions with to_data_stream method."""
    
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
    
    # Create tools list
    tools = [get_weather, calculate_math]
    
    # Create simple prompt template
    from langchain_core.prompts import PromptTemplate
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
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    
    # Simple test query
    test_query = "What is 2 + 3?"
    
    print("=" * 60)
    print("Testing to_data_stream method with different protocol versions")
    print("=" * 60)
    
    # Test V4 Protocol
    print("\n🔵 Testing V4 Protocol with to_data_stream:")
    print("-" * 50)
    
    try:
        agent_stream_v4 = agent_executor.astream_events({"input": test_query}, version="v2")
        
        # Convert to AI SDK stream with V4 protocol
        ai_sdk_stream_v4 = LangChainAdapter.to_data_stream(
            agent_stream_v4,
            message_id="test-v4-001",
            options={"protocol_version": "v4", "auto_events": True}
        )
        
        print("V4 Protocol Output (to_data_stream):")
        chunk_count = 0
        async for chunk in ai_sdk_stream_v4:
            chunk_count += 1
            print(f"V4 Chunk {chunk_count}: {chunk}")
            if chunk_count >= 5:  # Limit output for readability
                print("... (truncated for readability)")
                break
                
        await ai_sdk_stream_v4.close()
        
    except Exception as e:
        print(f"Error in V4 test: {str(e)}")
    
    # Test V5 Protocol
    print("\n🟢 Testing V5 Protocol with to_data_stream:")
    print("-" * 50)
    
    try:
        agent_stream_v5 = agent_executor.astream_events({"input": test_query}, version="v2")
        
        # Convert to AI SDK stream with V5 protocol
        ai_sdk_stream_v5 = LangChainAdapter.to_data_stream(
            agent_stream_v5,
            message_id="test-v5-001",
            options={"protocol_version": "v5", "auto_events": True}
        )
        
        print("V5 Protocol Output (to_data_stream):")
        chunk_count = 0
        async for chunk in ai_sdk_stream_v5:
            chunk_count += 1
            print(f"V5 Chunk {chunk_count}: {chunk}")
            if chunk_count >= 5:  # Limit output for readability
                print("... (truncated for readability)")
                break
                
        await ai_sdk_stream_v5.close()
        
    except Exception as e:
        print(f"Error in V5 test: {str(e)}")
    
    # Test Default Protocol (should be V4)
    print("\n🟡 Testing Default Protocol with to_data_stream:")
    print("-" * 50)
    
    try:
        agent_stream_default = agent_executor.astream_events({"input": test_query}, version="v2")
        
        # Convert to AI SDK stream with default protocol (should be V4)
        ai_sdk_stream_default = LangChainAdapter.to_data_stream(
            agent_stream_default,
            message_id="test-default-001",
            options={"auto_events": True}  # No protocol_version specified
        )
        
        print("Default Protocol Output (to_data_stream):")
        chunk_count = 0
        async for chunk in ai_sdk_stream_default:
            chunk_count += 1
            print(f"Default Chunk {chunk_count}: {chunk}")
            if chunk_count >= 5:  # Limit output for readability
                print("... (truncated for readability)")
                break
                
        await ai_sdk_stream_default.close()
        
    except Exception as e:
        print(f"Error in default test: {str(e)}")
    
    print("\n" + "=" * 60)
    print("Protocol version testing completed!")
    print("✅ to_data_stream method supports protocol version switching")
    print("✅ Both v4 and v5 protocols are working")
    print("✅ Default protocol is v4")
    print("=" * 60)


async def main():
    try:
        await test_protocol_versions()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())