#!/usr/bin/env python3
"""
Test script to validate the enhanced AI SDK protocol output.
This script checks if our protocol matches the expected format from real streams.
"""

import asyncio
import json
import os
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain import hub
from langchain_aisdk_adapter import to_data_stream

# Mock tools for testing
@tool
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"The weather in {location} is sunny with 22°C temperature."

@tool
def calculate_math(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression.replace('*', '*').replace('x', '*'))
        return f"The result of {expression} is {result}"
    except:
        return f"Could not calculate {expression}"

async def test_protocol_validation():
    """Test and validate the AI SDK protocol output."""
    
    # Check for API key
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("Error: DEEPSEEK_API_KEY environment variable not set")
        return []
    
    # Initialize the LLM
    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com",
        temperature=0,
        streaming=True
    )
    
    # Create tools
    tools = [get_weather, calculate_math]
    
    # Get the prompt template
    prompt = hub.pull("hwchase17/react")
    
    # Create the agent
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Test query
    query = "What's the weather in Beijing and what is 15*24?"
    
    print("=== Protocol Validation Test ===")
    print(f"Query: {query}")
    print("\n=== Expected Protocol Events ===")
    print("1. start")
    print("2. start-step")
    print("3. text-start, text-delta, text-end")
    print("4. tool-input-start")
    print("5. tool-input-delta")
    print("6. tool-input-available")
    print("7. tool-output-available")
    print("8. finish-step")
    print("9. finish")
    
    print("\n=== Actual Protocol Output ===")
    
    # Collect protocol events
    protocol_events = []
    
    try:
        # Temporarily disable LangSmith tracing to avoid warnings
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        
        # Stream the response using astream_events
        agent_stream = agent_executor.astream_events({"input": query}, version="v2")
        
        async for chunk in to_data_stream(agent_stream):
            protocol_events.append(chunk)
            print(f"Protocol: {chunk}")
            
    except Exception as e:
        print(f"Error during streaming: {e}")
    
    print("\n=== Protocol Analysis ===")
    
    # Analyze the protocol events
    event_types = [event.get('type') for event in protocol_events]
    event_counts = {}
    for event_type in event_types:
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
    
    print(f"Total events: {len(protocol_events)}")
    print(f"Event types: {list(set(event_types))}")
    print(f"Event counts: {event_counts}")
    
    # Check for required events
    required_events = ['start', 'start-step', 'text-start', 'text-delta', 'finish']
    tool_events = ['tool-input-start', 'tool-input-delta', 'tool-input-available', 'tool-output-available']
    
    print("\n=== Validation Results ===")
    
    for event in required_events:
        if event in event_types:
            print(f"✅ {event}: Found")
        else:
            print(f"❌ {event}: Missing")
    
    print("\nTool Events:")
    for event in tool_events:
        count = event_counts.get(event, 0)
        if count > 0:
            print(f"✅ {event}: Found ({count} times)")
        else:
            print(f"❌ {event}: Missing")
    
    # Check for tool-input-delta specifically
    tool_input_deltas = [e for e in protocol_events if e.get('type') == 'tool-input-delta']
    if tool_input_deltas:
        print(f"\n=== Tool Input Delta Analysis ===")
        for i, delta_event in enumerate(tool_input_deltas):
            print(f"Delta {i+1}: {delta_event.get('inputTextDelta', 'N/A')}")
    
    return protocol_events

if __name__ == "__main__":
    asyncio.run(test_protocol_validation())