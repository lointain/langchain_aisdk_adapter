#!/usr/bin/env python3
"""
Test script to validate the fixed AI SDK protocol output.
This script specifically tests:
1. Correct finish-step timing (after tool-output-available)
2. Proper tool name extraction (no more unknown_tool)
3. Correct protocol sequence matching real streams
"""

import asyncio
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

async def test_protocol_fixes():
    """Test the fixed protocol output."""
    
    # Temporarily disable LangSmith tracing to avoid warnings
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    
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
    try:
        prompt = hub.pull("hwchase17/react")
    except:
        # Fallback prompt if hub is not available
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
    
    # Create the agent
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Test query
    query = "What's the weather in Tokyo?"
    
    print("=== Protocol Fixes Test ===")
    print(f"Query: {query}")
    print("\n=== Expected Protocol Sequence ===")
    print("1. start")
    print("2. start-step")
    print("3. text-start, text-delta, text-end")
    print("4. tool-input-start")
    print("5. tool-input-delta")
    print("6. tool-input-available")
    print("7. tool-output-available")
    print("8. finish-step (IMMEDIATELY after tool-output-available)")
    print("9. start-step (for next step)")
    print("10. text-start, text-delta, text-end")
    print("11. finish-step")
    print("12. finish")
    
    print("\n=== Actual Protocol Output ===")
    
    # Collect protocol events
    protocol_events = []
    tool_events = []
    
    try:
        # Stream the response using astream_events
        agent_stream = agent_executor.astream_events({"input": query}, version="v2")
        
        async for chunk in to_data_stream(agent_stream):
            protocol_events.append(chunk)
            print(f"Protocol: {chunk}")
            
            # Track tool-related events
            if chunk.get('type') in ['tool-input-start', 'tool-input-available', 'tool-output-available', 'finish-step']:
                tool_events.append(chunk)
            
    except Exception as e:
        print(f"Error during streaming: {e}")
        return []
    
    print("\n=== Protocol Analysis ===")
    
    # Analyze the protocol events
    event_types = [event.get('type') for event in protocol_events]
    event_counts = {}
    for event_type in event_types:
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
    
    print(f"Total events: {len(protocol_events)}")
    print(f"Event types: {list(set(event_types))}")
    print(f"Event counts: {event_counts}")
    
    print("\n=== Tool Events Sequence Analysis ===")
    for i, event in enumerate(tool_events):
        event_type = event.get('type')
        tool_name = event.get('toolName', 'N/A')
        tool_call_id = event.get('toolCallId', 'N/A')
        print(f"{i+1}. {event_type} - Tool: {tool_name} - ID: {tool_call_id}")
    
    # Check for unknown_tool issues
    unknown_tools = [e for e in protocol_events if e.get('toolName') == 'unknown_tool']
    if unknown_tools:
        print(f"\n❌ Found {len(unknown_tools)} events with 'unknown_tool'")
        for event in unknown_tools:
            print(f"   - {event}")
    else:
        print("\n✅ No 'unknown_tool' issues found")
    
    # Check finish-step timing
    print("\n=== Finish-Step Timing Analysis ===")
    for i, event in enumerate(protocol_events):
        if event.get('type') == 'tool-output-available':
            # Find the next finish-step event after tool-output-available
            finish_step_found = False
            for j in range(i + 1, len(protocol_events)):
                if protocol_events[j].get('type') == 'finish-step':
                    finish_step_found = True
                    # Check if there are only text events between tool-output and finish-step
                    intermediate_events = [protocol_events[k].get('type') for k in range(i + 1, j)]
                    text_only = all(event_type in ['text-start', 'text-delta', 'text-end'] for event_type in intermediate_events)
                    
                    if text_only:
                        print(f"✅ finish-step correctly follows tool-output-available (with text generation in between)")
                        print(f"   Intermediate events: {intermediate_events}")
                    else:
                        print(f"❌ Non-text events found between tool-output-available and finish-step")
                        print(f"   Intermediate events: {intermediate_events}")
                    break
            
            if not finish_step_found:
                print(f"❌ No finish-step found after tool-output-available")
    
    return protocol_events

if __name__ == "__main__":
    asyncio.run(test_protocol_fixes())