#!/usr/bin/env python3
"""
Simple test to check event sequence, especially finish-step timing.
"""

import asyncio
import os
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

async def test_event_sequence():
    """Test the event sequence."""
    
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
    tools = [get_weather]
    
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
    
    print("=== Event Sequence Test ===")
    print(f"Query: {query}")
    print("\n=== Events ===")
    
    # Collect protocol events
    protocol_events = []
    
    try:
        # Stream the response using astream_events
        agent_stream = agent_executor.astream_events({"input": query}, version="v2")
        
        async for chunk in to_data_stream(agent_stream):
            protocol_events.append(chunk)
            event_type = chunk.get('type')
            
            # Print key events
            if event_type in ['start-step', 'tool-output-available', 'finish-step', 'finish']:
                print(f"{len(protocol_events):3d}. {event_type}")
                if event_type == 'tool-output-available':
                    tool_id = chunk.get('toolCallId', 'N/A')
                    print(f"     Tool ID: {tool_id}")
            
    except Exception as e:
        print(f"Error during streaming: {e}")
        return []
    
    print("\n=== Analysis ===")
    
    # Find tool-output-available events and check what follows
    for i, event in enumerate(protocol_events):
        if event.get('type') == 'tool-output-available':
            print(f"\nFound tool-output-available at position {i}:")
            print(f"  Event: {event}")
            
            # Check next few events
            for j in range(1, min(4, len(protocol_events) - i)):
                next_event = protocol_events[i + j]
                print(f"  +{j}: {next_event.get('type')}")
                if next_event.get('type') == 'finish-step':
                    print(f"  --> finish-step found at +{j} position")
                    break
    
    # Count events
    event_counts = {}
    for event in protocol_events:
        event_type = event.get('type')
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
    
    print(f"\nTotal events: {len(protocol_events)}")
    print(f"Event counts: {event_counts}")
    
    return protocol_events

if __name__ == "__main__":
    asyncio.run(test_event_sequence())