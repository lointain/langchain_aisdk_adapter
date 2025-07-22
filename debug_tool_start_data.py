#!/usr/bin/env python3
"""
Debug script to examine on_tool_start event data structure.
"""

import asyncio
import os
import json
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain import hub

# Mock tools for testing
@tool
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"The weather in {location} is sunny with 22°C temperature."

async def debug_tool_start_data():
    """Debug on_tool_start event data structure."""
    
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
    
    print("=== Debug Tool Start Data ===")
    print(f"Query: {query}")
    print("\n=== Tool Start Events ===")
    
    try:
        # Stream the response using astream_events to see raw LangChain events
        agent_stream = agent_executor.astream_events({"input": query}, version="v2")
        
        event_count = 0
        async for event in agent_stream:
            event_count += 1
            event_type = event.get("event", "unknown")
            
            # Print tool start events with full data
            if event_type == "on_tool_start":
                print(f"\n=== Tool Start Event {event_count} ===")
                data = event.get("data", {})
                print(f"Full event data:")
                print(json.dumps(data, indent=2, default=str))
                
                # Try different extraction methods
                print(f"\nExtraction attempts:")
                print(f"  data.get('name'): {data.get('name')}")
                print(f"  data.get('tool_name'): {data.get('tool_name')}")
                print(f"  data.get('tool'): {data.get('tool')}")
                
                serialized = data.get('serialized', {})
                print(f"  serialized.get('name'): {serialized.get('name')}")
                print(f"  serialized.get('id'): {serialized.get('id')}")
                print(f"  serialized.get('_type'): {serialized.get('_type')}")
                print(f"  serialized.get('class_name'): {serialized.get('class_name')}")
                
                if 'kwargs' in serialized:
                    kwargs = serialized['kwargs']
                    print(f"  serialized.kwargs.get('name'): {kwargs.get('name')}")
                
                metadata = data.get('metadata', {})
                print(f"  metadata.get('name'): {metadata.get('name')}")
                
                print(f"  inputs: {data.get('inputs')}")
                print(f"  run_id: {data.get('run_id')}")
                
    except Exception as e:
        print(f"Error during streaming: {e}")
        return []
    
    print(f"\nTotal events processed: {event_count}")

if __name__ == "__main__":
    asyncio.run(debug_tool_start_data())