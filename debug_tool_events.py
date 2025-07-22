#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to understand LangChain tool events structure
"""

import asyncio
import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny with 22°C temperature."


async def debug_tool_events():
    """Debug function to examine tool events"""
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
    tools = [get_weather]
    
    # Create simple prompt
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
    
    # Simple test query
    test_query = "What's the weather in Beijing?"
    
    try:
        # Create agent stream using astream_events
        agent_stream = agent_executor.astream_events({"input": test_query}, version="v2")
        
        print("\n=== LangChain Events Debug ===")
        async for event in agent_stream:
            event_type = event.get("event", "unknown")
            data = event.get("data", {})
            
            # Print all events to understand structure
            print(f"\nEvent: {event_type}")
            print(f"Data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            
            # Look for tool-related events
            if "tool" in event_type.lower():
                print(f"*** TOOL EVENT FOUND: {event_type} ***")
                print(f"Full data: {data}")
            
            # Look for events that might contain tool information
            if isinstance(data, dict):
                for key, value in data.items():
                    if "tool" in str(key).lower() or "tool" in str(value).lower():
                        print(f"*** TOOL-RELATED DATA: {key} = {value} ***")
        
        print("\n=== End Debug ===")
        
    except Exception as e:
        print(f"Error during streaming: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(debug_tool_events())