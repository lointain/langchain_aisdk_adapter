#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test protocol output to verify AI SDK compatibility
"""

import asyncio
import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate

from langchain_aisdk_adapter import to_data_stream


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny with 22°C temperature."


@tool
def calculate_math(expression: str) -> str:
    """Calculate a mathematical expression safely."""
    try:
        allowed_chars = set('0123456789+-*/.() ')
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"The result of {expression} is {result}"
        else:
            return "Invalid mathematical expression"
    except Exception as e:
        return f"Error calculating: {str(e)}"


async def test_protocol_output():
    """Test the actual protocol output"""
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
    
    # Create ReAct prompt template
    try:
        from langchain import hub
        prompt = hub.pull("hwchase17/react")
    except:
        # Fallback prompt
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
    
    # Test query
    test_query = "Please use get_weather tool to check Beijing's weather and calculate_math tool to compute 15 * 24"
    
    try:
        # Create agent stream
        agent_stream = agent_executor.astream_events({"input": test_query}, version="v2")
        
        # Convert to AI SDK stream
        ai_sdk_stream = to_data_stream(agent_stream)
        
        # Collect and analyze events
        events = []
        async for chunk in ai_sdk_stream:
            events.append(chunk)
            print(f"Event: {chunk}")
        
        print("\n" + "="*60)
        print("PROTOCOL ANALYSIS")
        print("="*60)
        
        # Count events
        start_steps = [e for e in events if e.get('type') == 'start-step']
        finish_steps = [e for e in events if e.get('type') == 'finish-step']
        tool_outputs = [e for e in events if e.get('type') == 'tool-output-available']
        text_starts = [e for e in events if e.get('type') == 'text-start']
        text_ends = [e for e in events if e.get('type') == 'text-end']
        
        print(f"Start steps: {len(start_steps)}")
        print(f"Finish steps: {len(finish_steps)}")
        print(f"Tool outputs: {len(tool_outputs)}")
        print(f"Text starts: {len(text_starts)}")
        print(f"Text ends: {len(text_ends)}")
        
        # Check expected pattern
        expected_pattern = [
            "start",
            "start-step",  # Step 1: Tool calls
            "text-start",
            "text-delta",
            "tool-input-start",
            "tool-input-delta", 
            "tool-input-available",
            "text-end",
            "tool-output-available",
            "finish-step",
            "start-step",  # Step 2: Final text
            "text-start",
            "text-delta",
            "text-end",
            "finish-step",
            "finish"
        ]
        
        actual_types = [e.get('type') for e in events]
        print(f"\nExpected multi-step pattern: {len(start_steps)} start-step events")
        print(f"Actual pattern matches: {'YES' if len(start_steps) >= 2 else 'NO'}")
        
        if len(start_steps) >= 2:
            print("✅ Multi-step protocol working correctly")
        else:
            print("❌ Multi-step protocol not working")
            
    except Exception as e:
        print(f"Error during streaming: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_protocol_output())