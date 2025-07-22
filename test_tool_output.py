#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Tool Output Events

This script tests if tool-output-available events are properly generated.
"""

import asyncio
import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_aisdk_adapter import to_data_stream


@tool
def simple_tool(input_text: str) -> str:
    """A simple tool for testing.
    
    Args:
        input_text: Input text to process
        
    Returns:
        Processed text
    """
    return f"Processed: {input_text}"


async def test_tool_output_events():
    """Test if tool-output-available events are generated"""
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
    tools = [simple_tool]
    
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
    
    # Test query
    test_query = "Use simple_tool to process the text 'hello world'"
    
    try:
        # Create agent stream
        agent_stream = agent_executor.astream_events({"input": test_query}, version="v2")
        
        # Convert to AI SDK stream
        ai_sdk_stream = to_data_stream(agent_stream)
        
        # Look for tool-output-available events and debug text
        tool_output_events = []
        tool_input_events = []
        text_content = ""
        async for chunk in ai_sdk_stream:
            if chunk.get('type') == 'tool-output-available':
                tool_output_events.append(chunk)
                print(f"Found tool-output-available: {chunk}")
            elif chunk.get('type') == 'tool-input-start':
                tool_input_events.append(chunk)
                print(f"Found tool-input-start: {chunk}")
            elif chunk.get('type') == 'text-delta':
                text_content += chunk.get('delta', '')
                print(f"Text delta: {repr(chunk.get('delta', ''))}")
        
        print(f"\nTotal tool-output-available events found: {len(tool_output_events)}")
        print(f"\nFull text content:\n{text_content}")
        
        # Check if text contains Observation
        if 'Observation:' in text_content:
            print("\n✓ Found 'Observation:' in text")
        else:
            print("\n✗ No 'Observation:' found in text")
        
    except Exception as e:
        print(f"Error during streaming: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_tool_output_events())