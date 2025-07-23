#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Agent Callback Test with Real DeepSeek API

This example demonstrates a complete agent workflow with:
- Real DeepSeek API calls (not mocked)
- Tool calls and agent execution
- AI SDK compatible callbacks
- step-start and tool-invocation protocols
"""

import asyncio
import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

from langchain_aisdk_adapter import LangChainAdapter
from langchain_aisdk_adapter.callbacks import StreamCallbacks


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


def create_test_callbacks():
    """Create test callbacks for demonstrating stream processing"""
    # 
    def on_start():
        print("\n" + "="*50)
        print("STREAM STARTED")
        print("="*50)
    
    # def on_token(token: str):
        # print(f"Token: {repr(token)}")
    
    # def on_text(text: str):
        # print(f"Text chunk: {repr(text)}")
    
    def on_final(final_text: str):
        print("\n" + "="*50)
        print("STREAM COMPLETED")
        print("="*50)
        print(f"\nFinal aggregated text:")
        print(f"{final_text}")
        print("\n--- End of Stream ---")
    
    return StreamCallbacks(
        on_start=on_start,
        # on_token=on_token,
        # on_text=on_text,
        on_final=on_final
    )


async def test_complete_agent_workflow():
    """Main test function"""
    # Temporarily disable LangSmith tracing to avoid warnings
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
    from langchain import hub
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
    
    # Create agent
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Initialize callback handler
    callbacks = create_test_callbacks()
    
    # Test query that should trigger tool usage
    test_query = "Please use the get_weather tool to check Beijing's weather and use calculate_math tool to compute 15 * 24"
    
    try:
        # Create agent stream using astream_events
        agent_stream = agent_executor.astream_events({"input": test_query}, version="v2")
        
        # Convert to AI SDK stream using LangChainAdapter.to_data_stream
        ai_sdk_stream = LangChainAdapter.to_data_stream(
            agent_stream,
            callbacks=callbacks,
            message_id="test-message-001",
            options={"auto_events": True}
        )
        
        # Stream the response and show AI SDK protocols
        print("\nAI SDK Protocol Output:")
        print("-" * 40)
        async for chunk in ai_sdk_stream:
            print(f"Protocol: {repr(chunk)}")
        print("-" * 40)
        
    except Exception as e:
        print(f"Error during streaming: {str(e)}")

async def main():
    try:
        await test_complete_agent_workflow()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())