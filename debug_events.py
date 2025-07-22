#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to analyze the exact event flow and identify missing finish-step events
"""

import asyncio
import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

from langchain_aisdk_adapter import (
    to_data_stream,
    BaseAICallbackHandler,
    Message,
    TextUIPart,
    ToolInvocationUIPart,
    StepStartUIPart
)


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


class DebugCallbackHandler(BaseAICallbackHandler):
    """Debug callback handler to track events"""
    
    def __init__(self):
        self.events = []
    
    async def on_finish(self, message: Message, options: Dict[str, Any]) -> None:
        """Handle completion events"""
        print("\n" + "="*50)
        print("FINAL ANALYSIS")
        print("="*50)
        
        step_starts = len([p for p in message.parts if isinstance(p, StepStartUIPart)])
        tool_invocations = len([p for p in message.parts if isinstance(p, ToolInvocationUIPart)])
        text_parts = len([p for p in message.parts if isinstance(p, TextUIPart)])
        
        print(f"Step starts: {step_starts}")
        print(f"Tool invocations: {tool_invocations}")
        print(f"Text parts: {text_parts}")
        
        finish_steps = len([e for e in self.events if e.get('type') == 'finish-step'])
        start_steps = len([e for e in self.events if e.get('type') == 'start-step'])
        
        print(f"Start-step events: {start_steps}")
        print(f"Finish-step events: {finish_steps}")
        print(f"Missing finish-step events: {start_steps - finish_steps}")
        
        if start_steps != finish_steps:
            print("\nERROR: Mismatch between start-step and finish-step events!")
            print("This indicates that some steps are not being properly closed.")
        else:
            print("\nSUCCESS: All steps are properly closed.")


async def debug_event_flow():
    """Debug the event flow to identify missing finish-step events"""
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
    from langchain import hub
    try:
        prompt = hub.pull("hwchase17/react")
    except:
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
    callback_handler = DebugCallbackHandler()
    
    # Test query
    test_query = "Please use the get_weather tool to check Beijing's weather and use calculate_math tool to compute 15 * 24"
    
    try:
        # Create agent stream
        agent_stream = agent_executor.astream_events({"input": test_query}, version="v2")
        
        # Convert to AI SDK stream
        ai_sdk_stream = to_data_stream(agent_stream, callbacks=callback_handler)
        
        # Track all events
        print("\nEvent Flow Analysis:")
        print("-" * 40)
        
        event_count = 0
        async for chunk in ai_sdk_stream:
            event_count += 1
            callback_handler.events.append(chunk)
            
            event_type = chunk.get('type', 'unknown')
            if event_type in ['start-step', 'finish-step']:
                print(f"[{event_count:3d}] {event_type}")
            elif event_type.startswith('tool-'):
                tool_name = chunk.get('toolName', chunk.get('toolCallId', 'unknown'))
                print(f"[{event_count:3d}] {event_type} ({tool_name})")
            elif event_type in ['text-start', 'text-end']:
                print(f"[{event_count:3d}] {event_type}")
            elif event_type == 'text-delta':
                # Skip text-delta for brevity
                pass
            else:
                print(f"[{event_count:3d}] {event_type}")
        
        print("-" * 40)
        print(f"Total events: {event_count}")
        
    except Exception as e:
        print(f"Error during streaming: {str(e)}")
        import traceback
        traceback.print_exc()


async def main():
    try:
        await debug_event_flow()
    except Exception as e:
        print(f"Error during debugging: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())