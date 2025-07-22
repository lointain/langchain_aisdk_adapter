#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Agent Example with AI SDK Data Stream Protocol

This example demonstrates the enhanced LangChain to AI SDK adapter that supports:
- Complete AI SDK UI Stream Protocol
- Tool invocation events (tool-input-start, tool-input-available, tool-output-available)
- Step control events (start-step, finish-step)
- Stream control events (start, finish)
- AI SDK compatible callbacks
"""

import asyncio
import json
from typing import Any, Dict, List

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from langchain_aisdk_adapter import (
    to_data_stream,
    BaseAICallbackHandler,
    Message,
    UIPart,
    TextUIPart,
    ToolInvocationUIPart,
)


# Define some example tools
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is sunny with a temperature of 22°C."


@tool
def calculate_math(expression: str) -> str:
    """Calculate a mathematical expression safely."""
    try:
        # Simple safe evaluation for demo purposes
        result = eval(expression.replace("^", "**"))
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for '{query}': Found 3 relevant articles about {query}."


class EnhancedAICallbackHandler(BaseAICallbackHandler):
    """Enhanced AI SDK compatible callback handler."""
    
    def __init__(self):
        self.tool_calls_count = 0
        self.text_length = 0
        self.steps_count = 0
    
    async def on_finish(self, message: Message, options: Dict[str, Any]) -> None:
        """Called when the stream finishes with the final message."""
        print("\n" + "="*60)
        print("🎉 STREAM COMPLETED - AI SDK Callback Summary")
        print("="*60)
        
        print(f"📝 Message ID: {message.id}")
        print(f"📄 Final Content Length: {len(message.content)} characters")
        print(f"🔧 Total Tool Calls: {self.tool_calls_count}")
        print(f"📊 Total Steps: {self.steps_count}")
        
        # Analyze message parts
        text_parts = []
        tool_parts = []
        step_parts = []
        
        for part in message.parts:
            if isinstance(part, TextUIPart):
                text_parts.append(part)
            elif isinstance(part, ToolInvocationUIPart):
                tool_parts.append(part)
                self.tool_calls_count += 1
            elif hasattr(part, 'type') and 'step' in str(part.type).lower():
                step_parts.append(part)
                self.steps_count += 1
        
        print(f"\n📋 Message Parts Analysis:")
        print(f"  • Text Parts: {len(text_parts)}")
        print(f"  • Tool Invocation Parts: {len(tool_parts)}")
        print(f"  • Step Parts: {len(step_parts)}")
        
        # Show tool details
        if tool_parts:
            print(f"\n🔧 Tool Invocations Details:")
            for i, tool_part in enumerate(tool_parts, 1):
                tool_inv = tool_part.toolInvocation
                print(f"  {i}. {tool_inv.toolName}")
                print(f"     • State: {tool_inv.state}")
                print(f"     • Call ID: {tool_inv.toolCallId}")
                if hasattr(tool_inv, 'args') and tool_inv.args:
                    print(f"     • Args: {tool_inv.args}")
                if hasattr(tool_inv, 'result') and tool_inv.result:
                    print(f"     • Result: {tool_inv.result}")
        
        # Show usage information if available
        if options:
            print(f"\n📊 Additional Options: {options}")
        
        print("="*60)
    
    async def on_error(self, error: Exception) -> None:
        """Called when an error occurs during streaming."""
        print(f"\n❌ ERROR in AI SDK Callback: {error}")
        print(f"Error type: {type(error).__name__}")


async def create_enhanced_agent():
    """Create an enhanced LangChain agent with tools."""
    
    # Initialize the language model
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        streaming=True,
    )
    
    # Define tools
    tools = [get_weather, calculate_math, search_web]
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can use tools to answer questions. "
                  "Use the available tools when needed to provide accurate information."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # Create the agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
    )
    
    return agent_executor


async def run_enhanced_example():
    """Run the enhanced agent example with AI SDK Data Stream Protocol."""
    
    print("🚀 Starting Enhanced LangChain to AI SDK Adapter Example")
    print("📡 Using AI SDK Data Stream Protocol with Tool Events")
    print("-" * 60)
    
    # Create the agent
    agent_executor = await create_enhanced_agent()
    
    # Create AI SDK callback handler
    callback_handler = EnhancedAICallbackHandler()
    
    # Define a complex query that will use multiple tools
    query = (
        "What's the weather like in Tokyo? "
        "Also, calculate what 15 * 8 + 32 equals, "
        "and search for information about 'machine learning trends 2024'."
    )
    
    print(f"🤔 Query: {query}")
    print("-" * 60)
    
    try:
        # Get the streaming response
        stream = agent_executor.astream(
            {"input": query},
            config={"callbacks": []}
        )
        
        print("\n📊 AI SDK Data Stream Protocol Output:")
        print("-" * 40)
        
        chunk_count = 0
        event_types = {}
        
        # Convert to AI SDK Data Stream Protocol with enhanced callbacks
        async for chunk in to_data_stream(stream, callbacks=callback_handler):
            chunk_count += 1
            chunk_type = chunk.get("type", "unknown")
            
            # Count event types
            event_types[chunk_type] = event_types.get(chunk_type, 0) + 1
            
            # Pretty print different chunk types
            if chunk_type == "start":
                print(f"🟢 START: Message ID {chunk.get('messageId')}")
            elif chunk_type == "start-step":
                print(f"🔵 STEP START")
            elif chunk_type == "finish-step":
                print(f"🔵 STEP FINISH")
            elif chunk_type == "text-start":
                print(f"📝 TEXT START: ID {chunk.get('id')}")
            elif chunk_type == "text-delta":
                delta = chunk.get("delta", "")
                print(f"📝 TEXT: {repr(delta)}", end="", flush=True)
            elif chunk_type == "text-end":
                print(f"\n📝 TEXT END: ID {chunk.get('id')}")
            elif chunk_type == "tool-input-start":
                print(f"🔧 TOOL INPUT START: {chunk.get('toolName')} (ID: {chunk.get('toolCallId')})")
            elif chunk_type == "tool-input-available":
                print(f"🔧 TOOL INPUT: {chunk.get('toolName')} - {chunk.get('input')}")
            elif chunk_type == "tool-output-available":
                print(f"🔧 TOOL OUTPUT: {chunk.get('output')}")
            elif chunk_type == "finish":
                print(f"🔴 FINISH")
            else:
                print(f"❓ {chunk_type.upper()}: {chunk}")
        
        print("\n" + "-" * 40)
        print(f"📊 Stream Statistics:")
        print(f"  • Total Chunks: {chunk_count}")
        print(f"  • Event Types: {dict(sorted(event_types.items()))}")
        
    except Exception as e:
        print(f"❌ Error during streaming: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the enhanced example
    asyncio.run(run_enhanced_example())