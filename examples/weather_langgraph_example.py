#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weather Query with LangGraph ReAct Agent Example

This example demonstrates using stream_text with a runnable_factory
to inject a LangGraph ReAct agent for weather queries using the get_weather tool.
"""

import asyncio
import os
import uuid
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
import json

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Please install it with: uv add python-dotenv")
    print("Falling back to manual environment variable reading.")

from langchain_aisdk_adapter.stream_text import stream_text
from langchain_aisdk_adapter import DataStreamContext, BaseAICallbackHandler
from langchain_aisdk_adapter.smooth_stream import smooth_stream

# Import the weather tool
from get_weather import WeatherTool


def generate_uuid() -> str:
    """
    Generate a unique UUID for message identification.
    
    This function mimics the AI SDK's generateMessageId functionality
    by creating a unique identifier for each message, which helps with
    message persistence and tracking.
    
    Returns:
        A unique UUID string
    """
    return str(uuid.uuid4())


class WeatherCallbackHandler(BaseAICallbackHandler):
    """Callback handler for weather query stream_text functionality"""
    
    async def on_start(self) -> None:
        """Called when the stream starts"""
        print("\n" + "="*50)
        print("WEATHER LANGGRAPH STREAM STARTED")
        print("="*50)
    
    async def on_finish(self, message, options: dict) -> None:
        """Called when the stream finishes"""
        print("\n" + "="*50)
        print("WEATHER LANGGRAPH STREAM COMPLETED")
        print("="*50)
        print(f"\nFinal message object:")
        print(f"Message ID: {message.id}")
        print(f"Message Role: {message.role}")
        print(f"Message Content: {message.content}")
        print(f"Message Parts: {len(message.parts)} parts")
        
        # Method 1: Use the new simplified method to get serialized parts
        serialized_parts = message.get_serialized_parts()
        json.dump(serialized_parts, open('weather_message_parts.json', 'w'), indent=2, ensure_ascii=False)
        print("Message parts saved to weather_message_parts.json")
        
        # Method 2: Use to_json() for complete message serialization
        message_json = message.to_json(indent=2, ensure_ascii=False)
        print(f"\nComplete message as JSON:\n{message_json}")
        
        # Method 3: Just the parts as JSON string
        parts_json = json.dumps(serialized_parts, indent=2, ensure_ascii=False)
        print(f"\nSerialized parts as JSON:\n{parts_json}")
        
        print(f"Options: {options}")
        print("\n--- End of Weather Stream ---")


def create_weather_agent_factory():
    """
    Create a runnable factory that returns a LangGraph ReAct agent for weather queries.
    
    This factory function will be called by stream_text to create
    the actual runnable (LangGraph agent) that will be used for streaming.
    
    Returns:
        A factory function that creates a LangGraph ReAct agent with weather tool
    """
    def agent_factory(model=None, system=None, messages=None, max_steps=None, 
                     tools=None, experimental_active_tools=None, context=None, **kwargs):
        """
        Factory function to create LangGraph ReAct agent for weather queries.
        
        Args:
            model: The base language model (optional, will create if not provided)
            system: System message (not used directly by LangGraph)
            messages: Input messages
            max_steps: Maximum steps (not used directly by LangGraph)
            tools: List of tools to bind to the agent
            experimental_active_tools: List of active tool names
            context: DataStreamContext for manual protocol sending
            **kwargs: Additional arguments
            
        Returns:
            LangGraph ReAct agent executor
        """
        # Check for API key
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")
        
        # Create or use provided model
        if model is None:
            model = ChatOpenAI(
                model="deepseek-chat",
                openai_api_key=api_key,
                openai_api_base="https://api.deepseek.com",
                temperature=0.1,
                streaming=True
            )
        
        # Create weather tool instance
        weather_tool = WeatherTool()
        
        # Use provided tools or default to weather tool
        agent_tools = tools or [weather_tool]
        
        # Filter tools based on experimental_active_tools if provided
        if experimental_active_tools:
            agent_tools = [
                tool for tool in agent_tools 
                if tool.name in experimental_active_tools
            ]
        
        # Create LangGraph ReAct agent
        agent_executor = create_react_agent(model, agent_tools)
        
        return agent_executor
    
    return agent_factory


async def test_weather_stream_text_with_langgraph():
    """Test stream_text with LangGraph ReAct agent for weather queries"""
    # Temporarily disable LangSmith tracing to avoid warnings
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    
    # Create callback handler
    callback_handler = WeatherCallbackHandler()
    
    # Test query for San Francisco weather
    test_query = "旧金山的坐标大约是纬度37.7749，经度-122.4194,那么旧金山的天气如何"
    
    # Create messages
    messages = [HumanMessage(content=test_query)]
    
    # Create weather tool instance
    weather_tool = WeatherTool()
    
    try:
        # Use stream_text with runnable_factory (no model parameter)
        result = stream_text(
            # No model parameter - using runnable_factory instead
            messages=messages,
            tools=[weather_tool],
            runnable_factory=create_weather_agent_factory(),
            experimental_transform=smooth_stream(chunking='word', delay_in_ms=20),
            experimental_generateMessageId=generate_uuid,
            message_id="weather-langgraph-001",
            on_finish=callback_handler.on_finish,
            on_start=callback_handler.on_start,
            protocol_version="v5"
        )
        
        # Stream the response and output v5 protocol
        print("\nWeather LangGraph v5 Protocol Output:")
        print("-" * 50)
        async for chunk in result:
            print(f"v5: {chunk}")
        print("-" * 50)
        
        print("Weather query with LangGraph completed successfully")
        
    except Exception as e:
        print(f"Error during weather stream_text with LangGraph: {str(e)}")
        import traceback
        traceback.print_exc()


async def main():
    """Main function"""
    try:
        await test_weather_stream_text_with_langgraph()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())