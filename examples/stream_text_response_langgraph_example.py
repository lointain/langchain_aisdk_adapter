#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stream Text Response with LangGraph ReAct Agent Example

This example demonstrates using stream_text_response with a LangGraph ReAct agent
for FastAPI integration. It combines the LangGraph ReAct functionality from
stream_text_langgraph_example.py with the FastAPI-ready stream_text_response
from test_stream_text_response.py.

This shows how to use LangGraph agents in FastAPI endpoints with proper streaming.
"""

import asyncio
import os
import uuid
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
import json

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Please install it with: uv add python-dotenv")
    print("Falling back to manual environment variable reading.")

# Import the stream_text_response function for FastAPI integration
from langchain_aisdk_adapter import stream_text_response


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


# Define tools for the LangGraph ReAct agent
@tool
def get_employee_age(employee_id: str) -> str:
    """Get the age of an internal employee by their ID.
    
    Args:
        employee_id: The unique identifier of the employee
        
    Returns:
        The age of the employee
    """
    # Simulate employee database lookup
    employees = {
        "EMP001": {"name": "Alice Johnson", "age": 28},
        "EMP002": {"name": "Bob Smith", "age": 35},
        "EMP003": {"name": "Carol Davis", "age": 42},
        "EMP004": {"name": "David Wilson", "age": 31}
    }
    
    if employee_id in employees:
        emp = employees[employee_id]
        return f"Employee {emp['name']} (ID: {employee_id}) is {emp['age']} years old."
    else:
        return f"Employee with ID {employee_id} not found in the database."


@tool
def get_employee_department(employee_id: str) -> str:
    """Get the department of an internal employee by their ID.
    
    Args:
        employee_id: The unique identifier of the employee
        
    Returns:
        The department of the employee
    """
    # Simulate employee database lookup
    departments = {
        "EMP001": "Engineering",
        "EMP002": "Marketing", 
        "EMP003": "Human Resources",
        "EMP004": "Finance"
    }
    
    if employee_id in departments:
        return f"Employee {employee_id} works in the {departments[employee_id]} department."
    else:
        return f"Department information for employee {employee_id} not found."


@tool
def calculate_salary_bonus(base_salary: float, bonus_percentage: float) -> str:
    """Calculate the total salary including bonus for an employee.
    
    Args:
        base_salary: The base salary amount
        bonus_percentage: The bonus percentage (e.g., 15 for 15%)
        
    Returns:
        The total salary including bonus
    """
    try:
        bonus_amount = base_salary * (bonus_percentage / 100)
        total_salary = base_salary + bonus_amount
        return f"Base salary: ${base_salary:,.2f}, Bonus: ${bonus_amount:,.2f} ({bonus_percentage}%), Total: ${total_salary:,.2f}"
    except Exception as e:
        return f"Error calculating salary: {str(e)}"


def create_langgraph_react_agent_factory():
    """
    Create a runnable factory that returns a LangGraph ReAct agent.
    
    This factory function will be called by stream_text_response to create
    the actual runnable (LangGraph agent) that will be used for streaming.
    This approach is needed because stream_text_response expects a runnable
    that can handle the specific message format.
    
    Returns:
        A factory function that creates a LangGraph ReAct agent
    """
    def agent_factory(model=None, system=None, messages=None, max_steps=None, 
                     tools=None, experimental_active_tools=None, context=None, **kwargs):
        """
        Factory function to create LangGraph ReAct agent.
        
        Args:
            model: The base language model (optional, will create if not provided)
            system: System message (will be prepended to messages)
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
        
        # Use provided tools or default tools
        agent_tools = tools or [get_employee_age, get_employee_department, calculate_salary_bonus]
        
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


async def test_stream_text_response_with_langgraph():
    """Test stream_text_response with LangGraph ReAct agent."""
    print("=== Testing Stream Text Response with LangGraph ReAct Agent ===")
    
    # Check for API key
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("⚠️  Warning: DEEPSEEK_API_KEY not set. Skipping test.")
        return
    
    # Temporarily disable LangSmith tracing to avoid warnings
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    
    # Create the LangGraph ReAct agent factory
    try:
        agent_factory = create_langgraph_react_agent_factory()
    except Exception as e:
        print(f"Error creating LangGraph agent factory: {e}")
        return
    
    # Test query that should trigger tool usage
    test_query = "I need information about employee EMP002. Please find out their age and department, then calculate their total salary if their base salary is $75000 and they get a 12% bonus."
    
    # Test messages
    messages = [
        HumanMessage(content=test_query)
    ]
    
    # Callback functions for testing
    async def on_chunk(text: str):
        if text.strip():
            print(f"Text: {text}", end="", flush=True)
    
    async def on_step_finish(step):
        print(f"\n[Step finished: {step}]")
    
    async def on_finish(message, options):
        print(f"\n\n=== LangGraph ReAct Test Finished ===")
        print(f"Final message: {message.content}")
        print(f"Message parts: {len(message.parts)}")
        
        # Show parts details
        for i, part in enumerate(message.parts):
            part_type = getattr(part, 'type', 'unknown')
            print(f"Part {i+1}: {part_type}")
    
    try:
        # Call stream_text_response with LangGraph ReAct agent factory
        # Note: We don't pass a model parameter, instead we use runnable_factory
        response = stream_text_response(
            # No model parameter - using runnable_factory instead
            system="You are a helpful HR assistant. Use the available tools to answer questions about employees.",
            messages=messages,
            tools=[get_employee_age, get_employee_department, calculate_salary_bonus],
            runnable_factory=agent_factory,
            on_chunk=on_chunk,
            on_step_finish=on_step_finish,
            on_finish=on_finish,
            message_id=f"langgraph-react-{generate_uuid()}"
        )
        
        print(f"\nLangGraph response type: {type(response)}")
        print(f"Protocol version: {response.protocol_config.version}")
        print(f"Status code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        
        # Test streaming the response
        print("\n=== Streaming LangGraph Response Content ===")
        chunk_count = 0
        iterator = response.body_iterator.__aiter__()
        try:
            while True:
                try:
                    chunk = await iterator.__anext__()
                    chunk_count += 1
                    chunk_str = chunk.decode('utf-8') if isinstance(chunk, bytes) else str(chunk)
                    
                    # Show first few chunks for debugging
                    if chunk_count <= 5:
                        print(f"Protocol chunk {chunk_count}: {chunk_str[:100]}..." if len(chunk_str) > 100 else f"Protocol chunk {chunk_count}: {chunk_str}")
                    elif chunk_count == 6:
                        print("... (remaining chunks hidden for brevity)")
                        
                except StopAsyncIteration:
                    break
        finally:
            # Properly close the async iterator
            if hasattr(iterator, 'aclose'):
                await iterator.aclose()
        
        print(f"\nTotal protocol chunks received: {chunk_count}")
        print("✓ LangGraph ReAct agent with stream_text_response test successful!")
        
    except Exception as e:
        print(f"Error in LangGraph test: {e}")
        import traceback
        traceback.print_exc()


async def test_fastapi_integration_with_langgraph():
    """Simulate FastAPI integration with LangGraph ReAct agent."""
    print("\n\n=== Simulating FastAPI Integration with LangGraph ReAct ===")
    
    # Check for API key
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("⚠️  Warning: DEEPSEEK_API_KEY not set. Skipping test.")
        return
    
    # This simulates how you would use stream_text_response with LangGraph in a FastAPI endpoint
    try:
        agent_factory = create_langgraph_react_agent_factory()
    except Exception as e:
        print(f"Error creating LangGraph agent factory: {e}")
        return
    
    messages = [
        HumanMessage(content="Please get information about employee EMP001 and calculate their salary with a 10% bonus if their base salary is $60000.")
    ]
    
    try:
        # This is what you would return from a FastAPI endpoint
        response = stream_text_response(
            # No model parameter - using runnable_factory instead
            system="You are an HR assistant with access to employee tools.",
            messages=messages,
            tools=[get_employee_age, get_employee_department, calculate_salary_bonus],
            runnable_factory=agent_factory,
            message_id=f"fastapi-langgraph-{generate_uuid()}"
        )
        
        print(f"FastAPI + LangGraph response type: {type(response)}")
        print(f"Status code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        print(f"Media type: {response.media_type}")
        
        # In FastAPI, this would be returned directly and handled by the framework
        print("\n=== FastAPI Response Content (first few chunks) ===")
        chunk_count = 0
        iterator = response.body_iterator.__aiter__()
        try:
            while True:
                try:
                    chunk = await iterator.__anext__()
                    chunk_count += 1
                    chunk_str = chunk.decode('utf-8') if isinstance(chunk, bytes) else str(chunk)
                    if chunk_count <= 3:
                        print(f"FastAPI chunk {chunk_count}: {repr(chunk_str[:80])}..." if len(chunk_str) > 80 else f"FastAPI chunk {chunk_count}: {repr(chunk_str)}")
                    elif chunk_count == 4:
                        print("... (FastAPI would handle the rest)")
                        break
                except StopAsyncIteration:
                    break
        finally:
            # Properly close the async iterator
            if hasattr(iterator, 'aclose'):
                await iterator.aclose()
        
        print("\n✓ FastAPI + LangGraph integration simulation successful!")
        
    except Exception as e:
        print(f"Error in FastAPI + LangGraph simulation: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all tests."""
    print("Stream Text Response with LangGraph ReAct Agent Test Suite")
    print("=" * 60)
    
    # Check if DeepSeek API key is set
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("⚠️  Warning: DEEPSEEK_API_KEY not set. Tests may fail.")
        print("Please set your DeepSeek API key in the environment or in .env file.")
        print("You can get your API key from: https://platform.deepseek.com/api_keys")
        print()
    
    # Run tests
    await test_stream_text_response_with_langgraph()
    await test_fastapi_integration_with_langgraph()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("\nTo use LangGraph ReAct agent in FastAPI:")
    print("""
from fastapi import FastAPI
from langchain_aisdk_adapter import stream_text_response
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

app = FastAPI()

def create_agent_factory():
    def agent_factory(model=None, system=None, messages=None, tools=None, **kwargs):
        if model is None:
            model = ChatOpenAI(model="deepseek-chat", ...)
        agent_tools = tools or [your_tools_here]
        return create_react_agent(model, agent_tools)
    return agent_factory

@app.post("/stream-langgraph")
async def stream_langgraph_endpoint():
    agent_factory = create_agent_factory()
    return stream_text_response(
        # No model parameter - using runnable_factory instead
        system="Your system prompt",
        messages=your_messages,
        tools=[your_tools_here],
        runnable_factory=agent_factory
    )
""")


if __name__ == "__main__":
    asyncio.run(main())