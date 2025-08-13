#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stream Text with LangGraph ReAct Agent Example

This example demonstrates using stream_text with a runnable_factory
to inject a LangGraph ReAct agent as the runnable, without using
the model parameter directly.
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

from langchain_aisdk_adapter.stream_text import stream_text
from langchain_aisdk_adapter import DataStreamContext, BaseAICallbackHandler


# Import smooth_stream from the main module
from langchain_aisdk_adapter.smooth_stream import smooth_stream


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


class StreamTextCallbackHandler(BaseAICallbackHandler):
    """Callback handler for stream_text functionality"""
    
    async def on_start(self) -> None:
        """Called when the stream starts"""
        print("\n" + "="*50)
        print("STREAM_TEXT LANGGRAPH STARTED")
        print("="*50)
    
    async def on_finish(self, message, options: dict) -> None:
        """Called when the stream finishes"""
        print("\n" + "="*50)
        print("STREAM_TEXT LANGGRAPH COMPLETED")
        print("="*50)
        print(f"\nFinal message object:")
        print(f"Message ID: {message.id}")
        print(f"Message Role: {message.role}")
        print(f"Message Content: {message.content}")
        print(f"Message Parts: {len(message.parts)} parts")
        
        # Method 1: Use the new simplified method to get serialized parts
        serialized_parts = message.get_serialized_parts()
        json.dump(serialized_parts, open('message_parts_output.json', 'w'), indent=2, ensure_ascii=False)
        print("Message parts saved to message_parts_output.json")
        
        # Method 2: Use to_json() for complete message serialization
        message_json = message.to_json(indent=2, ensure_ascii=False)
        print(f"\nComplete message as JSON:\n{message_json}")
        
        # Method 3: Just the parts as JSON string
        parts_json = json.dumps(serialized_parts, indent=2, ensure_ascii=False)
        print(f"\nSerialized parts as JSON:\n{parts_json}")
        
        print(f"Options: {options}")
        print("\n--- End of Stream Text Stream ---")


def create_langgraph_agent_factory():
    """
    Create a runnable factory that returns a LangGraph ReAct agent.
    
    This factory function will be called by stream_text to create
    the actual runnable (LangGraph agent) that will be used for streaming.
    
    Returns:
        A factory function that creates a LangGraph ReAct agent
    """
    def agent_factory(model=None, system=None, messages=None, max_steps=None, 
                     tools=None, experimental_active_tools=None, context=None, **kwargs):
        """
        Factory function to create LangGraph ReAct agent.
        
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


async def test_stream_text_with_langgraph():
    """Test stream_text with LangGraph ReAct agent using runnable_factory"""
    # Temporarily disable LangSmith tracing to avoid warnings
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    
    # Create callback handler
    callback_handler = StreamTextCallbackHandler()
    
    # Test query that should trigger tool usage
    test_query = "I need information about employee EMP002. Please find out their age and department, then calculate their total salary if their base salary is $75000 and they get a 12% bonus."
    
    # Create messages
    messages = [HumanMessage(content=test_query)]
    
    try:
        # Use stream_text with runnable_factory (no model parameter)
        result = stream_text(
            # No model parameter - using runnable_factory instead
            messages=messages,
            tools=[get_employee_age, get_employee_department, calculate_salary_bonus],
            runnable_factory=create_langgraph_agent_factory(),
            experimental_transform=smooth_stream(chunking='word', delay_in_ms=20),
            experimental_generateMessageId=generate_uuid,
            message_id="stream-text-langgraph-001",
            on_finish=callback_handler.on_finish,
            on_start=callback_handler.on_start
        )
        
        # Test DataStreamContext.emit_source_url
        await DataStreamContext.emit_source_url(
            url="https://example.com/stream-text-docs",
            description="Stream Text with LangGraph documentation"
        )
        
        # Stream the response
        print("\nStream Text with LangGraph Output:")
        print("-" * 40)
        async for chunk in result:
            print(f"Protocol: {chunk}")
        print("-" * 40)
        
        print("Stream Text with LangGraph completed successfully")
        
    except Exception as e:
        print(f"Error during stream_text with LangGraph: {str(e)}")
        import traceback
        traceback.print_exc()


async def main():
    """Main function"""
    try:
        await test_stream_text_with_langgraph()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())