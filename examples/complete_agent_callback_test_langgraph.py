#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Agent Callback Test with LangGraph ReAct Agent

This example demonstrates a complete agent workflow with:
- LangGraph ReAct agent implementation
- Real DeepSeek API calls (not mocked)
- Tool calls and agent execution
- AI SDK compatible callbacks
- step-start and tool-invocation protocols
"""

import asyncio
import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Please install it with: uv add python-dotenv")
    print("Falling back to manual environment variable reading.")

from langchain_aisdk_adapter import LangChainAdapter, DataStreamContext, BaseAICallbackHandler, Message


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


class TestCallbackHandler(BaseAICallbackHandler):
    """Test callback handler for demonstrating stream processing"""
    
    async def on_start(self) -> None:
        """Called when the stream starts"""
        print("\n" + "="*50)
        print("LANGGRAPH STREAM STARTED")
        print("="*50)
    
    async def on_finish(self, message: Message, options: dict) -> None:
        """Called when the stream finishes"""
        print("\n" + "="*50)
        print("LANGGRAPH STREAM COMPLETED")
        print("="*50)
        print(f"\nFinal message object:")
        print(f"Message ID: {message.id}")
        print(f"Message Role: {message.role}")
        print(f"Message Content: {message.content}")
        print(f"Message Parts: {len(message.parts)} parts")
        print(f"All Message Parts: {message.parts}")
        for i, part in enumerate(message.parts):
            print(f"  Part {i}: {part.type} - {getattr(part, 'text', getattr(part, 'content', str(part)))}")
        print(f"Options: {options}")
        print("\n--- End of LangGraph Stream ---")


def create_test_callbacks():
    """Create test callbacks for demonstrating stream processing"""
    return TestCallbackHandler()


async def test_complete_langgraph_agent_workflow():
    """Main test function using LangGraph ReAct agent"""
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
    tools = [get_employee_age, get_employee_department, calculate_salary_bonus]
    
    # Create LangGraph ReAct agent
    agent_executor = create_react_agent(llm, tools)
    
    # Initialize callback handler
    callbacks = create_test_callbacks()
    
    # Test query that should trigger tool usage - this MUST use tools as LLM cannot know internal employee data
    test_query = "I need information about employee EMP002. Please find out their age and department, then calculate their total salary if their base salary is $75000 and they get a 12% bonus."
    
    try:
        # Create agent stream using astream_events
        # LangGraph agents use different input format - messages list
        messages = [HumanMessage(content=test_query)]
        agent_stream = agent_executor.astream_events({"messages": messages}, version="v2")
        
        # Convert to AI SDK stream using LangChainAdapter.to_data_stream
        ai_sdk_stream = LangChainAdapter.to_data_stream(
            agent_stream,
            callbacks=callbacks,
            message_id="test-langgraph-message-001",
            options={"auto_events": True, "protocol_version": "v5"}
        )
        
        # Test DataStreamContext.emit_source_url
        await DataStreamContext.emit_source_url(
            url="https://example.com/langgraph-docs",
            description="LangGraph documentation"
        )
        
        # Stream the response and show AI SDK protocols
        print("\nLangGraph AI SDK Protocol Output:")
        print("-" * 40)
        async for chunk in ai_sdk_stream:
            print(f"Protocol: {chunk}")
        print("-" * 40)
        
        # Explicitly close the stream to trigger on_finish callback
        await ai_sdk_stream.close()
        print("LangGraph stream closed explicitly")
        
    except Exception as e:
        print(f"Error during LangGraph streaming: {str(e)}")
        import traceback
        traceback.print_exc()


async def main():
    try:
        await test_complete_langgraph_agent_workflow()
        
    except Exception as e:
        print(f"Error during LangGraph testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())