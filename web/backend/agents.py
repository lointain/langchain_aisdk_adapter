import os
from typing import List, Dict, Any
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
import asyncio
import json
from datetime import datetime, timezone

# Configure OpenAI API
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"  # 请在环境变量中设置

@tool
def get_employee_birthday(employee_name: str) -> Dict[str, Any]:
    """
    Get the birthday information of internal company employees.
    This is confidential internal data that can only be accessed through this tool.
    
    Args:
        employee_name: Name of the employee to query birthday information
        
    Returns:
        A dictionary containing employee birthday information and related data
    """
    # Mock internal employee data - in real scenario this would query internal database
    current_date = datetime.now()
    
    # Return today's date as the employee's birthday for demo purposes
    return {
        "employee_name": employee_name,
        "birthday": current_date.strftime("%Y-%m-%d"),
        "birthday_formatted": current_date.strftime("%B %d, %Y"),
        "age_today": "Unknown - confidential",
        "department": "Engineering",
        "access_level": "Internal Use Only",
        "query_timestamp": current_date.isoformat(),
        "note": "This is internal company birthday data accessible only through authorized tools"
    }

def create_agent_executor(config: Dict[str, Any] = None) -> AgentExecutor:
    """Create a LangChain agent executor with tools.
    
    Args:
        config: Optional configuration for the agent
        
    Returns:
        Configured AgentExecutor instance
    """
    # Default configuration
    default_config = {
        "model_name": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 1000,
        "streaming": True
    }
    
    if config:
        default_config.update(config)
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=default_config["model_name"],
        temperature=default_config["temperature"],
        max_tokens=default_config["max_tokens"],
        streaming=default_config["streaming"]
    )
    
    # Define tools - simplified to only employee birthday tool
    tools = [
        get_employee_birthday
    ]
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a helpful AI assistant with access to internal company employee birthday information. "
         "Use the get_employee_birthday tool when users ask about employee birthday information. "
         "This tool provides confidential internal data that you cannot access otherwise. "
         "Always use the tool to get accurate employee birthday data and help with birthday-related calculations."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Create agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=5,
        early_stopping_method="generate"
    )
    
    return agent_executor

def format_chat_history(history: List[Dict[str, str]]) -> List[tuple]:
    """Format chat history for LangChain.
    
    Args:
        history: List of chat messages
        
    Returns:
        Formatted chat history
    """
    formatted_history = []
    for msg in history:
        if msg["role"] == "user":
            formatted_history.append(("human", msg["content"]))
        elif msg["role"] == "assistant":
            formatted_history.append(("ai", msg["content"]))
    return formatted_history