"""FastAPI Example with LangChain AI SDK Adapter

This example demonstrates how to integrate LangChain with AI SDK streaming
protocol using FastAPI for web applications.

Installation:
    pip install fastapi uvicorn langchain-openai

Usage:
    uvicorn web.fastapi_example:app --reload
"""

import os
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor

# Import the adapter
from langchain_aisdk_adapter import LangChainAdapter, AdapterConfig, ThreadSafeAdapterConfig, safe_config

# Load environment variables
load_dotenv()

app = FastAPI(
    title="LangChain AI SDK Adapter - FastAPI Example",
    description="Demonstrates LangChain integration with AI SDK streaming protocol",
    version="0.1.0"
)


# Request/Response models
class ChatRequest(BaseModel):
    message: str
    system_prompt: Optional[str] = "You are a helpful assistant."
    model: Optional[str] = "gpt-3.5-turbo"
    temperature: Optional[float] = 0.7


class ChainRequest(BaseModel):
    question: str
    topic: Optional[str] = "artificial intelligence"
    model: Optional[str] = "gpt-3.5-turbo"


class AgentRequest(BaseModel):
    input: str
    model: Optional[str] = "gpt-3.5-turbo"


def get_llm(model: str = "gpt-3.5-turbo", temperature: float = 0.7) -> ChatOpenAI:
    """Create and configure LLM instance"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY not found in environment variables"
        )
    
    return ChatOpenAI(
        api_key=api_key,
        model=model,
        temperature=temperature,
        streaming=True
    )


# Define tools for agent examples
@tool
def get_weather(location: str) -> str:
    """Get weather information for a location.
    
    Args:
        location: The location to get weather for
    
    Returns:
        Weather information (simulated)
    """
    # Simulated weather data
    weather_data = {
        "new york": "Sunny, 22°C",
        "london": "Cloudy, 15°C",
        "tokyo": "Rainy, 18°C",
        "paris": "Partly cloudy, 20°C",
        "beijing": "Clear, 25°C",
        "shanghai": "Overcast, 19°C"
    }
    
    location_lower = location.lower()
    if location_lower in weather_data:
        return f"Weather in {location}: {weather_data[location_lower]}"
    else:
        return f"Weather data not available for {location}. Available cities: {', '.join(weather_data.keys())}"


@tool
def calculate_math(expression: str) -> str:
    """Calculate a mathematical expression safely.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 3 * 4")
    
    Returns:
        The result of the calculation
    """
    try:
        # Simple safe evaluation for basic math
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "LangChain AI SDK Adapter - FastAPI Example",
        "endpoints": {
            "/chat": "Basic chat with LLM",
            "/chat-with-config": "Chat with custom adapter configuration",
            "/chat-thread-safe": "Chat with thread-safe configuration (multi-user safe)",
            "/chat-selective-protocols": "Chat with selective protocol tracking",
            "/chat-chain": "Chat using LangChain chains",
            "/chat-agent": "Chat with agent and tools",
            "/health": "Health check endpoint"
        },
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "adapter_version": "0.1.0"}


@app.post("/chat")
async def chat(request: ChatRequest):
    """Basic chat endpoint with LLM streaming
    
    Example usage:
    ```bash
    curl -X POST "http://localhost:8000/chat" \
         -H "Content-Type: application/json" \
         -d '{"message": "Hello, how are you?"}'
    ```
    """
    try:
        # Create LangChain model
        llm = get_llm(request.model, request.temperature)
        
        # Create messages
        messages = [
            SystemMessage(content=request.system_prompt),
            HumanMessage(content=request.message)
        ]
        
        # Get stream and convert to AI SDK response
        stream = llm.astream(messages)
        
        return StreamingResponse(
            LangChainAdapter.to_data_stream_response(stream),
            media_type="text/plain"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat-with-config")
async def chat_with_config(request: ChatRequest):
    """Chat with custom adapter configuration
    
    This example demonstrates how to use AdapterConfig to control
    which AI SDK protocols are generated.
    
    Example usage:
    ```bash
    curl -X POST "http://localhost:8000/chat-with-config" \
         -H "Content-Type: application/json" \
         -d '{"message": "Explain quantum computing"}'
    ```
    """
    try:
        llm = get_llm(request.model, request.temperature)
        
        messages = [
            SystemMessage(content=request.system_prompt),
            HumanMessage(content=request.message)
        ]
        
        # Create custom configuration - minimal output
        config = AdapterConfig.minimal()
        
        stream = llm.astream(messages)
        
        return StreamingResponse(
            LangChainAdapter.to_data_stream_response(stream, config=config),
            media_type="text/plain"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat-thread-safe")
async def chat_thread_safe(request: ChatRequest):
    """Chat with thread-safe configuration for multi-user environments
    
    This example demonstrates how to use ThreadSafeAdapterConfig to ensure
    protocol state isolation in multi-user FastAPI applications.
    Each request gets its own isolated configuration context.
    
    Example usage:
    ```bash
    curl -X POST "http://localhost:8000/chat-thread-safe" \
         -H "Content-Type: application/json" \
         -d '{"message": "Hello from user 1"}'
    ```
    """
    try:
        llm = get_llm(request.model, request.temperature)
        
        messages = [
            SystemMessage(content=request.system_prompt),
            HumanMessage(content=request.message)
        ]
        
        # Create thread-safe configuration for this request
        # Each request gets isolated protocol state
        thread_safe_config = ThreadSafeAdapterConfig.for_request(safe_config)
        
        # Use pause_protocols context manager for demonstration
        with thread_safe_config.pause_protocols(['2', '6']):
            stream = llm.astream(messages)
            
            return StreamingResponse(
                LangChainAdapter.to_data_stream_response(stream, config=thread_safe_config.base_config),
                media_type="text/plain"
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat-selective-protocols")
async def chat_selective_protocols(request: ChatRequest):
    """Chat with selective protocol tracking
    
    This example demonstrates how to use the protocols() context manager
    to enable only specific AI SDK protocols for this request.
    
    Example usage:
    ```bash
    curl -X POST "http://localhost:8000/chat-selective-protocols" \
         -H "Content-Type: application/json" \
         -d '{"message": "Tell me about AI"}'
    ```
    """
    try:
        llm = get_llm(request.model, request.temperature)
        
        messages = [
            SystemMessage(content=request.system_prompt),
            HumanMessage(content=request.message)
        ]
        
        # Create thread-safe configuration for this request
        thread_safe_config = ThreadSafeAdapterConfig.for_request(safe_config)
        
        # Only enable text content and finish reason protocols
        with thread_safe_config.protocols(['0', '3']):
            stream = llm.astream(messages)
            
            return StreamingResponse(
                LangChainAdapter.to_data_stream_response(stream, config=thread_safe_config.base_config),
                media_type="text/plain"
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat-chain")
async def chat_chain(request: ChainRequest):
    """Chat using LangChain chains
    
    This example demonstrates how to use LangChain chains with
    the AI SDK adapter for more complex prompt processing.
    
    Example usage:
    ```bash
    curl -X POST "http://localhost:8000/chat-chain" \
         -H "Content-Type: application/json" \
         -d '{"question": "What are the latest developments?", "topic": "machine learning"}'
    ```
    """
    try:
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert on {topic}. Provide detailed and accurate information."),
            ("human", "{question}")
        ])
        
        llm = get_llm(request.model)
        output_parser = StrOutputParser()
        
        # Build chain
        chain = (
            {
                "topic": lambda x: request.topic,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | output_parser
        )
        
        stream = chain.astream(request.question)
        
        return StreamingResponse(
            LangChainAdapter.to_data_stream_response(stream),
            media_type="text/plain"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat-agent")
async def chat_agent(request: AgentRequest):
    """Chat with agent and tools
    
    This example demonstrates how to use LangChain agents with tools
    and stream the results using the AI SDK adapter.
    
    Example usage:
    ```bash
    curl -X POST "http://localhost:8000/chat-agent" \
         -H "Content-Type: application/json" \
         -d '{"input": "What is the weather like in Tokyo and calculate 15 * 7?"}'
    ```
    """
    try:
        # Create tools
        tools = [get_weather, calculate_math]
        
        llm = get_llm(request.model)
        
        # Create agent prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant with access to tools. Use them when needed to provide accurate information."),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create agent
        agent = create_openai_functions_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        # Use tools_only configuration to focus on tool interactions
        config = AdapterConfig.tools_only()
        
        stream = agent_executor.astream({"input": request.input})
        
        return StreamingResponse(
            LangChainAdapter.to_data_stream_response(stream, config=config),
            media_type="text/plain"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    print("Starting FastAPI server...")
    print("Make sure to set OPENAI_API_KEY in your environment variables")
    print("Visit http://localhost:8000/docs for interactive API documentation")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)