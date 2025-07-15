"""LangGraph Workflow Example with AI SDK Adapter

Demonstrates advanced usage with LangGraph workflows and AI SDK streaming protocol.
Includes tool usage, multi-step reasoning, and state management.
"""

import asyncio
import os
from typing import Dict, Any, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from typing_extensions import TypedDict
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("Warning: LangGraph not available. Install with: pip install langgraph")

from langchain_aisdk_adapter import AISDKAdapter, factory


def setup_environment():
    """Load environment variables and validate configuration"""
    load_dotenv()
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY not found in environment variables. "
            "Please copy .env.example to .env and configure your API key."
        )
    
    return api_key, base_url


def create_llm(api_key: str, base_url: str) -> ChatOpenAI:
    """Create and configure the LLM instance"""
    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=os.getenv("EXAMPLE_MODEL_NAME", "deepseek-chat"),
        temperature=float(os.getenv("EXAMPLE_TEMPERATURE", "0.3")),
        max_tokens=int(os.getenv("EXAMPLE_MAX_TOKENS", "1000")),
        streaming=True
    )


# Define tools for the agent
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


@tool
def get_weather_info(location: str) -> str:
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
        "paris": "Partly cloudy, 20°C"
    }
    
    location_lower = location.lower()
    if location_lower in weather_data:
        return f"Weather in {location}: {weather_data[location_lower]}"
    else:
        return f"Weather data not available for {location}. Try: New York, London, Tokyo, or Paris."


@tool
def search_knowledge(query: str) -> str:
    """Search for information on a topic.
    
    Args:
        query: The search query
    
    Returns:
        Relevant information (simulated)
    """
    # Simulated knowledge base
    knowledge = {
        "python": "Python is a high-level programming language known for its simplicity and readability.",
        "ai": "Artificial Intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence.",
        "machine learning": "Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.",
        "langchain": "LangChain is a framework for developing applications powered by language models."
    }
    
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return f"Information about {key}: {value}"
    
    return f"No specific information found for '{query}'. Try searching for: Python, AI, Machine Learning, or LangChain."


async def agent_executor_example():
    """Demonstrate AgentExecutor with tools and AI SDK streaming"""
    print("=== Agent Executor with Tools Example ===")
    
    # Setup
    api_key, base_url = setup_environment()
    llm = create_llm(api_key, base_url)
    
    # Create tools
    tools = [calculate_math, get_weather_info, search_knowledge]
    
    # Create agent prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with access to tools. Use them when needed to provide accurate information."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Create agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Test queries that should trigger tool usage
    queries = [
        "What's the weather like in Tokyo? Please use the weather tool.",
        "Calculate 15 * 7 + 23 using the math tool",
        "Search for information about machine learning using the search tool"
    ]
    
    for query in queries:
        print(f"\nUser: {query}")
        print("Assistant: ", end="")
        
        part_stats = {}
        try:
            # Use agent_executor directly with astream
            async for part in AISDKAdapter.astream(agent_executor.astream({"input": query, "chat_history": []})):
                # Parse part type and update statistics
                if ':' in part:
                    part_type = part.split(':', 1)[0]
                    part_stats[part_type] = part_stats.get(part_type, 0) + 1
                    
                    # Only show non-text protocols for diagnosis
                    if not part.startswith('0:'):
                        print(f"[Protocol {part_type}]: {part.strip()}")
                else:
                    # Show any non-protocol parts
                    if part.strip():
                        print(f"[Raw]: {part}")
        except Exception as e:
            print(f"\nError: {e}")
        
        print("\n")
        print("Query Statistics:")
        for part_type, count in sorted(part_stats.items()):
            print(f"  Type {part_type}: {count} parts")
        print("-" * 50)


if LANGGRAPH_AVAILABLE:
    # LangGraph state definition
    class GraphState(TypedDict):
        messages: List[Any]
        current_step: str
        analysis_result: str
    
    
    async def langgraph_workflow_example():
        """Demonstrate LangGraph workflow with AI SDK streaming"""
        print("\n=== LangGraph Workflow Example ===")
        
        # Setup
        api_key, base_url = setup_environment()
        llm = create_llm(api_key, base_url)
        
        # Create tools for LangGraph
        tools = [calculate_math, get_weather_info, search_knowledge]
        
        # Bind tools to LLM for function calling
        llm_with_tools = llm.bind_tools(tools)
        
        # Define workflow nodes
        def analyze_input(state: GraphState) -> GraphState:
            """Analyze the user input"""
            messages = state["messages"]
            last_message = messages[-1].content if messages else ""
            
            # Simple analysis logic
            if any(word in last_message.lower() for word in ["calculate", "math", "compute"]):
                analysis = "mathematical_query"
            elif any(word in last_message.lower() for word in ["weather", "temperature", "climate"]):
                analysis = "weather_query"
            else:
                analysis = "general_query"
            
            return {
                **state,
                "current_step": "analysis_complete",
                "analysis_result": analysis
            }
        
        def process_mathematical_query(state: GraphState) -> GraphState:
            """Process mathematical queries with tool calling"""
            messages = state["messages"]
            system_msg = SystemMessage(content="You MUST use the calculate_math tool to solve any mathematical problem. Do not solve it manually. Always call the calculate_math tool with the expression parameter.")
            new_messages = [system_msg] + messages
            
            return {
                **state,
                "messages": new_messages,
                "current_step": "math_processing"
            }
        
        def process_weather_query(state: GraphState) -> GraphState:
            """Process weather queries with tool calling"""
            messages = state["messages"]
            system_msg = SystemMessage(content="You MUST use the get_weather_info tool to get weather information. Do not provide weather information without calling the tool first. Always call get_weather_info with the location parameter.")
            new_messages = [system_msg] + messages
            
            return {
                **state,
                "messages": new_messages,
                "current_step": "weather_processing"
            }
        
        def process_general_query(state: GraphState) -> GraphState:
            """Process general queries with tool calling"""
            messages = state["messages"]
            system_msg = SystemMessage(content="You are a helpful assistant. Use the search_knowledge tool when you need to find information.")
            new_messages = [system_msg] + messages
            
            return {
                **state,
                "messages": new_messages,
                "current_step": "general_processing"
            }
        
        def route_query(state: GraphState) -> str:
            """Route to appropriate processor based on analysis"""
            analysis = state.get("analysis_result", "general_query")
            
            if analysis == "mathematical_query":
                return "math_processor"
            elif analysis == "weather_query":
                return "weather_processor"
            else:
                return "general_processor"
        
        def generate_response(state: GraphState) -> GraphState:
            """Generate final response using LLM with tools"""
            messages = state["messages"]
            
            # Generate response using the LLM with tools
            response = llm_with_tools.invoke(messages)
            updated_messages = messages + [response]
            
            # Check if the response contains tool calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # Execute tool calls and add tool messages
                for tool_call in response.tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']
                    tool_call_id = tool_call['id']
                    
                    # Find and execute the appropriate tool
                    tool_result = None
                    for tool in tools:
                        if tool.name == tool_name:
                            try:
                                tool_result = tool.invoke(tool_args)
                            except Exception as e:
                                tool_result = f"Error executing {tool_name}: {str(e)}"
                            break
                    
                    if tool_result is None:
                        tool_result = f"Tool {tool_name} not found"
                    
                    # Add tool message
                    from langchain_core.messages import ToolMessage
                    tool_message = ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call_id
                    )
                    updated_messages.append(tool_message)
                
                # Generate final response after tool execution
                final_response = llm.invoke(updated_messages)
                updated_messages.append(final_response)
            
            return {
                **state,
                "messages": updated_messages,
                "current_step": "response_generated"
            }
        
        # Build the graph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("analyzer", analyze_input)
        workflow.add_node("math_processor", process_mathematical_query)
        workflow.add_node("weather_processor", process_weather_query)
        workflow.add_node("general_processor", process_general_query)
        workflow.add_node("response_generator", generate_response)
        
        # Add edges
        workflow.set_entry_point("analyzer")
        workflow.add_conditional_edges(
            "analyzer",
            route_query,
            {
                "math_processor": "math_processor",
                "weather_processor": "weather_processor",
                "general_processor": "general_processor"
            }
        )
        
        # All processors go to response generator
        workflow.add_edge("math_processor", "response_generator")
        workflow.add_edge("weather_processor", "response_generator")
        workflow.add_edge("general_processor", "response_generator")
        workflow.add_edge("response_generator", END)
        
        # Compile the graph
        app = workflow.compile()
        
        # Test the workflow
        test_queries = [
            "What's 25 * 4 + 17?",
            "How's the weather in Paris?",
            "Tell me about artificial intelligence"
        ]
        
        for query in test_queries:
            print(f"\nUser: {query}")
            print("Processing through workflow...")
            
            # Run the workflow
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "current_step": "start",
                "analysis_result": ""
            }
            
            try:
                print("Assistant: ", end="")
                
                part_stats = {}
                # Use AISDKAdapter.astream to process the entire LangGraph workflow stream
                async for part in AISDKAdapter.astream(app.astream(initial_state)):
                    # Parse part type and update statistics
                    if ':' in part:
                        part_type = part.split(':', 1)[0]
                        part_stats[part_type] = part_stats.get(part_type, 0) + 1
                        
                        # Only show non-text protocols for diagnosis
                        if not part.startswith('0:'):
                            print(f"[Protocol {part_type}]: {part.strip()}")
                    else:
                        # Show any non-protocol parts
                        if part.strip():
                            print(f"[Raw]: {part}")
                
                print("\n")
                print("Workflow Statistics:")
                for part_type, count in sorted(part_stats.items()):
                    print(f"  Type {part_type}: {count} parts")
                
            except Exception as e:
                print(f"Error in workflow: {e}")
            
            print("-" * 50)


async def main():
    """Run all examples"""
    try:
        await agent_executor_example()
        
        if LANGGRAPH_AVAILABLE:
            await langgraph_workflow_example()
        else:
            print("\nSkipping LangGraph example (LangGraph not installed)")
            print("To run LangGraph examples, install with: pip install langgraph")
        
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("\nPlease ensure you have:")
        print("1. Copied .env.example to .env")
        print("2. Added your DeepSeek API key to the .env file")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease check your API configuration and network connection.")


if __name__ == "__main__":
    asyncio.run(main())