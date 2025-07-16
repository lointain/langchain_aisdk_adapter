"""Simple LangGraph Protocol Test Example

A minimal LangGraph workflow for testing AI SDK protocol generation.
Focuses on protocol output rather than conversation quality.
"""

import asyncio
import os
from typing import Dict, Any, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

try:
    from langgraph.graph import StateGraph, END
    from typing_extensions import TypedDict
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("Warning: LangGraph not available. Install with: pip install langgraph")

from langchain_aisdk_adapter import LangChainAdapter


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
        temperature=0.1,
        max_tokens=500,
        streaming=True
    )


# Mock internal knowledge base tool
@tool
def query_internal_docs(query: str) -> str:
    """Query internal documentation database.
    
    Args:
        query: Search query for internal docs
    
    Returns:
        Relevant documentation content
    """
    # Mock internal documentation data
    docs_db = {
        "api": "Internal API Documentation: Our REST API supports GET, POST, PUT, DELETE methods. Base URL: https://api.internal.com",
        "auth": "Authentication Guide: Use Bearer tokens in Authorization header. Tokens expire after 24 hours.",
        "database": "Database Schema: Users table has id, name, email, created_at columns. Products table has id, name, price, category.",
        "deployment": "Deployment Process: 1. Run tests 2. Build Docker image 3. Deploy to staging 4. Run integration tests 5. Deploy to production",
        "security": "Security Guidelines: Always validate input, use HTTPS, implement rate limiting, log security events."
    }
    
    query_lower = query.lower()
    for key, content in docs_db.items():
        if key in query_lower:
            return f"Found documentation for '{key}': {content}"
    
    return f"No internal documentation found for query: '{query}'. Available topics: api, auth, database, deployment, security"


if LANGGRAPH_AVAILABLE:
    # Simple state definition
    class WorkflowState(TypedDict):
        messages: List[Any]
        query: str
        docs_result: str
        agent_result: str
    
    
    async def simple_protocol_test():
        """Test LangGraph workflow with ReAct agent and protocol output"""
        print("=== Simple LangGraph Protocol Test ===")
        print("Focus: AI SDK Protocol Generation\n")
        
        # Setup
        api_key, base_url = setup_environment()
        llm = create_llm(api_key, base_url)
        
        # Create tools for ReAct agent
        tools = [query_internal_docs]
        
        # Create ReAct agent prompt
        react_prompt = PromptTemplate.from_template(
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
        
        # Create ReAct agent
        react_agent = create_react_agent(llm, tools, react_prompt)
        agent_executor = AgentExecutor(
            agent=react_agent, 
            tools=tools, 
            verbose=False,
            max_iterations=3,
            handle_parsing_errors=True
        )
        
        # Define workflow nodes
        def docs_lookup_node(state: WorkflowState) -> WorkflowState:
            """Node 1: Direct documentation lookup"""
            query = state["query"]
            result = query_internal_docs.invoke({"query": query})
            
            return {
                **state,
                "docs_result": result
            }
        
        def react_agent_node(state: WorkflowState) -> WorkflowState:
            """Node 2: ReAct agent processing with tools"""
            query = state["query"]
            docs_context = state["docs_result"]
            
            # Enhanced query with context
            enhanced_query = f"Based on this context: {docs_context}\n\nQuestion: {query}\n\nPlease use the query_internal_docs tool to find additional relevant information and provide a comprehensive answer."
            
            # Run ReAct agent
            result = agent_executor.invoke({"input": enhanced_query})
            
            return {
                **state,
                "agent_result": result.get("output", "No result from agent")
            }
        
        # Build the graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("docs_lookup", docs_lookup_node)
        workflow.add_node("react_agent", react_agent_node)
        
        # Add edges
        workflow.set_entry_point("docs_lookup")
        workflow.add_edge("docs_lookup", "react_agent")
        workflow.add_edge("react_agent", END)
        
        # Compile the graph
        app = workflow.compile()
        
        # Test query
        test_query = "How do I authenticate with our internal API?"
        
        print(f"Query: {test_query}")
        print("\n=== AI SDK Protocol Stream ===")
        
        # Initial state
        initial_state = {
            "messages": [HumanMessage(content=test_query)],
            "query": test_query,
            "docs_result": "",
            "agent_result": ""
        }
        
        # Track protocol statistics
        protocol_stats = {}
        current_text_buffer = ""
        text_chunk_count = 0
        
        try:
            # Use astream_events for proper streaming
            async for chunk in LangChainAdapter.to_data_stream_response(app.astream_events(initial_state, version="v1")):
                if isinstance(chunk, str) and ':' in chunk:
                    protocol_type = chunk.split(':', 1)[0]
                    protocol_stats[protocol_type] = protocol_stats.get(protocol_type, 0) + 1
                    
                    if chunk.startswith('0:'):
                        # Accumulate text content but only show summary
                        text_content = chunk[2:].strip('"\n')
                        current_text_buffer += text_content
                        text_chunk_count += 1
                    else:
                        # Output accumulated text summary if any
                        if current_text_buffer:
                            print(f"0: [Text content accumulated from {text_chunk_count} chunks, total length: {len(current_text_buffer)} chars]")
                            current_text_buffer = ""
                            text_chunk_count = 0
                        
                        # Output non-text protocol
                        print(f"{protocol_type}: {chunk[len(protocol_type)+1:].strip()}")
                else:
                    # Handle non-protocol chunks
                    if chunk.strip():
                        print(f"RAW: {repr(chunk)}")
            
            # Output any remaining text buffer summary
            if current_text_buffer:
                print(f"0: [Final text content from {text_chunk_count} chunks, total length: {len(current_text_buffer)} chars]")
            
        except Exception as e:
            print(f"Error in workflow: {e}")
        
        print("\n=== Protocol Statistics ===")
        for protocol_type, count in sorted(protocol_stats.items()):
            print(f"  {protocol_type}: {count} occurrences")
        
        print("\n=== Test Complete ===")


async def main():
    """Run the protocol test"""
    if not LANGGRAPH_AVAILABLE:
        print("LangGraph not available. Please install with: pip install langgraph")
        return
    
    try:
        await simple_protocol_test()
        
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