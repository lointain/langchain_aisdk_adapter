import asyncio
import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import tool

@tool
def get_weather(input: str) -> str:
    """Get weather information for a city."""
    return f"The weather in {input} is sunny with 22°C temperature."

@tool
def calculate_math(input: str) -> str:
    """Calculate mathematical expressions."""
    try:
        result = eval(input)
        return f"The result of {input} is {result}"
    except Exception as e:
        return f"Error calculating {input}: {str(e)}"

async def debug_langchain_chunks():
    """Debug LangChain chunk content to understand text accumulation."""
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
    
    # Test query
    test_query = "Please check Beijing's weather and calculate 15 * 24"
    
    print("=== Debugging LangChain astream_events ===\n")
    
    accumulated_text = ""
    event_count = 0
    
    try:
        # Create agent stream using astream_events
        agent_stream = agent_executor.astream_events({"input": test_query}, version="v2")
        
        async for event in agent_stream:
            event_count += 1
            event_type = event.get("event", "")
            data = event.get("data", {})
            
            print(f"Event {event_count}: {event_type}")
            
            if event_type == "on_chat_model_stream":
                chunk_data = data.get("chunk")
                if chunk_data:
                    # Extract text from chunk
                    if isinstance(chunk_data, dict):
                        content = chunk_data.get("content", "")
                    else:
                        content = getattr(chunk_data, "content", "")
                    
                    if isinstance(content, str) and content:
                        print(f"  Chunk content: {repr(content)}")
                        print(f"  Previous accumulated: {repr(accumulated_text)}")
                        
                        # Calculate delta
                        if len(content) > len(accumulated_text):
                            delta = content[len(accumulated_text):]
                            print(f"  Calculated delta: {repr(delta)}")
                            accumulated_text = content
                            print(f"  New accumulated: {repr(accumulated_text)}")
                        else:
                            print(f"  No delta (content length {len(content)} <= accumulated length {len(accumulated_text)})")
                        print()
            
            elif event_type == "on_chat_model_start":
                print("  Resetting accumulated text")
                accumulated_text = ""
                print()
            
            elif event_type == "on_chat_model_end":
                print(f"  Final accumulated text: {repr(accumulated_text)}")
                print(f"  Event data: {data}")
                print()
                
    except Exception as e:
        print(f"Error during streaming: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_langchain_chunks())