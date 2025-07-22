import asyncio
from langchain_aisdk_adapter import to_data_stream
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os

# Set up API key
os.environ["OPENAI_API_KEY"] = "sk-xxx"  # Placeholder
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"

@tool
def simple_tool(text: str) -> str:
    """A simple tool that processes text."""
    return f"Processed: {text}"

async def main():
    try:
        # Create LLM
        llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0,
            streaming=True
        )
        
        # Create prompt
        prompt = PromptTemplate.from_template(
            "Answer the following questions as best you can. You have access to the following tools:\n\n"
            "{tools}\n\n"
            "Use the following format:\n\n"
            "Question: the input question you must answer\n"
            "Thought: you should always think about what to do\n"
            "Action: the action to take, should be one of [{tool_names}]\n"
            "Action Input: the input to the action\n"
            "Observation: the result of the action\n"
            "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
            "Thought: I now know the final answer\n"
            "Final Answer: the final answer to the original input question\n\n"
            "Begin!\n\n"
            "Question: {input}\n"
            "Thought: {agent_scratchpad}"
        )
        
        # Create agent
        agent = create_react_agent(llm, [simple_tool], prompt)
        agent_executor = AgentExecutor(agent=agent, tools=[simple_tool], verbose=True)
        
        # Stream the agent execution
        agent_stream = agent_executor.astream({"input": "Process the text 'hello world'"})
        
        # Convert to AI SDK stream
        ai_sdk_stream = to_data_stream(agent_stream)
        
        # Collect all events
        all_events = []
        async for chunk in ai_sdk_stream:
            all_events.append(chunk)
            print(f"Event: {chunk}")
        
        # Check for tool events
        tool_input_events = [e for e in all_events if e.get('type') == 'tool-input-start']
        tool_output_events = [e for e in all_events if e.get('type') == 'tool-output-available']
        
        print(f"\nSummary:")
        print(f"Tool input events: {len(tool_input_events)}")
        print(f"Tool output events: {len(tool_output_events)}")
        
        if tool_input_events:
            print(f"First tool input: {tool_input_events[0]}")
        if tool_output_events:
            print(f"First tool output: {tool_output_events[0]}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())