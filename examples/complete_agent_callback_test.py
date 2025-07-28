#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Agent Callback Test with Real DeepSeek API

This example demonstrates a complete agent workflow with:
- Real DeepSeek API calls (not mocked)
- Tool calls and agent execution
- AI SDK compatible callbacks
- step-start and tool-invocation protocols
"""

import asyncio
import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Please install it with: uv add python-dotenv")
    print("Falling back to manual environment variable reading.")

from langchain_aisdk_adapter import LangChainAdapter,DataStreamContext, BaseAICallbackHandler, Message


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.
    
    Args:
        city: The name of the city to get weather for
        
    Returns:
        Weather information for the city
    """
    return f"The weather in {city} is sunny with 22°C temperature."


@tool
def calculate_math(expression: str) -> str:
    """Calculate a mathematical expression safely.
    
    Args:
        expression: Mathematical expression to calculate (e.g., "2+2", "10*5")
        
    Returns:
        The result of the calculation
    """
    try:
        # Simple safe evaluation for basic math
        allowed_chars = set('0123456789+-*/.() ')
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"The result of {expression} is {result}"
        else:
            return "Invalid mathematical expression"
    except Exception as e:
        return f"Error calculating: {str(e)}"


class TestCallbackHandler(BaseAICallbackHandler):
    """Test callback handler for demonstrating stream processing"""
    
    async def on_start(self) -> None:
        """Called when the stream starts"""
        print("\n" + "="*50)
        print("STREAM STARTED")
        print("="*50)
    
    async def on_finish(self, message: Message, options: dict) -> None:
        """Called when the stream finishes"""
        print("\n" + "="*50)
        print("STREAM COMPLETED")
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
        print("\n--- End of Stream ---")


def create_test_callbacks():
    """Create test callbacks for demonstrating stream processing"""
    return TestCallbackHandler()


async def test_complete_agent_workflow():
    """Main test function"""
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
    tools = [get_weather, calculate_math]
    
    # Create ReAct prompt template
    from langchain import hub
    try:
        prompt = hub.pull("hwchase17/react")
    except:
        # Fallback prompt if hub is not available
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
    
    # Initialize callback handler
    callbacks = create_test_callbacks()
    
    # Test query that should trigger tool usage
    test_query = "Please use the get_weather tool to check Beijing's weather and use calculate_math tool to compute 15 * 24"
    
    try:
        # Create agent stream using astream_events
        agent_stream = agent_executor.astream_events({"input": test_query}, version="v2")
        
        # Convert to AI SDK stream using LangChainAdapter.to_data_stream
        ai_sdk_stream = LangChainAdapter.to_data_stream(
            agent_stream,
            callbacks=callbacks,
            message_id="test-message-001",
            options={"auto_events": True, "protocol_version": "v5"}
        )
        
        # Test DataStreamContext.emit_file with v4 protocol (requires data and mime_type)
        # await DataStreamContext.emit_file(data="report.pdf", mime_type="application/pdf")
        await DataStreamContext.emit_source_url(
            url="https://example.com/docs",
            description="Example documentation"
        )
        
        # Test various emit methods directly on the stream with proper sequence
        print("\nTesting manual emit methods with proper sequence:")
        
        # # 1. Start message
        # await ai_sdk_stream.emit_start()
        # print("✓ Emitted start")
        
        # # 2. Text sequence: start -> delta -> end
        # text_id = "text-001"
        # await ai_sdk_stream.emit_text_start(text_id=text_id)
        # print("✓ Emitted text-start")
        
        # await ai_sdk_stream.emit_text_delta(delta="Hello ", text_id=text_id)
        # await ai_sdk_stream.emit_text_delta(delta="world!", text_id=text_id)
        # print("✓ Emitted text-delta chunks")
        
        # await ai_sdk_stream.emit_text_end(text="Hello world!", text_id=text_id)
        # print("✓ Emitted text-end")
        
        # # 3. Tool sequence: start -> delta -> available -> output
        # tool_call_id = "tool-001"
        # await ai_sdk_stream.emit_tool_input_start(tool_call_id=tool_call_id, tool_name="get_weather")
        # print("✓ Emitted tool-input-start")
        
        # await ai_sdk_stream.emit_tool_input_delta(tool_call_id=tool_call_id, delta='{"city": "Beijing"}')
        # print("✓ Emitted tool-input-delta")
        
        # await ai_sdk_stream.emit_tool_input_available(
        #     tool_call_id=tool_call_id, 
        #     tool_name="get_weather", 
        #     input_data={"city": "Beijing"}
        # )
        # print("✓ Emitted tool-input-available")
        
        # await ai_sdk_stream.emit_tool_output_available(
        #     tool_call_id=tool_call_id, 
        #     output="The weather in Beijing is sunny with 22°C temperature."
        # )
        # print("✓ Emitted tool-output-available")
        
        # # 4. Step sequence
        # await ai_sdk_stream.emit_start_step(step_type="reasoning")
        # print("✓ Emitted start-step")
        
        # await ai_sdk_stream.emit_finish_step(step_type="reasoning")
        # print("✓ Emitted finish-step")
        
        # # 5. File attachments
        # await ai_sdk_stream.emit_file(url="report.pdf", mediaType="application/pdf")
        # print("✓ Emitted file: report.pdf")
        
        # await ai_sdk_stream.emit_file(url="data.json", mediaType="application/json")
        # print("✓ Emitted file: data.json")
        
        # # 6. Source documents
        # await ai_sdk_stream.emit_source_document(
        #     source_id="doc-001",
        #     media_type="text/plain",
        #     title="Example Document",
        #     filename="example.txt"
        # )
        # print("✓ Emitted source-document")
        
        # # 7. Custom data
        # await ai_sdk_stream.emit_data(data={"key": "value", "number": 42}, data_type="data-custom")
        # print("✓ Emitted data chunk")
        
        # # 8. Error handling
        # await ai_sdk_stream.emit_error(error_text="This is a test error message.")
        # print("✓ Emitted error")
        
        # # 9. Finish message
        # await ai_sdk_stream.emit_finish()
        # print("✓ Emitted finish")
        
        # print("All manual emit tests completed with proper sequence.\n")
        
        # Stream the response and show AI SDK protocols
        print("\nAI SDK Protocol Output:")
        print("-" * 40)
        async for chunk in ai_sdk_stream:
            print(f"Protocol: {chunk}")
        print("-" * 40)
        
        # Explicitly close the stream to trigger on_finish callback
        await ai_sdk_stream.close()
        print("Stream closed explicitly")
        
    except Exception as e:
        print(f"Error during streaming: {str(e)}")

async def main():
    try:
        await test_complete_agent_workflow()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())