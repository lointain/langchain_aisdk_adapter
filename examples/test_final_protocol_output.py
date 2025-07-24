#!/usr/bin/env python3
"""
Test script to verify the final protocol output format differences between v4 and v5.
This shows the actual protocol format that gets sent over the wire.
"""

import asyncio
import os
from typing import AsyncIterable
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_aisdk_adapter import LangChainAdapter


@tool
def simple_math(expression: str) -> str:
    """Calculate a simple mathematical expression."""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error: {str(e)}"


async def test_final_protocol_formats():
    """Test the final protocol output formats for v4 and v5."""
    
    # Temporarily disable LangSmith tracing
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
    tools = [simple_math]
    
    # Create simple prompt template
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
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    
    # Simple test query
    test_query = "What is 5 + 3?"
    
    print("=" * 80)
    print("Testing FINAL PROTOCOL OUTPUT formats (v4 vs v5)")
    print("=" * 80)
    
    # Test V4 Protocol with DataStreamResponse
    print("\n🔵 V4 Protocol - Final Output Format (to_data_stream_response):")
    print("-" * 70)
    
    try:
        agent_stream_v4 = agent_executor.astream_events({"input": test_query}, version="v2")
        
        # Convert to DataStreamResponse with V4 protocol
        response_v4 = await LangChainAdapter.to_data_stream_response(
            agent_stream_v4,
            message_id="test-v4-response",
            options={"protocol_version": "v4", "auto_events": True}
        )
        
        print("V4 Final Protocol Output:")
        chunk_count = 0
        async for protocol_chunk in response_v4.body_iterator:
            chunk_count += 1
            # Decode bytes to string for display
            if isinstance(protocol_chunk, bytes):
                protocol_text = protocol_chunk.decode('utf-8')
            else:
                protocol_text = str(protocol_chunk)
            print(f"V4 Protocol Line {chunk_count}: {repr(protocol_text)}")
            if chunk_count >= 8:  # Limit output for readability
                print("... (truncated for readability)")
                break
                
    except Exception as e:
        print(f"Error in V4 test: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test V5 Protocol with DataStreamResponse
    print("\n🟢 V5 Protocol - Final Output Format (to_data_stream_response):")
    print("-" * 70)
    
    try:
        agent_stream_v5 = agent_executor.astream_events({"input": test_query}, version="v2")
        
        # Convert to DataStreamResponse with V5 protocol
        response_v5 = await LangChainAdapter.to_data_stream_response(
            agent_stream_v5,
            message_id="test-v5-response",
            options={"protocol_version": "v5", "auto_events": True}
        )
        
        print("V5 Final Protocol Output:")
        chunk_count = 0
        async for protocol_chunk in response_v5.body_iterator:
            chunk_count += 1
            # Decode bytes to string for display
            if isinstance(protocol_chunk, bytes):
                protocol_text = protocol_chunk.decode('utf-8')
            else:
                protocol_text = str(protocol_chunk)
            print(f"V5 Protocol Line {chunk_count}: {repr(protocol_text)}")
            if chunk_count >= 8:  # Limit output for readability
                print("... (truncated for readability)")
                break
                
    except Exception as e:
        print(f"Error in V5 test: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test Default Protocol
    print("\n🟡 Default Protocol - Final Output Format (to_data_stream_response):")
    print("-" * 70)
    
    try:
        agent_stream_default = agent_executor.astream_events({"input": test_query}, version="v2")
        
        # Convert to DataStreamResponse with default protocol
        response_default = await LangChainAdapter.to_data_stream_response(
            agent_stream_default,
            message_id="test-default-response",
            options={"auto_events": True}  # No protocol_version specified
        )
        
        print("Default Final Protocol Output:")
        chunk_count = 0
        async for protocol_chunk in response_default.body_iterator:
            chunk_count += 1
            # Decode bytes to string for display
            if isinstance(protocol_chunk, bytes):
                protocol_text = protocol_chunk.decode('utf-8')
            else:
                protocol_text = str(protocol_chunk)
            print(f"Default Protocol Line {chunk_count}: {repr(protocol_text)}")
            if chunk_count >= 8:  # Limit output for readability
                print("... (truncated for readability)")
                break
                
    except Exception as e:
        print(f"Error in default test: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("PROTOCOL FORMAT ANALYSIS:")
    print("• V4 Protocol: Uses '0:' and 'd:' prefixes (custom format)")
    print("• V5 Protocol: Uses 'data:' prefix (Server-Sent Events format)")
    print("• to_data_stream_response() converts UIMessageChunk to final protocol")
    print("• to_data_stream() returns UIMessageChunk objects (not final protocol)")
    print("=" * 80)


async def main():
    try:
        await test_final_protocol_formats()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())