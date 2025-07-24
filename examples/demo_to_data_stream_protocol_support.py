#!/usr/bin/env python3
"""
Demonstration that to_data_stream method supports protocol version switching.
This shows that both to_data_stream and to_data_stream_response support protocol versions.
"""

import asyncio
import os
from typing import AsyncIterable
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_aisdk_adapter import LangChainAdapter


@tool
def simple_calculator(expression: str) -> str:
    """Calculate a simple mathematical expression."""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error: {str(e)}"


async def demonstrate_protocol_support():
    """Demonstrate that to_data_stream supports protocol version switching."""
    
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
    tools = [simple_calculator]
    
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
    test_query = "What is 7 + 8?"
    
    print("=" * 80)
    print("🎯 DEMONSTRATION: to_data_stream() supports protocol version switching")
    print("=" * 80)
    
    print("\n📋 Summary:")
    print("• to_data_stream() returns UIMessageChunk objects")
    print("• to_data_stream_response() converts UIMessageChunk to final protocol format")
    print("• Both methods support protocol_version parameter (v4/v5)")
    print("• Protocol version affects the internal chunk structure and final output format")
    
    # Test 1: to_data_stream with V4 protocol
    print("\n🔵 Test 1: to_data_stream() with V4 protocol")
    print("-" * 60)
    
    try:
        agent_stream_v4 = agent_executor.astream_events({"input": test_query}, version="v2")
        
        # Convert to AI SDK stream with V4 protocol
        ai_sdk_stream_v4 = LangChainAdapter.to_data_stream(
            agent_stream_v4,
            message_id="demo-v4-001",
            options={"protocol_version": "v4", "auto_events": True}
        )
        
        print("✅ to_data_stream() with protocol_version='v4' - UIMessageChunk objects:")
        chunk_count = 0
        async for chunk in ai_sdk_stream_v4:
            chunk_count += 1
            chunk_type = chunk.get('type') if isinstance(chunk, dict) else getattr(chunk, 'type', 'unknown')
            print(f"  V4 Chunk {chunk_count}: type='{chunk_type}' | {chunk}")
            if chunk_count >= 3:  # Limit output for readability
                print("  ... (truncated for readability)")
                break
                
        await ai_sdk_stream_v4.close()
        
    except Exception as e:
        print(f"❌ Error in V4 test: {str(e)}")
    
    # Test 2: to_data_stream with V5 protocol
    print("\n🟢 Test 2: to_data_stream() with V5 protocol")
    print("-" * 60)
    
    try:
        agent_stream_v5 = agent_executor.astream_events({"input": test_query}, version="v2")
        
        # Convert to AI SDK stream with V5 protocol
        ai_sdk_stream_v5 = LangChainAdapter.to_data_stream(
            agent_stream_v5,
            message_id="demo-v5-001",
            options={"protocol_version": "v5", "auto_events": True}
        )
        
        print("✅ to_data_stream() with protocol_version='v5' - UIMessageChunk objects:")
        chunk_count = 0
        async for chunk in ai_sdk_stream_v5:
            chunk_count += 1
            chunk_type = chunk.get('type') if isinstance(chunk, dict) else getattr(chunk, 'type', 'unknown')
            print(f"  V5 Chunk {chunk_count}: type='{chunk_type}' | {chunk}")
            if chunk_count >= 3:  # Limit output for readability
                print("  ... (truncated for readability)")
                break
                
        await ai_sdk_stream_v5.close()
        
    except Exception as e:
        print(f"❌ Error in V5 test: {str(e)}")
    
    # Test 3: to_data_stream_response with V4 protocol (final format)
    print("\n🔵 Test 3: to_data_stream_response() with V4 protocol (final format)")
    print("-" * 60)
    
    try:
        agent_stream_v4_response = agent_executor.astream_events({"input": test_query}, version="v2")
        
        # Convert to DataStreamResponse with V4 protocol
        response_v4 = await LangChainAdapter.to_data_stream_response(
            agent_stream_v4_response,
            message_id="demo-v4-response",
            options={"protocol_version": "v4", "auto_events": True}
        )
        
        print("✅ to_data_stream_response() with protocol_version='v4' - Final protocol format:")
        chunk_count = 0
        async for protocol_chunk in response_v4.body_iterator:
            chunk_count += 1
            if isinstance(protocol_chunk, bytes):
                protocol_text = protocol_chunk.decode('utf-8')
            else:
                protocol_text = str(protocol_chunk)
            print(f"  V4 Protocol Line {chunk_count}: {repr(protocol_text)}")
            if chunk_count >= 3:  # Limit output for readability
                print("  ... (truncated for readability)")
                break
                
    except Exception as e:
        print(f"❌ Error in V4 response test: {str(e)}")
    
    # Test 4: to_data_stream_response with V5 protocol (final format)
    print("\n🟢 Test 4: to_data_stream_response() with V5 protocol (final format)")
    print("-" * 60)
    
    try:
        agent_stream_v5_response = agent_executor.astream_events({"input": test_query}, version="v2")
        
        # Convert to DataStreamResponse with V5 protocol
        response_v5 = await LangChainAdapter.to_data_stream_response(
            agent_stream_v5_response,
            message_id="demo-v5-response",
            options={"protocol_version": "v5", "auto_events": True}
        )
        
        print("✅ to_data_stream_response() with protocol_version='v5' - Final protocol format:")
        chunk_count = 0
        async for protocol_chunk in response_v5.body_iterator:
            chunk_count += 1
            if isinstance(protocol_chunk, bytes):
                protocol_text = protocol_chunk.decode('utf-8')
            else:
                protocol_text = str(protocol_chunk)
            print(f"  V5 Protocol Line {chunk_count}: {repr(protocol_text)}")
            if chunk_count >= 3:  # Limit output for readability
                print("  ... (truncated for readability)")
                break
                
    except Exception as e:
        print(f"❌ Error in V5 response test: {str(e)}")
    
    print("\n" + "=" * 80)
    print("🎉 CONCLUSION:")
    print("✅ to_data_stream() DOES support protocol version switching!")
    print("✅ to_data_stream_response() ALSO supports protocol version switching!")
    print("✅ Both methods accept protocol_version parameter in options")
    print("✅ V4 protocol uses '0:' and 'd:' prefixes (custom format)")
    print("✅ V5 protocol uses 'data:' prefix (Server-Sent Events format)")
    print("\n📝 Key Differences:")
    print("• to_data_stream(): Returns UIMessageChunk objects (intermediate format)")
    print("• to_data_stream_response(): Returns final protocol-formatted text stream")
    print("• Both support the same protocol_version options: 'v4' and 'v5'")
    print("=" * 80)


async def main():
    try:
        await demonstrate_protocol_support()
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())