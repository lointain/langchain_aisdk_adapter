#!/usr/bin/env python3
"""
Simple test script to check protocol output order without real API calls
"""

import asyncio
import os
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_aisdk_adapter import LangChainAdapter

# Mock stream that simulates agent events
async def mock_agent_stream():
    """Mock agent stream that simulates the events we expect"""
    
    # Simulate agent executor start
    yield {
        "event": "on_chain_start",
        "name": "AgentExecutor",
        "run_id": "test-run-1",
        "data": {"input": {"input": "test query"}}
    }
    
    # Simulate LLM start
    yield {
        "event": "on_llm_start",
        "name": "ChatOpenAI",
        "run_id": "test-llm-1",
        "data": {"input": {"messages": [["human", "test query"]]}}
    }
    
    # Simulate LLM stream chunks
    yield {
        "event": "on_llm_stream",
        "name": "ChatOpenAI",
        "run_id": "test-llm-1",
        "data": {"chunk": AIMessage(content="I need to")}
    }
    
    yield {
        "event": "on_llm_stream",
        "name": "ChatOpenAI",
        "run_id": "test-llm-1",
        "data": {"chunk": AIMessage(content=" use tools")}
    }
    
    # Simulate LLM end
    yield {
        "event": "on_llm_end",
        "name": "ChatOpenAI",
        "run_id": "test-llm-1",
        "data": {"output": AIMessage(content="I need to use tools")}
    }
    
    # Simulate tool start
    yield {
        "event": "on_tool_start",
        "name": "get_weather",
        "run_id": "test-tool-1",
        "data": {"input": "Beijing"}
    }
    
    # Simulate tool end
    yield {
        "event": "on_tool_end",
        "name": "get_weather",
        "run_id": "test-tool-1",
        "data": {"output": "The weather in Beijing is sunny with 22°C temperature."}
    }
    
    # Simulate agent action
    yield {
        "event": "on_agent_action",
        "name": "AgentExecutor",
        "run_id": "test-run-1",
        "data": {
            "output": {
                "tool": "get_weather",
                "tool_input": "Beijing",
                "log": "Action: get_weather\nAction Input: Beijing"
            }
        }
    }
    
    # Simulate final LLM start for conclusion
    yield {
        "event": "on_llm_start",
        "name": "ChatOpenAI",
        "run_id": "test-llm-2",
        "data": {"input": {"messages": [["human", "test query"]]}}
    }
    
    # Simulate final LLM stream
    yield {
        "event": "on_llm_stream",
        "name": "ChatOpenAI",
        "run_id": "test-llm-2",
        "data": {"chunk": AIMessage(content="Final answer: Weather is sunny")}
    }
    
    # Simulate final LLM end
    yield {
        "event": "on_llm_end",
        "name": "ChatOpenAI",
        "run_id": "test-llm-2",
        "data": {"output": AIMessage(content="Final answer: Weather is sunny")}
    }
    
    # Simulate agent finish
    yield {
        "event": "on_agent_finish",
        "name": "AgentExecutor",
        "run_id": "test-run-1",
        "data": {
            "output": {
                "output": "Final answer: Weather is sunny"
            }
        }
    }
    
    # Simulate chain end
    yield {
        "event": "on_chain_end",
        "name": "AgentExecutor",
        "run_id": "test-run-1",
        "data": {"output": {"output": "Final answer: Weather is sunny"}}
    }

async def test_protocol_order():
    """Test protocol output order"""
    
    # Create adapter
    adapter = LangChainAdapter()
    
    print("Protocol Output:")
    print("=" * 50)
    
    try:
        # Convert mock stream to AI SDK stream
        ai_sdk_stream = adapter.to_data_stream_response(mock_agent_stream())
        
        # Stream the response and analyze protocols
        async for chunk in ai_sdk_stream:
            protocol_line = chunk.strip()
            if protocol_line:
                # Extract protocol type
                if protocol_line.startswith('f:'):
                    print(f"STEP_START: {protocol_line}")
                elif protocol_line.startswith('0:'):
                    print(f"TEXT: {protocol_line[:50]}...")
                elif protocol_line.startswith('b:'):
                    print(f"TOOL_CALL_START: {protocol_line}")
                elif protocol_line.startswith('9:'):
                    print(f"TOOL_CALL_COMPLETE: {protocol_line}")
                elif protocol_line.startswith('a:'):
                    print(f"TOOL_RESULT: {protocol_line}")
                elif protocol_line.startswith('e:'):
                    print(f"STEP_END: {protocol_line}")
                elif protocol_line.startswith('d:'):
                    print(f"STREAM_END: {protocol_line}")
                else:
                    print(f"OTHER: {protocol_line[:50]}...")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_protocol_order())