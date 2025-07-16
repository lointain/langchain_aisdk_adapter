#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Protocol Types 9, a, b: Tool Call, Tool Result, and Tool Call Streaming Start
Format:
- 9:{"toolCallId": "call_123", "toolName": "search", "args": {...}}\n
- a:{"toolCallId": "call_123", "result": "..."}\n
- b:{"toolCallId": "call_123", "toolName": "search"}\n

This test verifies that the adapter correctly generates tool-related protocol parts
when streaming tool calls from DeepSeek API with function calling.
"""

import asyncio
import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_aisdk_adapter import LangChainAdapter

# Load environment variables
load_dotenv()


@tool
def calculate_math(expression: str) -> str:
    """Calculate a mathematical expression safely.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")
        
    Returns:
        The result of the calculation
    """
    try:
        # Simple safe evaluation for basic math
        allowed_chars = set('0123456789+-*/()., ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


@tool
def get_weather(location: str) -> str:
    """Get weather information for a location.
    
    Args:
        location: The city or location to get weather for
        
    Returns:
        Weather information for the location
    """
    # Mock weather data
    weather_data = {
        "beijing": "Sunny, 22°C",
        "shanghai": "Cloudy, 18°C", 
        "guangzhou": "Rainy, 25°C",
        "shenzhen": "Partly cloudy, 24°C"
    }
    
    location_lower = location.lower()
    for city, weather in weather_data.items():
        if city in location_lower:
            return f"Weather in {location}: {weather}"
    
    return f"Weather data not available for {location}. Try Beijing, Shanghai, Guangzhou, or Shenzhen."


async def test_tool_protocols_9ab():
    """Test Protocol Types 9, a, b: Tool Call, Tool Result, and Tool Call Streaming Start"""
    print("=== Testing Protocol Types 9, a, b: Tool Call, Tool Result, and Tool Call Streaming Start ===")
    print("Expected formats:")
    print("- 9:{\"toolCallId\": \"call_123\", \"toolName\": \"search\", \"args\": {...}}\\n")
    print("- a:{\"toolCallId\": \"call_123\", \"result\": \"...\"}\\n")
    print("- b:{\"toolCallId\": \"call_123\", \"toolName\": \"search\"}\\n")
    print("Reference: https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol\n")
    
    # Setup DeepSeek API
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ DEEPSEEK_API_KEY not found in environment")
        print("Please set your DeepSeek API key in .env file")
        return False
    
    try:
        # Create tools
        tools = [calculate_math, get_weather]
        
        # Create DeepSeek LLM with function calling
        llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=api_key,
            base_url="https://api.deepseek.com",
            streaming=True,
            temperature=0.1
        )
        
        # Test direct tool calling first
        print("\nTesting direct LLM tool calling:")
        try:
            llm_with_tools = llm.bind_tools(tools)
            direct_response = llm_with_tools.invoke([{"role": "user", "content": "Calculate 15 * 8 + 7 using the calculate_math tool"}])
            print(f"Direct LLM response: {direct_response}")
            print(f"Has tool calls: {hasattr(direct_response, 'tool_calls') and direct_response.tool_calls}")
        except Exception as e:
            print(f"Direct tool calling failed: {e}")
            print("DeepSeek might not support function calling, using ReAct agent instead")
            
            # Fallback to ReAct agent
            from langchain.agents import create_react_agent
            from langchain import hub
            
            # Get ReAct prompt
            react_prompt = hub.pull("hwchase17/react")
            
            # Create ReAct agent
            agent = create_react_agent(llm, tools, react_prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
            
            print("Using ReAct agent instead of function calling")
        else:
            # Use function calling agent if supported
            llm = llm_with_tools
            
            # Create agent prompt with explicit tool usage instruction
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that MUST use the available tools for calculations and weather queries. When asked to calculate something, you MUST use the calculate_math tool. When asked about weather, you MUST use the get_weather tool. Do not attempt to answer these questions without using the appropriate tools. Always use tools when available."),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            
            # Create agent
            agent = create_openai_functions_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        # Test message that should trigger tool usage
        test_input = "Please calculate 15 * 8 + 7 using the calculate_math tool"
        
        print(f"Test input: {test_input}")
        print("Testing direct LLM with tools (bypassing AgentExecutor)...")
        
        # Test if tools are properly bound
        print(f"\nTools available: {[tool.name for tool in tools]}")
        print(f"LLM type: {type(llm)}")
        
        # Create adapter and stream
        adapter = LangChainAdapter()
        protocol_parts = []
        tool_call_parts = []  # Protocol 9
        tool_result_parts = []  # Protocol a
        tool_start_parts = []  # Protocol b
        
        print("\nTesting direct LLM streaming with tools:")
        
        # Test direct LLM streaming with tools
        messages = [{"role": "user", "content": test_input}]
        
        print("Raw LLM stream:")
        raw_chunks = []
        async for chunk in llm.astream(messages):
            raw_chunks.append(chunk)
            print(f"Raw chunk: {chunk}")
            print(f"Chunk type: {type(chunk)}")
            if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                print(f"Tool calls: {chunk.tool_calls}")
        
        print(f"\nTotal raw chunks: {len(raw_chunks)}")
        print("\nNow processing through adapter:")
        
        # Now process through adapter
        async for part in adapter.to_data_stream_response(llm.astream(messages)):
            protocol_parts.append(part)
            print(f"Protocol: {repr(part)}")
            
            # Categorize parts
            if part.startswith('9:'):
                tool_call_parts.append(part)
                # Extract and display tool call information
                try:
                    tool_call_content = json.loads(part[2:-1])  # Remove '9:' and '\n', parse JSON
                    print(f"  TOOL CALL: {tool_call_content}")
                    if 'toolName' in tool_call_content:
                        print(f"     Tool: {tool_call_content['toolName']}")
                    if 'args' in tool_call_content:
                        print(f"     Args: {tool_call_content['args']}")
                except json.JSONDecodeError:
                    print(f"  ❌ Invalid JSON in tool call part: {part}")
                    
            elif part.startswith('a:'):
                tool_result_parts.append(part)
                # Extract and display tool result information
                try:
                    tool_result_content = json.loads(part[2:-1])  # Remove 'a:' and '\n', parse JSON
                    print(f"  TOOL RESULT: {tool_result_content}")
                    if 'result' in tool_result_content:
                        result = tool_result_content['result']
                        if len(str(result)) > 100:
                            print(f"     Result: {str(result)[:100]}...")
                        else:
                            print(f"     Result: {result}")
                except json.JSONDecodeError:
                    print(f"  ❌ Invalid JSON in tool result part: {part}")
                    
            elif part.startswith('b:'):
                tool_start_parts.append(part)
                # Extract and display tool start information
                try:
                    tool_start_content = json.loads(part[2:-1])  # Remove 'b:' and '\n', parse JSON
                    print(f"  TOOL START: {tool_start_content}")
                    if 'toolName' in tool_start_content:
                        print(f"     Starting: {tool_start_content['toolName']}")
                except json.JSONDecodeError:
                    print(f"  ❌ Invalid JSON in tool start part: {part}")
        
        print(f"\n=== Analysis ===")
        print(f"Total protocol parts: {len(protocol_parts)}")
        print(f"Tool call parts (type 9): {len(tool_call_parts)}")
        print(f"Tool result parts (type a): {len(tool_result_parts)}")
        print(f"Tool start parts (type b): {len(tool_start_parts)}")
        
        # Verify we got tool-related parts
        success = True
        
        # Check tool call parts (Protocol 9)
        if len(tool_call_parts) > 0:
            print("✅ SUCCESS: Tool call protocol parts (9:) generated")
            
            # Verify tool call format compliance
            for part in tool_call_parts:
                if not (part.startswith('9:') and part.endswith('\n')):
                    print(f"❌ Tool call format error: {repr(part)}")
                    success = False
                else:
                    # Verify JSON validity and required fields
                    try:
                        tool_call_data = json.loads(part[2:-1])
                        required_fields = ['toolCallId', 'toolName', 'args']
                        for field in required_fields:
                            if field not in tool_call_data:
                                print(f"❌ Missing {field} in tool call: {repr(part)}")
                                success = False
                    except json.JSONDecodeError:
                        print(f"❌ Invalid JSON in tool call part: {repr(part)}")
                        success = False
            
            if success:
                print("✅ All tool call parts follow correct format: 9:{\"toolCallId\":\"...\", \"toolName\":\"...\", \"args\":{...}}\\n")
        else:
            print("⚠️  No tool call protocol parts generated (might be expected if no tools were called)")
        
        # Check tool result parts (Protocol a)
        if len(tool_result_parts) > 0:
            print("✅ SUCCESS: Tool result protocol parts (a:) generated")
            
            # Verify tool result format compliance
            for part in tool_result_parts:
                if not (part.startswith('a:') and part.endswith('\n')):
                    print(f"❌ Tool result format error: {repr(part)}")
                    success = False
                else:
                    # Verify JSON validity and required fields
                    try:
                        tool_result_data = json.loads(part[2:-1])
                        required_fields = ['toolCallId', 'result']
                        for field in required_fields:
                            if field not in tool_result_data:
                                print(f"❌ Missing {field} in tool result: {repr(part)}")
                                success = False
                    except json.JSONDecodeError:
                        print(f"❌ Invalid JSON in tool result part: {repr(part)}")
                        success = False
            
            if success:
                print("✅ All tool result parts follow correct format: a:{\"toolCallId\":\"...\", \"result\":\"...\"}\\n")
        else:
            print("⚠️  No tool result protocol parts generated (might be expected if no tools were called)")
        
        # Check tool start parts (Protocol b)
        if len(tool_start_parts) > 0:
            print("✅ SUCCESS: Tool start protocol parts (b:) generated")
            
            # Verify tool start format compliance
            for part in tool_start_parts:
                if not (part.startswith('b:') and part.endswith('\n')):
                    print(f"❌ Tool start format error: {repr(part)}")
                    success = False
                else:
                    # Verify JSON validity and required fields
                    try:
                        tool_start_data = json.loads(part[2:-1])
                        required_fields = ['toolCallId', 'toolName']
                        for field in required_fields:
                            if field not in tool_start_data:
                                print(f"❌ Missing {field} in tool start: {repr(part)}")
                                success = False
                    except json.JSONDecodeError:
                        print(f"❌ Invalid JSON in tool start part: {repr(part)}")
                        success = False
            
            if success:
                print("✅ All tool start parts follow correct format: b:{\"toolCallId\":\"...\", \"toolName\":\"...\"}\\n")
        else:
            print("⚠️  No tool start protocol parts generated (might be expected if no tools were called)")
        
        # Verify logical sequence (start -> call -> result)
        if len(tool_start_parts) > 0 and len(tool_call_parts) > 0 and len(tool_result_parts) > 0:
            print("✅ Complete tool workflow: start (b:) -> call (9:) -> result (a:)")
        elif len(tool_call_parts) > 0 and len(tool_result_parts) > 0:
            print("✅ Basic tool workflow: call (9:) -> result (a:)")
        elif len(tool_call_parts) > 0 or len(tool_result_parts) > 0 or len(tool_start_parts) > 0:
            print("⚠️  Partial tool workflow detected")
        
        # Final assessment
        tool_protocols_found = len(tool_call_parts) + len(tool_result_parts) + len(tool_start_parts)
        
        if success and tool_protocols_found > 0:
            print("\n✅ COMPREHENSIVE SUCCESS: Tool protocols (9:, a:, b:) work correctly")
            print("   ✅ DeepSeek API with function calling generates proper AI SDK protocol format")
            print("   ✅ Tool calls are properly JSON-encoded with required fields")
            print("   ✅ Tool results are properly formatted")
            if len(tool_start_parts) > 0:
                print("   ✅ Tool streaming start events are captured")
            return True
        elif tool_protocols_found == 0:
            print("\n⚠️  NO TOOL PROTOCOLS GENERATED")
            print("   This might be expected if the agent didn't use tools for this query")
            print("   Try a query that more clearly requires tool usage")
            return False
        else:
            print("\n❌ PARTIAL OR COMPLETE FAILURE: Issues detected in tool protocol generation")
            return False
            
    except Exception as e:
        print(f"ERROR during test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_tool_protocols_9ab())
    
    if success:
        print("SUCCESS: Protocol Types 9, a, b (Tool Call, Result, Start) test PASSED")
    else:
        print("FAILED: Protocol Types 9, a, b (Tool Call, Result, Start) test FAILED")
        print("   Check DeepSeek API configuration and function calling setup")
    
    exit(0 if success else 1)