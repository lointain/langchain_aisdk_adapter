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

from langchain_aisdk_adapter import LangChainAdapter, DataStreamContext, BaseAICallbackHandler, Message

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
            options={"auto_events": False, "auto_context": True}
        )
        
        # Test various emit methods using DataStreamContext with proper sequence
        print("\nTesting DataStreamContext emit methods with proper sequence:")
        
        # Test DataStreamContext emit methods (using await for async methods)
        print("\nTesting DataStreamContext emit methods with proper sequence:")
        
        # Test basic emit functionality
        try:
            # 1. Start message
            await DataStreamContext.emit_start()
            print("✓ Emitted start")
            
            # 2. Text sequence: start -> delta -> end
            text_id = "text-001"
            await DataStreamContext.emit_text_start(text_id=text_id)
            print("✓ Emitted text-start")
            
            await DataStreamContext.emit_text_delta(text_delta="Hello ", text_id=text_id)
            await DataStreamContext.emit_text_delta(text_delta="world!", text_id=text_id)
            print("✓ Emitted text-delta chunks")
            
            await DataStreamContext.emit_text_end(text="Hello world!", text_id=text_id)
            print("✓ Emitted text-end")
            
            # 3. Tool sequence: start -> delta -> available -> output
            tool_call_id = "tool-001"
            await DataStreamContext.emit_tool_input_start(tool_call_id=tool_call_id, tool_name="get_weather")
            print("✓ Emitted tool-input-start")
            
            await DataStreamContext.emit_tool_input_delta(tool_call_id=tool_call_id, input_text_delta='{"city": "Beijing"}')
            print("✓ Emitted tool-input-delta")
            
            await DataStreamContext.emit_tool_input_available(
                tool_call_id=tool_call_id, 
                tool_name="get_weather", 
                input_data={"city": "Beijing"}
            )
            print("✓ Emitted tool-input-available")
            
            await DataStreamContext.emit_tool_output_available(
                tool_call_id=tool_call_id, 
                output="The weather in Beijing is sunny with 22°C temperature."
            )
            print("✓ Emitted tool-output-available")
            
            # Test tool output error
            tool_call_id_2 = "tool-002"
            await DataStreamContext.emit_tool_output_error(
                tool_call_id=tool_call_id_2,
                error_text="Tool execution failed: timeout"
            )
            print("✓ Emitted tool-output-error")
            
            # 4. File attachments - Test both v4 and v5 protocols
            # v5 protocol (default)
            await DataStreamContext.emit_file(url="https://example.com/report.pdf", media_type="application/pdf")
            print("✓ Emitted file (v5): report.pdf")
            
            # v4 protocol with base64 data
            import base64
            sample_data = "This is sample file content for testing"
            base64_data = base64.b64encode(sample_data.encode()).decode()
            await DataStreamContext.emit_file(
                data=base64_data, 
                mime_type="text/plain", 
                protocol_version="v4"
            )
            print("✓ Emitted file (v4): base64 data")
            
            # v5 protocol explicitly
            await DataStreamContext.emit_file(
                url="https://example.com/data.json", 
                media_type="application/json",
                protocol_version="v5"
            )
            print("✓ Emitted file (v5): data.json")
            
            # Test fallback: v4 params with v5 protocol
            await DataStreamContext.emit_file(
                data=base64_data,
                mime_type="application/pdf",
                protocol_version="v5"
            )
            print("✓ Emitted file (v5 fallback): converted from v4 params")
            
            # 5. Step control
            await DataStreamContext.emit_start_step(step_type="reasoning", step_id="step-001")
            print("✓ Emitted start-step")
            
            await DataStreamContext.emit_finish_step(step_type="reasoning", step_id="step-001")
            print("✓ Emitted finish-step")
            
            # 6. Reasoning sequence
            reasoning_id = "reasoning-001"
            await DataStreamContext.emit_reasoning_start(reasoning_id=reasoning_id)
            print("✓ Emitted reasoning-start")
            
            await DataStreamContext.emit_reasoning_delta(delta="Let me think about this...", reasoning_id=reasoning_id)
            print("✓ Emitted reasoning-delta")
            
            await DataStreamContext.emit_reasoning_end(reasoning_id=reasoning_id)
            print("✓ Emitted reasoning-end")
            
            # 7. Source documents and URLs
            await DataStreamContext.emit_source_url(
                url="https://example.com/docs",
                description="Example documentation"
            )
            print("✓ Emitted source-url")
            
            # Note: emit_source_document is only available on DataStreamWithEmitters, not DataStreamContext
            # This demonstrates the difference between context-based and stream-based emit methods
            
            # 8. Message metadata
            await DataStreamContext.emit_message_metadata(
                metadata={"model": "deepseek-chat", "temperature": 0.1}
            )
            print("✓ Emitted message-metadata")
            
            # 9. Custom data
            await DataStreamContext.emit_data(data={"key": "value", "number": 42})
            print("✓ Emitted data chunk")
            
            # 10. Error handling
            await DataStreamContext.emit_error(error_text="This is a test error message.")
            print("✓ Emitted error")
            
            await DataStreamContext.emit_abort(reason="User requested abort")
            print("✓ Emitted abort")
            
            # 11. Finish message
            await DataStreamContext.emit_finish()
            print("✓ Emitted finish")
            
            print("All DataStreamContext emit tests completed with proper sequence.\n")
        except Exception as e:
            print(f"DataStreamContext emit test failed: {e}\n")
        
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