#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual Agent Callback Test with Real DeepSeek API

This example demonstrates a complete agent workflow with manual event handling:
- Real DeepSeek API calls (not mocked)
- Tool calls and agent execution
- AI SDK compatible callbacks
- Manual event handling (auto_events=False)
- Testing parts accumulation in manual mode
"""

import asyncio
import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

from langchain_aisdk_adapter import LangChainAdapter
from langchain_aisdk_adapter.callbacks import BaseAICallbackHandler, Message


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


class ManualTestCallbackHandler(BaseAICallbackHandler):
    """Test callback handler for demonstrating manual stream processing"""
    
    def __init__(self):
        self.chunk_count = 0
        self.step_start_count = 0
        self.tool_invocation_count = 0
        self.text_count = 0
    
    async def on_start(self) -> None:
        """Called when the stream starts"""
        print("\n" + "="*50)
        print("MANUAL STREAM STARTED (auto_events=False)")
        print("="*50)
        self.chunk_count = 0
        self.step_start_count = 0
        self.tool_invocation_count = 0
        self.text_count = 0
    
    async def on_finish(self, message: Message, options: dict) -> None:
        """Called when the stream finishes"""
        print("\n" + "="*50)
        print("MANUAL STREAM COMPLETED")
        print("="*50)
        print(f"\nStream Statistics:")
        print(f"Total chunks processed: {self.chunk_count}")
        print(f"Step-start parts: {self.step_start_count}")
        print(f"Tool-invocation parts: {self.tool_invocation_count}")
        print(f"Text parts: {self.text_count}")
        
        print(f"\nFinal message object:")
        print(f"Message ID: {message.id}")
        print(f"Message Role: {message.role}")
        print(f"Message Content: {message.content}")
        print(f"Message Parts: {len(message.parts)} parts")
        print(f"All Message Parts: {message.parts}")
        
        # Detailed parts analysis
        for i, part in enumerate(message.parts):
            part_type = getattr(part, 'type', 'unknown')
            if part_type == 'step-start':
                self.step_start_count += 1
                print(f"  Part {i}: step-start - {part}")
            elif part_type == 'tool-invocation':
                self.tool_invocation_count += 1
                tool_inv = getattr(part, 'toolInvocation', None)
                if tool_inv:
                    print(f"  Part {i}: tool-invocation - {tool_inv.toolName} (step={tool_inv.step})")
                else:
                    print(f"  Part {i}: tool-invocation - {part}")
            elif part_type == 'text':
                self.text_count += 1
                text_content = getattr(part, 'text', '')
                preview = text_content[:100] + '...' if len(text_content) > 100 else text_content
                print(f"  Part {i}: text - {preview}")
            else:
                print(f"  Part {i}: {part_type} - {part}")
        
        print(f"\nParts Validation:")
        print(f"Expected step-start parts: >= 1")
        print(f"Actual step-start parts: {self.step_start_count}")
        print(f"Expected tool-invocation parts: >= 2 (weather + math)")
        print(f"Actual tool-invocation parts: {self.tool_invocation_count}")
        print(f"Expected text parts: >= 1")
        print(f"Actual text parts: {self.text_count}")
        
        # Validation results
        validation_passed = (
            self.step_start_count >= 1 and
            self.tool_invocation_count >= 2 and
            self.text_count >= 1
        )
        
        print(f"\nValidation Result: {'PASSED' if validation_passed else 'FAILED'}")
        print(f"Options: {options}")
        print("\n--- End of Manual Stream ---")


def create_manual_test_callbacks():
    """Create test callbacks for demonstrating manual stream processing"""
    return ManualTestCallbackHandler()


async def test_manual_agent_workflow():
    """Main test function for manual event handling"""
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
    callbacks = create_manual_test_callbacks()
    
    # Test query that should trigger tool usage
    test_query = "Please use the get_weather tool to check Beijing's weather and use calculate_math tool to compute 15 * 24"
    
    try:
        # Create agent stream using astream_events
        agent_stream = agent_executor.astream_events({"input": test_query}, version="v2")
        
        # Convert to AI SDK stream using LangChainAdapter.to_data_stream with manual events
        ai_sdk_stream = LangChainAdapter.to_data_stream(
            agent_stream,
            callbacks=callbacks,
            message_id="manual-test-message-001",
            options={"auto_events": False}  # Manual event handling
        )
        
        # Import ManualStreamController for manual event emission
        from langchain_aisdk_adapter import ManualStreamController
        
        # Create manual stream controller
        manual_controller = ManualStreamController()
        
        # Process the stream manually and emit some custom events
        print("\nManual AI SDK Protocol Output:")
        print("-" * 40)
        
        # 手动发射一些事件
        # 发射start事件
        await manual_controller.emit_start(message_id="manual-message-001")
        print("Manual start chunk emitted")
        
        # 发射start-step事件
        await manual_controller.emit_start_step(
            step_type="tool",
            step_id="step-1",
            message_id="manual-message-001"
        )
        print("Manual start-step chunk emitted")
        
        # 发射text-start事件
        await manual_controller.emit_text_start(
            text_id="manual-text-001",
            message_id="manual-message-001"
        )
        print("Manual text-start chunk emitted")
        
        # 发射text-delta事件
        await manual_controller.emit_text_delta(
            text_delta="Processing your request manually...",
            text_id="manual-text-001",
            message_id="manual-message-001"
        )
        print("Manual text-delta chunk emitted")
        
        # 发射text-end事件
        await manual_controller.emit_text_end(
            text="Processing your request manually...",
            text_id="manual-text-001",
            message_id="manual-message-001"
        )
        print("Manual text-end chunk emitted")
        
        # Process the actual stream (this will accumulate parts but not emit events)
        chunk_counter = 0
        async for chunk in ai_sdk_stream:
            chunk_counter += 1
            callbacks.chunk_count = chunk_counter
            # In manual mode, no chunks should be yielded
            print(f"Unexpected Chunk {chunk_counter}: {repr(chunk)}")
        
        # 发射finish-step事件
        await manual_controller.emit_finish_step(
            step_type="tool",
            step_id="step-1",
            message_id="manual-message-001"
        )
        print("Manual finish-step chunk emitted")
        
        # 发射finish事件
        await manual_controller.emit_finish(
            message_id="manual-message-001",
            finish_reason="stop"
        )
        print("Manual finish chunk emitted")
        
        # 关闭手动控制器
        await manual_controller.close()
        
        # 创建一个新的MessageBuilder来处理手动事件
        from langchain_aisdk_adapter.adapter import MessageBuilder
        manual_message_builder = MessageBuilder("manual-message-001")
        
        # 迭代手动控制器的事件并处理它们
        print("\n=== Manual Controller Events ===")
        async for chunk in manual_controller:
            print(f"Manual chunk: {chunk}")
            await manual_message_builder.add_chunk(chunk)
        
        # 构建手动消息并显示parts
        manual_message = manual_message_builder.build_message()
        print(f"\n=== Manual Message Parts ===")
        print(f"Manual Message ID: {manual_message.id}")
        print(f"Manual Message Content: {manual_message.content}")
        print(f"Manual Message Parts: {len(manual_message.parts)} parts")
        for i, part in enumerate(manual_message.parts):
            if hasattr(part, 'text'):
                print(f"  Part {i}: {part.type} - {part.text[:100]}..." if len(part.text) > 100 else f"  Part {i}: {part.type} - {part.text}")
            elif hasattr(part, 'toolInvocation'):
                print(f"  Part {i}: {part.type} - {part.toolInvocation.toolName} (step={part.toolInvocation.step})")
            else:
                print(f"  Part {i}: {part.type} - {part}")
        
        print("-" * 40)
        print(f"Total manual events emitted: 6")
        print(f"Total unexpected chunks from auto stream: {chunk_counter}")
        
    except Exception as e:
        print(f"Error during streaming: {str(e)}")
        import traceback
        traceback.print_exc()


async def main():
    try:
        await test_manual_agent_workflow()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())