#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison Test: Auto vs Manual Event Handling

This example demonstrates the difference between auto and manual event handling:
- Runs the same agent workflow with both auto_events=True and auto_events=False
- Compares the parts accumulation in both modes
- Validates that both modes produce correct results
"""

import asyncio
import os
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate

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


class ComparisonCallbackHandler(BaseAICallbackHandler):
    """Callback handler for comparing auto vs manual modes"""
    
    def __init__(self, mode: str):
        self.mode = mode  # "auto" or "manual"
        self.chunk_count = 0
        self.parts_summary = {
            'step-start': 0,
            'tool-invocation': 0,
            'text': 0,
            'other': 0
        }
        self.final_message = None
    
    async def on_start(self) -> None:
        """Called when the stream starts"""
        print(f"\n{'='*60}")
        print(f"STARTING {self.mode.upper()} MODE TEST (auto_events={self.mode == 'auto'})")
        print(f"{'='*60}")
        self.chunk_count = 0
        self.parts_summary = {'step-start': 0, 'tool-invocation': 0, 'text': 0, 'other': 0}
    
    async def on_finish(self, message: Message, options: dict) -> None:
        """Called when the stream finishes"""
        self.final_message = message
        
        # Count parts by type
        for part in message.parts:
            part_type = getattr(part, 'type', 'unknown')
            if part_type in self.parts_summary:
                self.parts_summary[part_type] += 1
            else:
                self.parts_summary['other'] += 1
        
        print(f"\n{'-'*60}")
        print(f"{self.mode.upper()} MODE RESULTS")
        print(f"{'-'*60}")
        print(f"Total chunks processed: {self.chunk_count}")
        print(f"Total parts in final message: {len(message.parts)}")
        print(f"Parts breakdown:")
        for part_type, count in self.parts_summary.items():
            print(f"  {part_type}: {count}")
        
        print(f"\nDetailed parts:")
        for i, part in enumerate(message.parts):
            part_type = getattr(part, 'type', 'unknown')
            if part_type == 'tool-invocation':
                tool_inv = getattr(part, 'toolInvocation', None)
                if tool_inv:
                    print(f"  [{i}] {part_type}: {tool_inv.toolName} (step={tool_inv.step})")
                else:
                    print(f"  [{i}] {part_type}: {part}")
            elif part_type == 'text':
                text_content = getattr(part, 'text', '')
                preview = text_content[:50] + '...' if len(text_content) > 50 else text_content
                print(f"  [{i}] {part_type}: {preview}")
            else:
                print(f"  [{i}] {part_type}: {part}")
        
        print(f"\nMessage content length: {len(message.content)} characters")
        print(f"Options: {options}")


async def run_test_mode(mode: str, test_query: str) -> ComparisonCallbackHandler:
    """Run test in specified mode (auto or manual)"""
    
    # Check for API key
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("Error: DEEPSEEK_API_KEY environment variable not set")
        return None
    
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
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)  # Disable verbose for cleaner output
    
    # Initialize callback handler
    callbacks = ComparisonCallbackHandler(mode)
    
    try:
        # Create agent stream using astream_events
        agent_stream = agent_executor.astream_events({"input": test_query}, version="v2")
        
        # Convert to AI SDK stream with appropriate options
        auto_events = (mode == "auto")
        
        if mode == "auto":
            # Auto mode: use to_data_stream_response for full compatibility
            ai_sdk_stream_response = await LangChainAdapter.to_data_stream_response(
                agent_stream,
                callbacks=callbacks,
                message_id=f"{mode}-test-message-001",
                options={"auto_events": auto_events}
            )
            
            # Add a file emission to test mixed mode functionality
            # This demonstrates that even in auto mode, manual emissions are supported
            from langchain_aisdk_adapter import ManualStreamController
            manual_controller = ManualStreamController()
            await manual_controller.emit_file(
                url="https://example.com/file.pdf",
                media_type="application/pdf"
            )
            
            # Extract the data stream from response for processing
            ai_sdk_stream = ai_sdk_stream_response.body_iterator
        else:
            # Manual mode: use to_data_stream for direct stream access
            ai_sdk_stream = LangChainAdapter.to_data_stream(
                agent_stream,
                callbacks=callbacks,
                message_id=f"{mode}-test-message-001",
                options={"auto_events": auto_events}
            )
        
        # Process the stream
        chunk_counter = 0
        async for chunk in ai_sdk_stream:
            chunk_counter += 1
            callbacks.chunk_count = chunk_counter
            # Only print first few chunks to avoid spam
            if chunk_counter <= 3:
                chunk_type = chunk.get('type', 'unknown') if hasattr(chunk, 'get') else str(type(chunk).__name__)
                print(f"  Chunk {chunk_counter}: {chunk_type}")
            elif chunk_counter == 4:
                print(f"  ... (processing more chunks)")
        
        return callbacks
        
    except Exception as e:
        print(f"Error during {mode} mode streaming: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


async def compare_modes():
    """Compare auto and manual modes"""
    # Temporarily disable LangSmith tracing to avoid warnings
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    
    # Test query that should trigger tool usage
    test_query = "Please use the get_weather tool to check Beijing's weather and use calculate_math tool to compute 15 * 24"
    
    print("\n" + "="*80)
    print("COMPARISON TEST: AUTO vs MANUAL EVENT HANDLING")
    print("="*80)
    print(f"Test Query: {test_query}")
    
    # Run auto mode test
    auto_handler = await run_test_mode("auto", test_query)
    
    # Run manual mode test
    manual_handler = await run_test_mode("manual", test_query)
    
    # Compare results
    if auto_handler and manual_handler:
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        print(f"\nChunk Processing:")
        print(f"  Auto mode chunks: {auto_handler.chunk_count}")
        print(f"  Manual mode chunks: {manual_handler.chunk_count}")
        
        print(f"\nParts Count Comparison:")
        print(f"{'Part Type':<20} {'Auto':<10} {'Manual':<10} {'Match':<10}")
        print("-" * 50)
        
        all_part_types = set(auto_handler.parts_summary.keys()) | set(manual_handler.parts_summary.keys())
        total_match = True
        
        for part_type in sorted(all_part_types):
            auto_count = auto_handler.parts_summary.get(part_type, 0)
            manual_count = manual_handler.parts_summary.get(part_type, 0)
            match = "✓" if auto_count == manual_count else "✗"
            if auto_count != manual_count:
                total_match = False
            print(f"{part_type:<20} {auto_count:<10} {manual_count:<10} {match:<10}")
        
        print(f"\nTotal Parts:")
        auto_total = len(auto_handler.final_message.parts) if auto_handler.final_message else 0
        manual_total = len(manual_handler.final_message.parts) if manual_handler.final_message else 0
        total_parts_match = "✓" if auto_total == manual_total else "✗"
        print(f"{'Total':<20} {auto_total:<10} {manual_total:<10} {total_parts_match:<10}")
        
        print(f"\nContent Comparison:")
        if auto_handler.final_message and manual_handler.final_message:
            auto_content_len = len(auto_handler.final_message.content)
            manual_content_len = len(manual_handler.final_message.content)
            content_match = "✓" if auto_content_len == manual_content_len else "✗"
            print(f"Content length: Auto={auto_content_len}, Manual={manual_content_len} {content_match}")
        
        print(f"\nOverall Result: {'PARTS ACCUMULATION CONSISTENT' if total_match and total_parts_match == '✓' else 'DIFFERENCES DETECTED'}")
        
        if not total_match or total_parts_match != '✓':
            print("\n⚠️  Note: Some differences were detected between auto and manual modes.")
            print("   This might be expected behavior depending on the implementation.")
        else:
            print("\n✅ Parts accumulation is consistent between auto and manual modes.")


async def main():
    try:
        await compare_modes()
        
    except Exception as e:
        print(f"Error during comparison testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())