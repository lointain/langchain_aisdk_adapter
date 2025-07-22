#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual Protocol Test Example

This example manually creates AI SDK protocol outputs and tests the on_finish callback
to verify the Message format matches expectations.
"""

import asyncio
from typing import Dict, Any, Optional
from langchain_aisdk_adapter import (
    LangChainAdapter, BaseAICallbackHandler, Message, LanguageModelUsage,
    create_text_part, create_tool_call_part, create_tool_result_part
)
from langchain_aisdk_adapter.factory import create_start_step_part, create_finish_step_part


class MessageFormatCallback(BaseAICallbackHandler):
    """Callback to verify Message format in on_finish"""
    
    async def on_tool_call(self, tool_call: Dict[str, Any]) -> Optional[Any]:
        return None
    
    async def on_response(self, response: Any) -> None:
        pass
    
    async def on_finish(self, message: Message, options: Dict[str, Any]) -> None:
        """Display message format similar to complete_agent_callback_test.py"""
        print("\n" + "="*50)
        print("🎯 ON_FINISH CALLBACK RESULTS")
        print("="*50)
        
        print(f"\n📝 Message Content:")
        print(message.content)
        
        # Display message parts
        if message.parts:
            print(f"\n🧩 Message Parts ({len(message.parts)}):")
            for i, part in enumerate(message.parts):
                print(f"  [{i+1}] Type: {part.type}")
                
                if part.type == 'text':
                    print(f"      Text: {part.text}")
                
                elif part.type == 'tool-invocation':
                    tool_inv = part.toolInvocation
                    print(f"      Tool: {tool_inv.toolName} (state: {tool_inv.state})")
                    print(f"      Args: {tool_inv.args}")
                    print(f"      Result: {tool_inv.result}")
        
        # Check if tools were called
        has_tools = any(part.type == 'tool-invocation' for part in message.parts) if message.parts else False
        print(f"\n🔧 Tools Called: {'Yes' if has_tools else 'No'}")
        
        # Display full message
        print(f"\n--- Full Message ---")
        print(message)
        print("="*50)
    
    async def on_error(self, error: Exception) -> None:
        print(f"\n[ERROR] {type(error)} - {error}")


async def create_manual_protocol_stream():
    """Create a stream that manually outputs AI SDK protocol strings with step protocols"""
    print("\n🔄 AI SDK Protocol Output:")
    print("-" * 40)
    
    # Step start protocol (f:)
    step_start_part = create_start_step_part(message_id="step_manual_123")
    protocol_output = step_start_part.ai_sdk_part_content
    yield protocol_output
    print(f"Protocol: {repr(protocol_output)}")
    
    # Text protocol (0:)
    text_part1 = create_text_part("The weather in Beijing is sunny with 22°C temperature, and I will calculate 15 * 24 = ")
    protocol_output = text_part1.ai_sdk_part_content
    yield protocol_output
    print(f"Protocol: {repr(protocol_output)}")
    
    # Tool call protocol (9:)
    tool_call_part = create_tool_call_part(
        tool_call_id="call_0",
        tool_name="get_weather",
        args={"input": "Beijing"}
    )
    protocol_output = tool_call_part.ai_sdk_part_content
    yield protocol_output
    print(f"Protocol: {repr(protocol_output)}")
    
    # Tool result protocol (a:)
    tool_result_part = create_tool_result_part(
        tool_call_id="call_0",
        result="The weather in Beijing is sunny with 22°C temperature."
    )
    protocol_output = tool_result_part.ai_sdk_part_content
    yield protocol_output
    print(f"Protocol: {repr(protocol_output)}")
    
    # Second tool call
    tool_call_part2 = create_tool_call_part(
        tool_call_id="call_1",
        tool_name="calculate_math",
        args={"input": "15*24"}
    )
    protocol_output = tool_call_part2.ai_sdk_part_content
    yield protocol_output
    print(f"Protocol: {repr(protocol_output)}")
    
    # Second tool result
    tool_result_part2 = create_tool_result_part(
        tool_call_id="call_1",
        result="The result of 15*24 is 360"
    )
    protocol_output = tool_result_part2.ai_sdk_part_content
    yield protocol_output
    print(f"Protocol: {repr(protocol_output)}")
    
    # Final text
    text_part2 = create_text_part("360.\n")
    protocol_output = text_part2.ai_sdk_part_content
    yield protocol_output
    print(f"Protocol: {repr(protocol_output)}")
    
    print("-" * 40)


async def test_manual_protocol():
    """Test manual protocol output with step protocols"""
    callback = MessageFormatCallback()
    
    # Create manual protocol stream
    langchain_stream = create_manual_protocol_stream()
    
    # Create adapter instance with callback
    adapter = LangChainAdapter(callback=callback)
    
    # Convert to AI SDK stream
    ai_sdk_stream = adapter.to_data_stream_response(langchain_stream)
    
    # Process the stream (silently, just like complete_agent_callback_test.py)
    async for chunk in ai_sdk_stream:
        pass  # Process silently, callback will handle the output


async def main():
    """Main function to test manual protocol with step protocols"""
    print("=== Manual Protocol Test - Simulating complete_agent_callback_test.py ===")
    print("This example manually creates AI SDK protocols including step-start and step-finish.")
    
    try:
        await test_manual_protocol()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())