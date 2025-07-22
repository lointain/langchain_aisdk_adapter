#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Adapter Demo with Mock Data

This demo shows the enhanced LangChain to AI SDK adapter functionality
using mock data instead of real API calls.
"""

import asyncio
import json
from typing import Any, Dict, AsyncGenerator

from langchain_aisdk_adapter import (
    to_data_stream,
    BaseAICallbackHandler,
    Message,
    UIPart,
    TextUIPart,
    ToolInvocationUIPart,
)


class DemoAICallbackHandler(BaseAICallbackHandler):
    """Demo AI SDK compatible callback handler."""
    
    def __init__(self):
        self.events = []
    
    async def on_finish(self, message: Message, options: Dict[str, Any]) -> None:
        """Called when the stream finishes."""
        print("\n" + "="*60)
        print("🎉 AI SDK CALLBACK - Stream Completed")
        print("="*60)
        
        print(f"📝 Message ID: {message.id}")
        print(f"📄 Content: {message.content}")
        print(f"🔧 Role: {message.role}")
        print(f"📊 Parts Count: {len(message.parts)}")
        
        # Analyze parts
        text_parts = []
        tool_parts = []
        
        for part in message.parts:
            if isinstance(part, TextUIPart):
                text_parts.append(part)
            elif isinstance(part, ToolInvocationUIPart):
                tool_parts.append(part)
        
        print(f"\n📋 Parts Analysis:")
        print(f"  • Text Parts: {len(text_parts)}")
        print(f"  • Tool Parts: {len(tool_parts)}")
        
        if tool_parts:
            print(f"\n🔧 Tool Invocations:")
            for i, tool_part in enumerate(tool_parts, 1):
                tool_inv = tool_part.toolInvocation
                print(f"  {i}. {tool_inv.toolName} (State: {tool_inv.state})")
                if hasattr(tool_inv, 'args'):
                    print(f"     Args: {tool_inv.args}")
                if hasattr(tool_inv, 'result'):
                    print(f"     Result: {tool_inv.result}")
        
        print("="*60)
    
    async def on_error(self, error: Exception) -> None:
        """Called when an error occurs."""
        print(f"\n❌ AI SDK Callback Error: {error}")


async def create_mock_langchain_stream() -> AsyncGenerator[Dict[str, Any], None]:
    """Create a mock LangChain stream with various event types."""
    
    # Simulate chat model start
    yield {
        "event": "on_chat_model_start",
        "data": {
            "input": {"messages": [["human", "What's the weather in Tokyo and calculate 15*8?"]]}
        }
    }
    
    # Simulate some initial text
    yield {
        "event": "on_chat_model_stream",
        "data": {
            "chunk": {
                "content": "I'll help you with both requests. Let me "
            }
        }
    }
    
    yield {
        "event": "on_chat_model_stream",
        "data": {
            "chunk": {
                "content": "check the weather in Tokyo first."
            }
        }
    }
    
    # Simulate tool start - weather tool
    yield {
        "event": "on_tool_start",
        "data": {
            "run_id": "tool_call_1",
            "name": "get_weather",
            "inputs": {"location": "Tokyo"}
        }
    }
    
    # Simulate tool end - weather tool
    yield {
        "event": "on_tool_end",
        "data": {
            "run_id": "tool_call_1",
            "outputs": "The weather in Tokyo is sunny with a temperature of 22°C."
        }
    }
    
    # More text after first tool
    yield {
        "event": "on_chat_model_stream",
        "data": {
            "chunk": {
                "content": "\n\nGreat! The weather in Tokyo is sunny with 22°C. "
            }
        }
    }
    
    yield {
        "event": "on_chat_model_stream",
        "data": {
            "chunk": {
                "content": "Now let me calculate 15 * 8 for you."
            }
        }
    }
    
    # Simulate tool start - math tool
    yield {
        "event": "on_tool_start",
        "data": {
            "run_id": "tool_call_2",
            "name": "calculate_math",
            "inputs": {"expression": "15 * 8"}
        }
    }
    
    # Simulate tool end - math tool
    yield {
        "event": "on_tool_end",
        "data": {
            "run_id": "tool_call_2",
            "outputs": "The result of 15 * 8 is 120"
        }
    }
    
    # Final text
    yield {
        "event": "on_chat_model_stream",
        "data": {
            "chunk": {
                "content": "\n\nPerfect! The calculation shows that 15 * 8 = 120. "
            }
        }
    }
    
    yield {
        "event": "on_chat_model_stream",
        "data": {
            "chunk": {
                "content": "To summarize:\n- Tokyo weather: Sunny, 22°C\n- 15 * 8 = 120"
            }
        }
    }
    
    # Simulate chat model end
    yield {
        "event": "on_chat_model_end",
        "data": {
            "output": {
                "content": "Complete response with weather and math calculation"
            }
        }
    }


async def run_enhanced_demo():
    """Run the enhanced adapter demo."""
    
    print("🚀 Enhanced LangChain to AI SDK Adapter Demo")
    print("📡 Testing AI SDK Data Stream Protocol with Mock Data")
    print("-" * 60)
    
    # Create callback handler
    callback_handler = DemoAICallbackHandler()
    
    # Create mock stream
    mock_stream = create_mock_langchain_stream()
    
    print("\n📊 AI SDK Data Stream Protocol Output:")
    print("-" * 40)
    
    chunk_count = 0
    event_types = {}
    
    try:
        # Convert to AI SDK Data Stream Protocol
        async for chunk in to_data_stream(mock_stream, callbacks=callback_handler):
            chunk_count += 1
            chunk_type = chunk.get("type", "unknown")
            
            # Count event types
            event_types[chunk_type] = event_types.get(chunk_type, 0) + 1
            
            # Pretty print different chunk types
            if chunk_type == "start":
                print(f"🟢 START: Message ID {chunk.get('messageId')}")
            elif chunk_type == "start-step":
                print(f"🔵 STEP START")
            elif chunk_type == "finish-step":
                print(f"🔵 STEP FINISH")
            elif chunk_type == "text-start":
                print(f"📝 TEXT START: ID {chunk.get('id')}")
            elif chunk_type == "text-delta":
                delta = chunk.get("delta", "")
                print(f"📝 TEXT: {repr(delta)}")
            elif chunk_type == "text-end":
                print(f"📝 TEXT END: ID {chunk.get('id')}")
            elif chunk_type == "tool-input-start":
                print(f"🔧 TOOL INPUT START: {chunk.get('toolName')} (ID: {chunk.get('toolCallId')})")
            elif chunk_type == "tool-input-available":
                print(f"🔧 TOOL INPUT: {chunk.get('toolName')} - {chunk.get('input')}")
            elif chunk_type == "tool-output-available":
                print(f"🔧 TOOL OUTPUT: {chunk.get('output')}")
            elif chunk_type == "finish":
                print(f"🔴 FINISH")
            else:
                print(f"❓ {chunk_type.upper()}: {json.dumps(chunk, indent=2)}")
        
        print("\n" + "-" * 40)
        print(f"📊 Stream Statistics:")
        print(f"  • Total Chunks: {chunk_count}")
        print(f"  • Event Types: {dict(sorted(event_types.items()))}")
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_enhanced_demo())