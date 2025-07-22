#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI SDK Compatible Callbacks Example

Demonstrates how to use the new AI SDK compatible callback system
with the LangChain AI SDK adapter.
"""

import asyncio
from typing import Dict, Any

from langchain_aisdk_adapter import (
    to_data_stream,
    BaseAICallbackHandler,
    Message,
    TextUIPart,
    ToolInvocationUIPart,
    StepStartUIPart,
)


class ExampleAICallback(BaseAICallbackHandler):
    """Example AI SDK compatible callback handler."""
    
    def __init__(self):
        self.step_count = 0
        self.tool_calls = []
        self.text_parts = []
    
    async def on_finish(self, message: Message, options: Dict[str, Any]) -> None:
        """Called when stream processing is complete."""
        print("\n=== Stream Processing Complete ===")
        print(f"Message ID: {message.id}")
        print(f"Role: {message.role}")
        print(f"Content: {message.content}")
        print(f"Created At: {message.createdAt}")
        print(f"Total Parts: {len(message.parts)}")
        
        # Analyze parts
        for i, part in enumerate(message.parts):
            print(f"\nPart {i + 1}: {part.type}")
            
            if isinstance(part, TextUIPart):
                print(f"  Text: {part.text[:100]}{'...' if len(part.text) > 100 else ''}")
                self.text_parts.append(part.text)
            
            elif isinstance(part, ToolInvocationUIPart):
                tool = part.toolInvocation
                print(f"  Tool: {tool.toolName}")
                print(f"  State: {tool.state}")
                print(f"  Step: {tool.step}")
                print(f"  Args: {tool.args}")
                if tool.result:
                    print(f"  Result: {tool.result}")
                self.tool_calls.append(tool)
            
            elif isinstance(part, StepStartUIPart):
                self.step_count += 1
                print(f"  Step {self.step_count} started")
        
        # Print usage statistics if available
        if 'usage' in options:
            usage = options['usage']
            print(f"\n=== Usage Statistics ===")
            print(f"Prompt Tokens: {usage.get('promptTokens', 'N/A')}")
            print(f"Completion Tokens: {usage.get('completionTokens', 'N/A')}")
            print(f"Total Tokens: {usage.get('totalTokens', 'N/A')}")
        
        print(f"\n=== Summary ===")
        print(f"Steps: {self.step_count}")
        print(f"Tool Calls: {len(self.tool_calls)}")
        print(f"Text Parts: {len(self.text_parts)}")
    
    async def on_error(self, error: Exception) -> None:
        """Called when an error occurs."""
        print(f"\n=== Error Occurred ===")
        print(f"Error Type: {type(error).__name__}")
        print(f"Error Message: {str(error)}")


async def simulate_langchain_stream():
    """Simulate a LangChain stream with text and tool calls."""
    # Simulate text streaming
    yield "Hello, "
    yield "I can help you "
    yield "with various tasks."
    
    # Simulate tool call events
    yield {
        "event": "on_tool_start",
        "run_id": "tool_run_123",
        "name": "weather_tool",
        "data": {
            "input": {
                "location": "San Francisco",
                "unit": "celsius"
            }
        }
    }
    
    yield {
        "event": "on_tool_end",
        "run_id": "tool_run_123",
        "data": {
            "output": {
                "temperature": 18,
                "condition": "partly cloudy",
                "humidity": 65
            }
        }
    }
    
    # More text after tool call
    yield "The weather in San Francisco is 18°C and partly cloudy."


async def example_basic_usage():
    """Example of basic usage with AI SDK callbacks."""
    print("=== Basic Usage Example ===")
    
    callback = ExampleAICallback()
    stream = simulate_langchain_stream()
    
    print("\nStreaming chunks:")
    async for chunk in to_data_stream(stream, callbacks=callback):
        print(f"Chunk: {chunk['type']}")
        if chunk['type'] == 'text-delta':
            print(f"  Delta: {chunk['delta']}")
        elif chunk['type'] in ['tool-input-start', 'tool-output-available']:
            print(f"  Tool: {chunk.get('toolName', 'N/A')}")


async def example_enhanced_usage():
    """Example of enhanced usage with full AI SDK support."""
    print("\n\n=== Enhanced Usage Example ===")
    
    callback = ExampleAICallback()
    stream = simulate_langchain_stream()
    
    print("\nStreaming chunks (enhanced):")
    async for chunk in to_data_stream(stream, callbacks=callback):
        print(f"Chunk: {chunk['type']}")
        if chunk['type'] == 'text-delta':
            print(f"  Delta: {chunk['delta']}")
        elif chunk['type'] == 'start':
            print(f"  Message ID: {chunk['messageId']}")
        elif chunk['type'] in ['tool-input-start', 'tool-output-available']:
            print(f"  Tool: {chunk.get('toolName', 'N/A')}")


async def main():
    """Main example function."""
    print("LangChain AI SDK Adapter - AI SDK Callbacks Example")
    print("====================================================")
    
    # Run basic example
    await example_basic_usage()
    
    # Run enhanced example
    await example_enhanced_usage()
    
    print("\n\nExample completed!")


if __name__ == "__main__":
    asyncio.run(main())