"""Example of integrating with real LangChain models."""

import asyncio
import os
from typing import AsyncGenerator

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain not installed. Install with: pip install langchain langchain-openai")

from langchain_aisdk_adapter import to_ui_message_stream


async def example_with_chat_model():
    """Example using ChatOpenAI model with streaming."""
    if not LANGCHAIN_AVAILABLE:
        print("Skipping ChatOpenAI example - LangChain not available")
        return
    
    print("=== Example: ChatOpenAI Model Streaming ===")
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found in environment variables")
        print("   Set your API key: export OPENAI_API_KEY='your-key-here'")
        return
    
    try:
        # Initialize the model
        model = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            streaming=True
        )
        
        # Create a message
        messages = [HumanMessage(content="Tell me a short joke about programming")]
        
        print("🤖 Asking ChatGPT for a programming joke...\n")
        
        # Stream the response
        async for ui_chunk in to_ui_message_stream(model.astream(messages)):
            if ui_chunk["type"] == "text-delta":
                print(ui_chunk["delta"], end="", flush=True)
            elif ui_chunk["type"] == "text-end":
                print("\n")
        
    except Exception as e:
        print(f"❌ Error with ChatOpenAI: {e}")
        print("   Make sure your OPENAI_API_KEY is valid")


async def example_with_string_output_parser():
    """Example using StringOutputParser with streaming."""
    if not LANGCHAIN_AVAILABLE:
        print("Skipping StringOutputParser example - LangChain not available")
        return
    
    print("\n=== Example: StringOutputParser Chain ===")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found - using mock data")
        
        # Mock the chain output
        async def mock_chain_stream():
            text = "This is a mock response from a LangChain chain with StringOutputParser."
            for word in text.split():
                yield word + " "
                await asyncio.sleep(0.1)
        
        print("📝 Mock chain output:")
        async for ui_chunk in to_ui_message_stream(mock_chain_stream()):
            if ui_chunk["type"] == "text-delta":
                print(ui_chunk["delta"], end="", flush=True)
            elif ui_chunk["type"] == "text-end":
                print("\n")
        return
    
    try:
        # Create a chain with StringOutputParser
        model = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)
        parser = StrOutputParser()
        chain = model | parser
        
        print("🔗 Running LangChain chain with StringOutputParser...\n")
        
        # Stream the chain output
        async for ui_chunk in to_ui_message_stream(
            chain.astream("Explain what LangChain is in one sentence")
        ):
            if ui_chunk["type"] == "text-delta":
                print(ui_chunk["delta"], end="", flush=True)
            elif ui_chunk["type"] == "text-end":
                print("\n")
        
    except Exception as e:
        print(f"❌ Error with chain: {e}")


async def example_with_callbacks_and_real_model():
    """Example using callbacks with a real model."""
    if not LANGCHAIN_AVAILABLE:
        print("Skipping callbacks example - LangChain not available")
        return
    
    print("\n=== Example: Real Model with Callbacks ===")
    
    class StreamCallbacks:
        def __init__(self):
            self.token_count = 0
            self.start_time = None
        
        def on_start(self):
            import time
            self.start_time = time.time()
            print("🚀 Stream started!")
        
        def on_token(self, token: str):
            self.token_count += 1
        
        def on_final(self, completion: str):
            import time
            duration = time.time() - self.start_time if self.start_time else 0
            print(f"\n✅ Stream completed!")
            print(f"   📊 Total tokens: {self.token_count}")
            print(f"   ⏱️  Duration: {duration:.2f}s")
            print(f"   📝 Total characters: {len(completion)}")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found - using mock data")
        
        async def mock_stream():
            words = ["Mock", " response", " with", " callbacks", " enabled", "!"]
            for word in words:
                yield word
                await asyncio.sleep(0.2)
        
        callbacks = StreamCallbacks()
        print("📝 Mock response with callbacks:")
        async for ui_chunk in to_ui_message_stream(mock_stream(), callbacks=callbacks):
            if ui_chunk["type"] == "text-delta":
                print(ui_chunk["delta"], end="", flush=True)
        return
    
    try:
        model = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)
        callbacks = StreamCallbacks()
        
        messages = [HumanMessage(content="Write a haiku about artificial intelligence")]
        
        print("🎋 Generating a haiku about AI...\n")
        
        async for ui_chunk in to_ui_message_stream(
            model.astream(messages), 
            callbacks=callbacks
        ):
            if ui_chunk["type"] == "text-delta":
                print(ui_chunk["delta"], end="", flush=True)
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def example_stream_events():
    """Example handling LangChain stream events (if supported)."""
    print("\n=== Example: Stream Events (Mock) ===")
    print("📡 This example shows how stream events would be handled")
    
    # Mock stream events as they would come from LangChain
    async def mock_stream_events():
        events = [
            {"event": "on_chat_model_start", "data": {}},
            {"event": "on_chat_model_stream", "data": {"chunk": {"content": "Stream"}}},
            {"event": "on_chat_model_stream", "data": {"chunk": {"content": " events"}}},
            {"event": "on_chat_model_stream", "data": {"chunk": {"content": " work"}}},
            {"event": "on_chat_model_stream", "data": {"chunk": {"content": " great!"}}},
            {"event": "on_chat_model_end", "data": {}},
        ]
        
        for event in events:
            yield event
            await asyncio.sleep(0.1)
    
    print("📝 Processing stream events:")
    async for ui_chunk in to_ui_message_stream(mock_stream_events()):
        if ui_chunk["type"] == "text-delta":
            print(ui_chunk["delta"], end="", flush=True)
        elif ui_chunk["type"] == "text-end":
            print("\n")


async def main():
    """Run all integration examples."""
    print("🔗 LangChain AI SDK Adapter - Integration Examples\n")
    
    await example_with_chat_model()
    await example_with_string_output_parser()
    await example_with_callbacks_and_real_model()
    await example_stream_events()
    
    print("\n🎉 All examples completed!")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n💡 Tip: Set OPENAI_API_KEY to try real model examples")


if __name__ == "__main__":
    asyncio.run(main())