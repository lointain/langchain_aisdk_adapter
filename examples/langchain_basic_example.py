"""Basic LangChain Example with AI SDK Adapter

Demonstrates basic usage of LangChain with AI SDK streaming protocol.
Uses DeepSeek API with environment variable configuration.
"""

import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_aisdk_adapter import AISDKAdapter, factory


def setup_environment():
    """Load environment variables and validate configuration"""
    load_dotenv()
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY not found in environment variables. "
            "Please copy .env.example to .env and configure your API key."
        )
    
    return api_key, base_url


def create_llm(api_key: str, base_url: str) -> ChatOpenAI:
    """Create and configure the LLM instance"""
    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=os.getenv("EXAMPLE_MODEL_NAME", "deepseek-chat"),
        temperature=float(os.getenv("EXAMPLE_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("EXAMPLE_MAX_TOKENS", "1000")),
        streaming=True
    )


async def basic_streaming_example():
    """Demonstrate basic streaming with AI SDK protocol"""
    print("=== Basic LangChain Streaming Example ===")
    
    # Setup
    api_key, base_url = setup_environment()
    llm = create_llm(api_key, base_url)
    
    # Create messages
    messages = [
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessage(content="Tell me a short story about a robot learning to paint.")
    ]
    
    print("\nStreaming response:")
    print("-" * 50)
    
    # Statistics tracking
    part_stats = {}
    
    # Stream with AI SDK protocol
    async for part in AISDKAdapter.astream(llm.astream(messages)):
        # Parse part type
        if ':' in part:
            part_type = part.split(':', 1)[0]
            part_stats[part_type] = part_stats.get(part_type, 0) + 1
            
            if part.startswith('0:'):
                # Text content - output horizontally without newline
                text = part[2:].strip('"')
                print(text, end="", flush=True)
            else:
                # Other types - output on separate lines
                print(f"\n[{part_type}]: {part}")
        else:
            # Handle parts without type prefix
            print(part, end="")
    
    print("\n" + "-" * 50)
    print("Streaming completed.")
    print("\nStatistics:")
    for part_type, count in sorted(part_stats.items()):
        print(f"  Type {part_type}: {count} parts")
    print()


async def factory_usage_example():
    """Demonstrate using the new factory interface"""
    print("=== Factory Usage Example ===")
    
    # Using the new factory interface
    text_part = factory.text("Hello from the new factory!")
    data_part = factory.data(["example", "data", "timestamp", "2024-01-01"])
    error_part = factory.error("This is an example error")
    
    print("\nFactory-created parts:")
    print(f"Text: {text_part.ai_sdk_part_content}", end="")
    print(f"Data: {data_part.ai_sdk_part_content}", end="")
    print(f"Error: {error_part.ai_sdk_part_content}", end="")
    
    # Using class methods directly
    reasoning_part = factory.reasoning("Let me think about this problem...")
    source_part = factory.source(
            url="https://example.com/docs",
            title="Example Documentation"
        )
    
    print(f"Reasoning: {reasoning_part.ai_sdk_part_content}", end="")
    print(f"Source: {source_part.ai_sdk_part_content}", end="")
    print()


async def conversation_example():
    """Demonstrate a multi-turn conversation"""
    print("=== Multi-turn Conversation Example ===")
    
    # Setup
    api_key, base_url = setup_environment()
    llm = create_llm(api_key, base_url)
    
    # Conversation history
    conversation = [
        SystemMessage(content="You are a helpful coding assistant.")
    ]
    
    # First turn
    conversation.append(HumanMessage(content="What is Python?"))
    
    print("\nUser: What is Python?")
    print("Assistant: ", end="")
    
    response_content = ""
    part_stats_1 = {}
    async for part in AISDKAdapter.astream(llm.astream(conversation)):
        # Parse part type and update statistics
        if ':' in part:
            part_type = part.split(':', 1)[0]
            part_stats_1[part_type] = part_stats_1.get(part_type, 0) + 1
            
            if part.startswith('0:'):
                # Text content - output horizontally
                text = part[2:].strip('"')
                response_content += text
                print(text, end="", flush=True)
            else:
                # Other types - output on separate lines
                print(f"\n[{part_type}]: {part}")
        else:
            print(part, end="")
    
    print("\n")
    print("Turn 1 Statistics:")
    for part_type, count in sorted(part_stats_1.items()):
        print(f"  Type {part_type}: {count} parts")
    
    # Add assistant response to conversation
    from langchain_core.messages import AIMessage
    conversation.append(AIMessage(content=response_content))
    
    # Second turn
    conversation.append(HumanMessage(content="Can you show me a simple Python function?"))
    
    print("User: Can you show me a simple Python function?")
    print("Assistant: ", end="")
    
    part_stats_2 = {}
    async for part in AISDKAdapter.astream(llm.astream(conversation)):
        # Parse part type and update statistics
        if ':' in part:
            part_type = part.split(':', 1)[0]
            part_stats_2[part_type] = part_stats_2.get(part_type, 0) + 1
            
            if part.startswith('0:'):
                # Text content - output horizontally
                text = part[2:].strip('"')
                print(text, end="", flush=True)
            else:
                # Other types - output on separate lines
                print(f"\n[{part_type}]: {part}")
        else:
            print(part, end="")
    
    print("\n")
    print("Turn 2 Statistics:")
    for part_type, count in sorted(part_stats_2.items()):
        print(f"  Type {part_type}: {count} parts")
    print()


async def main():
    """Run all examples"""
    try:
        await basic_streaming_example()
        await factory_usage_example()
        await conversation_example()
        
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("\nPlease ensure you have:")
        print("1. Copied .env.example to .env")
        print("2. Added your DeepSeek API key to the .env file")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease check your API configuration and network connection.")


if __name__ == "__main__":
    asyncio.run(main())