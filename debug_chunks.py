import asyncio
from typing import AsyncGenerator, Dict, Any

# Mock LangChain chunk data to simulate the actual behavior
async def mock_langchain_stream() -> AsyncGenerator[Dict[str, Any], None]:
    """Mock LangChain stream events to simulate actual behavior."""
    
    # Simulate on_chat_model_start
    yield {
        "event": "on_chat_model_start",
        "data": {}
    }
    
    # Simulate on_chat_model_stream events with cumulative text
    cumulative_texts = [
        "I",
        "I need",
        "I need to",
        "I need to perform",
        "I need to perform two",
        "I need to perform two tasks"
    ]
    
    for text in cumulative_texts:
        yield {
            "event": "on_chat_model_stream",
            "data": {
                "chunk": {
                    "content": text
                }
            }
        }
        await asyncio.sleep(0.1)  # Simulate streaming delay
    
    # Simulate on_chat_model_end
    yield {
        "event": "on_chat_model_end",
        "data": {}
    }

async def test_chunk_processing():
    """Test how our adapter processes chunks."""
    print("=== Testing Chunk Processing ===")
    
    accumulated_text = ""
    
    async for event in mock_langchain_stream():
        if event.get("event") == "on_chat_model_stream":
            chunk_data = event.get("data", {}).get("chunk")
            if chunk_data:
                text = chunk_data.get("content", "")
                
                print(f"Chunk text: '{text}'")
                print(f"Previous accumulated: '{accumulated_text}'")
                
                # Current logic (wrong): treat chunk as delta
                print(f"If treated as delta: '{text}'")
                
                # Correct logic: calculate delta from cumulative
                if len(text) > len(accumulated_text):
                    delta = text[len(accumulated_text):]
                    accumulated_text = text
                    print(f"Correct delta: '{delta}'")
                else:
                    print("No new content")
                
                print("-" * 40)

if __name__ == "__main__":
    asyncio.run(test_chunk_processing())