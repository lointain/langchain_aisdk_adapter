#!/usr/bin/env python3

import asyncio
import uuid
from src.langchain_aisdk_adapter.enhanced_adapter import EnhancedStreamProcessor

async def test_step_creation():
    """Test step creation with simulated events."""
    
    print("=== Step Creation Debug ===")
    print("Simulating LangChain events for tool call + text generation")
    print()
    
    # Create processor
    processor = EnhancedStreamProcessor()
    
    # Simulate LangChain events
    events = [
        # Chat model start
        {"event": "on_chat_model_start", "data": {}},
        
        # Tool start
        {
            "event": "on_tool_start", 
            "data": {
                "run_id": "tool_123",
                "name": "get_weather",
                "inputs": {"location": "Tokyo"}
            }
        },
        
        # Tool end
        {
            "event": "on_tool_end", 
            "data": {
                "run_id": "tool_123",
                "outputs": "The weather in Tokyo is sunny with 22°C temperature."
            }
        },
        
        # Chat model end (should trigger finish-step)
        {"event": "on_chat_model_end", "data": {}},
        
        # Text content (should trigger new start-step)
        "Final Answer: The weather in Tokyo is sunny with a temperature of 22°C."
    ]
    
    step_events = []
    event_count = 0
    
    async def simulate_stream():
        for event in events:
            yield event
    
    async for chunk in processor.process_stream(simulate_stream()):
        event_count += 1
        
        if chunk["type"] in ["start-step", "finish-step"]:
            step_events.append((event_count, chunk["type"]))
            print(f"  {event_count:2d}. {chunk['type']}")
        elif chunk["type"] == "tool-output-available":
            print(f"  {event_count:2d}. tool-output-available")
            print(f"     Tool ID: {chunk.get('toolCallId', 'N/A')}")
        elif chunk["type"] in ["text-start", "text-end"]:
            print(f"  {event_count:2d}. {chunk['type']}")
        elif chunk["type"] in ["start", "finish"]:
            print(f"  {event_count:2d}. {chunk['type']}")
    
    print()
    print("=== Step Events Summary ===")
    for event_num, event_type in step_events:
        print(f"  Event {event_num}: {event_type}")
    
    print(f"\nTotal step events: {len(step_events)}")
    print(f"Total events: {event_count}")
    
    # Check if we have the expected pattern
    expected_pattern = ["start-step", "finish-step", "start-step", "finish-step"]
    actual_pattern = [e[1] for e in step_events]
    
    if actual_pattern == expected_pattern:
        print("\n✅ Expected multi-step pattern found")
        print(f"   Pattern: {' -> '.join(actual_pattern)}")
    else:
        print("\n❌ Missing expected multi-step pattern")
        print(f"   Expected: {' -> '.join(expected_pattern)}")
        print(f"   Found: {' -> '.join(actual_pattern)}")
        
        # Debug processor state
        print("\n=== Processor State Debug ===")
        print(f"   current_step_active: {processor.current_step_active}")
        print(f"   tool_completed_in_current_step: {processor.tool_completed_in_current_step}")
        print(f"   need_new_step_for_text: {processor.need_new_step_for_text}")
        print(f"   has_text_started: {processor.has_text_started}")

if __name__ == "__main__":
    asyncio.run(test_step_creation())