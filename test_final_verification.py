#!/usr/bin/env python3

import asyncio
import uuid
from src.langchain_aisdk_adapter.enhanced_adapter import EnhancedStreamProcessor

async def test_comprehensive_fixes():
    """Comprehensive test for all fixes."""
    
    print("=== Comprehensive Fix Verification ===")
    print("Testing: step-finish timing, multi-step flow, text-id reset")
    print()
    
    # Create processor
    processor = EnhancedStreamProcessor()
    
    # Simulate complete LangChain events matching real flow
    events = [
        # Chat model start - triggers first step
        {"event": "on_chat_model_start", "data": {}},
        
        # Initial text (thinking)
        "I am retrieving the current weather for the location you provided.",
        
        # Tool start
        {
            "event": "on_tool_start", 
            "data": {
                "run_id": "tool_123",
                "name": "getWeather",
                "inputs": {"latitude": 51.9235, "longitude": 4.4813}
            }
        },
        
        # Tool end
        {
            "event": "on_tool_end", 
            "data": {
                "run_id": "tool_123",
                "outputs": {"latitude": 51.916, "longitude": 4.495, "temperature": 18.2}
            }
        },
        
        # Chat model end (should trigger finish-step for tool step)
        {"event": "on_chat_model_end", "data": {}},
        
        # Final answer text (should trigger new start-step)
        "The current temperature at your location (lat: 51.9235, lon: 4.4813) is 18.2°C."
    ]
    
    all_events = []
    step_events = []
    text_events = []
    tool_events = []
    
    async def simulate_stream():
        for event in events:
            yield event
    
    async for chunk in processor.process_stream(simulate_stream()):
        all_events.append(chunk)
        
        if chunk["type"] in ["start-step", "finish-step"]:
            step_events.append(chunk)
        elif chunk["type"] in ["text-start", "text-end"]:
            text_events.append(chunk)
        elif "tool" in chunk["type"]:
            tool_events.append(chunk)
    
    print("=== Event Sequence ===")
    for i, event in enumerate(all_events, 1):
        event_type = event["type"]
        if event_type in ["start-step", "finish-step"]:
            print(f"  {i:2d}. {event_type} ⭐")
        elif event_type in ["text-start", "text-end"]:
            text_id = event.get("id", "N/A")
            print(f"  {i:2d}. {event_type} (id: {text_id[-8:]})")
        elif "tool" in event_type:
            tool_id = event.get("toolCallId", "N/A")
            print(f"  {i:2d}. {event_type} (tool: {tool_id})")
        else:
            print(f"  {i:2d}. {event_type}")
    
    print()
    print("=== Verification Results ===")
    
    # Check 1: Multi-step pattern
    step_types = [e["type"] for e in step_events]
    expected_steps = ["start-step", "finish-step", "start-step", "finish-step"]
    
    if step_types == expected_steps:
        print("✅ Multi-step pattern: CORRECT")
        print(f"   Pattern: {' -> '.join(step_types)}")
    else:
        print("❌ Multi-step pattern: INCORRECT")
        print(f"   Expected: {' -> '.join(expected_steps)}")
        print(f"   Found: {' -> '.join(step_types)}")
    
    # Check 2: Text ID reset between steps
    text_start_events = [e for e in text_events if e["type"] == "text-start"]
    if len(text_start_events) >= 2:
        first_text_id = text_start_events[0]["id"]
        second_text_id = text_start_events[1]["id"]
        
        if first_text_id != second_text_id:
            print("✅ Text ID reset: CORRECT")
            print(f"   First step text ID: {first_text_id[-8:]}")
            print(f"   Second step text ID: {second_text_id[-8:]}")
        else:
            print("❌ Text ID reset: INCORRECT")
            print(f"   Both steps use same text ID: {first_text_id[-8:]}")
    else:
        print("❌ Text ID reset: INSUFFICIENT DATA")
        print(f"   Found {len(text_start_events)} text-start events, expected 2+")
    
    # Check 3: Tool events sequence
    tool_sequence = [e["type"] for e in tool_events]
    expected_tool_sequence = ["tool-input-start", "tool-input-delta", "tool-input-available", "tool-output-available"]
    
    if tool_sequence == expected_tool_sequence:
        print("✅ Tool event sequence: CORRECT")
        print(f"   Sequence: {' -> '.join(tool_sequence)}")
    else:
        print("❌ Tool event sequence: INCORRECT")
        print(f"   Expected: {' -> '.join(expected_tool_sequence)}")
        print(f"   Found: {' -> '.join(tool_sequence)}")
    
    # Check 4: Step-finish timing (should come after tool-output-available)
    tool_output_index = None
    first_finish_step_index = None
    
    for i, event in enumerate(all_events):
        if event["type"] == "tool-output-available" and tool_output_index is None:
            tool_output_index = i
        elif event["type"] == "finish-step" and first_finish_step_index is None:
            first_finish_step_index = i
    
    if tool_output_index is not None and first_finish_step_index is not None:
        if first_finish_step_index > tool_output_index:
            print("✅ Step-finish timing: CORRECT")
            print(f"   tool-output-available at position {tool_output_index + 1}")
            print(f"   finish-step at position {first_finish_step_index + 1}")
        else:
            print("❌ Step-finish timing: INCORRECT")
            print(f"   finish-step at {first_finish_step_index + 1} should come after tool-output-available at {tool_output_index + 1}")
    else:
        print("❌ Step-finish timing: MISSING EVENTS")
    
    print()
    print(f"Total events generated: {len(all_events)}")
    print(f"Step events: {len(step_events)}")
    print(f"Text events: {len(text_events)}")
    print(f"Tool events: {len(tool_events)}")

if __name__ == "__main__":
    asyncio.run(test_comprehensive_fixes())