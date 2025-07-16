#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Protocol Types 0 & d: Text Content and Finish Message
Format: 
- 0:"text content"\n
- d:{"finishReason": "stop", "usage": {"promptTokens": 10, "completionTokens": 20}}\n

This test verifies that the adapter correctly generates both text protocol parts
and finish message protocol parts when streaming from DeepSeek Chat API.
"""

import asyncio
import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_aisdk_adapter import LangChainAdapter

# Load environment variables
load_dotenv()


async def test_text_and_finish_protocols_0d():
    """Test Protocol Types 0 & d: Text Content and Finish Message with real DeepSeek API call"""
    print("=== Testing Protocol Types 0 & d: Text Content and Finish Message ===")
    print("Expected formats:")
    print("- 0:\"text content\"\\n")
    print("- d:{\"finishReason\": \"stop\", \"usage\": {...}}\\n")
    print("Reference: https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol\n")
    
    # Setup DeepSeek API
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not found in environment")
        print("Please set your DeepSeek API key in .env file")
        return False
    
    try:
        # Create DeepSeek LLM with streaming
        llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=api_key,
            base_url="https://api.deepseek.com",
            streaming=True,
            temperature=0.1
        )
        
        # Test message
        messages = [HumanMessage(content="Please write a short greeting message in exactly 2 sentences.")]
        
        print("Calling DeepSeek API with streaming...")
        
        # Create adapter and stream
        adapter = LangChainAdapter()
        protocol_parts = []
        text_parts = []
        finish_parts = []
        
        print("\nStreaming response:")
        async for part in adapter.to_data_stream_response(llm.astream(messages)):
            protocol_parts.append(part)
            print(f"Protocol: {repr(part)}")
            
            # Categorize parts
            if part.startswith('0:'):
                text_parts.append(part)
                # Extract and display text content
                try:
                    text_content = json.loads(part[2:-1])  # Remove '0:' and '\n', parse JSON
                    print(f"  TEXT: {text_content}")
                except json.JSONDecodeError:
                    print(f"  ERROR: Invalid JSON in text part: {part}")
                    
            elif part.startswith('d:'):
                finish_parts.append(part)
                # Extract and display finish information
                try:
                    finish_content = json.loads(part[2:-1])  # Remove 'd:' and '\n', parse JSON
                    print(f"  FINISH: {finish_content}")
                    if 'finishReason' in finish_content:
                        print(f"     Reason: {finish_content['finishReason']}")
                    if 'usage' in finish_content:
                        usage = finish_content['usage']
                        print(f"     Usage: {usage}")
                except json.JSONDecodeError:
                    print(f"  ERROR: Invalid JSON in finish part: {part}")
        
        print(f"\n=== Analysis ===")
        print(f"Total protocol parts: {len(protocol_parts)}")
        print(f"Text parts (type 0): {len(text_parts)}")
        print(f"Finish parts (type d): {len(finish_parts)}")
        
        # Verify we got both text and finish parts
        success = True
        
        # Check text parts
        if len(text_parts) > 0:
            print("SUCCESS: Text protocol parts (0:) generated")
            
            # Verify text format compliance
            for part in text_parts:
                if not (part.startswith('0:') and part.endswith('\n')):
                    print(f"ERROR: Text format error: {repr(part)}")
                    success = False
                else:
                    # Verify JSON validity
                    try:
                        json.loads(part[2:-1])
                    except json.JSONDecodeError:
                        print(f"ERROR: Invalid JSON in text part: {repr(part)}")
                        success = False
            
            if success:
                print("SUCCESS: All text parts follow correct format: 0:\"text\"\\n")
        else:
            print("ERROR: No text protocol parts generated")
            success = False
        
        # Check finish parts
        if len(finish_parts) > 0:
            print("SUCCESS: Finish protocol parts (d:) generated")
            
            # Verify finish format compliance
            for part in finish_parts:
                if not (part.startswith('d:') and part.endswith('\n')):
                    print(f"ERROR: Finish format error: {repr(part)}")
                    success = False
                else:
                    # Verify JSON validity and required fields
                    try:
                        finish_data = json.loads(part[2:-1])
                        if 'finishReason' not in finish_data:
                            print(f"ERROR: Missing finishReason in: {repr(part)}")
                            success = False
                        # usage is optional but should be valid if present
                        if 'usage' in finish_data:
                            usage = finish_data['usage']
                            if not isinstance(usage, dict):
                                print(f"ERROR: Invalid usage format in: {repr(part)}")
                                success = False
                    except json.JSONDecodeError:
                        print(f"ERROR: Invalid JSON in finish part: {repr(part)}")
                        success = False
            
            if success:
                print("SUCCESS: All finish parts follow correct format: d:{\"finishReason\":\"...\"}\\n")
        else:
            print("ERROR: No finish protocol parts generated")
            success = False
        
        # Verify logical sequence (text should come before finish)
        if success and len(text_parts) > 0 and len(finish_parts) > 0:
            last_text_index = -1
            first_finish_index = -1
            
            for i, part in enumerate(protocol_parts):
                if part.startswith('0:'):
                    last_text_index = i
                elif part.startswith('d:') and first_finish_index == -1:
                    first_finish_index = i
            
            if last_text_index < first_finish_index:
                print("SUCCESS: Correct sequence: Text parts come before finish parts")
            else:
                print("WARNING: Finish part appeared before last text part")
        
        # Final assessment
        if success:
            print("\nCOMPREHENSIVE SUCCESS: Both text (0:) and finish (d:) protocols work correctly")
            print("   SUCCESS: DeepSeek API streaming generates proper AI SDK protocol format")
            print("   SUCCESS: Text content is properly JSON-encoded")
            print("   SUCCESS: Finish message includes required finishReason")
            return True
        else:
            print("ERROR: PARTIAL OR COMPLETE FAILURE: Issues detected in protocol generation")
            return False
            
    except Exception as e:
        print(f"ERROR: Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_text_and_finish_protocols_0d())
    
    if success:
        print("\nProtocol Types 0 & d (Text and Finish) test PASSED")
    else:
        print("\nProtocol Types 0 & d (Text and Finish) test FAILED")
        print("   Check DeepSeek API configuration and network connection")
    
    exit(0 if success else 1)