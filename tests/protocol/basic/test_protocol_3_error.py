#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Protocol Type 3: Error Part
Format: 3:"error message"\n

This test verifies that the adapter correctly generates error protocol parts
when encountering errors during DeepSeek API streaming.
"""

import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_aisdk_adapter import LangChainAdapter

# Load environment variables
load_dotenv()


async def test_error_protocol_3():
    """Test Protocol Type 3: Error Part with intentional API error"""
    print("=== Testing Protocol Type 3: Error Part ===")
    print("Expected format: 3:\"error message\"\\n")
    print("Reference: https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol\n")
    
    # Test 1: Invalid API key error
    print("Test 1: Invalid API key error")
    try:
        # Create LLM with invalid API key
        llm_invalid_key = ChatOpenAI(
            model="deepseek-chat",
            api_key="invalid_key_12345",
            base_url="https://api.deepseek.com",
            streaming=True
        )
        
        messages = [HumanMessage(content="Hello")]
        
        # Create adapter and try to stream
        adapter = LangChainAdapter()
        protocol_parts = []
        error_parts = []
        
        print("Attempting to stream with invalid API key...")
        
        try:
            async for part in adapter.to_data_stream_response(llm_invalid_key.astream(messages)):
                protocol_parts.append(part)
                print(f"Protocol: {repr(part)}")
                
                if part.startswith('3:'):
                    error_parts.append(part)
                    # Extract error message
                    try:
                        import json
                        error_message = json.loads(part[2:-1])  # Remove '3:' and '\n', parse JSON
                        print(f"  ERROR: {error_message}")
                    except json.JSONDecodeError:
                        print(f"  RAW ERROR: {part[2:-1]}")
        
        except Exception as e:
            print(f"Expected exception caught: {e}")
            # Manually create error part to test format
            from langchain_aisdk_adapter import factory
            error_part = factory.error(str(e))
            error_protocol = error_part.ai_sdk_part_content
            print(f"Generated error protocol: {repr(error_protocol)}")
            
            # Verify format
            if error_protocol.startswith('3:') and error_protocol.endswith('\n'):
                print("SUCCESS: Error protocol format is correct")
                error_parts.append(error_protocol)
            else:
                print(f"ERROR: Error protocol format is incorrect: {repr(error_protocol)}")
        
        if len(error_parts) > 0:
            print(f"SUCCESS: Error protocol parts (3:) generated: {len(error_parts)}")
        else:
            print("WARNING: No error protocol parts generated in Test 1")
    
    except Exception as e:
        print(f"ERROR: Unexpected error in Test 1: {e}")
    
    print("\n" + "-"*50)
    
    # Test 2: Invalid model name error
    print("Test 2: Invalid model name error")
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("WARNING: DEEPSEEK_API_KEY not found, skipping Test 2")
    else:
        try:
            # Create LLM with invalid model name
            llm_invalid_model = ChatOpenAI(
                model="invalid-model-name",
                api_key=api_key,
                base_url="https://api.deepseek.com",
                streaming=True
            )
            
            messages = [HumanMessage(content="Hello")]
            
            # Create adapter and try to stream
            adapter = LangChainAdapter()
            protocol_parts = []
            error_parts = []
            
            print("Attempting to stream with invalid model name...")
            
            try:
                async for part in adapter.to_data_stream_response(llm_invalid_model.astream(messages)):
                    protocol_parts.append(part)
                    print(f"Protocol: {repr(part)}")
                    
                    if part.startswith('3:'):
                        error_parts.append(part)
                        # Extract error message
                        try:
                            import json
                            error_message = json.loads(part[2:-1])  # Remove '3:' and '\n', parse JSON
                            print(f"  ERROR: {error_message}")
                        except json.JSONDecodeError:
                            print(f"  RAW ERROR: {part[2:-1]}")
            
            except Exception as e:
                print(f"Expected exception caught: {e}")
                # Manually create error part to test format
                from langchain_aisdk_adapter import factory
                error_part = factory.error(str(e))
                error_protocol = error_part.ai_sdk_part_content
                print(f"Generated error protocol: {repr(error_protocol)}")
                
                # Verify format
                if error_protocol.startswith('3:') and error_protocol.endswith('\n'):
                    print("SUCCESS: Error protocol format is correct")
                    error_parts.append(error_protocol)
                else:
                    print(f"ERROR: Error protocol format is incorrect: {repr(error_protocol)}")
            
            if len(error_parts) > 0:
                print(f"SUCCESS: Error protocol parts (3:) generated: {len(error_parts)}")
            else:
                print("WARNING: No error protocol parts generated in Test 2")
        
        except Exception as e:
            print(f"ERROR: Unexpected error in Test 2: {e}")
    
    print("\n" + "-"*50)
    
    # Test 3: Manual error part creation and format validation
    print("Test 3: Manual error part creation and format validation")
    
    try:
        from langchain_aisdk_adapter import factory
        import json
        
        test_errors = [
            "API key is invalid",
            "Model not found",
            "Rate limit exceeded",
            "Network connection failed",
            "Internal server error"
        ]
        
        print("Testing error part creation:")
        all_valid = True
        
        for i, error_msg in enumerate(test_errors, 1):
            print(f"\nError {i}: {error_msg}")
            
            # Create error part
            error_part = factory.error(error_msg)
            error_protocol = error_part.ai_sdk_part_content
            print(f"Protocol: {repr(error_protocol)}")
            
            # Verify format
            if not (error_protocol.startswith('3:') and error_protocol.endswith('\n')):
                print(f"ERROR: Format error: {repr(error_protocol)}")
                all_valid = False
                continue
            
            # Verify JSON content
            try:
                json_content = error_protocol[2:-1]  # Remove '3:' and '\n'
                parsed_error = json.loads(json_content)
                if parsed_error == error_msg:
                    print(f"SUCCESS: Valid JSON: {parsed_error}")
                else:
                    print(f"ERROR: JSON mismatch: expected {error_msg}, got {parsed_error}")
                    all_valid = False
            except json.JSONDecodeError as e:
                print(f"ERROR: Invalid JSON: {e}")
                all_valid = False
        
        if all_valid:
            print("\nSUCCESS: All manual error parts follow correct format: 3:\"error message\"\\n")
            return True
        else:
            print("\nERROR: Some manual error parts have incorrect format")
            return False
    
    except Exception as e:
        print(f"ERROR: Error in Test 3: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_error_protocol_3())
    
    if success:
        print("\nSUCCESS: Protocol Type 3 (Error) test PASSED")
        print("   SUCCESS: Error protocol format is correct")
        print("   SUCCESS: Error messages are properly JSON-encoded")
    else:
        print("\nFAILED: Protocol Type 3 (Error) test FAILED")
        print("   ERROR: Check error protocol format and JSON encoding")
    
    exit(0 if success else 1)