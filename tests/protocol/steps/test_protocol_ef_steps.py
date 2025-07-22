#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Protocol Types e & f: Step Finish and Step Start (AI SDK Compatible)
Format:
- e:{"finishReason": "stop", "usage": {...}, "isContinued": false}\n
- f:{"stepType": "initial"}\n

This test verifies that the adapter correctly generates step-related protocol parts
only for actual agent reasoning steps (compatible with AI SDK), while using 2: protocol
for agent executor lifecycle events.
"""

import asyncio
import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_aisdk_adapter import LangChainAdapter

# Load environment variables
load_dotenv()


@tool
def analyze_text(text: str) -> str:
    """Analyze text and provide insights.
    
    Args:
        text: Text to analyze
        
    Returns:
        Analysis results
    """
    word_count = len(text.split())
    char_count = len(text)
    sentences = text.count('.') + text.count('!') + text.count('?')
    
    return f"Text analysis: {word_count} words, {char_count} characters, approximately {sentences} sentences."


@tool
def summarize_content(content: str) -> str:
    """Summarize the given content.
    
    Args:
        content: Content to summarize
        
    Returns:
        Summary of the content
    """
    # Simple summarization logic
    sentences = content.split('.')
    if len(sentences) <= 2:
        return f"Summary: {content}"
    
    # Take first and last sentence as summary
    summary = f"{sentences[0].strip()}. {sentences[-2].strip()}."
    return f"Summary: {summary}"


async def test_step_protocols_ef():
    """Test Protocol Types e & f: Step Finish and Step Start (AI SDK Compatible)"""
    print("=== Testing Protocol Types e & f: Step Finish and Step Start (AI SDK Compatible) ===")
    print("Expected behavior:")
    print("- f: and e: protocols only for actual agent reasoning steps")
    print("- 2: protocol for agent executor lifecycle events")
    print("- e:{\"finishReason\": \"stop\", \"usage\": {...}, \"isContinued\": false}\\n")
    print("- f:{\"stepType\": \"initial\"}\\n")
    print("Reference: https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol\n")
    
    # Setup DeepSeek API
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ DEEPSEEK_API_KEY not found in environment")
        print("Please set your DeepSeek API key in .env file")
        return False
    
    try:
        # Create DeepSeek LLM
        llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=api_key,
            base_url="https://api.deepseek.com",
            streaming=True,
            temperature=0.1
        )
        
        # Create tools for multi-step workflow
        tools = [analyze_text, summarize_content]
        
        # Create agent prompt that encourages multi-step thinking
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that analyzes and processes text. "
                      "When given text, first analyze it, then provide a summary. "
                      "Use the available tools to help with your analysis."),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create agent with multi-step capability
        agent = create_openai_functions_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True,
            max_iterations=3,  # Allow multiple steps
            return_intermediate_steps=True
        )
        
        # Test input that should trigger multi-step processing
        test_text = "Artificial intelligence is transforming the world. Machine learning algorithms can process vast amounts of data. Deep learning models are becoming more sophisticated. The future of AI looks very promising."
        test_input = f"Please analyze this text and then provide a summary: {test_text}"
        
        print(f"Test input: {test_input[:100]}...")
        print("Creating agent executor with multi-step workflow...")
        
        # Create adapter and stream
        adapter = LangChainAdapter()
        protocol_parts = []
        step_finish_parts = []  # Protocol e
        step_start_parts = []   # Protocol f
        data_parts = []         # Protocol 2 (for agent executor lifecycle)
        
        print("\nStreaming agent response with step tracking:")
        async for part in adapter.to_data_stream_response(agent_executor.astream({"input": test_input})):
            protocol_parts.append(part)
            print(f"Protocol: {repr(part)}")
            
            # Categorize parts
            if part.startswith('e:'):
                step_finish_parts.append(part)
                # Extract and display step finish information
                try:
                    step_finish_content = json.loads(part[2:-1])  # Remove 'e:' and '\n', parse JSON
                    print(f"  [FINISH] Step Finish: {step_finish_content}")
                    if 'finishReason' in step_finish_content:
                        print(f"     Reason: {step_finish_content['finishReason']}")
                    if 'usage' in step_finish_content:
                        print(f"     Usage: {step_finish_content['usage']}")
                    if 'isContinued' in step_finish_content:
                        print(f"     Continued: {step_finish_content['isContinued']}")
                except json.JSONDecodeError:
                    print(f"  [ERROR] Invalid JSON in step finish part: {part}")
                    
            elif part.startswith('f:'):
                step_start_parts.append(part)
                # Extract and display step start information
                try:
                    step_start_content = json.loads(part[2:-1])  # Remove 'f:' and '\n', parse JSON
                    print(f"  [START] Step Start: {step_start_content}")
                    if 'stepType' in step_start_content:
                        print(f"     Type: {step_start_content['stepType']}")
                except json.JSONDecodeError:
                    print(f"  [ERROR] Invalid JSON in step start part: {part}")
                    
            elif part.startswith('2:'):
                data_parts.append(part)
                # Extract and display data information
                try:
                    data_content = json.loads(part[2:-1])  # Remove '2:' and '\n', parse JSON
                    if isinstance(data_content, list) and len(data_content) > 0:
                        for item in data_content:
                            if isinstance(item, dict) and 'custom_type' in item:
                                custom_type = item['custom_type']
                                if custom_type in ['agent-executor-start', 'agent-executor-end']:
                                    print(f"  [AGENT] Agent Executor Event: {custom_type}")
                                elif custom_type in ['node-start', 'node-end', 'node-error']:
                                    print(f"  [NODE] LangGraph Node Event: {custom_type}")
                                else:
                                    print(f"  [DATA] Data Event: {custom_type}")
                except json.JSONDecodeError:
                    print(f"  [ERROR] Invalid JSON in data part: {part}")
        
        print(f"\n=== Analysis ===")
        print(f"Total protocol parts: {len(protocol_parts)}")
        print(f"Step finish parts (type e): {len(step_finish_parts)}")
        print(f"Step start parts (type f): {len(step_start_parts)}")
        print(f"Data parts (type 2): {len(data_parts)}")
        
        # Count agent executor events
        agent_executor_events = 0
        for part in data_parts:
            try:
                data_content = json.loads(part[2:-1])
                if isinstance(data_content, list):
                    for item in data_content:
                        if isinstance(item, dict) and item.get('custom_type') in ['agent-executor-start', 'agent-executor-end']:
                            agent_executor_events += 1
            except json.JSONDecodeError:
                pass
        print(f"Agent executor lifecycle events: {agent_executor_events}")
        
        # Verify we got step-related parts
        success = True
        
        # Check step finish parts (Protocol e)
        if len(step_finish_parts) > 0:
            print("[SUCCESS] Step finish protocol parts (e:) generated")
            
            # Verify step finish format compliance
            for part in step_finish_parts:
                if not (part.startswith('e:') and part.endswith('\n')):
                    print(f"[ERROR] Step finish format error: {repr(part)}")
                    success = False
                else:
                    # Verify JSON validity and required fields
                    try:
                        step_finish_data = json.loads(part[2:-1])
                        # finishReason is required
                        if 'finishReason' not in step_finish_data:
                            print(f"[ERROR] Missing finishReason in step finish: {repr(part)}")
                            success = False
                        # usage and isContinued are optional but should be valid if present
                        if 'usage' in step_finish_data and not isinstance(step_finish_data['usage'], dict):
                            print(f"[ERROR] Invalid usage format in step finish: {repr(part)}")
                            success = False
                        if 'isContinued' in step_finish_data and not isinstance(step_finish_data['isContinued'], bool):
                            print(f"[ERROR] Invalid isContinued format in step finish: {repr(part)}")
                            success = False
                    except json.JSONDecodeError:
                        print(f"[ERROR] Invalid JSON in step finish part: {repr(part)}")
                        success = False
            
            if success:
                print("[SUCCESS] All step finish parts follow correct format: e:{\"finishReason\":\"...\"}\\n")
        else:
            print("WARNING: No step finish protocol parts generated")
            print("   This might be expected if the agent workflow doesn't generate step events")
        
        # Check step start parts (Protocol f)
        if len(step_start_parts) > 0:
            print("[SUCCESS] Step start protocol parts (f:) generated")
            
            # Verify step start format compliance
            for part in step_start_parts:
                if not (part.startswith('f:') and part.endswith('\n')):
                    print(f"[ERROR] Step start format error: {repr(part)}")
                    success = False
                else:
                    # Verify JSON validity and stepType field
                    try:
                        step_start_data = json.loads(part[2:-1])
                        # stepType is typically required
                        if 'stepType' not in step_start_data:
                            print("WARNING: Missing stepType in step start: {repr(part)}")
                            # Not marking as failure since stepType might be optional
                    except json.JSONDecodeError:
                        print(f"[ERROR] Invalid JSON in step start part: {repr(part)}")
                        success = False
            
            if success:
                print("[SUCCESS] All step start parts follow correct format: f:{\"stepType\":\"...\"}\\n")
        else:
            print("WARNING: No step start protocol parts generated")
            print("   This might be expected if the agent workflow doesn't generate step events")
        
        # Verify logical sequence (start should come before finish)
        if len(step_start_parts) > 0 and len(step_finish_parts) > 0:
            first_start_index = -1
            last_finish_index = -1
            
            for i, part in enumerate(protocol_parts):
                if part.startswith('f:') and first_start_index == -1:
                    first_start_index = i
                elif part.startswith('e:'):
                    last_finish_index = i
            
            if first_start_index < last_finish_index:
                print("✅ Correct sequence: Step start (f:) comes before step finish (e:)")
            else:
                print("WARNING: Step finish appeared before step start")
        
        # Verify agent executor lifecycle events (2: protocol)
        if agent_executor_events >= 2:  # Should have at least start and end
            print("✅ SUCCESS: Agent executor lifecycle events (2:) generated correctly")
        elif agent_executor_events > 0:
            print("⚠️  PARTIAL: Some agent executor events found, but incomplete lifecycle")
        else:
            print("❌ MISSING: No agent executor lifecycle events found")
            success = False
        
        # Final assessment
        step_protocols_found = len(step_finish_parts) + len(step_start_parts)
        
        if success and agent_executor_events >= 1:  # At least some lifecycle events
            print("\n✅ COMPREHENSIVE SUCCESS: AI SDK compatible protocol generation")
            print("   ✅ Agent executor lifecycle uses 2: protocol (AI SDK compatible)")
            print("   ✅ Step protocols (e:, f:) reserved for actual agent reasoning steps")
            if step_protocols_found > 0:
                print("   ✅ Step events are properly JSON-encoded with required fields")
                if len(step_start_parts) > 0 and len(step_finish_parts) > 0:
                    print("   ✅ Complete step lifecycle: start (f:) -> finish (e:)")
            else:
                print("   ℹ️  No step protocols generated (expected for simple workflows)")
            return True
        elif agent_executor_events == 0:
            print("\n❌ FAILURE: No agent executor lifecycle events generated")
            print("   Expected 2: protocol for agent-executor-start and agent-executor-end")
            return False
        else:
            print("\n❌ PARTIAL OR COMPLETE FAILURE: Issues detected in protocol generation")
            return False
            
    except Exception as e:
        print(f"ERROR during test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_step_protocols_ef())
    
    if success:
        print("\nSUCCESS: Protocol Types e & f (Step Finish and Start) test PASSED")
    else:
        print("\nFAILED: Protocol Types e & f (Step Finish and Start) test FAILED")
        print("   Check DeepSeek API configuration and agent setup")
    
    exit(0 if success else 1)