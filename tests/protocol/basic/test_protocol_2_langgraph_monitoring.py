#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Protocol Type 2: Lifecycle Monitoring (AI SDK Compatible)
Format: 2:Array<JSONValue>\n

This test verifies that the adapter correctly generates protocol 2 data parts
for agent executor and LangGraph node lifecycle monitoring in DeepSeek environment.
This follows AI SDK standards where 2: protocol is used for lifecycle events.
"""

import asyncio
import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_aisdk_adapter import LangChainAdapter

try:
    from langgraph.graph import StateGraph, END
    from typing_extensions import TypedDict
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("Warning: LangGraph not available, skipping LangGraph tests")

# Load environment variables
load_dotenv()


class GraphState(TypedDict):
    """State for the graph"""
    messages: list
    current_step: str
    result: str


def create_test_graph(llm):
    """Create a simple test graph for monitoring"""
    
    def step_1(state: GraphState) -> GraphState:
        """First processing step"""
        print("Executing step_1")
        state["current_step"] = "step_1"
        state["result"] = "Step 1 completed"
        return state
    
    def step_2(state: GraphState) -> GraphState:
        """Second processing step"""
        print("Executing step_2")
        state["current_step"] = "step_2"
        
        # Call LLM in this step
        response = llm.invoke(state["messages"])
        state["messages"].append(response)
        state["result"] = "Step 2 with LLM completed"
        return state
    
    def step_3(state: GraphState) -> GraphState:
        """Final processing step"""
        print("Executing step_3")
        state["current_step"] = "step_3"
        state["result"] = "All steps completed"
        return state
    
    # Create graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("step_1", step_1)
    workflow.add_node("step_2", step_2)
    workflow.add_node("step_3", step_3)
    
    # Add edges
    workflow.set_entry_point("step_1")
    workflow.add_edge("step_1", "step_2")
    workflow.add_edge("step_2", "step_3")
    workflow.add_edge("step_3", END)
    
    return workflow.compile()


async def test_langgraph_monitoring_protocol_2():
    """Test Protocol Type 2: Lifecycle Monitoring (AI SDK Compatible) in DeepSeek environment"""
    print("=== Testing Protocol Type 2: Lifecycle Monitoring (AI SDK Compatible) ===")
    print("Expected format: 2:Array<JSONValue>\\n")
    print("Testing agent executor and LangGraph node lifecycle events in DeepSeek environment")
    print("Following AI SDK standards where 2: protocol is used for lifecycle events\n")
    
    if not LANGGRAPH_AVAILABLE:
        print("ERROR: LangGraph not available, skipping test")
        return False
    
    # Setup DeepSeek API
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not found in environment")
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
        
        # Create test graph
        graph = create_test_graph(llm)
        
        # Initial state
        initial_state = {
            "messages": [HumanMessage(content="Hello, please process this message through the workflow.")],
            "current_step": "",
            "result": ""
        }
        
        print("Starting LangGraph workflow with monitoring...")
        
        # Create adapter
        adapter = LangChainAdapter()
        
        # Stream with events to capture node monitoring
        protocol_parts = []
        node_events = []
        
        print("\nStreaming LangGraph events:")
        
        # Create an async generator from the events
        async def event_generator():
            async for event in graph.astream_events(initial_state, version="v1"):
                print(f"Event: {event.get('event', 'unknown')} - {event.get('name', 'unnamed')}")
                print(f"  Tags: {event.get('tags', [])}")
                print(f"  Run ID: {event.get('run_id', 'unknown')}")
                print(f"  Data keys: {list(event.get('data', {}).keys())}")
                yield event
        
        # Process through adapter with proper async generator
        async for part in adapter.to_data_stream_response(event_generator()):
            protocol_parts.append(part)
            print(f"Protocol: {repr(part)}")
            
            # Check for protocol 2 (data parts)
            if part.startswith('2:'):
                node_events.append(part)
                
                # Parse and analyze the data
                try:
                    json_content = part[2:-1]  # Remove '2:' and '\n'
                    data = json.loads(json_content)
                    print(f"  NODE DATA: {data}")
                    
                    # Verify data structure for lifecycle monitoring
                    if isinstance(data, list) and len(data) > 0:
                        for item in data:
                            if isinstance(item, dict) and 'custom_type' in item:
                                custom_type = item['custom_type']
                                if custom_type in ['node-start', 'node-end', 'node-error']:
                                    print(f"  SUCCESS: Valid LangGraph node monitoring data: {custom_type}")
                                    if 'node_id' in item:
                                        print(f"     Node ID: {item['node_id']}")
                                    if 'name' in item:
                                        print(f"     Node Name: {item['name']}")
                                elif custom_type in ['agent-executor-start', 'agent-executor-end']:
                                    print(f"  SUCCESS: Valid agent executor monitoring data: {custom_type}")
                                    if 'name' in item:
                                        print(f"     Agent Name: {item['name']}")
                                    if 'inputs' in item:
                                        print(f"     Inputs: {item['inputs']}")
                                    if 'output' in item:
                                        print(f"     Output: {item['output']}")
                                else:
                                    print(f"  INFO: Other lifecycle event: {custom_type}")
                except json.JSONDecodeError as e:
                    print(f"  ERROR: Invalid JSON in protocol 2: {e}")
        
        print(f"\n=== Analysis ===")
        print(f"Total protocol parts: {len(protocol_parts)}")
        print(f"Protocol 2 (lifecycle monitoring) parts: {len(node_events)}")
        
        # Count different types of lifecycle events
        node_monitoring_events = 0
        agent_executor_events = 0
        other_lifecycle_events = 0
        
        for part in node_events:
            try:
                json_content = part[2:-1]
                data = json.loads(json_content)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'custom_type' in item:
                            custom_type = item['custom_type']
                            if custom_type in ['node-start', 'node-end', 'node-error']:
                                node_monitoring_events += 1
                            elif custom_type in ['agent-executor-start', 'agent-executor-end']:
                                agent_executor_events += 1
                            else:
                                other_lifecycle_events += 1
            except json.JSONDecodeError:
                pass
        
        print(f"LangGraph node monitoring events: {node_monitoring_events}")
        print(f"Agent executor lifecycle events: {agent_executor_events}")
        print(f"Other lifecycle events: {other_lifecycle_events}")
        
        # Verify we got lifecycle monitoring data
        if len(node_events) > 0:
            print("SUCCESS: Lifecycle monitoring data captured via protocol 2")
            
            # Verify format compliance
            format_correct = True
            valid_monitoring_data = node_monitoring_events + agent_executor_events + other_lifecycle_events
            
            for part in node_events:
                if not (part.startswith('2:') and part.endswith('\n')):
                    print(f"ERROR: Format error in: {repr(part)}")
                    format_correct = False
                else:
                    # Verify JSON validity
                    try:
                        json_content = part[2:-1]
                        data = json.loads(json_content)
                        if not isinstance(data, list):
                            print(f"ERROR: Protocol 2 data should be an array: {repr(part)}")
                            format_correct = False
                    except json.JSONDecodeError:
                        print(f"ERROR: Invalid JSON in: {repr(part)}")
                        format_correct = False
            
            print(f"Total valid lifecycle monitoring events: {valid_monitoring_data}")
            
            if format_correct and valid_monitoring_data > 0:
                # Additional check: ensure we have both node and agent executor events if expected
                if node_monitoring_events > 0:
                    print("\nTEST PASSED: Protocol 2 correctly captures lifecycle monitoring data")
                    print(f"  - LangGraph node events: {node_monitoring_events}")
                    print(f"  - Agent executor events: {agent_executor_events}")
                    print(f"  - Other lifecycle events: {other_lifecycle_events}")
                    return True
                else:
                    print("\nTEST FAILED: Expected LangGraph node monitoring events but found none")
                    return False
            else:
                print("\nTEST FAILED: Protocol 2 format or data validation failed")
                return False
        else:
            print("\nTEST FAILED: No protocol 2 lifecycle monitoring data found")
            return False
            
    except Exception as e:
        print(f"ERROR: Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_langgraph_monitoring_protocol_2())
    
    if success:
        print("\nProtocol Type 2 (LangGraph Monitoring) test PASSED")
        print("   SUCCESS: DeepSeek environment correctly generates node monitoring data")
    else:
        print("\nProtocol Type 2 (LangGraph Monitoring) test FAILED")
        print("   ERROR: Check DeepSeek API configuration and LangGraph setup")
    
    exit(0 if success else 1)