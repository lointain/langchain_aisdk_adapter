#!/usr/bin/env python3
"""
调试 complete_agent_callback_test.py 中的协议版本
"""

import asyncio
import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_aisdk_adapter import LangChainAdapter

# 简单的工具定义
@tool
def get_weather(city: str) -> str:
    """Get weather information for a city."""
    return f"The weather in {city} is sunny with 22°C temperature."

@tool
def calculate_math(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"

async def debug_protocol_version():
    """调试协议版本"""
    # 检查API密钥
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("Error: DEEPSEEK_API_KEY environment variable not set")
        return
    
    # 初始化语言模型
    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com",
        temperature=0.1,
        streaming=True
    )
    
    # 创建工具列表
    tools = [get_weather, calculate_math]
    
    # 创建简单的提示模板
    from langchain_core.prompts import PromptTemplate
    prompt = PromptTemplate.from_template(
        """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
    )
    
    # 创建代理
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # 测试查询
    test_query = "What is 2 + 2?"
    
    try:
        # 创建代理流
        agent_stream = agent_executor.astream_events({"input": test_query}, version="v2")
        
        # 测试1: 使用默认设置（应该是v4）
        print("\n=== 测试1: 默认设置 ===")
        ai_sdk_stream_default = LangChainAdapter.to_data_stream(
            agent_stream,
            message_id="test-default",
            options={"auto_events": True}
        )
        
        # 检查协议版本
        print(f"默认协议版本: {ai_sdk_stream_default._protocol_version}")
        
        # 获取第一个chunk来验证格式
        chunk_count = 0
        async for chunk in ai_sdk_stream_default:
            print(f"Chunk {chunk_count}: {chunk}")
            chunk_count += 1
            if chunk_count >= 3:  # 只看前3个chunk
                break
        
        # 重新创建流进行测试2
        agent_stream2 = agent_executor.astream_events({"input": test_query}, version="v2")
        
        # 测试2: 显式指定v5协议
        print("\n=== 测试2: 显式指定v5协议 ===")
        ai_sdk_stream_v5 = LangChainAdapter.to_data_stream(
            agent_stream2,
            message_id="test-v5",
            options={"auto_events": True, "protocol_version": "v5"}
        )
        
        print(f"v5协议版本: {ai_sdk_stream_v5._protocol_version}")
        
        # 获取第一个chunk来验证格式
        chunk_count = 0
        async for chunk in ai_sdk_stream_v5:
            print(f"Chunk {chunk_count}: {chunk}")
            chunk_count += 1
            if chunk_count >= 3:  # 只看前3个chunk
                break
                
    except Exception as e:
        print(f"Error during streaming: {str(e)}")
        import traceback
        traceback.print_exc()

async def main():
    await debug_protocol_version()

if __name__ == "__main__":
    asyncio.run(main())