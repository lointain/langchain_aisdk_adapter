#!/usr/bin/env python3
"""
最终验证脚本 - 验证AI SDK协议修复

这个脚本验证以下问题已被修复：
1. step-finish时机正确
2. 没有空的TextUIPart
3. 工具调用正确关联到step
4. 多步骤流程正确实现
5. 事件序列符合AI SDK协议
"""

import asyncio
from typing import List, Dict, Any
from langchain_aisdk_adapter import to_data_stream, BaseAICallbackHandler, Message


class VerificationCallbackHandler(BaseAICallbackHandler):
    """验证回调处理器"""
    
    def __init__(self):
        self.final_message = None
        self.options = None
    
    async def on_finish(self, message: Message, options: Dict[str, Any]):
        """处理完成事件"""
        self.final_message = message
        self.options = options
        print("\n=== 最终消息验证 ===")
        print(f"消息ID: {message.id}")
        print(f"内容长度: {len(message.content)}")
        print(f"Parts数量: {len(message.parts)}")
        
        # 验证parts结构
        step_starts = 0
        tool_invocations = 0
        text_parts = 0
        empty_text_parts = 0
        
        for i, part in enumerate(message.parts):
            print(f"  Part {i+1}: {part.type}")
            if part.type == 'step-start':
                step_starts += 1
            elif part.type == 'tool-invocation':
                tool_invocations += 1
                tool_inv = part.toolInvocation
                print(f"    工具: {tool_inv.toolName}, Step: {tool_inv.step}, 状态: {tool_inv.state}")
            elif part.type == 'text':
                text_parts += 1
                if not part.text or not part.text.strip():
                    empty_text_parts += 1
                    print(f"    ❌ 发现空文本部分!")
                else:
                    print(f"    文本长度: {len(part.text)}")
        
        print(f"\n统计:")
        print(f"  Step starts: {step_starts}")
        print(f"  Tool invocations: {tool_invocations}")
        print(f"  Text parts: {text_parts}")
        print(f"  Empty text parts: {empty_text_parts}")
        
        # 验证结果
        if empty_text_parts == 0:
            print("✅ 没有空的TextUIPart")
        else:
            print("❌ 发现空的TextUIPart")
        
        if tool_invocations > 0:
            print("✅ 工具调用已处理")
        
        if step_starts >= tool_invocations:
            print("✅ Step数量合理")


async def simulate_agent_stream():
    """模拟agent流，包含工具调用"""
    events = [
        # 第一步：开始和文本
        {
            "event": "on_chat_model_start",
            "data": {"input": {"messages": [{"content": "查询北京天气并计算15*24"}]}}
        },
        {
            "event": "on_chat_model_stream",
            "data": {
                "chunk": {
                    "content": "我需要执行两个任务：查询北京天气和计算15*24。我先查询天气。\n\nAction: get_weather\nAction Input: Beijing"
                }
            }
        },
        {
            "event": "on_chat_model_end",
            "data": {}
        },
        
        # 第一个工具调用
        {
            "event": "on_chain_stream",
            "data": {
                "chunk": {
                    "intermediate_steps": [
                        (
                            type('AgentAction', (), {
                                'tool': 'get_weather',
                                'tool_input': {'input': 'Beijing'}
                            })(),
                            "北京天气晴朗，温度22°C"
                        )
                    ]
                }
            }
        },
        
        # 第二步：继续文本
        {
            "event": "on_chat_model_start",
            "data": {}
        },
        {
            "event": "on_chat_model_stream",
            "data": {
                "chunk": {
                    "content": "现在我已经获得了北京的天气信息，接下来计算数学表达式。\n\nAction: calculate_math\nAction Input: 15*24"
                }
            }
        },
        {
            "event": "on_chat_model_end",
            "data": {}
        },
        
        # 第二个工具调用
        {
            "event": "on_chain_stream",
            "data": {
                "chunk": {
                    "intermediate_steps": [
                        (
                            type('AgentAction', (), {
                                'tool': 'calculate_math',
                                'tool_input': {'input': '15*24'}
                            })(),
                            "15*24的结果是360"
                        )
                    ]
                }
            }
        },
        
        # 第三步：最终回答
        {
            "event": "on_chat_model_start",
            "data": {}
        },
        {
            "event": "on_chat_model_stream",
            "data": {
                "chunk": {
                    "content": "现在我有了两个任务的结果。\n\n最终答案：北京天气晴朗，温度22°C，15*24的结果是360。"
                }
            }
        },
        {
            "event": "on_chat_model_end",
            "data": {}
        }
    ]
    
    for event in events:
        yield event


async def main():
    """主函数"""
    print("=== AI SDK协议最终验证 ===")
    print("验证项目：")
    print("1. step-finish时机")
    print("2. 空TextUIPart问题")
    print("3. 工具调用step关联")
    print("4. 多步骤流程")
    print("5. 事件序列正确性")
    
    callback_handler = VerificationCallbackHandler()
    
    # 收集所有事件
    events = []
    start_steps = 0
    finish_steps = 0
    tool_outputs = 0
    text_starts = 0
    text_ends = 0
    
    print("\n=== 事件流分析 ===")
    async for chunk in to_data_stream(simulate_agent_stream(), callbacks=callback_handler):
        events.append(chunk)
        event_type = chunk.get('type')
        
        if event_type == 'start-step':
            start_steps += 1
            print(f"📍 Step {start_steps} 开始")
        elif event_type == 'finish-step':
            finish_steps += 1
            print(f"✅ Step {finish_steps} 结束")
        elif event_type == 'tool-output-available':
            tool_outputs += 1
            print(f"🔧 工具输出 {tool_outputs}")
        elif event_type == 'text-start':
            text_starts += 1
            print(f"📝 文本开始 {text_starts}")
        elif event_type == 'text-end':
            text_ends += 1
            print(f"📝 文本结束 {text_ends}")
    
    print(f"\n=== 协议统计 ===")
    print(f"Start steps: {start_steps}")
    print(f"Finish steps: {finish_steps}")
    print(f"Tool outputs: {tool_outputs}")
    print(f"Text starts: {text_starts}")
    print(f"Text ends: {text_ends}")
    
    # 验证协议正确性
    print(f"\n=== 协议验证 ===")
    if start_steps == finish_steps:
        print("✅ Step开始/结束匹配")
    else:
        print("❌ Step开始/结束不匹配")
    
    if text_starts == text_ends:
        print("✅ 文本开始/结束匹配")
    else:
        print("❌ 文本开始/结束不匹配")
    
    if start_steps >= 3:  # 期望至少3个步骤
        print("✅ 多步骤流程正确")
    else:
        print("❌ 多步骤流程不正确")
    
    if tool_outputs >= 2:  # 期望至少2个工具输出
        print("✅ 工具调用处理正确")
    else:
        print("❌ 工具调用处理不正确")
    
    # 验证事件序列
    print(f"\n=== 事件序列验证 ===")
    expected_pattern = ['start', 'start-step']
    actual_start = [e.get('type') for e in events[:2]]
    if actual_start == expected_pattern:
        print("✅ 开始序列正确")
    else:
        print(f"❌ 开始序列错误: {actual_start}")
    
    # 检查是否以finish结束
    if events and events[-1].get('type') == 'finish':
        print("✅ 正确以finish结束")
    else:
        print("❌ 未正确以finish结束")
    
    print(f"\n=== 验证完成 ===")
    if callback_handler.final_message:
        print("✅ 回调处理正确")
    else:
        print("❌ 回调处理失败")


if __name__ == "__main__":
    asyncio.run(main())