import asyncio
import uuid
import json
import sys
import os
from typing import AsyncGenerator, Dict, Any, Optional, List
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
from langchain.schema.messages import BaseMessage

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# 使用真正的 langchain_aisdk_adapter
from langchain_aisdk_adapter import (
    to_data_stream_response,
    BaseAICallbackHandler,
    StreamCallbacks
)
class ManualStreamController:
    """手动流控制器 - 线程安全"""
    
    def __init__(self, message_id: Optional[str] = None):
        self.message_id = message_id or str(uuid.uuid4())
        self._queue = asyncio.Queue(maxsize=1000)
        self._finished = False
        self._lock = asyncio.Lock()
        self._step_count = 0
        self._current_text_id = None
        self._current_step_active = False
        self._error_occurred = False
    
    async def emit_start(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """发送流开始事件"""
        async with self._lock:
            if not self._finished and not self._error_occurred:
                chunk = {"type": "start", "messageId": self.message_id}
                if metadata:
                    chunk["messageMetadata"] = metadata
                await self._queue.put(chunk)
    
    async def emit_start_step(self) -> None:
        """发送步骤开始事件"""
        async with self._lock:
            if not self._finished and not self._error_occurred:
                self._step_count += 1
                self._current_step_active = True
                await self._queue.put({"type": "start-step"})
    
    async def emit_finish_step(self) -> None:
        """发送步骤结束事件"""
        async with self._lock:
            if not self._finished and not self._error_occurred and self._current_step_active:
                self._current_step_active = False
                await self._queue.put({"type": "finish-step"})
    
    async def emit_text_start(self, text_id: Optional[str] = None, 
                             metadata: Optional[Dict[str, Any]] = None) -> str:
        """发送文本开始事件"""
        async with self._lock:
            if not self._finished and not self._error_occurred:
                text_id = text_id or f"text-{uuid.uuid4()}"
                self._current_text_id = text_id
                chunk = {"type": "text-start", "id": text_id}
                if metadata:
                    chunk["providerMetadata"] = metadata
                await self._queue.put(chunk)
                return text_id
    
    async def emit_text_delta(self, delta: str, text_id: Optional[str] = None,
                             metadata: Optional[Dict[str, Any]] = None) -> None:
        """发送文本增量事件"""
        async with self._lock:
            if not self._finished and not self._error_occurred:
                text_id = text_id or self._current_text_id
                if text_id:
                    chunk = {"type": "text-delta", "id": text_id, "delta": delta}
                    if metadata:
                        chunk["providerMetadata"] = metadata
                    await self._queue.put(chunk)
    
    async def emit_text_end(self, text_id: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """发送文本结束事件"""
        async with self._lock:
            if not self._finished and not self._error_occurred:
                text_id = text_id or self._current_text_id
                if text_id:
                    chunk = {"type": "text-end", "id": text_id}
                    if metadata:
                        chunk["providerMetadata"] = metadata
                    await self._queue.put(chunk)
                    self._current_text_id = None
    
    async def emit_tool_input_start(self, tool_call_id: str, tool_name: str,
                                   provider_executed: bool = False) -> None:
        """发送工具输入开始事件"""
        async with self._lock:
            if not self._finished and not self._error_occurred:
                chunk = {
                    "type": "tool-input-start",
                    "toolCallId": tool_call_id,
                    "toolName": tool_name
                }
                if provider_executed:
                    chunk["providerExecuted"] = provider_executed
                await self._queue.put(chunk)
    
    async def emit_tool_input_available(self, tool_call_id: str, tool_name: str, 
                                       input_data: Any, provider_executed: bool = False,
                                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """发送工具输入可用事件"""
        async with self._lock:
            if not self._finished and not self._error_occurred:
                chunk = {
                    "type": "tool-input-available",
                    "toolCallId": tool_call_id,
                    "toolName": tool_name,
                    "input": input_data
                }
                if provider_executed:
                    chunk["providerExecuted"] = provider_executed
                if metadata:
                    chunk["providerMetadata"] = metadata
                await self._queue.put(chunk)
    
    async def emit_tool_output_available(self, tool_call_id: str, output_data: Any,
                                        provider_executed: bool = False) -> None:
        """发送工具输出可用事件"""
        async with self._lock:
            if not self._finished and not self._error_occurred:
                chunk = {
                    "type": "tool-output-available",
                    "toolCallId": tool_call_id,
                    "output": output_data
                }
                if provider_executed:
                    chunk["providerExecuted"] = provider_executed
                await self._queue.put(chunk)
    
    async def emit_tool_output_error(self, tool_call_id: str, error_text: str,
                                    provider_executed: bool = False) -> None:
        """发送工具输出错误事件"""
        async with self._lock:
            if not self._finished and not self._error_occurred:
                chunk = {
                    "type": "tool-output-error",
                    "toolCallId": tool_call_id,
                    "errorText": error_text
                }
                if provider_executed:
                    chunk["providerExecuted"] = provider_executed
                await self._queue.put(chunk)
    
    async def emit_error(self, error_text: str) -> None:
        """发送错误事件"""
        async with self._lock:
            if not self._finished:
                self._error_occurred = True
                await self._queue.put({
                    "type": "error",
                    "errorText": error_text
                })
    
    async def finish(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """结束流"""
        async with self._lock:
            if not self._finished:
                self._finished = True
                # 确保当前步骤已结束
                if self._current_step_active:
                    await self._queue.put({"type": "finish-step"})
                    self._current_step_active = False
                
                # 发送结束事件
                chunk = {"type": "finish"}
                if metadata:
                    chunk["messageMetadata"] = metadata
                await self._queue.put(chunk)
                await self._queue.put(None)  # 结束标记
    
    async def get_stream(self) -> AsyncGenerator[Dict[str, Any], None]:
        """获取流生成器"""
        try:
            while True:
                chunk = await self._queue.get()
                if chunk is None:  # 结束标记
                    break
                yield chunk
        except Exception as e:
            # 确保在异常情况下也能正确结束
            await self.emit_error(f"Stream error: {str(e)}")
            raise

class manual_stream_context:
    """手动流控制上下文管理器"""
    
    def __init__(self, message_id: Optional[str] = None,
                 auto_start: bool = True,
                 auto_finish: bool = True):
        self.message_id = message_id
        self.auto_start = auto_start
        self.auto_finish = auto_finish
        self.controller = None
    
    async def __aenter__(self):
        self.controller = ManualStreamController(self.message_id)
        if self.auto_start:
            await self.controller.emit_start()
        return self.controller
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            await self.controller.emit_error(str(exc_val))
        elif self.auto_finish and not self.controller._finished:
            await self.controller.finish()

# 模拟 to_data_stream_response 函数
# 注意：to_data_stream_response 现在从 langchain_aisdk_adapter 导入
# 不再需要自定义实现

class AutoStreamHandler:
    """Automatic stream handler using to_data_stream_response"""
    
    @staticmethod
    async def create_auto_stream(
        agent_executor,
        user_input: str,
        chat_history: List[tuple] = None,
        message_id: str = None
    ) -> AsyncGenerator[str, None]:
        """Create automatic stream using to_data_stream_response.
        
        Args:
            agent_executor: LangChain agent executor
            user_input: User input message
            chat_history: Optional chat history
            message_id: Optional message ID
            
        Yields:
            Stream chunks in AI SDK format
        """
        try:
            # Prepare input for agent
            agent_input = {
                "input": user_input,
                "chat_history": chat_history or []
            }
            
            # Create LangChain stream
            langchain_stream = agent_executor.astream(agent_input)
            
            # Convert to AI SDK stream with automatic handling
            ai_sdk_stream = to_data_stream_response(
                langchain_stream,
                message_id=message_id
            )
            
            # Yield formatted chunks
            async for chunk in ai_sdk_stream:
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                
        except Exception as e:
            error_chunk = {
                "type": "error",
                "errorText": f"Auto stream error: {str(e)}"
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"

class ManualStreamCallbackHandler(BaseCallbackHandler):
    """Manual stream callback handler for precise control"""
    
    def __init__(self, controller: ManualStreamController):
        super().__init__()
        self.controller = controller
        self._text_runs = {}
        self._tool_runs = {}
        self._current_step_active = False
        self._agent_thinking = False
    
    async def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs
    ) -> None:
        """Called when chain starts"""
        if not self._current_step_active:
            await self.controller.emit_start_step()
            self._current_step_active = True
    
    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Called when chain ends"""
        if self._current_step_active:
            await self.controller.emit_finish_step()
            self._current_step_active = False
    
    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs
    ) -> None:
        """Called when LLM starts"""
        if not self._agent_thinking:
            self._agent_thinking = True
            # Start thinking step if not already active
            if not self._current_step_active:
                await self.controller.emit_start_step()
                self._current_step_active = True
    
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when LLM generates a new token"""
        run_id = kwargs.get('run_id', 'default')
        
        # Create text stream if not exists
        if run_id not in self._text_runs:
            self._text_runs[run_id] = await self.controller.emit_text_start(
                metadata={"type": "llm_response"}
            )
        
        # Emit token
        await self.controller.emit_text_delta(token, self._text_runs[run_id])
    
    async def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when LLM ends"""
        run_id = kwargs.get('run_id', 'default')
        
        # End text stream if exists
        if run_id in self._text_runs:
            await self.controller.emit_text_end(
                self._text_runs[run_id],
                metadata={"finish_reason": "stop"}
            )
            del self._text_runs[run_id]
        
        self._agent_thinking = False
    
    async def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Called when agent decides to use a tool"""
        run_id = str(kwargs.get('run_id', uuid.uuid4()))
        tool_name = action.tool
        tool_input = action.tool_input
        
        # Store tool run info
        self._tool_runs[run_id] = {
            "tool_name": tool_name,
            "started": True
        }
        
        # Emit tool input events
        await self.controller.emit_tool_input_start(run_id, tool_name)
        await self.controller.emit_tool_input_available(
            run_id, tool_name, tool_input,
            metadata={"reasoning": action.log}
        )
    
    async def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs
    ) -> None:
        """Called when tool starts executing"""
        run_id = str(kwargs.get('run_id', uuid.uuid4()))
        tool_name = serialized.get('name', 'unknown')
        
        # Update tool run info if exists, otherwise create new
        if run_id not in self._tool_runs:
            self._tool_runs[run_id] = {
                "tool_name": tool_name,
                "started": True
            }
            await self.controller.emit_tool_input_start(run_id, tool_name)
            await self.controller.emit_tool_input_available(run_id, tool_name, input_str)
    
    async def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when tool execution ends"""
        run_id = str(kwargs.get('run_id', ''))
        
        if run_id in self._tool_runs:
            # Emit tool output
            await self.controller.emit_tool_output_available(run_id, output)
            del self._tool_runs[run_id]
    
    async def on_tool_error(self, error: Exception, **kwargs) -> None:
        """Called when tool execution fails"""
        run_id = str(kwargs.get('run_id', ''))
        
        if run_id in self._tool_runs:
            await self.controller.emit_tool_output_error(run_id, str(error))
            del self._tool_runs[run_id]
    
    async def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """Called when agent finishes"""
        # Ensure any remaining text streams are closed
        for run_id, text_id in list(self._text_runs.items()):
            await self.controller.emit_text_end(text_id)
            del self._text_runs[run_id]
        
        # End current step if active
        if self._current_step_active:
            await self.controller.emit_finish_step()
            self._current_step_active = False

class ManualStreamHandler:
    """Manual stream handler with precise control"""
    
    @staticmethod
    async def create_manual_stream(
        agent_executor,
        user_input: str,
        chat_history: List[tuple] = None,
        message_id: str = None
    ) -> AsyncGenerator[str, None]:
        """Create manual stream with precise control.
        
        Args:
            agent_executor: LangChain agent executor
            user_input: User input message
            chat_history: Optional chat history
            message_id: Optional message ID
            
        Yields:
            Stream chunks in AI SDK format
        """
        async with manual_stream_context(
            message_id=message_id,
            auto_start=True,
            auto_finish=True
        ) as controller:
            try:
                # Create callback handler
                callback_handler = ManualStreamCallbackHandler(controller)
                
                # Prepare input for agent
                agent_input = {
                    "input": user_input,
                    "chat_history": chat_history or []
                }
                
                # Execute agent with manual callback
                result = await agent_executor.ainvoke(
                    agent_input,
                    config={"callbacks": [callback_handler]}
                )
                
                # Emit final result if needed
                if result.get("output"):
                    text_id = await controller.emit_text_start(
                        metadata={"type": "final_answer"}
                    )
                    await controller.emit_text_delta(result["output"], text_id)
                    await controller.emit_text_end(text_id)
                
            except Exception as e:
                await controller.emit_error(f"Manual stream error: {str(e)}")
        
        # Convert controller stream to SSE format
        async for chunk in controller.get_stream():
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"