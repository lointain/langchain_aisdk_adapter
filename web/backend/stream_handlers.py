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
    StreamCallbacks,
    ManualStreamController,
    manual_stream_context,
    StreamMode,
    Message
)
# ManualStreamController and manual_stream_context are now imported from langchain_aisdk_adapter
# This provides the complete implementation with instance isolation and context management

# 模拟 to_data_stream_response 函数
# 注意：to_data_stream_response 现在从 langchain_aisdk_adapter 导入
# 不再需要自定义实现

class AutoStreamCallbackHandler(BaseCallbackHandler):
    """Callback handler for auto stream mode"""
    
    def __init__(self):
        super().__init__()
        # Add required attributes for LangChain compatibility
        self.run_inline = True
    
    async def on_finish(self, message: Message, options: Dict[str, Any]) -> None:
        """Called when stream processing is complete"""
        # Convert message to dict for JSON serialization with datetime handling
        message_dict = message.dict()
        # Convert datetime to ISO string for JSON serialization
        if 'createdAt' in message_dict and hasattr(message_dict['createdAt'], 'isoformat'):
            message_dict['createdAt'] = message_dict['createdAt'].isoformat()
        print(f"[AUTO MODE] Serialized message: {json.dumps(message_dict, ensure_ascii=False, indent=2)}")
    
    # Add LangChain callback methods for compatibility
    async def on_llm_start(self, serialized, prompts, **kwargs):
        pass
    
    async def on_llm_new_token(self, token, **kwargs):
        pass
    
    async def on_llm_end(self, response, **kwargs):
        pass
    
    async def on_chain_start(self, serialized, inputs, **kwargs):
        pass
    
    async def on_chain_end(self, outputs, **kwargs):
        pass

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
            
            # Create callback handler for auto mode
            callback_handler = AutoStreamCallbackHandler()
            
            # Create LangChain stream with callback
            langchain_stream = agent_executor.astream(
                agent_input,
                config={"callbacks": [callback_handler]}
            )
            
            # Convert to AI SDK stream response with automatic handling
            response = await to_data_stream_response(
                langchain_stream,
                callbacks=callback_handler,
                message_id=message_id or str(uuid.uuid4())
            )
            
            # Yield formatted chunks from the response stream
            async for chunk in response.stream:
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
            message_id=message_id or str(uuid.uuid4()),
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