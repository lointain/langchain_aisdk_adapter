import asyncio
import json
import logging
import sys
import os
import uuid
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from models import ChatRequest, ChatResponse, ErrorResponse, StreamMode
from agents import create_agent_executor, format_chat_history
from langchain_aisdk_adapter import LangChainAdapter, Message

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper function to create empty async generator
async def async_generator_from_list(items: List) -> AsyncGenerator:
    """Create an async generator from a list of items."""
    for item in items:
        yield item

# 全局变量存储 agent executor
agent_executor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global agent_executor
    
    # 启动时初始化 agent
    logger.info("Initializing LangChain Agent...")
    try:
        agent_executor = create_agent_executor()
        logger.info("Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        raise
    
    yield
    
    # 关闭时清理资源
    logger.info("Shutting down...")

# 创建 FastAPI 应用
app = FastAPI(
    title="LangChain AI SDK Adapter Demo",
    description="FastAPI backend with LangChain Agent integration",
    version="1.0.0",
    lifespan=lifespan
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """根路径健康检查"""
    return {
        "message": "LangChain AI SDK Adapter Demo",
        "status": "running",
        "agent_ready": agent_executor is not None
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "agent_ready": agent_executor is not None
    }

@app.get("/api/health")
async def api_health_check():
    """API健康检查端点"""
    return {
        "status": "healthy",
        "agent_ready": agent_executor is not None
    }

@app.post("/api/chat/auto", response_class=StreamingResponse)
async def chat_auto_stream(request: ChatRequest):
    """自动模式 (默认) - 使用 LangChainAdapter.to_data_stream_response
    
    自动处理text-delta, tool相关事件等，适合简单的对话场景。
    """
    if not agent_executor:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        # 格式化聊天历史
        chat_history = format_chat_history([msg.dict() for msg in request.messages[:-1]]) if len(request.messages) > 1 else []
        user_input = request.messages[-1].get_content()
        
        logger.info(f"Auto stream request: {user_input[:100]}...")
        
        # 1. 获取LangChain流
        agent_stream = agent_executor.astream_events(
            {"input": user_input, "chat_history": chat_history},
            version="v2"
        )
         
        # 2. 转换为AI SDK流（自动处理text-delta, tool相关事件等）
        ai_sdk_stream = LangChainAdapter.to_data_stream_response(
            agent_stream,
            callbacks=None,
            message_id=request.message_id,
            options={"protocol_version": request.protocol_version, "output_format": "protocol"}
        )
         
        # 3. 可选：在流中手动添加额外内容
         # 比如文件、自定义数据等无法自动检测的内容
         # await ai_sdk_stream.emit_file(url="report.pdf", mediaType="application/pdf")
         
        return ai_sdk_stream
        
    except Exception as e:
        logger.error(f"Auto stream error: {e}")
        raise HTTPException(status_code=500, detail=f"Auto stream processing failed: {str(e)}")

@app.post("/api/chat/manual", response_class=StreamingResponse)
async def chat_manual_stream(request: ChatRequest):
    """手动模式 - 使用 LangChain callback 调用 emit 方法发送协议
    
    在这种模式下，通过LangChain callback精确控制流的每个阶段，
    在文本输出和工具调用的位置手动调用emit方法发送自定义协议。
    """
    if not agent_executor:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        # 格式化聊天历史
        chat_history = format_chat_history([msg.dict() for msg in request.messages[:-1]]) if len(request.messages) > 1 else []
        user_input = request.messages[-1].get_content()
        
        logger.info(f"Manual stream request: {user_input[:100]}...")
        
        # 1. 首先创建一个空的流（关闭自动事件）
        empty_stream = async_generator_from_list([])
        ai_sdk_stream = LangChainAdapter.to_data_stream_response(
            empty_stream,
            callbacks=None,
            message_id=request.message_id,
            options={
                "auto_events": False,  # 关闭自动事件
                "protocol_version": request.protocol_version,
                "output_format": "protocol"
            }
        )
        
        # 2. 后台任务：直接在事件循环中处理并调用 adapter 的 emit 方法
        async def process_langchain_stream():
            try:
                # Send initial events
                await ai_sdk_stream.emit_start(request.message_id)
                await ai_sdk_stream.emit_start_step("agent", str(uuid.uuid4()))
                
                # Track state for text and tool events
                current_text_id = None
                current_tool_call_id = None
                accumulated_text = ""
                step_active = True
                
                # Process LangChain events and call adapter emit methods directly
                async for event in agent_executor.astream_events(
                    {"input": user_input, "chat_history": chat_history},
                    version="v2"
                ):
                    event_type = event.get("event")
                    event_data = event.get("data", {})
                    event_name = event.get("name", "")
                    
                    # Handle LLM events
                    if event_type == "on_llm_start":
                        current_text_id = f"text-{uuid.uuid4()}"
                        await ai_sdk_stream.emit_text_start(current_text_id)
                        
                    elif event_type == "on_llm_stream":
                        chunk = event_data.get("chunk")
                        if chunk and hasattr(chunk, 'content') and current_text_id:
                            content = chunk.content
                            if content:
                                await ai_sdk_stream.emit_text_delta(content, current_text_id)
                                accumulated_text += content
                                
                    elif event_type == "on_llm_end":
                        if current_text_id:
                            await ai_sdk_stream.emit_text_end(accumulated_text, current_text_id)
                            current_text_id = None
                            accumulated_text = ""
                            
                    # Handle tool events
                    elif event_type == "on_tool_start":
                        tool_name = event_name or "unknown"
                        current_tool_call_id = f"tool_{int(asyncio.get_event_loop().time() * 1000000)}"
                        
                        await ai_sdk_stream.emit_tool_input_start(
                            current_tool_call_id,
                            tool_name
                        )
                        
                        tool_input = event_data.get("input")
                        if tool_input:
                            # Emit input delta first
                            input_str = json.dumps(tool_input) if not isinstance(tool_input, str) else tool_input
                            await ai_sdk_stream.emit_tool_input_delta(current_tool_call_id, input_str)
                            
                            # Then emit input available
                            await ai_sdk_stream.emit_tool_input_available(
                                current_tool_call_id,
                                tool_name,
                                tool_input
                            )
                            
                    elif event_type == "on_tool_end":
                        if current_tool_call_id:
                            output = event_data.get("output")
                            if output:
                                await ai_sdk_stream.emit_tool_output_available(
                                    current_tool_call_id,
                                    output
                                )
                            current_tool_call_id = None
                            
                    elif event_type == "on_tool_error":
                        if current_tool_call_id:
                            error = event_data.get("error", "Unknown tool error")
                            await ai_sdk_stream.emit_tool_output_error(
                                current_tool_call_id,
                                str(error)
                            )
                            current_tool_call_id = None
                
                # Finish the stream
                if step_active:
                    await ai_sdk_stream.emit_finish_step("agent", str(uuid.uuid4()))
                await ai_sdk_stream.emit_finish(request.message_id)
                
            except Exception as e:
                logger.error(f"Error in manual stream processing: {e}")
                await ai_sdk_stream.emit_error(str(e))
                await ai_sdk_stream.emit_abort()
        
        # 3. 启动后台任务
        asyncio.create_task(process_langchain_stream())
        
        return ai_sdk_stream
        
    except Exception as e:
        logger.error(f"Manual stream error: {e}")
        raise HTTPException(status_code=500, detail=f"Manual stream processing failed: {str(e)}")

@app.post("/api/chat", response_class=StreamingResponse)
async def chat_unified(request: ChatRequest):
    """统一聊天端点 - 根据请求参数选择流处理模式
    
    支持通过 stream_mode 参数选择自动或手动模式。
    默认使用自动模式以保持简单性。
    """
    if not agent_executor:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    # 根据模式选择处理方式
    if request.stream_mode == StreamMode.MANUAL:
        return await chat_manual_stream(request)
    else:
        return await chat_auto_stream(request)

@app.post("/api/chat/sync")
async def chat_sync(request: ChatRequest) -> ChatResponse:
    """同步聊天端点 - 非流式响应
    
    用于不需要流式输出的场景，返回完整的响应。
    """
    if not agent_executor:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        # 格式化聊天历史
        chat_history = format_chat_history([msg.dict() for msg in request.messages[:-1]]) if len(request.messages) > 1 else []
        user_input = request.messages[-1].get_content()
        
        logger.info(f"Sync request: {user_input[:100]}...")
        
        # 准备输入
        agent_input = {
            "input": user_input,
            "chat_history": chat_history
        }
        
        # 执行 agent
        result = await agent_executor.ainvoke(agent_input)
        
        return ChatResponse(
            message_id=request.message_id,
            content=result.get("output", ""),
            metadata={
                "mode": "sync",
                "steps": result.get("intermediate_steps", [])
            }
        )
        
    except Exception as e:
        logger.error(f"Sync chat error: {e}")
        error_response = ErrorResponse(
            error="sync_chat_failed",
            message=f"Sync chat processing failed: {str(e)}"
        )
        raise HTTPException(status_code=500, detail=error_response.dict())

@app.get("/api/models")
async def get_models():
    """获取可用模型列表"""
    return {
        "models": [
            {
                "id": "gpt-3.5-turbo",
                "name": "GPT-3.5 Turbo",
                "description": "Fast and efficient model for most tasks"
            },
            {
                "id": "gpt-4",
                "name": "GPT-4",
                "description": "Most capable model for complex reasoning"
            }
        ],
        "default": "gpt-3.5-turbo"
    }

@app.get("/api/tools")
async def get_tools():
    """获取可用工具列表"""
    return {
        "tools": [
            {
                "name": "search_web",
                "description": "Search the web for current information",
                "parameters": ["query"]
            },
            {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "parameters": ["expression"]
            },
            {
                "name": "get_current_time",
                "description": "Get the current date and time",
                "parameters": []
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )