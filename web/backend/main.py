import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import ChatRequest, ChatResponse, ErrorResponse, StreamMode
from agents import create_agent_executor, format_chat_history
from stream_handlers import AutoStreamHandler, ManualStreamHandler

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    """自动流处理模式 - 使用 to_data_stream_response
    
    这种模式下，流的控制完全自动化，适合简单的对话场景。
    线程安全通过实例隔离保证。
    """
    if not agent_executor:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        # 格式化聊天历史
        chat_history = format_chat_history([msg.dict() for msg in request.messages[:-1]]) if len(request.messages) > 1 else []
        user_input = request.messages[-1].content
        
        logger.info(f"Auto stream request: {user_input[:100]}...")
        
        # 创建自动流
        stream = AutoStreamHandler.create_auto_stream(
            agent_executor=agent_executor,
            user_input=user_input,
            chat_history=chat_history,
            message_id=request.message_id
        )
        
        return StreamingResponse(
            stream,
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        logger.error(f"Auto stream error: {e}")
        # Return AI SDK compatible error format
        error_chunk = {
            "type": "error",
            "errorText": f"Auto stream processing failed: {str(e)}"
        }
        async def error_stream():
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
        
        return StreamingResponse(
            error_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

@app.post("/api/chat/manual", response_class=StreamingResponse)
async def chat_manual_stream(request: ChatRequest):
    """手动流处理模式 - 使用 LangChain 回调精确控制
    
    这种模式下，可以精确控制流的每个阶段，适合复杂的工作流场景。
    线程安全通过实例隔离和上下文管理器保证。
    """
    if not agent_executor:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        # 格式化聊天历史
        chat_history = format_chat_history([msg.dict() for msg in request.messages[:-1]]) if len(request.messages) > 1 else []
        user_input = request.messages[-1].content
        
        logger.info(f"Manual stream request: {user_input[:100]}...")
        
        # 创建手动流
        stream = ManualStreamHandler.create_manual_stream(
            agent_executor=agent_executor,
            user_input=user_input,
            chat_history=chat_history,
            message_id=request.message_id
        )
        
        return StreamingResponse(
            stream,
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        logger.error(f"Manual stream error: {e}")
        # Return AI SDK compatible error format
        error_chunk = {
            "type": "error",
            "errorText": f"Manual stream processing failed: {str(e)}"
        }
        async def error_stream():
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
        
        return StreamingResponse(
            error_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

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
        user_input = request.messages[-1].content
        
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