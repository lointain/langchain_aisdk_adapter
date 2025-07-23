import asyncio
import json
import logging
import sys
import os
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

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
from langchain_aisdk_adapter import LangChainAdapter, BaseAICallbackHandler, Message

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
    """自动模式 (默认) - 使用 LangChainAdapter.to_data_stream_response
    
    自动处理text-delta, tool相关事件等，适合简单的对话场景。
    """
    if not agent_executor:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        # 格式化聊天历史
        chat_history = format_chat_history([msg.dict() for msg in request.messages[:-1]]) if len(request.messages) > 1 else []
        user_input = request.messages[-1].content
        
        logger.info(f"Auto stream request: {user_input[:100]}...")
        
        # 1. 定义回调
        def on_start():
            print("Stream started")
         
        def on_final(final_text: str, message: Message):
            print(f"Final text: {final_text}")
            # 存储完整的message对象到数据库
            # save_to_database(message)
         
        class CustomCallbacks(BaseAICallbackHandler):
             async def on_start(self, message_id: str, options: Dict[str, Any]) -> None:
                 on_start()
             
             async def on_finish(self, message: Message, options: Dict[str, Any]) -> None:
                 on_final(message.content, message)
         
        callbacks = CustomCallbacks()
         
        # 2. 获取LangChain流
        agent_stream = agent_executor.astream_events(
            {"input": user_input, "chat_history": chat_history},
            version="v2"
        )
         
        # 3. 转换为AI SDK流（自动处理text-delta, tool相关事件等）
        ai_sdk_stream = await LangChainAdapter.to_data_stream_response(
            agent_stream,
            callbacks=callbacks,
            message_id=request.message_id
        )
         
        # 4. 可选：在流中手动添加额外内容
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
    可以在适当的时机调用emit方法发送自定义协议。
    """
    if not agent_executor:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        # 格式化聊天历史
        chat_history = format_chat_history([msg.dict() for msg in request.messages[:-1]]) if len(request.messages) > 1 else []
        user_input = request.messages[-1].content
        
        logger.info(f"Manual stream request: {user_input[:100]}...")
        
        # 1. 定义手动控制的回调
        class ManualCallbacks(BaseAICallbackHandler):
            def __init__(self, stream_emitter):
                self.stream_emitter = stream_emitter
            
            async def on_start(self, message_id: str, options: Dict[str, Any]) -> None:
                print("Manual stream started")
                # 可以在这里发送自定义协议
                await self.stream_emitter.emit_data({
                    "status": "started",
                    "mode": "manual",
                    "timestamp": str(asyncio.get_event_loop().time())
                })
            
            async def on_finish(self, message: Message, options: Dict[str, Any]) -> None:
                print(f"Manual stream finished: {message.content[:50]}...")
                # 发送完成状态和额外信息
                await self.stream_emitter.emit_data({
                    "status": "completed",
                    "final_message_length": len(message.content),
                    "parts_count": len(message.parts),
                    "timestamp": str(asyncio.get_event_loop().time())
                })
                
                # 可选：发送文件或其他资源
                # await self.stream_emitter.emit_file(
                #     url="https://example.com/report.pdf",
                #     mediaType="application/pdf"
                # )
        
        # 2. 获取LangChain流
        agent_stream = agent_executor.astream_events(
            {"input": user_input, "chat_history": chat_history},
            version="v2"
        )
        
        # 3. 转换为AI SDK流（关闭自动事件，手动控制）
        ai_sdk_stream = await LangChainAdapter.to_data_stream_response(
            agent_stream,
            callbacks=None,  # 先不传callbacks，稍后设置
            message_id=request.message_id,
            options={"auto_close": False}  # 手动控制关闭
        )
        
        # 4. 设置手动回调（传入stream emitter）
        manual_callbacks = ManualCallbacks(ai_sdk_stream)
        # 注意：这里需要重新创建stream，因为callbacks需要在创建时传入
        agent_stream = agent_executor.astream_events(
            {"input": user_input, "chat_history": chat_history},
            version="v2"
        )
        
        ai_sdk_stream = await LangChainAdapter.to_data_stream_response(
            agent_stream,
            callbacks=manual_callbacks,
            message_id=request.message_id,
            options={"auto_close": False}
        )
        
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