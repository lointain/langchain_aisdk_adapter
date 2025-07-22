"""LangChain to AI SDK Adapter.

A Python package that converts LangChain/LangGraph event streams to AI SDK UI Stream Protocol format.
"""

from .adapter import to_ui_message_stream
from .enhanced_adapter import to_data_stream
from .callbacks import StreamCallbacks
from .ai_sdk_callbacks import (
    BaseAICallbackHandler,
    Message,
    UIPart,
    TextUIPart,
    ToolInvocationUIPart,
    ToolInvocation,
    StepStartUIPart,
    Attachment,
    LanguageModelUsage,
)
from .models import (
    LangChainAIMessageChunk,
    LangChainMessageContent,
    LangChainMessageContentComplex,
    LangChainMessageContentImageUrl,
    LangChainMessageContentText,
    LangChainStreamEvent,
    UIMessageChunk,
    ProviderMetadata,
    UIMessageChunkTextStart,
    UIMessageChunkTextDelta,
    UIMessageChunkTextEnd,
    UIMessageChunkError,
    UIMessageChunkToolInputStart,
    UIMessageChunkToolInputDelta,
    UIMessageChunkToolInputAvailable,
    UIMessageChunkToolOutputAvailable,
    UIMessageChunkToolOutputError,
    UIMessageChunkReasoning,
    UIMessageChunkReasoningStart,
    UIMessageChunkReasoningDelta,
    UIMessageChunkReasoningEnd,
    UIMessageChunkReasoningPartFinish,
    UIMessageChunkSourceUrl,
    UIMessageChunkSourceDocument,
    UIMessageChunkFile,
    UIMessageChunkData,
    UIMessageChunkStartStep,
    UIMessageChunkFinishStep,
    UIMessageChunkStart,
    UIMessageChunkFinish,
    UIMessageChunkAbort,
    UIMessageChunkMessageMetadata,
)

__version__ = "0.0.1a1"

__all__ = [
    # Main adapter functions
    "to_ui_message_stream",
    "to_data_stream",
    
    # Callback systems
    "StreamCallbacks",
    "BaseAICallbackHandler",
    
    # AI SDK models
    "Message",
    "UIPart",
    "TextUIPart",
    "ToolInvocationUIPart",
    "ToolInvocation",
    "StepStartUIPart",
    "Attachment",
    "LanguageModelUsage",
    
    # LangChain types
    "LangChainAIMessageChunk",
    "LangChainMessageContent",
    "LangChainMessageContentComplex",
    "LangChainMessageContentImageUrl",
    "LangChainMessageContentText",
    "LangChainStreamEvent",
    
    # UI Message Chunk types
    "UIMessageChunk",
    "ProviderMetadata",
    "UIMessageChunkTextStart",
    "UIMessageChunkTextDelta",
    "UIMessageChunkTextEnd",
    "UIMessageChunkError",
    "UIMessageChunkToolInputStart",
    "UIMessageChunkToolInputDelta",
    "UIMessageChunkToolInputAvailable",
    "UIMessageChunkToolOutputAvailable",
    "UIMessageChunkToolOutputError",
    "UIMessageChunkReasoning",
    "UIMessageChunkReasoningStart",
    "UIMessageChunkReasoningDelta",
    "UIMessageChunkReasoningEnd",
    "UIMessageChunkReasoningPartFinish",
    "UIMessageChunkSourceUrl",
    "UIMessageChunkSourceDocument",
    "UIMessageChunkFile",
    "UIMessageChunkData",
    "UIMessageChunkStartStep",
    "UIMessageChunkFinishStep",
    "UIMessageChunkStart",
    "UIMessageChunkFinish",
    "UIMessageChunkAbort",
    "UIMessageChunkMessageMetadata",
]