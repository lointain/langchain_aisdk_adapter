"""LangChain to AI SDK Adapter.

A Python package that converts LangChain/LangGraph event streams to AI SDK UI Stream Protocol format.
Provides comprehensive support for tool invocations, step control, and AI SDK callbacks.
"""

from .adapter import to_data_stream, to_data_stream_response, merge_into_data_stream, DataStreamResponse, DataStreamWriter
from .callbacks import (
    BaseAICallbackHandler,
    Message,
    UIPart,
    TextUIPart,
    ReasoningUIPart,
    ToolInvocationUIPart,
    ToolInvocation,
    SourceUIPart,
    FileUIPart,
    StepStartUIPart,
    ErrorUIPart,
    Attachment,
    LanguageModelUsage,
    StreamCallbacks,
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
    # Main adapter functions (AI SDK compatible)
    "to_data_stream",
    "to_data_stream_response", 
    "merge_into_data_stream",
    
    # Response and Writer classes
    "DataStreamResponse",
    "DataStreamWriter",
    
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