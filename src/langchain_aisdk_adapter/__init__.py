"""LangChain to AI SDK Adapter - V1 Implementation.

This package provides adapters to convert LangChain streaming outputs
to AI SDK compatible data streams with full protocol support.
"""

# Import from individual modules
from .message_builder import MessageBuilder
from .protocol_generator import ProtocolGenerator
from .stream_processor import StreamProcessor
from .data_stream import DataStreamWithEmitters, DataStreamResponse, DataStreamWriter
from .protocol_strategy import ProtocolStrategy, AISDKv4Strategy, AISDKv5Strategy, ProtocolConfig
from .text_processing_adapter import TextProcessingAdapter
from .langchain_adapter import (
    LangChainAdapter,
    AdapterOptions
)

from .context import DataStreamContext
from .lifecycle import ContextLifecycleManager
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
    # Main adapter class (V1 Implementation)
    "LangChainAdapter",
    "AdapterOptions",
    
    # Core processing classes
    "MessageBuilder",
    "ProtocolGenerator",
    "StreamProcessor",
    
    # Protocol strategy classes
    "ProtocolStrategy",
    "AISDKv4Strategy",
    "AISDKv5Strategy",
    "ProtocolConfig",
    "TextProcessingAdapter",
    

    
    # Response and Writer classes
    "DataStreamResponse",
    "DataStreamWriter",
    "DataStreamWithEmitters",
    
    # Manual stream control

    
    # Context and lifecycle management
    "DataStreamContext",
    "ContextLifecycleManager",
    
    # Callback systems
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