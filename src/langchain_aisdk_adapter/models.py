"""Type definitions for LangChain to AI SDK adapter."""

from typing import Any, Dict, List, Literal, Union
from typing_extensions import TypedDict, NotRequired


class LangChainMessageContentText(TypedDict):
    """Text content in LangChain message."""
    type: Literal["text"]
    text: str


class LangChainImageUrlSimple(TypedDict):
    """Simple image URL format."""
    url: str
    detail: NotRequired[Literal["auto", "low", "high"]]


class LangChainMessageContentImageUrl(TypedDict):
    """Image URL content in LangChain message."""
    type: Literal["image_url"]
    image_url: Union[str, LangChainImageUrlSimple]


class LangChainMessageContentOther(TypedDict, total=False):
    """Other content types in LangChain message."""
    type: str


LangChainMessageContentComplex = Union[
    LangChainMessageContentText,
    LangChainMessageContentImageUrl,
    LangChainMessageContentOther,
    Dict[str, Any],
]

LangChainMessageContent = Union[str, List[LangChainMessageContentComplex]]


class LangChainAIMessageChunk(TypedDict):
    """LangChain AI message chunk."""
    content: LangChainMessageContent


class LangChainStreamEventData(TypedDict, total=False):
    """Data field in LangChain stream event."""
    chunk: LangChainAIMessageChunk


class LangChainStreamEvent(TypedDict):
    """LangChain stream event (v2 format)."""
    event: str
    data: LangChainStreamEventData


class ProviderMetadata(TypedDict, total=False):
    """Provider metadata for UI message chunks."""
    pass  # Can be extended with specific provider metadata fields


class UIMessageChunkTextStart(TypedDict):
    """UI message chunk for text start."""
    type: Literal["text-start"]
    id: str
    providerMetadata: NotRequired[ProviderMetadata]


class UIMessageChunkTextDelta(TypedDict):
    """UI message chunk for text delta."""
    type: Literal["text-delta"]
    id: str
    delta: str
    providerMetadata: NotRequired[ProviderMetadata]


class UIMessageChunkTextEnd(TypedDict):
    """UI message chunk for text end."""
    type: Literal["text-end"]
    id: str
    providerMetadata: NotRequired[ProviderMetadata]


class UIMessageChunkError(TypedDict):
    """UI message chunk for error."""
    type: Literal["error"]
    errorText: str


class UIMessageChunkToolInputStart(TypedDict):
    """UI message chunk for tool input start."""
    type: Literal["tool-input-start"]
    toolCallId: str
    toolName: str
    providerExecuted: NotRequired[bool]


class UIMessageChunkToolInputDelta(TypedDict):
    """UI message chunk for tool input delta."""
    type: Literal["tool-input-delta"]
    toolCallId: str
    inputTextDelta: str


class UIMessageChunkToolInputAvailable(TypedDict):
    """UI message chunk for tool input available."""
    type: Literal["tool-input-available"]
    toolCallId: str
    toolName: str
    input: Any
    providerExecuted: NotRequired[bool]
    providerMetadata: NotRequired[ProviderMetadata]


class UIMessageChunkToolOutputAvailable(TypedDict):
    """UI message chunk for tool output available."""
    type: Literal["tool-output-available"]
    toolCallId: str
    output: Any
    providerExecuted: NotRequired[bool]


class UIMessageChunkToolOutputError(TypedDict):
    """UI message chunk for tool output error."""
    type: Literal["tool-output-error"]
    toolCallId: str
    errorText: str
    providerExecuted: NotRequired[bool]


class UIMessageChunkReasoning(TypedDict):
    """UI message chunk for reasoning."""
    type: Literal["reasoning"]
    text: str
    providerMetadata: NotRequired[ProviderMetadata]


class UIMessageChunkReasoningStart(TypedDict):
    """UI message chunk for reasoning start."""
    type: Literal["reasoning-start"]
    id: str
    providerMetadata: NotRequired[ProviderMetadata]


class UIMessageChunkReasoningDelta(TypedDict):
    """UI message chunk for reasoning delta."""
    type: Literal["reasoning-delta"]
    id: str
    delta: str
    providerMetadata: NotRequired[ProviderMetadata]


class UIMessageChunkReasoningEnd(TypedDict):
    """UI message chunk for reasoning end."""
    type: Literal["reasoning-end"]
    id: str
    providerMetadata: NotRequired[ProviderMetadata]


class UIMessageChunkReasoningPartFinish(TypedDict):
    """UI message chunk for reasoning part finish."""
    type: Literal["reasoning-part-finish"]


class UIMessageChunkSourceUrl(TypedDict):
    """UI message chunk for source URL."""
    type: Literal["source-url"]
    sourceId: str
    url: str
    title: NotRequired[str]
    providerMetadata: NotRequired[ProviderMetadata]


class UIMessageChunkSourceDocument(TypedDict):
    """UI message chunk for source document."""
    type: Literal["source-document"]
    sourceId: str
    mediaType: str
    title: str
    filename: NotRequired[str]
    providerMetadata: NotRequired[ProviderMetadata]


class UIMessageChunkFile(TypedDict):
    """UI message chunk for file."""
    type: Literal["file"]
    url: str
    mediaType: str
    providerMetadata: NotRequired[ProviderMetadata]


class UIMessageChunkData(TypedDict):
    """UI message chunk for custom data."""
    type: str  # Should start with 'data-'
    id: NotRequired[str]
    data: Any
    transient: NotRequired[bool]


class UIMessageChunkStartStep(TypedDict):
    """UI message chunk for start step."""
    type: Literal["start-step"]


class UIMessageChunkFinishStep(TypedDict):
    """UI message chunk for finish step."""
    type: Literal["finish-step"]


class UIMessageChunkStart(TypedDict):
    """UI message chunk for start."""
    type: Literal["start"]
    messageId: NotRequired[str]
    messageMetadata: NotRequired[Any]


class UIMessageChunkFinish(TypedDict):
    """UI message chunk for finish."""
    type: Literal["finish"]
    messageMetadata: NotRequired[Any]


class UIMessageChunkAbort(TypedDict):
    """UI message chunk for abort."""
    type: Literal["abort"]


class UIMessageChunkMessageStart(TypedDict):
    """UI message chunk for message start."""
    type: Literal["message-start"]
    role: str
    messageId: NotRequired[str]


class UIMessageChunkMessageEnd(TypedDict):
    """UI message chunk for message end."""
    type: Literal["message-end"]
    messageId: NotRequired[str]


class UIMessageChunkMessageMetadata(TypedDict):
    """UI message chunk for message metadata."""
    type: Literal["message-metadata"]
    messageMetadata: Any


UIMessageChunk = Union[
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
    UIMessageChunkMessageStart,
    UIMessageChunkMessageEnd,
    UIMessageChunkAbort,
    UIMessageChunkMessageMetadata,
]


# Union type for all supported input stream types
LangChainStreamInput = Union[
    LangChainStreamEvent,
    LangChainAIMessageChunk,
    str,
]