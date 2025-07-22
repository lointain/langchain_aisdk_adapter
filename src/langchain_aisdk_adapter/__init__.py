"""LangChain AI SDK Adapter

A Python package that converts LangChain/LangGraph event streams to AI SDK UI Stream Protocol format.
"""

__version__ = "0.0.1a1"

# Import components from modules
from .adapter import LangChainAdapter
from .config import AdapterConfig
from .callbacks import (
    BaseAICallbackHandler, Message, LanguageModelUsage, UIPart, TextUIPart,
    ToolInvocationUIPart, ReasoningUIPart, SourceUIPart, FileUIPart,
    StepStartUIPart, ErrorUIPart, Attachment
)
from .factory import AISDKFactory, factory

__all__ = [
    "__version__",
    # Core adapter
    "LangChainAdapter",
    # Configuration
    "AdapterConfig",
    # Callback system
    "BaseAICallbackHandler",
    # Pydantic models
    "Message",
    "LanguageModelUsage",
    "UIPart",
    "TextUIPart",
    "ToolInvocationUIPart",
    "ReasoningUIPart",
    "SourceUIPart",
    "FileUIPart",
    "StepStartUIPart",
    "ErrorUIPart",
    "Attachment",
    # Factory for manual part creation
    "AISDKFactory",
    "factory",
]