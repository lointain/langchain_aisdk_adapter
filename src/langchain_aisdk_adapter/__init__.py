"""LangChain AI SDK Adapter

A Python package that converts LangChain/LangGraph event streams to AI SDK UI Stream Protocol format.
"""

__version__ = "0.0.1a1"

# Import components from modules
from .adapter import LangChainAdapter
from .config import AdapterConfig, ThreadSafeAdapterConfig, default_config, safe_config
from .emitter import AISDKPartEmitter
from .factory import (
    AISDKFactory,
    factory,
    create_ai_sdk_part,
    create_text_part,
    create_reasoning_part,
    create_redacted_reasoning_part,
    create_reasoning_signature_part,
    create_source_part,
    create_file_part,
    create_data_part,
    create_message_annotation_part,
    create_error_part,
    create_tool_call_streaming_start_part,
    create_tool_call_delta_part,
    create_tool_call_part,
    create_tool_result_part,
    create_start_step_part,
    create_finish_step_part,
    create_finish_message_part,
)

__all__ = [
    "__version__",
    "LangChainAdapter",
    "AdapterConfig",
    "ThreadSafeAdapterConfig",
    "default_config",
    "safe_config",
    "AISDKPartEmitter",
    # Factory class and convenience instance
    "AISDKFactory",
    "factory",
    # Factory functions for all AI SDK protocol parts (backward compatibility)
    "create_ai_sdk_part",
    "create_text_part",
    "create_reasoning_part",
    "create_redacted_reasoning_part",
    "create_reasoning_signature_part",
    "create_source_part",
    "create_file_part",
    "create_data_part",
    "create_message_annotation_part",
    "create_error_part",
    "create_tool_call_streaming_start_part",
    "create_tool_call_delta_part",
    "create_tool_call_part",
    "create_tool_result_part",
    "create_start_step_part",
    "create_finish_step_part",
    "create_finish_message_part",
]