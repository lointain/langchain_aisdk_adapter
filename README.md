# LangChain AI SDK Adapter

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/langchain-aisdk-adapter.svg)](https://pypi.org/project/langchain-aisdk-adapter/)

A Python package that converts LangChain/LangGraph event streams to [AI SDK UI Stream Protocol](https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol) format, enabling seamless integration between LangChain applications and AI SDK UI components.

## Features

- 🔄 **Stream Conversion**: Convert LangChain/LangGraph streams to AI SDK UI Stream Protocol
- 🛠️ **Tool Call Support**: Handle tool calls and results with proper streaming
- 📊 **Usage Tracking**: Track token usage and model performance
- 🎯 **Type Safety**: Full TypeScript-style type hints with Pydantic models
- 🔧 **Easy Integration**: Simple adapter functions for quick setup
- 📝 **Comprehensive**: Support for text, reasoning, sources, files, and more

## Installation

### Basic Installation

```bash
pip install langchain-aisdk-adapter
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/lointain/langchain_aisdk_adapter.git
cd langchain_aisdk_adapter

# Install with uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[all]"

# Or install with pip
pip install -e ".[all]"
```

### Optional Dependencies

- **Examples**: `pip install langchain-aisdk-adapter[examples]` - For running example scripts
- **Testing**: `pip install langchain-aisdk-adapter[test]` - For running tests
- **Development**: `pip install langchain-aisdk-adapter[dev]` - For development tools
- **All**: `pip install langchain-aisdk-adapter[all]` - Install everything

For development:

```bash
pip install langchain-aisdk-adapter[dev]
```

## Quick Start

### Environment Setup

First, copy the example environment file and configure your API keys:

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### Basic Usage

```python
import asyncio
import os
from dotenv import load_dotenv
from langchain_aisdk_adapter import AISDKAdapter
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

async def main():
    # Load environment variables
    load_dotenv()
    
    # Create a LangChain model with environment variables
    model = ChatOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        model="deepseek-chat",
        streaming=True
    )
    
    # Create a stream
    stream = model.astream([HumanMessage(content="Hello, world!")])
    
    # Convert to AI SDK format
    async for ai_sdk_chunk in AISDKAdapter.astream(stream):
        print(ai_sdk_chunk, end="")

asyncio.run(main())
```

### With LangGraph

```python
import asyncio
from langchain_aisdk_adapter import AISDKAdapter
from langgraph import StateGraph
from langchain_openai import ChatOpenAI

async def main():
    # Create your LangGraph workflow
    workflow = StateGraph(...)
    app = workflow.compile()
    
    # Stream events from LangGraph
    stream = app.astream_events(
        {"messages": [{"role": "user", "content": "Hello!"}]},
        version="v1"
    )
    
    # Convert to AI SDK format
    async for ai_sdk_chunk in AISDKAdapter.astream(stream):
        print(ai_sdk_chunk, end="")

asyncio.run(main())
```

### Using AISDKPartEmitter

For more control over the streaming process:

```python
from langchain_aisdk_adapter import AISDKPartEmitter
from langchain_openai import ChatOpenAI

# Create an emitter
emitter = AISDKPartEmitter()

# Use with any LangChain runnable
model = ChatOpenAI(model="gpt-3.5-turbo")
chain = emitter.wrap(model)

# Stream with AI SDK format
async for chunk in chain.astream([{"role": "user", "content": "Hello!"}]):
    print(chunk, end="")
```

## AI SDK UI Stream Protocol

This adapter converts LangChain streams to the [AI SDK UI Stream Protocol](https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol) format. The protocol uses type identifiers to identify different types of content:

- `0:` - Text content
- `2:` - Data array
- `3:` - Error information
- `8:` - Message annotations
- `9:` - Tool call
- `a:` - Tool result
- `b:` - Tool call streaming start
- `c:` - Tool call delta
- `d:` - Finish message
- `e:` - Finish step
- `f:` - Start step
- `g:` - Reasoning content (see limitations below)
- `h:` - Source/reference
- `i:` - Redacted reasoning
- `j:` - Reasoning signature
- `k:` - File content
- And more...

### Protocol Support Status

**Fully Supported Protocols:**
- `0:` Text content - ✅ Full support
- `2:` Data array - ✅ Full support
- `3:` Error information - ✅ Full support
- `9:` Tool call - ✅ Full support
- `a:` Tool result - ✅ Full support
- `b:` Tool call streaming start - ✅ Full support
- `d:` Finish message - ✅ Full support
- `e:` Finish step - ✅ Full support
- `f:` Start step - ✅ Full support

**Not Currently Supported:**
- `1:` Text delta - ❌ Not implemented
- `4:` Function call - ❌ Not implemented
- `5:` Function call result - ❌ Not implemented
- `6:` Function call streaming start - ❌ Not implemented
- `7:` Function call delta - ❌ Not implemented
- `8:` Message annotations - ❌ Not implemented
- `c:` Tool call delta - ❌ Not implemented
- `g:` Reasoning content - ❌ Not implemented (inconsistent format across models)
- `h:` Source/reference - ❌ Not implemented
- `i:` Redacted reasoning - ❌ Not implemented
- `j:` Reasoning signature - ❌ Not implemented
- `k:` File content - ❌ Not implemented

The adapter focuses on the most commonly used protocols in LangChain workflows. Additional protocols can be implemented based on community needs.



## Supported LangChain Components

- **LangChain Models**: ChatOpenAI, ChatAnthropic, and other streaming models
- **LangGraph**: Full support for astream_events() output
- **Tool Calls**: Automatic handling of tool invocations and results
- **Agent Executors**: Support for agent reasoning and actions
- **Custom Chains**: Any LangChain Runnable with streaming support

## Advanced Usage

### Custom Part Creation

#### Using the New Factory Class (Recommended)

```python
from langchain_aisdk_adapter import AISDKFactory, factory

# Method 1: Using the class directly
text_part = AISDKFactory.text("Hello, world!")
tool_part = AISDKFactory.tool_call(
    tool_call_id="call_123",
    tool_name="search",
    args={"query": "Python"}
)
# Method 2: Using the convenience factory instance
text_part = factory.text("Hello, world!")
error_part = factory.error("Something went wrong")
data_part = factory.data([1, 2, 3])

# Method 3: Generic part creation
generic_part = factory.part('0', 'Any content')

# Get AI SDK format
print(text_part.ai_sdk_string)
print(tool_part.ai_sdk_string)
```

#### Using Legacy Functions (Backward Compatibility)

```python
from langchain_aisdk_adapter import (
    create_text_part,
    create_tool_call_part
)

# Create custom parts (legacy way)
text_part = create_text_part("Hello, world!")
tool_part = create_tool_call_part(
    tool_call_id="call_123",
    tool_name="search",
    args={"query": "Python"}
)

# Get AI SDK format
print(text_part.ai_sdk_string)
print(tool_part.ai_sdk_string)
```

#### Factory Benefits

- **Shorter method names**: `factory.text()` vs `create_text_part()`
- **Better IDE support**: Organized methods with auto-completion
- **Logical grouping**: Related methods grouped together
- **Multiple usage patterns**: Class methods, instance methods, or convenience instance

## Examples

The `examples/` directory contains comprehensive examples:

### Running Examples

1. **Setup environment**:
   ```bash
   cp .env.example .env
   # Edit .env and add your DeepSeek API key
   ```

2. **Install example dependencies**:
   ```bash
   pip install langchain-aisdk-adapter[examples]
   ```

3. **Run examples**:
   ```bash
   # Basic LangChain example
   python examples/langchain_basic_example.py
   
   # Advanced LangGraph workflow example
   python examples/langgraph_workflow_example.py
   
   # Factory usage patterns
   python examples/factory_usage.py
   ```

### Example Features

- **LangChain Integration**: Basic streaming with DeepSeek API
- **LangGraph Workflows**: Complex multi-step reasoning and tool usage
- **Tool Integration**: Mathematical calculations, weather queries, knowledge search
- **Environment Configuration**: Secure API key management with dotenv
- **Error Handling**: Comprehensive error handling and validation

### Error Handling

```python
import asyncio
from langchain_aisdk_adapter import AISDKAdapter

async def main():
    try:
        async for chunk in AISDKAdapter.astream(your_stream):
            print(chunk, end="")
    except Exception as e:
        print(f"Stream error: {e}")

asyncio.run(main())
```

## Development

### Setup

```bash
git clone https://github.com/lointain/langchain_aisdk_adapter.git
cd langchain_aisdk_adapter
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/langchain_aisdk_adapter

# Run specific test types
pytest -m unit
pytest -m integration
```

### Code Quality

```bash
# Format code
black src tests
isort src tests

# Lint
flake8 src tests
mypy src
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the amazing framework
- [AI SDK](https://ai-sdk.dev/) for the stream protocol specification
- [Vercel](https://vercel.com/) for the AI SDK UI components

## Links

- [GitHub Repository](https://github.com/lointain/langchain_aisdk_adapter)
- [PyPI Package](https://pypi.org/project/langchain-aisdk-adapter/)
- [AI SDK Documentation](https://ai-sdk.dev/docs)
- [LangChain Documentation](https://python.langchain.com/)
