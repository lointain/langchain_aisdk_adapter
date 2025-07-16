# FastAPI Web Examples

This directory contains FastAPI examples demonstrating how to use the LangChain AI SDK Adapter in web applications.

## Setup

1. Install the web dependencies:
   ```bash
   pip install -r web/requirements.txt
   ```

2. Set up your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   
   Or create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Running the Examples

1. Start the FastAPI server:
   ```bash
   uvicorn web.fastapi_example:app --reload
   ```

2. Visit the interactive API documentation:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## Available Endpoints

### Basic Chat (`/chat`)
Simple chat with LLM streaming:
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello, how are you?"}'
```

### Chat with Configuration (`/chat-with-config`)
Demonstrates custom adapter configuration:
```bash
curl -X POST "http://localhost:8000/chat-with-config" \
     -H "Content-Type: application/json" \
     -d '{"message": "Explain quantum computing"}'
```

### Chain-based Chat (`/chat-chain`)
Uses LangChain chains for complex prompt processing:
```bash
curl -X POST "http://localhost:8000/chat-chain" \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the latest developments?", "topic": "machine learning"}'
```

### Agent Chat (`/chat-agent`)
Demonstrates agents with tools (weather and calculator):
```bash
curl -X POST "http://localhost:8000/chat-agent" \
     -H "Content-Type: application/json" \
     -d '{"input": "What is the weather like in Tokyo and calculate 15 * 7?"}'
```

## Features Demonstrated

- **Basic LLM Streaming**: Simple chat with OpenAI models
- **Custom Configuration**: Using `AdapterConfig` to control protocol generation
- **LangChain Chains**: Complex prompt templates and processing pipelines
- **Agent Integration**: Tools and function calling with streaming responses
- **Error Handling**: Proper HTTP error responses
- **Type Safety**: Pydantic models for request/response validation

## Notes

- All endpoints return streaming responses using the AI SDK protocol
- The adapter automatically handles LangChain stream conversion
- Different configuration presets are demonstrated (minimal, tools_only)
- Tools include weather lookup and mathematical calculations