# Dependencies Guide

## Core Dependencies

The `langchain-aisdk-adapter` has minimal core dependencies to keep the package lightweight:

- `pydantic>=2.0.0` - For data validation and serialization
- `typing-extensions>=4.0.0` - For enhanced type hints

## Optional Dependencies

All other dependencies are optional and can be installed based on your specific needs:

### Web Integration (`web`)
For FastAPI integration and web server functionality:
```bash
uv add langchain-aisdk-adapter[web]
```
Includes: `fastapi`, `uvicorn`, `python-multipart`, `aiofiles`

### LangChain Integration (`langchain`)
For LangChain compatibility:
```bash
uv add langchain-aisdk-adapter[langchain]
```
Includes: `langchain-core`, `langchain-community`

### LangGraph Integration (`langgraph`)
For LangGraph compatibility:
```bash
uv add langchain-aisdk-adapter[langgraph]
```
Includes: `langgraph`

### HTTP Client (`http`)
For external HTTP requests:
```bash
uv add langchain-aisdk-adapter[http]
```
Includes: `aiohttp`

### Configuration (`config`)
For environment variable management:
```bash
uv add langchain-aisdk-adapter[config]
```
Includes: `python-dotenv`

### Development (`dev`)
For development and testing:
```bash
uv add langchain-aisdk-adapter[dev]
```
Includes: `pytest`, `black`, `isort`, `flake8`, `mypy`, `pre-commit`, etc.

### Examples (`examples`)
For running the example scripts:
```bash
uv add langchain-aisdk-adapter[examples]
```
Includes: `langchain`, `langgraph`, `langchain-openai`, `openai`, `python-dotenv`

### All Dependencies (`all`)
To install all optional dependencies:
```bash
uv add langchain-aisdk-adapter[all]
```

## Common Installation Patterns

### Minimal Installation (Core Only)
```bash
uv add langchain-aisdk-adapter
```
Use this if you only need the core adapter functionality without web or AI framework integration.

### Web Application
```bash
uv add langchain-aisdk-adapter[web,langchain,config]
```
For building web applications with LangChain integration.

### LangGraph Development
```bash
uv add langchain-aisdk-adapter[langgraph,config]
```
For LangGraph-based applications.

### Full Development Setup
```bash
uv add langchain-aisdk-adapter[all]
```
For contributors or when you need all features.

## Migration from Previous Versions

If you were using a previous version where all dependencies were required:

1. **For web applications**: Add `[web,langchain,config]` to your installation
2. **For examples**: Add `[examples]` to your installation  
3. **For development**: Add `[dev]` to your installation

This change significantly reduces the package size and installation time for users who don't need all features.

## Dependency Rationale

- **Core**: Only essential dependencies for the adapter's core functionality
- **Web**: FastAPI and related web server dependencies
- **LangChain/LangGraph**: AI framework specific dependencies
- **HTTP**: External HTTP client for advanced use cases
- **Config**: Environment configuration utilities
- **Dev**: Development and testing tools
- **Examples**: Dependencies needed to run example scripts

This modular approach follows Python packaging best practices and allows users to install only what they need.