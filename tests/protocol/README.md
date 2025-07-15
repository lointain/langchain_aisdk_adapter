# Protocol Tests

This directory contains tests for the AI SDK UI Stream Protocol implementation. The tests are organized by protocol categories and use cases.

## Test Structure

```
tests/protocol/
├── basic/          # Basic protocols (0, 3, d)
├── tools/          # Tool-related protocols (9, a, b)
├── steps/          # Step-related protocols (e, f)
├── reasoning/      # Reasoning protocol (g)
├── run_tests.py    # Test runner script
└── README.md       # This file
```

## Protocol Categories

### Basic Protocols (`basic/`)
- **Protocol 0 (Text)**: `test_protocol_0_text.py` - Basic text streaming
- **Protocol 3 (Error)**: `test_protocol_3_error.py` - Error handling
- **Protocol d (Finish)**: `test_protocol_d_finish.py` - Stream completion

### Tool Protocols (`tools/`)
- **Protocol 9 (Tool Call)**: `test_protocol_9_tool_call.py` - Function/tool invocation
- **Protocol a (Tool Result)**: `test_protocol_a_tool_result.py` - Tool execution results
- **Protocol b (Tool Start)**: `test_protocol_b_tool_start.py` - Tool execution start

### Step Protocols (`steps/`)
- **Protocol e (Step Finish)**: `test_protocol_e_step_finish.py` - Agent step completion
- **Protocol f (Step Start)**: `test_protocol_f_step_start.py` - Agent step initiation

**Use Cases for Step Protocols:**
- **Agent Workflows**: Multi-step reasoning and planning
- **LangGraph Integration**: State machine transitions
- **Complex Task Decomposition**: Breaking down tasks into steps
- **Progress Tracking**: Monitoring agent execution progress

### Reasoning Protocol (`reasoning/`)
- **Protocol g (Reasoning)**: `test_protocol_g_reasoning.py` - Reasoning traces

**Use Cases for Reasoning Protocol:**
- **DeepSeek R1 Model**: Exposing internal reasoning steps
- **Chain-of-Thought**: Displaying thinking process
- **Debugging AI Logic**: Understanding model decisions
- **Educational Tools**: Showing problem-solving steps

## Running Tests

Use the test runner to execute specific categories:

```bash
# Run basic protocol tests
python tests/protocol/run_tests.py basic

# Run tool protocol tests
python tests/protocol/run_tests.py tools

# Run step protocol tests (for agents)
python tests/protocol/run_tests.py steps

# Run reasoning protocol tests (for DeepSeek R1)
python tests/protocol/run_tests.py reasoning

# Run all tests
python tests/protocol/run_tests.py all
```

## Protocol Applications

### When to Use Each Protocol

1. **Basic Protocols (0, 3, d)**
   - Standard chat applications
   - Simple Q&A systems
   - Basic streaming interfaces

2. **Tool Protocols (9, a, b)**
   - Function calling applications
   - API integrations
   - External service interactions
   - Calculator, weather, database queries

3. **Step Protocols (e, f)**
   - **Agent Systems**: Multi-step reasoning agents
   - **LangGraph Workflows**: State-based execution
   - **Planning Applications**: Task decomposition
   - **Progress Monitoring**: Step-by-step execution tracking

4. **Reasoning Protocol (g)**
   - **DeepSeek R1**: Internal reasoning exposure
   - **Educational Tools**: Showing problem-solving process
   - **Debugging**: Understanding AI decision-making
   - **Research**: Analyzing model reasoning patterns

## Environment Setup

Before running tests, ensure you have the required API keys:

```bash
# For OpenAI tests
export OPENAI_API_KEY="your_openai_key"

# For DeepSeek tests (reasoning protocol)
export DEEPSEEK_API_KEY="your_deepseek_key"

# For Anthropic tests
export ANTHROPIC_API_KEY="your_anthropic_key"
```

## Notes

- **Step Protocols (e, f)**: Best suited for agent frameworks like LangGraph where you need to track multi-step execution
- **Reasoning Protocol (g)**: Currently only supported by DeepSeek R1 model with specific limitations (see main README)
- **Tool Protocols (9, a, b)**: Work with any model that supports function calling
- **Basic Protocols (0, 3, d)**: Universal support across all models