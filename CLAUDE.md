# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Development Environment Setup
```bash
# Install dependencies (creates isolated .venv)
make install

# Run unit tests
make test

# Run specific test file
pytest jentic_agents/tests/test_reasoner.py -v

# Run with coverage
pytest --cov=jentic_agents

# Static analysis and linting
make lint

# Strict static analysis with mypy
make lint-strict

# Auto-fix common issues
ruff check . --fix
```

### Running the Application
```bash
# Run the main demo with live services
python main.py

# Run escalation system example
python -m jentic_agents.examples.escalation_example

# Run human-in-the-loop example
python -m jentic_agents.examples.hitl_example
```

## Architecture Overview

ActBots is a modular AI agent library built around the ReAct pattern (plan → select_tool → act → observe → evaluate → reflect) with Jentic platform integration.

### Core Components

1. **Reasoners** (`jentic_agents/reasoners/`):
   - `BaseReasoner`: Abstract interface defining the reasoning contract
   - `BulletPlanReasoner`: Implements bullet-point planning with escalation
   - `StandardReasoner`: Basic ReAct implementation
   - `FreeformReasoner`: Flexible reasoning with escalation support

2. **Agents** (`jentic_agents/agents/`):
   - `BaseAgent`: Abstract agent interface with `spin()` main loop
   - `InteractiveCLIAgent`: CLI-based agent for interactive use

3. **Memory** (`jentic_agents/memory/`):
   - `BaseMemory`: Simple key-value storage interface
   - `ScratchPadMemory`: In-memory dict-based implementation
   - `AgentMemory`: Enhanced memory with search capabilities

4. **Communication System** (`jentic_agents/communication/`):
   - **Inbox** (`inbox/`): Receives goals/tasks from various sources
   - **Outbox** (`outbox/`): Sends progress updates and results
   - **HITL** (`hitl/`): Human-in-the-loop intervention system

5. **Platform Integration** (`jentic_agents/platform/`):
   - `JenticClient`: Wrapper around jentic-sdk with auth, retries, and logging
   - Supports both workflows and operations
   - Handles search, load, and execute operations

### Key Design Patterns

- **Dependency Injection**: All components are injected via constructors
- **Abstract Interfaces**: Clean separation between contracts and implementations
- **Escalation System**: Agents can request human help when they choose to
- **Stateless Reasoning**: `BaseReasoner.run()` returns packaged results
- **Single Source of Truth**: Only `JenticClient` contacts the Jentic SDK

### Escalation vs HITL Systems

The codebase includes two human interaction approaches:

1. **Escalation System** (Recommended):
   - Agent-driven: Agent chooses when to escalate
   - Simple interface: Just `ask_human(question, context)`
   - Located in `jentic_agents/communication/hitl/base_intervention_hub.py`

2. **HITL System** (Deprecated):
   - Complex automatic triggers based on heuristics
   - Three-component system (inbox, outbox, intervention hub)
   - Still available but not recommended for new projects

## Configuration

### Environment Variables
Required for live demo (`main.py`):
- `JENTIC_API_KEY`: API key for Jentic platform
- `OPENAI_API_KEY`: API key for OpenAI (if using OpenAI models)
- `GEMINI_API_KEY`: API key for Gemini (if using Gemini models)
- `LLM_MODEL`: Model name (default: "gpt-4o")

### Project Structure
```
jentic_agents/
├── agents/              # Agent orchestration layer
├── communication/       # Human interaction systems
│   ├── inbox/          # Goal input
│   ├── outbox/         # Progress output
│   └── hitl/           # Human-in-the-loop/escalation
├── memory/             # Memory backends
├── platform/           # External service adapters
├── prompts/            # System prompts for reasoning
├── reasoners/          # Reasoning loop implementations
├── tests/              # Comprehensive test suite
└── utils/              # Shared utilities (LLM, logging, etc.)
```

### Adding New Components

1. **New Reasoner**: Extend `BaseReasoner` and implement all abstract methods
2. **New Agent**: Extend `BaseAgent` and override I/O methods
3. **New Memory**: Extend `BaseMemory` with your storage backend
4. **New Communication**: Extend base classes in `communication/`

## Testing

- Target: >90% coverage on core modules
- Use `pytest-mock` for mocking external dependencies
- Test files follow pattern: `test_*.py` in `jentic_agents/tests/`
- All external calls are injectable for easy testing

## Code Quality Standards

- Python 3.13+ required
- Type hints mandatory (mypy configuration in `mypy.ini`)
- Ruff for linting and formatting
- No global dependencies - clean `.venv` usage
- Strict interfaces with abstract base classes
- Comprehensive error handling with context

## Important Notes

- The JenticClient requires the `jentic` SDK to be installed
- All components support both synchronous and asynchronous operation
- The escalation system gives agents full autonomy to choose when to request help
- Memory and platform components are designed to be easily swapped out
- The codebase follows strict dependency isolation principles