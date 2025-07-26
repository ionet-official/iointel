# IOIntel Architecture

## Directory Structure
```
iointel/
├── src/          # Core implementation
├── client/       # Client mode implementation
├── __init__.py   # Package initialization and version
tests/            # Test suite
```

## Execution Modes

### Local Mode
- Direct execution in local environment
- Uses `run_agents()` for task execution
- Full access to local resources and tools
- Suitable for development and standalone usage

### Client Mode
- Remote API execution
- Tasks delegate to remote endpoints
- Suitable for production deployment
- Allows distributed execution

## Core Components

### Agent System
- Configurable with different model providers
- Supports custom personas via `PersonaConfig`
- Extensible tool system
- Configurable instructions and behaviors

### Workflow Engine
- Task chaining and orchestration
- Support for YAML/JSON workflow definitions
- Sequential task execution
- Error handling and state management
- ⚠️ **Known Issue**: Variable resolution in task configs not implemented (see workflow_planner_design.md)

### Integration Points
- OpenAI API compatibility
- Custom model support
- External tool integration
- Database integration (pgvector, SQLAlchemy)

## Configuration
- Environment-based configuration
- Support for `.env` files
- Configurable logging levels
- Model-specific settings

## Dependencies
The project uses several key technologies:
- pydantic for data validation
- langchain for LLM integration
- pgvector for vector storage
- SQLAlchemy for database operations
- httpx for HTTP client operations 