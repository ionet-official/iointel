# IOIntel Project Overview

## Project Purpose
IOIntel is a flexible framework for building and orchestrating AI agents and workflows. It provides a structured way to create, manage, and execute AI-powered tasks using different models and tools.

## Key Features
- Supports both local and client (remote API) execution modes
- Configurable agents with custom personas and tools
- Chainable workflow system
- YAML/JSON workflow definition support
- Extensive model provider support (OpenAI, Llama, etc.)

## Main Components
1. **Agents**: Core entities that execute tasks, configured with:
   - Model providers
   - Custom tools
   - Persona profiles
   - Instructions and capabilities

2. **Tasks**: Individual operations that agents can perform
   - Examples: sentiment analysis, text translation, summarization
   - Can be chained together in workflows
   - Support both local and remote execution

3. **Workflows**: Orchestration of multiple tasks
   - Sequential execution of tasks
   - Support for YAML/JSON definition
   - Chainable API in Python code

## Project Status
- Currently in beta development
- Actively maintained
- Requires Python 3.10 or higher 