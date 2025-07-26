# IOIntel Dependencies

## Core Dependencies
- **pydantic** (≥2.10.3): Data validation and settings management
- **pydantic-ai** (≥0.1.0): AI model integration with Pydantic
- **langchain-core** (≥0.3.28): Core LLM functionality
- **langchain-openai** (≥0.2.14): OpenAI integration

## Database & Storage
- **pgvector** (≥0.4.0): Vector storage capabilities
- **sqlalchemy** (≥2.0.36): Database ORM
- **sqlalchemy-utils** (≥0.41.2): SQLAlchemy utilities

## HTTP & API
- **httpx**: Modern HTTP client
- **python-multipart** (≥0.0.20): Multipart form data handling
- **python-dotenv** (≥1.0.1): Environment variable management

## Search & Tools
- **duckduckgo-search** (=8.0.0): Web search capabilities
- **wolframalpha** (≥5.1.3): Computational knowledge engine
- **firecrawl** & **firecrawl-py** (≥2.2.0): Web crawling functionality

## Development Dependencies
- **pre-commit** (≥4.1.0): Git hooks management
- **pytest** suite: Comprehensive testing tools
  - pytest-asyncio
  - pytest-dotenv
  - pytest-rerunfailures
  - pytest-retry
  - pytest-sugar
  - pytest-timeout
  - pytest-xdist

## System Requirements
- Python ≥ 3.10
- YAML support (pyyaml ≥6.0.2)
- Rich terminal output (rich ≥13.9.4)

## Environment Variables
Key environment variables needed:
- `OPENAI_API_KEY` or `IO_API_KEY`
- `AGENT_LOGGING_LEVEL` (optional)
- `OPENAI_API_BASE_URL` or `IO_API_BASE_URL` (optional)
- `OPENAI_API_MODEL` or `IO_API_MODEL` (optional) 