# RL Model Evaluation API

A production-grade FastAPI application for evaluating reinforcement learning models with tool usage capabilities.

## Features

- **Synchronous and Asynchronous Evaluation**: Run evaluations that wait for completion or start background tasks
- **Multiple Model Support**: Evaluate multiple models in a single request
- **Comprehensive Metrics**: Get detailed feedback from critic and oracle agents
- **Task Management**: Track evaluation progress and retrieve results later
- **Production Ready**: Rate limiting, authentication, error handling, and configuration management
- **API Documentation**: Auto-generated OpenAPI/Swagger documentation

## Installation

```bash
cd iointel/src/RL/api
uv sync  # or pip install -r requirements.txt
```

## Prerequisites

1. **Environment Setup**: Copy the project's `creds.env` file to the project root with valid credentials:

```bash
# Required environment variables (in creds.env at project root)
IO_API_KEY=your_valid_io_api_key_here
IO_BASE_URL=https://api.intelligence-dev.io.solutions/api/v1
```

2. **Validation**: Ensure your IO API key is valid by testing with existing CLI tools first.

## Configuration

The API loads environment variables from `creds.env` in the project root. Additional configuration:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Evaluation Settings
DEFAULT_NUM_TASKS=3
DEFAULT_TIMEOUT=120
MAX_TIMEOUT=600

# Security (optional)
REQUIRE_API_KEY=false
API_KEYS=key1,key2,key3

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=3600
```

## Running the API

### Development Mode

**Important**: Run from the project root directory so `creds.env` is found:

```bash
# From project root (where creds.env is located)
cd /path/to/iointel
uv run uvicorn iointel.src.RL.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## API Endpoints

### Health Check

```bash
GET /health
```

### Synchronous Evaluation

Evaluate models and wait for results:

```bash
POST /evaluate
Content-Type: application/json

{
  "models": ["meta-llama/Llama-3.3-70B-Instruct"],
  "num_tasks": 3,
  "timeout": 120
}
```

### Asynchronous Evaluation

Start evaluation in background:

```bash
POST /evaluate/async
Content-Type: application/json

{
  "models": [
    "meta-llama/Llama-3.3-70B-Instruct",
    "microsoft/phi-4"
  ],
  "num_tasks": 5,
  "timeout": 180
}
```

Response:
```json
{
  "task_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "pending",
  "total_models": 2,
  "models_completed": 0
}
```

### Check Evaluation Status

```bash
GET /evaluate/{task_id}/status
```

### Get Evaluation Results

```bash
GET /evaluate/{task_id}/results
```

### List Recommended Models

```bash
GET /models
```

## Example Usage

### Using the Included Example Client

The easiest way to test the API:

```bash
# From project root
uv run python iointel/src/RL/api/example_client.py
```

### Python Client Example

```python
import requests
import time

# Start evaluation
response = requests.post(
    "http://localhost:8000/evaluate/async",
    json={
        "models": ["meta-llama/Llama-3.3-70B-Instruct"],
        "num_tasks": 3,
        "timeout": 120,
        "api_key": "your_api_key",  # Optional if set in creds.env
        "base_url": "https://api.intelligence-dev.io.solutions/api/v1"  # Optional if set in creds.env
    }
)

task = response.json()
task_id = task["task_id"]

# Poll for status
while True:
    status_response = requests.get(f"http://localhost:8000/evaluate/{task_id}/status")
    status = status_response.json()
    
    print(f"Status: {status['status']}, Models completed: {status['models_completed']}/{status['total_models']}")
    
    if status["status"] in ["completed", "failed"]:
        break
    
    time.sleep(5)

# Get results
if status["status"] == "completed":
    results_response = requests.get(f"http://localhost:8000/evaluate/{task_id}/results")
    results = results_response.json()
    
    print(f"Total tasks evaluated: {results['total_tasks']}")
    print(f"Summary: {results['summary']}")
```

### cURL Examples

```bash
# Synchronous evaluation
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["microsoft/phi-4"],
    "num_tasks": 2,
    "timeout": 60
  }'

# Start async evaluation
curl -X POST "http://localhost:8000/evaluate/async" \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["meta-llama/Llama-3.3-70B-Instruct", "microsoft/phi-4"],
    "num_tasks": 3
  }'

# Check status
curl "http://localhost:8000/evaluate/123e4567-e89b-12d3-a456-426614174000/status"

# Get models
curl "http://localhost:8000/models"
```

## Response Schema

### Evaluation Response

```json
{
  "status": "completed",
  "total_models": 2,
  "total_tasks": 6,
  "results": [
    {
      "model": "meta-llama/Llama-3.3-70B-Instruct",
      "task_id": "task_001",
      "task_description": "Calculate the sum of 15 and 27",
      "task_difficulty": "easy",
      "task_required_tools": ["add"],
      "step_count": 1,
      "critic_feedback": {
        "score": 0.9,
        "better_query": "What is 15 + 27?",
        "metrics": {...}
      },
      "oracle_result": {
        "correct": true,
        "score": 1.0,
        "feedback": "Correct result"
      },
      "execution_time": 2.34
    }
  ],
  "summary": {
    "models": {
      "meta-llama/Llama-3.3-70B-Instruct": {
        "total_tasks": 3,
        "successful": 3,
        "failed": 0,
        "average_oracle_score": 0.95,
        "oracle_accuracy": 0.67
      }
    }
  }
}
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Missing API key (if required)
- `403 Forbidden`: Invalid API key
- `404 Not Found`: Task or resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

Error response format:
```json
{
  "error": "Error message",
  "detail": "Detailed error information",
  "status_code": 400
}
```

## Development

### Running Tests

```bash
pytest tests/
```

### API Documentation

When running, access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Production Deployment

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

### Environment Variables for Production

```bash
# Required
IO_API_KEY=your_production_api_key
IO_BASE_URL=https://api.intelligence-dev.io.solutions/api/v1

# Recommended
REQUIRE_API_KEY=true
API_KEYS=production_key_1,production_key_2
LOG_LEVEL=WARNING
RATE_LIMIT_REQUESTS=50
```

## Troubleshooting

### Common Issues

1. **401 Invalid API Key**: 
   - Verify your IO_API_KEY in `creds.env` is valid
   - Test the key with existing CLI tools first
   - Check if the key has expired

2. **404 Model Not Found**:
   - Verify model names are correct and available on the IO API
   - Check that base URL is correct

3. **Environment Variables Not Loading**:
   - Ensure you're running the server from the project root directory
   - Verify `creds.env` exists in the project root
   - Check that `load_dotenv("creds.env")` is working

4. **Tool Choice Errors**:
   - Models requiring special settings are configured in `config.py`
   - Some models may need specific tool configuration

### Validation Steps

1. Test API key with existing tools:
   ```bash
   uv run python iointel/src/RL/tests/testing.py
   ```

2. Check environment variable loading:
   ```bash
   curl http://localhost:8000/health
   # Should show io_api_key_set: true
   ```

3. Test with a simple model first:
   ```bash
   curl -X POST http://localhost:8000/evaluate \
     -H "Content-Type: application/json" \
     -d '{"models": ["microsoft/phi-4"], "num_tasks": 1}'
   ```