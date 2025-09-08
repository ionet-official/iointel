import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import os

# Set test environment variables
os.environ["IO_API_KEY"] = "test_key"
os.environ["IO_BASE_URL"] = "http://test.api"

from iointel.src.RL.api.main import app
from iointel.src.RL.api.models import TaskResult, TaskDifficulty

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "environment" in data


def test_get_models():
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "recommended_models" in data
    assert "models_requiring_settings" in data
    assert isinstance(data["recommended_models"], list)


def test_evaluate_missing_models():
    response = client.post("/evaluate", json={})
    assert response.status_code == 422  # Validation error


@patch("iointel.src.RL.api.service.EvaluationService.evaluate_models")
async def test_evaluate_sync(mock_evaluate):
    # Mock the evaluation response
    mock_result = TaskResult(
        model="test-model",
        task_id="task-1",
        task_description="Test task",
        task_difficulty=TaskDifficulty.EASY,
        step_count=1,
        execution_time=1.5
    )
    
    mock_evaluate.return_value = {
        "status": "completed",
        "total_models": 1,
        "total_tasks": 1,
        "results": [mock_result.model_dump()],
        "summary": {
            "total_evaluations": 1,
            "successful_evaluations": 1,
            "failed_evaluations": 0
        }
    }
    
    response = client.post("/evaluate", json={
        "models": ["test-model"],
        "num_tasks": 1,
        "timeout": 60
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["total_models"] == 1


def test_evaluate_async():
    response = client.post("/evaluate/async", json={
        "models": ["test-model"],
        "num_tasks": 1
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "pending"
    assert data["total_models"] == 1


def test_get_status_not_found():
    response = client.get("/evaluate/nonexistent-task/status")
    assert response.status_code == 404


def test_rate_limit_disabled():
    # Rate limiting is disabled in test config
    for _ in range(10):
        response = client.get("/health")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])