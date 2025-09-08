"""
Tests for WorkflowPlanner web API endpoints.
"""

import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
import asyncio

from iointel.src.web.workflow_server import app
from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec
)


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_workflow_data():
    """Sample workflow data for testing."""
    return {
        "id": str(uuid.uuid4()),
        "rev": 1,
        "title": "Test API Workflow",
        "description": "A workflow for testing the API",
        "nodes": [
            {
                "id": "test_node",
                "type": "tool",
                "label": "Test Node",
                "data": {
                    "config": {"param": "value"},
                    "ins": ["input"],
                    "outs": ["output"],
                    "tool_name": "test_tool",
                    "agent_instructions": None,
                    "workflow_id": None
                }
            }
        ],
        "edges": [],
        "metadata": {}
    }


@pytest.fixture
def mock_planner():
    """Mock WorkflowPlanner for testing."""
    planner = MagicMock()
    planner.generate_workflow = AsyncMock()
    planner.refine_workflow = AsyncMock()
    planner.create_example_workflow = MagicMock()
    return planner


class TestWorkflowWebAPI:
    """Test cases for workflow web API endpoints."""

    @patch('iointel.src.web.workflow_server.planner')
    @patch('iointel.src.web.workflow_server.tool_catalog')
    def test_get_tools_endpoint(self, mock_catalog, mock_planner, client):
        """Test the /api/tools endpoint."""
        mock_catalog.update({
            "tool1": {"name": "tool1", "description": "Test tool 1"},
            "tool2": {"name": "tool2", "description": "Test tool 2"}
        })
        
        response = client.get("/api/tools")
        
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert len(data["tools"]) == 2

    @patch('iointel.src.web.workflow_server.current_workflow')
    def test_get_workflow_endpoint_with_workflow(self, mock_workflow, client, sample_workflow_data):
        """Test /api/workflow endpoint when workflow exists."""
        # Create a WorkflowSpec object
        workflow_spec = WorkflowSpec.model_validate(sample_workflow_data)
        mock_workflow.__bool__ = lambda: True
        mock_workflow.model_dump.return_value = sample_workflow_data
        
        # Monkey patch the global variable
        import iointel.src.web.workflow_server as server_module
        server_module.current_workflow = workflow_spec
        
        response = client.get("/api/workflow")
        
        assert response.status_code == 200
        data = response.json()
        assert data["workflow"]["title"] == "Test API Workflow"

    @patch('iointel.src.web.workflow_server.current_workflow', None)
    def test_get_workflow_endpoint_no_workflow(self, client):
        """Test /api/workflow endpoint when no workflow exists."""
        response = client.get("/api/workflow")
        
        assert response.status_code == 200
        data = response.json()
        assert data["workflow"] is None

    @patch('iointel.src.web.workflow_server.workflow_history')
    def test_get_history_endpoint(self, mock_history, client, sample_workflow_data):
        """Test /api/history endpoint."""
        workflow_spec = WorkflowSpec.model_validate(sample_workflow_data)
        mock_history.__iter__ = lambda: iter([workflow_spec])
        mock_history.__getitem__ = lambda self, key: [workflow_spec][key]
        
        response = client.get("/api/history")
        
        assert response.status_code == 200
        data = response.json()
        assert "history" in data
        assert isinstance(data["history"], list)

    @patch('iointel.src.web.workflow_server.planner')
    @patch('iointel.src.web.workflow_server.tool_catalog')
    def test_generate_workflow_endpoint_success(self, mock_catalog, mock_planner_global, client, sample_workflow_data):
        """Test successful workflow generation via API."""
        # Mock the planner instance
        mock_planner_global.generate_workflow = AsyncMock()
        workflow_spec = WorkflowSpec.model_validate(sample_workflow_data)
        mock_planner_global.generate_workflow.return_value = workflow_spec
        
        request_data = {
            "query": "Create a test workflow",
            "refine": False
        }
        
        response = client.post("/api/generate", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["workflow"]["title"] == "Test API Workflow"

    @patch('iointel.src.web.workflow_server.planner')
    def test_generate_workflow_endpoint_error(self, mock_planner_global, client):
        """Test workflow generation API error handling."""
        mock_planner_global.generate_workflow = AsyncMock(side_effect=Exception("Generation failed"))
        
        request_data = {
            "query": "Create a test workflow",
            "refine": False
        }
        
        response = client.post("/api/generate", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "Generation failed" in data["error"]

    @patch('iointel.src.web.workflow_server.planner')
    @patch('iointel.src.web.workflow_server.current_workflow')
    def test_refine_workflow_endpoint(self, mock_current, mock_planner_global, client, sample_workflow_data):
        """Test workflow refinement via API."""
        # Setup current workflow
        original_workflow = WorkflowSpec.model_validate(sample_workflow_data)
        mock_current.__bool__ = lambda: True
        
        # Setup refined workflow
        refined_data = sample_workflow_data.copy()
        refined_data["rev"] = 2
        refined_data["title"] = "Refined Test Workflow"
        refined_workflow = WorkflowSpec.model_validate(refined_data)
        
        mock_planner_global.refine_workflow = AsyncMock(return_value=refined_workflow)
        
        # Monkey patch the current workflow
        import iointel.src.web.workflow_server as server_module
        server_module.current_workflow = original_workflow
        
        request_data = {
            "query": "Add error handling",
            "refine": True
        }
        
        response = client.post("/api/generate", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["workflow"]["title"] == "Refined Test Workflow"
        assert data["workflow"]["rev"] == 2

    @patch('iointel.src.web.workflow_server.planner')
    def test_example_workflow_endpoint(self, mock_planner_global, client, sample_workflow_data):
        """Test example workflow creation endpoint."""
        workflow_spec = WorkflowSpec.model_validate(sample_workflow_data)
        mock_planner_global.create_example_workflow.return_value = workflow_spec
        
        response = client.get("/api/example")
        
        assert response.status_code == 200
        data = response.json()
        assert data["workflow"]["title"] == "Test API Workflow"

    def test_clear_workflow_endpoint(self, client):
        """Test workflow clearing endpoint."""
        response = client.post("/api/clear")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_root_endpoint(self, client):
        """Test root endpoint serves HTML."""
        response = client.get("/")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestWorkflowWebSocket:
    """Test WebSocket functionality."""

    def test_websocket_connection(self, client):
        """Test basic WebSocket connection."""
        with client.websocket_connect("/ws"):
            # Connection should be established without errors
            pass

    @patch('iointel.src.web.workflow_server.current_workflow')
    def test_websocket_sends_current_workflow(self, mock_workflow, client, sample_workflow_data):
        """Test WebSocket sends current workflow on connection."""
        workflow_spec = WorkflowSpec.model_validate(sample_workflow_data)
        
        # Monkey patch the current workflow
        import iointel.src.web.workflow_server as server_module
        server_module.current_workflow = workflow_spec
        
        with client.websocket_connect("/ws") as websocket:
            data = websocket.receive_json()
            
            assert data["type"] == "workflow_update"
            assert data["workflow"]["title"] == "Test API Workflow"

    def test_websocket_no_current_workflow(self, client):
        """Test WebSocket behavior when no current workflow exists."""
        # Ensure no current workflow
        import iointel.src.web.workflow_server as server_module
        server_module.current_workflow = None
        
        with client.websocket_connect("/ws"):
            # Should connect without sending workflow data
            pass


class TestWorkflowAPIValidation:
    """Test API input validation."""

    def test_generate_workflow_missing_query(self, client):
        """Test generate endpoint with missing query."""
        request_data = {"refine": False}  # Missing "query"
        
        response = client.post("/api/generate", json=request_data)
        
        assert response.status_code == 422  # Validation error

    def test_generate_workflow_invalid_json(self, client):
        """Test generate endpoint with invalid JSON."""
        response = client.post(
            "/api/generate",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422

    def test_generate_workflow_empty_query(self, client):
        """Test generate endpoint with empty query."""
        request_data = {"query": "", "refine": False}
        
        # Should accept empty query (let the planner handle it)
        response = client.post("/api/generate", json=request_data)
        
        # Will fail at planner level, but validation should pass
        assert response.status_code == 200


class TestWorkflowAPIIntegration:
    """Integration tests for the workflow API."""

    @patch('iointel.src.web.workflow_server.AsyncMemory')
    @patch('iointel.src.web.workflow_server.load_tools_from_env')
    @patch('iointel.src.web.workflow_server.WorkflowPlanner')
    def test_startup_event(self, mock_planner_class, mock_load_tools, mock_memory, client):
        """Test application startup event."""
        mock_memory_instance = AsyncMock()
        mock_memory.return_value = mock_memory_instance
        
        mock_load_tools.return_value = ["tool1", "tool2"]
        
        mock_planner_instance = MagicMock()
        mock_planner_class.return_value = mock_planner_instance
        
        # The startup event should have run during app initialization
        # We can't easily test it directly, but we can verify the app starts
        response = client.get("/")
        assert response.status_code == 200

    @patch('iointel.src.web.workflow_server.planner')
    def test_full_workflow_lifecycle(self, mock_planner_global, client, sample_workflow_data):
        """Test complete workflow lifecycle via API."""
        workflow_spec = WorkflowSpec.model_validate(sample_workflow_data)
        mock_planner_global.generate_workflow = AsyncMock(return_value=workflow_spec)
        
        # 1. Check initially no workflow
        response = client.get("/api/workflow")
        assert response.json()["workflow"] is None
        
        # 2. Generate workflow
        generate_response = client.post("/api/generate", json={
            "query": "Create test workflow",
            "refine": False
        })
        assert generate_response.status_code == 200
        assert generate_response.json()["success"] is True
        
        # 3. Check workflow now exists (in practice, would be set by the endpoint)
        # Note: In actual test, current_workflow would be updated by the endpoint
        
        # 4. Clear workflow
        clear_response = client.post("/api/clear")
        assert clear_response.status_code == 200
        assert clear_response.json()["success"] is True


class TestWorkflowAPIErrorScenarios:
    """Test various error scenarios."""

    @patch('iointel.src.web.workflow_server.planner', None)
    def test_endpoints_with_no_planner(self, client):
        """Test endpoints when planner is not initialized."""
        response = client.post("/api/generate", json={
            "query": "test",
            "refine": False
        })
        
        assert response.status_code == 500

    def test_unsupported_http_methods(self, client):
        """Test unsupported HTTP methods."""
        # Try DELETE on generate endpoint
        response = client.delete("/api/generate")
        assert response.status_code == 405  # Method not allowed
        
        # Try PUT on tools endpoint
        response = client.put("/api/tools")
        assert response.status_code == 405

    def test_nonexistent_endpoints(self, client):
        """Test requests to non-existent endpoints."""
        response = client.get("/api/nonexistent")
        assert response.status_code == 404
        
        response = client.post("/api/invalid")
        assert response.status_code == 404


class TestWorkflowAPIPerformance:
    """Performance and stress tests for the API."""

    @patch('iointel.src.web.workflow_server.planner')
    def test_concurrent_workflow_generation(self, mock_planner_global, client, sample_workflow_data):
        """Test handling of concurrent workflow generation requests."""
        import threading
        
        workflow_spec = WorkflowSpec.model_validate(sample_workflow_data)
        
        # Simulate slow workflow generation
        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(0.1)  # 100ms delay
            return workflow_spec
        
        mock_planner_global.generate_workflow = slow_generate
        
        results = []
        
        def make_request():
            response = client.post("/api/generate", json={
                "query": f"test query {threading.current_thread().ident}",
                "refine": False
            })
            results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 5

    @patch('iointel.src.web.workflow_server.planner')
    def test_large_workflow_handling(self, mock_planner_global, client):
        """Test handling of large workflow specifications."""
        # Create a large workflow with many nodes
        large_workflow_data = {
            "id": str(uuid.uuid4()),
            "rev": 1,
            "title": "Large Workflow",
            "description": "A workflow with many nodes",
            "nodes": [],
            "edges": [],
            "metadata": {}
        }
        
        # Add 100 nodes
        for i in range(100):
            large_workflow_data["nodes"].append({
                "id": f"node_{i}",
                "type": "tool",
                "label": f"Node {i}",
                "data": {
                    "config": {"param": f"value_{i}"},
                    "ins": [f"input_{i}"],
                    "outs": [f"output_{i}"],
                    "tool_name": f"tool_{i}",
                    "agent_instructions": None,
                    "workflow_id": None
                }
            })
        
        # Add edges connecting consecutive nodes
        for i in range(99):
            large_workflow_data["edges"].append({
                "id": f"edge_{i}",
                "source": f"node_{i}",
                "target": f"node_{i+1}",
                "sourceHandle": f"output_{i}",
                "targetHandle": f"input_{i+1}",
                "data": {"condition": None}
            })
        
        large_workflow = WorkflowSpec.model_validate(large_workflow_data)
        mock_planner_global.generate_workflow = AsyncMock(return_value=large_workflow)
        
        response = client.post("/api/generate", json={
            "query": "Create large workflow",
            "refine": False
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["workflow"]["nodes"]) == 100


if __name__ == "__main__":
    pytest.main([__file__])