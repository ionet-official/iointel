"""
Workflow-as-API Service (WaaS - Workflow as a Service)
======================================================

This service dynamically creates bespoke API endpoints for workflow specifications.
Each workflow becomes a containerized service with proper REST API routing.

API Structure:
- POST /api/v1/orgs/{org_id}/users/{user_id}/workflows/{workflow_id}/runs
- GET  /api/v1/orgs/{org_id}/users/{user_id}/workflows/{workflow_id}/runs/{run_id}
- GET  /api/v1/orgs/{org_id}/users/{user_id}/workflows/{workflow_id}/spec

User inputs become query parameters or request body fields automatically.
"""

import sys
import asyncio
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec
from iointel.src.workflow import Workflow
from iointel.src.utilities.io_logger import get_component_logger
from iointel.src.web.workflow_storage import WorkflowStorage

logger = get_component_logger("WORKFLOW_API_SERVICE")

# Pydantic models for API requests/responses
class WorkflowRunRequest(BaseModel):
    """Request to execute a workflow with optional inputs."""
    inputs: Optional[Dict[str, Any]] = Field(default_factory=dict, description="User inputs for the workflow")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional execution metadata")
    async_execution: bool = Field(default=True, description="Whether to run asynchronously")

class WorkflowRunResponse(BaseModel):
    """Response from workflow execution."""
    run_id: str = Field(description="Unique execution ID")
    status: str = Field(description="Execution status: pending, running, completed, failed")
    workflow_id: str = Field(description="Workflow specification ID")
    started_at: datetime = Field(description="Execution start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Execution completion timestamp")
    results: Optional[Dict[str, Any]] = Field(default=None, description="Execution results")
    error: Optional[str] = Field(default=None, description="Error message if failed")

class WorkflowSpecResponse(BaseModel):
    """Response containing workflow specification."""
    workflow_id: str
    title: str
    description: str
    spec: WorkflowSpec
    created_at: datetime
    updated_at: datetime

class WorkflowAPIRegistry:
    """Registry for managing workflow API endpoints."""
    
    def __init__(self):
        self.registered_workflows: Dict[str, Dict[str, Any]] = {}
        self.active_runs: Dict[str, WorkflowRunResponse] = {}
        self.storage = WorkflowStorage()
    
    def register_workflow(
        self, 
        org_id: str, 
        user_id: str, 
        workflow_id: str, 
        workflow_spec: WorkflowSpec
    ) -> Dict[str, str]:
        """Register a workflow for API access."""
        workflow_key = f"{org_id}/{user_id}/{workflow_id}"
        
        # Extract user input nodes to determine API parameters
        user_input_params = self._extract_user_input_parameters(workflow_spec)
        
        self.registered_workflows[workflow_key] = {
            "org_id": org_id,
            "user_id": user_id, 
            "workflow_id": workflow_id,
            "workflow_spec": workflow_spec,
            "user_input_params": user_input_params,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        
        logger.info("Registered workflow API endpoint", data={
            "workflow_key": workflow_key,
            "title": workflow_spec.title,
            "user_inputs": len(user_input_params),
            "nodes": len(workflow_spec.nodes),
            "edges": len(workflow_spec.edges)
        })
        
        return {
            "workflow_key": workflow_key,
            "run_endpoint": f"/api/v1/orgs/{org_id}/users/{user_id}/workflows/{workflow_id}/runs",
            "spec_endpoint": f"/api/v1/orgs/{org_id}/users/{user_id}/workflows/{workflow_id}/spec"
        }
    
    def _extract_user_input_parameters(self, workflow_spec: WorkflowSpec) -> List[Dict[str, Any]]:
        """Extract user input nodes to create API parameter schema."""
        user_input_params = []
        
        for node in workflow_spec.nodes:
            if node.type == "data_source" and node.data.source_name == "user_input":
                # Extract parameter details from node configuration
                param_config = {
                    "node_id": node.id,
                    "label": node.label,
                    "description": node.data.config.get("prompt", "User input parameter"),
                    "required": node.data.config.get("required", True),
                    "type": node.data.config.get("input_type", "text"),
                    "default": node.data.config.get("default_value")
                }
                user_input_params.append(param_config)
        
        return user_input_params
    
    async def execute_workflow_api(
        self,
        org_id: str,
        user_id: str, 
        workflow_id: str,
        run_request: WorkflowRunRequest
    ) -> WorkflowRunResponse:
        """Execute a registered workflow via API."""
        workflow_key = f"{org_id}/{user_id}/{workflow_id}"
        
        if workflow_key not in self.registered_workflows:
            raise HTTPException(status_code=404, detail=f"Workflow not found: {workflow_key}")
        
        workflow_info = self.registered_workflows[workflow_key]
        workflow_spec = workflow_info["workflow_spec"]
        
        # Generate unique run ID
        run_id = str(uuid.uuid4())
        
        # Create initial run response
        run_response = WorkflowRunResponse(
            run_id=run_id,
            status="pending",
            workflow_id=workflow_id,
            started_at=datetime.now()
        )
        
        self.active_runs[run_id] = run_response
        
        # Map user inputs to workflow user_input nodes
        user_inputs = self._map_inputs_to_workflow(
            run_request.inputs, 
            workflow_info["user_input_params"]
        )
        
        logger.info("Starting workflow execution via API", data={
            "run_id": run_id,
            "workflow_key": workflow_key,
            "user_inputs": user_inputs,
            "async": run_request.async_execution
        })
        
        try:
            if run_request.async_execution:
                # Start async execution
                run_response.status = "running"
                self.active_runs[run_id] = run_response
                
                # Execute in background
                asyncio.create_task(
                    self._execute_workflow_background(run_id, workflow_spec, user_inputs)
                )
                
                return run_response
            else:
                # Synchronous execution
                run_response.status = "running"
                results = await self._execute_workflow_sync(workflow_spec, user_inputs)
                
                run_response.status = "completed"
                run_response.completed_at = datetime.now()
                run_response.results = results
                self.active_runs[run_id] = run_response
                
                return run_response
                
        except Exception as e:
            logger.error("Workflow execution failed", data={
                "run_id": run_id,
                "workflow_key": workflow_key,
                "error": str(e)
            })
            
            run_response.status = "failed"
            run_response.completed_at = datetime.now()
            run_response.error = str(e)
            self.active_runs[run_id] = run_response
            
            return run_response
    
    def _map_inputs_to_workflow(
        self, 
        api_inputs: Dict[str, Any], 
        user_input_params: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Map API inputs to workflow user_input node IDs."""
        mapped_inputs = {}
        
        for param in user_input_params:
            node_id = param["node_id"]
            
            # Try multiple ways to match input
            input_value = None
            
            # 1. Direct node_id match
            if node_id in api_inputs:
                input_value = api_inputs[node_id]
            # 2. Label-based match (lowercase, underscored)
            elif param["label"].lower().replace(" ", "_") in api_inputs:
                input_value = api_inputs[param["label"].lower().replace(" ", "_")]
            # 3. First available input if only one param
            elif len(user_input_params) == 1 and api_inputs:
                input_value = list(api_inputs.values())[0]
            # 4. Use default value
            elif param.get("default") is not None:
                input_value = param["default"]
            
            if input_value is not None:
                mapped_inputs[node_id] = input_value
            elif param.get("required", True):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Required parameter missing: {param['label']} (node_id: {node_id})"
                )
        
        return mapped_inputs
    
    async def _execute_workflow_background(
        self, 
        run_id: str, 
        workflow_spec: WorkflowSpec, 
        user_inputs: Dict[str, Any]
    ):
        """Execute workflow in background and update run status."""
        try:
            results = await self._execute_workflow_sync(workflow_spec, user_inputs)
            
            if run_id in self.active_runs:
                run_response = self.active_runs[run_id]
                run_response.status = "completed"
                run_response.completed_at = datetime.now()
                run_response.results = results
                self.active_runs[run_id] = run_response
                
        except Exception as e:
            if run_id in self.active_runs:
                run_response = self.active_runs[run_id]
                run_response.status = "failed"
                run_response.completed_at = datetime.now()
                run_response.error = str(e)
                self.active_runs[run_id] = run_response
    
    async def _execute_workflow_sync(
        self, 
        workflow_spec: WorkflowSpec, 
        user_inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute workflow synchronously and return results."""
        # Create workflow instance
        workflow = Workflow(workflow_spec.title)
        
        # Configure execution metadata
        execution_metadata = {
            "execution_id": str(uuid.uuid4()),
            "api_execution": True,
            "user_inputs": user_inputs
        }
        
        # Execute workflow using DAG executor
        results = await workflow.execute_workflow_spec(
            workflow_spec=workflow_spec,
            user_inputs=user_inputs,
            execution_metadata=execution_metadata
        )
        
        return results
    
    def get_run_status(self, run_id: str) -> WorkflowRunResponse:
        """Get status of a workflow run."""
        if run_id not in self.active_runs:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        
        return self.active_runs[run_id]
    
    def get_workflow_spec(self, org_id: str, user_id: str, workflow_id: str) -> WorkflowSpecResponse:
        """Get workflow specification."""
        workflow_key = f"{org_id}/{user_id}/{workflow_id}"
        
        if workflow_key not in self.registered_workflows:
            raise HTTPException(status_code=404, detail=f"Workflow not found: {workflow_key}")
        
        workflow_info = self.registered_workflows[workflow_key]
        
        return WorkflowSpecResponse(
            workflow_id=workflow_id,
            title=workflow_info["workflow_spec"].title,
            description=workflow_info["workflow_spec"].description,
            spec=workflow_info["workflow_spec"],
            created_at=workflow_info["created_at"],
            updated_at=workflow_info["updated_at"]
        )

# Global registry instance
workflow_api_registry = WorkflowAPIRegistry()

# FastAPI app for workflow APIs
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for the FastAPI app."""
    logger.info("Starting Workflow API Service")
    yield
    logger.info("Shutting down Workflow API Service")

app = FastAPI(
    title="Workflow API Service",
    description="Container-as-a-Service for Workflow Specifications",
    version="1.0.0",
    lifespan=lifespan
)

# API Route Handlers
@app.post("/api/v1/orgs/{org_id}/users/{user_id}/workflows/{workflow_id}/register")
async def register_workflow_endpoint(
    org_id: str,
    user_id: str, 
    workflow_id: str,
    workflow_spec: WorkflowSpec
) -> Dict[str, Any]:
    """Register a workflow for API access."""
    try:
        result = workflow_api_registry.register_workflow(
            org_id=org_id,
            user_id=user_id,
            workflow_id=workflow_id,
            workflow_spec=workflow_spec
        )
        return {
            "success": True,
            "message": "Workflow registered successfully",
            "endpoints": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/orgs/{org_id}/users/{user_id}/workflows/{workflow_id}/runs")
async def execute_workflow_endpoint(
    org_id: str,
    user_id: str,
    workflow_id: str,
    run_request: WorkflowRunRequest,
    background_tasks: BackgroundTasks
) -> WorkflowRunResponse:
    """Execute a workflow via API."""
    return await workflow_api_registry.execute_workflow_api(
        org_id=org_id,
        user_id=user_id,
        workflow_id=workflow_id,
        run_request=run_request
    )

@app.get("/api/v1/orgs/{org_id}/users/{user_id}/workflows/{workflow_id}/runs/{run_id}")
async def get_run_status_endpoint(
    org_id: str,
    user_id: str,
    workflow_id: str,
    run_id: str
) -> WorkflowRunResponse:
    """Get status of a workflow run."""
    return workflow_api_registry.get_run_status(run_id)

@app.get("/api/v1/orgs/{org_id}/users/{user_id}/workflows/{workflow_id}/spec")
async def get_workflow_spec_endpoint(
    org_id: str,
    user_id: str,
    workflow_id: str
) -> WorkflowSpecResponse:
    """Get workflow specification."""
    return workflow_api_registry.get_workflow_spec(org_id, user_id, workflow_id)

# Convenience endpoints for query parameter style
@app.get("/api/v1/orgs/{org_id}/users/{user_id}/workflows/{workflow_id}/run")
async def execute_workflow_with_query_params(
    org_id: str,
    user_id: str,
    workflow_id: str,
    request: Request
) -> WorkflowRunResponse:
    """Execute workflow with query parameters (GET request)."""
    # Convert query parameters to inputs
    query_params = dict(request.query_params)
    
    # Remove system parameters
    system_params = {"async_execution"}
    async_execution = query_params.pop("async_execution", "true").lower() == "true"
    
    # Remaining params are workflow inputs
    user_inputs = {k: v for k, v in query_params.items() if k not in system_params}
    
    run_request = WorkflowRunRequest(
        inputs=user_inputs,
        async_execution=async_execution
    )
    
    return await workflow_api_registry.execute_workflow_api(
        org_id=org_id,
        user_id=user_id,
        workflow_id=workflow_id,
        run_request=run_request
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "registered_workflows": len(workflow_api_registry.registered_workflows),
        "active_runs": len(workflow_api_registry.active_runs),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)