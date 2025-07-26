"""
Workflow RAG Router
===================

Router version of the workflow RAG service that can be integrated into the main workflow server.
Provides semantic search endpoints for saved workflows.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from .workflow_rag_service import WorkflowRAGService, WorkflowSearchRequest, WorkflowSearchResponse
from ..agent_methods.data_models.workflow_spec import WorkflowSpec

# Create router for workflow RAG endpoints
workflow_rag_router = APIRouter(prefix="/api/workflow-rag", tags=["workflow-rag"])

# Initialize RAG service (will be shared across requests)
rag_service: Optional[WorkflowRAGService] = None


def get_rag_service() -> WorkflowRAGService:
    """Get or initialize the RAG service."""
    global rag_service
    if rag_service is None:
        print("🔧 Initializing Workflow RAG Service...")
        rag_service = WorkflowRAGService()
        print("✅ Workflow RAG Service initialized")
    return rag_service


@workflow_rag_router.get("/")
async def rag_root():
    """Root endpoint with RAG service info."""
    service = get_rag_service()
    stats = service.get_stats()
    return {
        "service": "Workflow RAG API",
        "description": "Semantic search over saved workflows",
        "stats": stats,
        "endpoints": {
            "/search": "Search workflows by semantic similarity",
            "/stats": "Get RAG collection statistics",
            "/refresh": "Refresh index with latest workflows"
        }
    }


@workflow_rag_router.post("/search", response_model=WorkflowSearchResponse)
async def search_workflows(request: WorkflowSearchRequest):
    """Search workflows using semantic similarity."""
    try:
        service = get_rag_service()
        return service.search_workflows(
            query=request.query,
            top_k=request.top_k,
            indices=request.indices
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@workflow_rag_router.get("/search", response_model=WorkflowSearchResponse)
async def search_workflows_get(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(5, description="Number of results to return"),
    indices: Optional[str] = Query(None, description="Comma-separated indices (title,description,both)")
):
    """Search workflows using semantic similarity (GET version)."""
    indices_list = indices.split(",") if indices else None
    return await search_workflows(WorkflowSearchRequest(
        query=query,
        top_k=top_k,
        indices=indices_list
    ))


@workflow_rag_router.get("/stats")
async def get_rag_stats():
    """Get RAG collection statistics."""
    try:
        service = get_rag_service()
        return service.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@workflow_rag_router.post("/refresh")
async def refresh_rag_index():
    """Refresh the RAG index with latest saved workflows."""
    try:
        service = get_rag_service()
        service.refresh_index()
        return {"status": "refreshed", "stats": service.get_stats()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@workflow_rag_router.post("/add_workflow")
async def add_workflow_to_rag(workflow_spec: WorkflowSpec):
    """Add a new workflow to the RAG collection."""
    try:
        service = get_rag_service()
        service.add_workflow(workflow_spec)
        return {
            "status": "added",
            "workflow_id": workflow_spec.id,
            "workflow_title": workflow_spec.title
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@workflow_rag_router.get("/health")
async def rag_health_check():
    """Health check for RAG service."""
    try:
        service = get_rag_service()
        stats = service.get_stats()
        return {
            "status": "healthy",
            "service": "workflow_rag",
            "collection_initialized": stats.get("status") != "not_initialized",
            "total_workflows": stats.get("total_records", 0),
            "available_indices": list(stats.get("indices", {}).keys()) if stats.get("indices") else []
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }