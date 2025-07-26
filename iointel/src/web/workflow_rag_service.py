"""
Workflow RAG Service
====================

FastAPI endpoint for semantic search over saved workflows using our RAG system.
Encodes title and description fields for similarity search and returns full WorkflowSpec objects.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
from pathlib import Path

from ..utilities.semantic_rag import RAGFactory, SemanticRAGCollection
from ..web.workflow_storage import WorkflowStorage
from ..agent_methods.data_models.workflow_spec import WorkflowSpec


class WorkflowSearchRequest(BaseModel):
    """Request model for workflow search."""
    query: str
    top_k: int = 5
    indices: Optional[List[str]] = None  # Which indices to search (title, description, both)


class WorkflowSearchResult(BaseModel):
    """Single search result with score and workflow."""
    workflow_spec: WorkflowSpec
    similarity_score: float
    matched_fields: Dict[str, Any]  # Which fields matched


class WorkflowSearchResponse(BaseModel):
    """Response model for workflow search."""
    query: str
    results: List[WorkflowSearchResult]
    total_found: int


class WorkflowRAGService:
    """Service for managing workflow RAG collection."""
    
    def __init__(self, storage_dir: str = "saved_workflows", fast_mode: bool = True):
        self.storage = WorkflowStorage(storage_dir=storage_dir)
        self.rag_collection: Optional[SemanticRAGCollection] = None
        self.fast_mode = fast_mode
        self._initialize_rag()
    
    def _initialize_rag(self):
        """Initialize RAG collection with saved workflows."""
        # Get all saved workflows
        workflows = self.storage.list_workflows()
        
        if not workflows:
            print("⚠️  No saved workflows found to index")
            return
        
        print(f"📁 Found {len(workflows)} workflow metadata entries")
        
        # Load WorkflowSpec objects
        workflow_specs = []
        for workflow_info in workflows:
            try:
                # Use id from metadata (full UUID)
                workflow_id = workflow_info.get('id')
                if not workflow_id:
                    print(f"⚠️  No id in metadata: {workflow_info.get('name', 'Unknown')}")
                    continue
                    
                # Load workflow using storage API
                spec = self.storage.load_workflow(workflow_id)
                    
                if spec:
                    workflow_specs.append(spec)
                    print(f"✅ Loaded: {spec.title}")
                else:
                    print(f"❌ Failed to load workflow: {workflow_id}")
            except Exception as e:
                print(f"❌ Error loading workflow {workflow_id}: {e}")
        
        if not workflow_specs:
            print("⚠️  No valid workflows could be loaded")
            return
        
        print(f"🔍 Indexing {len(workflow_specs)} workflows for semantic search...")
        
        # Create RAG collection with field encodings for title and description
        self.rag_collection = RAGFactory.from_pydantic(
            models=workflow_specs,
            collection_name="saved_workflows",
            field_encodings={
                "title": "title",
                "description": "description", 
                "both": ["title", "description"]  # Combined index
            },
            fast_mode=self.fast_mode
        )
        
        print(f"✅ RAG collection initialized with {len(workflow_specs)} workflows")
        print(f"   Indices: {list(self.rag_collection.vector_indices.keys())}")
    
    def search_workflows(
        self, 
        query: str, 
        top_k: int = 5,
        indices: Optional[List[str]] = None
    ) -> WorkflowSearchResponse:
        """Search for workflows using semantic similarity."""
        if not self.rag_collection:
            raise HTTPException(status_code=503, detail="RAG collection not initialized")
        
        # Default to searching both title and description
        if not indices:
            indices = ["both"]
        
        # Validate indices
        available_indices = list(self.rag_collection.vector_indices.keys())
        for index in indices:
            if index not in available_indices:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid index '{index}'. Available: {available_indices}"
                )
        
        # Search across specified indices
        if len(indices) == 1:
            # Single index search
            results = self.rag_collection.search_single_index(query, indices[0], top_k=top_k)
        else:
            # Multi-index search with reranking
            results = self.rag_collection.search_multi_index(
                query, 
                indices, 
                top_k=top_k,
                rerank_method="borda"
            )
        
        # Convert results to response format
        search_results = []
        for result in results:
            workflow_spec = result['data']  # The WorkflowSpec object
            search_results.append(WorkflowSearchResult(
                workflow_spec=workflow_spec,
                similarity_score=result.get('similarity', result.get('final_score', 0)),
                matched_fields={
                    'title': workflow_spec.title,
                    'description': workflow_spec.description,
                    'indices_used': result.get('indices_used', indices)
                }
            ))
        
        return WorkflowSearchResponse(
            query=query,
            results=search_results,
            total_found=len(search_results)
        )
    
    def add_workflow(self, workflow_spec: WorkflowSpec):
        """Add a new workflow to the RAG collection."""
        if not self.rag_collection:
            # Initialize with this single workflow
            self.rag_collection = RAGFactory.from_pydantic(
                models=[workflow_spec],
                collection_name="saved_workflows",
                field_encodings={
                    "title": "title",
                    "description": "description",
                    "both": ["title", "description"]
                },
                fast_mode=self.fast_mode
            )
        else:
            # Add to existing collection
            self.rag_collection.add_record(workflow_spec)
    
    def refresh_index(self):
        """Refresh the RAG index with latest saved workflows."""
        self._initialize_rag()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG collection."""
        if not self.rag_collection:
            return {"status": "not_initialized"}
        
        return self.rag_collection.get_stats()


# Create FastAPI app
app = FastAPI(title="Workflow RAG API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service
rag_service = WorkflowRAGService()


@app.get("/")
async def root():
    """Root endpoint with service info."""
    stats = rag_service.get_stats()
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


@app.post("/search", response_model=WorkflowSearchResponse)
async def search_workflows(request: WorkflowSearchRequest):
    """Search workflows using semantic similarity."""
    try:
        return rag_service.search_workflows(
            query=request.query,
            top_k=request.top_k,
            indices=request.indices
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search", response_model=WorkflowSearchResponse)
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


@app.get("/stats")
async def get_stats():
    """Get RAG collection statistics."""
    return rag_service.get_stats()


@app.post("/refresh")
async def refresh_index():
    """Refresh the RAG index with latest saved workflows."""
    rag_service.refresh_index()
    return {"status": "refreshed", "stats": rag_service.get_stats()}


@app.post("/add_workflow")
async def add_workflow(workflow_spec: WorkflowSpec):
    """Add a new workflow to the RAG collection."""
    try:
        rag_service.add_workflow(workflow_spec)
        return {"status": "added", "workflow_id": workflow_spec.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_server(host: str = "0.0.0.0", port: int = 8101):
    """Run the RAG service server."""
    print(f"🚀 Starting Workflow RAG Service on {host}:{port}")
    print(f"📄 API docs available at http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()