"""
Unified Search Service
=====================

Provides semantic search across multiple data types:
- Workflows (from saved examples and test repository)
- Tools (from tool registry)
- Test cases (known working examples)

Uses the RAG factory to create type-specific search indices with proper field encoding.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Literal
import uvicorn

from ..utilities.semantic_rag import RAGFactory, SemanticRAGCollection
from ..web.workflow_storage import WorkflowStorage
from ..utilities.tool_registry_utils import create_tool_catalog
from ..utilities.workflow_test_repository import WorkflowTestRepository, TestLayer


class UnifiedSearchRequest(BaseModel):
    """Request model for unified search."""
    query: str
    search_types: List[Literal["workflows", "tools", "tests"]] = ["workflows", "tools"]
    top_k: int = 5
    indices: Optional[List[str]] = None


class SearchResult(BaseModel):
    """Single search result with metadata."""
    result_type: Literal["workflow", "tool", "test"]
    title: str
    description: str
    similarity_score: float
    data: Dict[str, Any]  # The actual object data
    metadata: Dict[str, Any] = {}


class UnifiedSearchResponse(BaseModel):
    """Response model for unified search."""
    query: str
    results: List[SearchResult]
    total_found: int
    results_by_type: Dict[str, int]


class UnifiedSearchService:
    """Service for unified semantic search across workflows, tools, and tests."""
    
    def __init__(self, 
                 storage_dir: str = "saved_workflows", 
                 test_repository_dir: str = "smart_test_repository",
                 fast_mode: bool = True):
        self.storage_dir = storage_dir
        self.test_repository_dir = test_repository_dir
        self.fast_mode = fast_mode
        
        # Initialize storage systems
        self.workflow_storage = WorkflowStorage(storage_dir=storage_dir)
        self.test_repository = WorkflowTestRepository(storage_dir=test_repository_dir)
        
        # RAG collections
        self.workflow_rag: Optional[SemanticRAGCollection] = None
        self.tool_rag: Optional[SemanticRAGCollection] = None
        self.test_rag: Optional[SemanticRAGCollection] = None
        
        # Initialize collections
        self._initialize_all()
    
    def _initialize_all(self):
        """Initialize all RAG collections."""
        print("🔧 Initializing unified search service...")
        try:
            self._initialize_workflow_rag()
            self._initialize_tool_rag()
            self._initialize_test_rag()
            print("✅ Unified search service initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing unified search: {e}")
    
    def _initialize_workflow_rag(self):
        """Initialize workflow RAG from saved workflows."""
        try:
            # Load saved workflows
            workflow_metadata = self.workflow_storage.list_workflows()
            workflow_specs = []
            
            for metadata in workflow_metadata:
                try:
                    workflow_id = metadata.get('id')
                    if workflow_id:
                        spec = self.workflow_storage.load_workflow(workflow_id)
                        if spec:
                            workflow_specs.append(spec)
                except Exception as e:
                    print(f"⚠️ Skipped workflow {workflow_id}: {e}")
                    continue
            
            if workflow_specs:
                self.workflow_rag = RAGFactory.from_pydantic(
                    models=workflow_specs,
                    collection_name="workflows",
                    field_encodings={
                        "title": "title",
                        "description": "description",
                        "combined": ["title", "description"]
                    },
                    fast_mode=self.fast_mode
                )
                print(f"✅ Indexed {len(workflow_specs)} workflows")
            else:
                print("⚠️ No workflows found to index")
                
        except Exception as e:
            print(f"❌ Error initializing workflow RAG: {e}")
    
    def _initialize_tool_rag(self):
        """Initialize tool RAG from tool registry."""
        try:
            # Use already-loaded tools instead of reloading
            # load_tools_from_env() is already called in workflow_server.py
            tool_catalog = create_tool_catalog(filter_broken=True, verbose_format=False, use_working_filter=True)
            
            print(f"🔍 Tool catalog keys: {list(tool_catalog.keys()) if tool_catalog else 'None'}")
            
            # Check different possible structures
            tools_data = None
            if tool_catalog:
                if 'tools' in tool_catalog:
                    tools_data = tool_catalog['tools']
                elif isinstance(tool_catalog, dict):
                    # Maybe tools are at the root level
                    tools_data = tool_catalog
            
            if not tools_data:
                print("⚠️ No tools found to index")
                return
            
            print(f"🔍 Found {len(tools_data)} tools to index")
            
            # Convert tools to indexable format
            tool_objects = []
            for tool_name, tool_data in tools_data.items():
                # Handle different tool data structures
                description = ""
                category = "general"
                parameters = {}
                
                if isinstance(tool_data, dict):
                    raw_description = tool_data.get('description', tool_data.get('summary', ''))
                    # Clean up XML-style descriptions
                    description = self._clean_description(raw_description)
                    category = tool_data.get('category', 'general')
                    parameters = tool_data.get('parameters', tool_data.get('input_schema', {}))
                
                tool_obj = {
                    'name': tool_name,
                    'description': description,
                    'category': category,
                    'parameters': parameters,
                    'full_data': tool_data
                }
                tool_objects.append(tool_obj)
                
            print(f"🔍 Converted {len(tool_objects)} tool objects")
            
            if tool_objects:
                self.tool_rag = RAGFactory.from_lists(
                    data=[[
                        obj['name'],
                        obj['description'], 
                        obj['category'],
                        str(obj['parameters'])
                    ] for obj in tool_objects],
                    collection_name="tools",
                    column_encodings={
                        "name": 0,
                        "description": 1,
                        "category": 2,
                        "name_desc": [0, 1],  # Combined name + description
                        "all": [0, 1, 2]      # Name + description + category
                    },
                    fast_mode=self.fast_mode
                )
                
                # Store original objects for retrieval
                self.tool_objects = tool_objects
                print(f"✅ Indexed {len(tool_objects)} tools")
            else:
                print("⚠️ No valid tools found to index")
                
        except Exception as e:
            print(f"❌ Error initializing tool RAG: {e}")
    
    def _initialize_test_rag(self):
        """Initialize test RAG from test repository."""
        try:
            # Load test cases from repository - get all test cases from all layers
            test_cases = []
            for layer in [TestLayer.LOGICAL, TestLayer.AGENTIC, TestLayer.ORCHESTRATION, TestLayer.FEEDBACK]:
                try:
                    layer_tests = self.test_repository.get_tests_by_layer(layer)
                    test_cases.extend(layer_tests)
                    print(f"📋 Loaded {len(layer_tests)} tests from {layer.value} layer")
                except Exception as e:
                    print(f"⚠️ Could not load tests for layer {layer.value}: {e}")
                    continue
            
            if not test_cases:
                print("⚠️ No test cases found to index")
                return
            
            # Convert test cases to indexable format
            test_objects = []
            for test_case in test_cases:
                test_obj = {
                    'name': test_case.name,
                    'description': test_case.description,
                    'category': test_case.category,
                    'layer': test_case.layer.value,
                    'tags': ', '.join(test_case.tags),
                    'full_data': test_case
                }
                test_objects.append(test_obj)
            
            if test_objects:
                self.test_rag = RAGFactory.from_lists(
                    data=[[
                        obj['name'],
                        obj['description'],
                        obj['category'],
                        obj['layer'],
                        obj['tags']
                    ] for obj in test_objects],
                    collection_name="tests",
                    column_encodings={
                        "name": 0,
                        "description": 1,
                        "category": 2,
                        "layer": 3,
                        "name_desc": [0, 1],
                        "all": [0, 1, 2, 3, 4]
                    },
                    fast_mode=self.fast_mode
                )
                
                # Store original objects for retrieval
                self.test_objects = test_objects
                print(f"✅ Indexed {len(test_objects)} test cases")
            else:
                print("⚠️ No valid test cases found to index")
                
        except Exception as e:
            print(f"❌ Error initializing test RAG: {e}")
    
    def search(self, 
               query: str, 
               search_types: List[str] = ["workflows", "tools"],
               top_k: int = 5,
               indices: Optional[List[str]] = None) -> UnifiedSearchResponse:
        """Perform unified search across specified types."""
        
        all_results = []
        results_by_type = {}
        
        # Search workflows
        if "workflows" in search_types and self.workflow_rag:
            try:
                # Use the correct index name for workflows
                workflow_indices = ["combined"]  # This is what we actually created
                workflow_results = self.workflow_rag.search_single_index(
                    query, workflow_indices[0], top_k=top_k
                )
                
                for result in workflow_results:
                    workflow_spec = result['data']
                    all_results.append(SearchResult(
                        result_type="workflow",
                        title=workflow_spec.title,
                        description=workflow_spec.description or "",
                        similarity_score=result.get('similarity', result.get('final_score', 0)),
                        data=workflow_spec.model_dump(),
                        metadata={
                            "id": str(workflow_spec.id),
                            "node_count": len(workflow_spec.nodes),
                            "edge_count": len(workflow_spec.edges)
                        }
                    ))
                
                results_by_type["workflows"] = len(workflow_results)
            except Exception as e:
                print(f"❌ Error searching workflows: {e}")
                results_by_type["workflows"] = 0
        
        # Search tools
        if "tools" in search_types and self.tool_rag:
            try:
                tool_indices = ["name_desc"]  # Default to name + description
                tool_results = self.tool_rag.search_single_index(
                    query, tool_indices[0], top_k=top_k
                )
                
                for result in tool_results:
                    # Get the actual index from the search result
                    result_idx = result.get('idx', result.get('index', -1))
                    if result_idx >= 0 and result_idx < len(self.tool_objects):
                        tool_obj = self.tool_objects[result_idx]
                        all_results.append(SearchResult(
                            result_type="tool",
                            title=tool_obj['name'],
                            description=tool_obj['description'],
                            similarity_score=result.get('similarity', result.get('final_score', 0)),
                            data=tool_obj['full_data'],
                            metadata={
                                "category": tool_obj['category'],
                                "parameter_count": len(tool_obj['parameters'])
                            }
                        ))
                
                results_by_type["tools"] = len(tool_results)
            except Exception as e:
                print(f"❌ Error searching tools: {e}")
                results_by_type["tools"] = 0
        
        # Search tests
        if "tests" in search_types and self.test_rag:
            try:
                test_indices = ["name_desc"]
                test_results = self.test_rag.search_single_index(
                    query, test_indices[0], top_k=top_k
                )
                
                for result in test_results:
                    # Get the actual index from the search result
                    result_idx = result.get('idx', result.get('index', -1))
                    if result_idx >= 0 and result_idx < len(self.test_objects):
                        test_obj = self.test_objects[result_idx]
                        all_results.append(SearchResult(
                            result_type="test",
                            title=test_obj['name'],
                            description=test_obj['description'],
                            similarity_score=result.get('similarity', 0),
                            data=test_obj['full_data'].__dict__ if hasattr(test_obj['full_data'], '__dict__') else test_obj['full_data'],
                            metadata={
                                "category": test_obj['category'],
                                "layer": test_obj['layer'],
                                "tags": test_obj['tags']
                            }
                        ))
                
                results_by_type["tests"] = len(test_results)
            except Exception as e:
                print(f"❌ Error searching tests: {e}")
                results_by_type["tests"] = 0
        
        # Sort all results by similarity score
        all_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return UnifiedSearchResponse(
            query=query,
            results=all_results[:top_k],
            total_found=len(all_results),
            results_by_type=results_by_type
        )
    
    def refresh_all(self):
        """Refresh all RAG collections."""
        self._initialize_all()
    
    def _clean_description(self, description: str) -> str:
        """Clean up verbose tool descriptions by extracting summary content."""
        if not description:
            return ""
        
        # Remove XML-style tags and extract just the summary content
        import re
        
        # Extract content from <summary> tags if present
        summary_match = re.search(r'<summary>(.*?)</summary>', description, re.DOTALL)
        if summary_match:
            return summary_match.group(1).strip()
        
        # If no summary tags, remove all XML-style tags
        clean_desc = re.sub(r'<[^>]+>', '', description)
        
        # Take first sentence or first 100 chars, whichever is shorter
        sentences = clean_desc.split('.')
        if sentences and len(sentences[0]) < 100:
            return sentences[0].strip() + '.'
        
        return clean_desc[:100].strip() + ('...' if len(clean_desc) > 100 else '')

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections."""
        stats = {"collections": {}}
        
        if self.workflow_rag:
            stats["collections"]["workflows"] = self.workflow_rag.get_stats()
        if self.tool_rag:
            stats["collections"]["tools"] = self.tool_rag.get_stats()
        if self.test_rag:
            stats["collections"]["tests"] = self.test_rag.get_stats()
            
        return stats


# Create FastAPI app
app = FastAPI(title="Unified Search API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service
search_service = UnifiedSearchService()


@app.get("/")
async def root():
    """Root endpoint with service info."""
    stats = search_service.get_stats()
    return {
        "service": "Unified Search API",
        "description": "Semantic search across workflows, tools, and tests",
        "stats": stats,
        "endpoints": {
            "/search": "Unified semantic search",
            "/stats": "Collection statistics",
            "/refresh": "Refresh all indices"
        }
    }


@app.post("/search", response_model=UnifiedSearchResponse)
async def unified_search(request: UnifiedSearchRequest):
    """Perform unified search across multiple data types."""
    try:
        return search_service.search(
            query=request.query,
            search_types=request.search_types,
            top_k=request.top_k,
            indices=request.indices
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search", response_model=UnifiedSearchResponse)
async def unified_search_get(
    query: str = Query(..., description="Search query"),
    search_types: str = Query("workflows,tools", description="Comma-separated types to search"),
    top_k: int = Query(5, description="Number of results to return"),
    indices: Optional[str] = Query(None, description="Comma-separated indices")
):
    """Unified search (GET version)."""
    search_types_list = search_types.split(",")
    indices_list = indices.split(",") if indices else None
    
    return await unified_search(UnifiedSearchRequest(
        query=query,
        search_types=search_types_list,
        top_k=top_k,
        indices=indices_list
    ))


@app.get("/search/tools", response_model=UnifiedSearchResponse)
async def search_tools_only(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(8, description="Number of results to return")
):
    """Search tools only."""
    return await unified_search(UnifiedSearchRequest(
        query=query,
        search_types=["tools"],
        top_k=top_k
    ))


@app.get("/search/workflows", response_model=UnifiedSearchResponse)
async def search_workflows_only(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(8, description="Number of results to return")
):
    """Search workflows only."""
    return await unified_search(UnifiedSearchRequest(
        query=query,
        search_types=["workflows"],
        top_k=top_k
    ))


@app.get("/stats")
async def get_stats():
    """Get statistics for all collections."""
    return search_service.get_stats()


@app.post("/refresh")
async def refresh_indices():
    """Refresh all search indices."""
    search_service.refresh_all()
    return {"status": "refreshed", "stats": search_service.get_stats()}


def run_server(host: str = "0.0.0.0", port: int = 8101):
    """Run the unified search service."""
    print(f"🚀 Starting Unified Search Service on {host}:{port}")
    print(f"📄 API docs available at http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()