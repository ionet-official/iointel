"""
FastAPI server for serving WorkflowSpecs to the web interface.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn
from pathlib import Path

import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from iointel.src.agent_methods.agents.workflow_planner import WorkflowPlanner
from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec
from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env
from iointel.src.utilities.registries import TOOLS_REGISTRY
from iointel.src.memory import AsyncMemory

# Initialize FastAPI app
app = FastAPI(title="WorkflowPlanner Web Interface", version="1.0.0")

# Global state
planner: Optional[WorkflowPlanner] = None
current_workflow: Optional[WorkflowSpec] = None
workflow_history: List[WorkflowSpec] = []
tool_catalog: Dict[str, Any] = {}

# WebSocket connections for real-time updates
connections: List[WebSocket] = []


class WorkflowRequest(BaseModel):
    query: str
    refine: bool = False


class WorkflowResponse(BaseModel):
    success: bool
    workflow: Optional[Dict] = None
    error: Optional[str] = None


async def broadcast_workflow_update(workflow: WorkflowSpec):
    """Broadcast workflow updates to all connected clients."""
    if connections:
        # Convert UUID objects to strings for JSON serialization
        workflow_data = None
        if workflow:
            workflow_data = workflow.model_dump()
            # Convert UUID to string if present
            if 'id' in workflow_data:
                workflow_data['id'] = str(workflow_data['id'])
        
        message = {
            "type": "workflow_update",
            "workflow": workflow_data,
            "timestamp": datetime.now().isoformat()
        }
        
        disconnected = []
        for connection in connections:
            try:
                await connection.send_json(message)
                print("📡 Sent workflow update to connection")
            except Exception as e:
                print(f"❌ Failed to send to connection: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            connections.remove(conn)
            print("🗑️ Removed disconnected client")


def create_tool_catalog() -> Dict[str, Any]:
    """Create a tool catalog from available tools."""
    catalog = {}
    
    for tool_name, tool in TOOLS_REGISTRY.items():
        catalog[tool_name] = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "is_async": tool.is_async
        }
    
    return catalog


@app.on_event("startup")
async def startup_event(conversation_id: str = "web_interface_session_01"):
    """Initialize the application on startup."""
    global planner, tool_catalog
    
    print("🚀 Starting WorkflowPlanner web server...")
    
    # Initialize memory (SQLite)
    try:
        memory = AsyncMemory("sqlite+aiosqlite:///conversations.db")
        await memory.init_models()
        print("✅ Memory initialized")
    except Exception as e:
        print(f"❌ Memory initialization failed: {e}")
        raise
    
    # Load tools
    try:
        from dotenv import load_dotenv
        load_dotenv("creds.env")
        available_tools = load_tools_from_env("creds.env")
        tool_catalog = create_tool_catalog()
        print(f"✅ Loaded {len(available_tools)} tools from environment")
        print(f"✅ Tool catalog contains {len(tool_catalog)} tools")
    except Exception as e:
        print(f"⚠️  Warning: Could not load tools: {e}")
        tool_catalog = {}
    
    # Initialize WorkflowPlanner
    try:
        planner = WorkflowPlanner(
            memory=memory,
            conversation_id=conversation_id,
            debug=False
        )
        print("✅ WorkflowPlanner initialized")
    except Exception as e:
        print(f"❌ WorkflowPlanner initialization failed: {e}")
        raise
    
    print("🎉 WorkflowPlanner web server initialized successfully!")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    connections.append(websocket)
    print(f"🔌 New WebSocket connection. Total: {len(connections)}")
    
    # Send current workflow on connection
    if current_workflow:
        try:
            workflow_data = current_workflow.model_dump()
            # Convert UUID to string for JSON serialization
            if 'id' in workflow_data:
                workflow_data['id'] = str(workflow_data['id'])
            
            await websocket.send_json({
                "type": "workflow_update",
                "workflow": workflow_data,
                "timestamp": datetime.now().isoformat()
            })
            print("📡 Sent current workflow to new connection")
        except Exception as e:
            print(f"❌ Failed to send current workflow to new connection: {e}")
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in connections:
            connections.remove(websocket)
        print(f"🔌 WebSocket disconnected. Total: {len(connections)}")


@app.get("/api/workflow")
async def get_current_workflow():
    """Get the current workflow."""
    if current_workflow:
        workflow_data = current_workflow.model_dump()
        # Convert UUID to string for JSON serialization
        if 'id' in workflow_data:
            workflow_data['id'] = str(workflow_data['id'])
        return {"workflow": workflow_data}
    return {"workflow": None}


@app.get("/api/tools")
async def get_tools():
    """Get available tools."""
    return {"tools": tool_catalog}


@app.get("/api/history")
async def get_workflow_history():
    """Get workflow history."""
    history_data = []
    for wf in workflow_history[-10:]:  # Last 10
        wf_data = wf.model_dump()
        # Convert UUID to string for JSON serialization
        if 'id' in wf_data:
            wf_data['id'] = str(wf_data['id'])
        history_data.append(wf_data)
    
    return {"history": history_data}


@app.post("/api/generate", response_model=WorkflowResponse)
async def generate_workflow(request: WorkflowRequest):
    """Generate or refine a workflow."""
    global current_workflow
    
    print(f"🔥 Generate workflow request: query='{request.query}', refine={request.refine}")
    
    if not planner:
        print("❌ WorkflowPlanner not initialized")
        raise HTTPException(status_code=500, detail="WorkflowPlanner not initialized")
    
    try:
        if request.refine and current_workflow:
            print(f"🔧 Refining existing workflow (rev {current_workflow.rev})")
            # Refine existing workflow
            new_workflow = await planner.refine_workflow(
                workflow_spec=current_workflow,
                feedback=request.query
            )
            new_workflow.rev = current_workflow.rev + 1
            
            # Store in history
            workflow_history.append(current_workflow)
            current_workflow = new_workflow
            print(f"✅ Workflow refined to rev {current_workflow.rev}")
            
        else:
            print(f"✨ Generating new workflow with {len(tool_catalog)} tools available")
            # Generate new workflow
            current_workflow = await planner.generate_workflow(
                query=request.query,
                tool_catalog=tool_catalog,
                context={"timestamp": datetime.now().isoformat()}
            )
            print(f"✅ New workflow generated: '{current_workflow.title}' with {len(current_workflow.nodes)} nodes")
            
            # Debug: Print the actual workflow structure
            print("🔍 Generated workflow details:")
            print(f"   Title: {current_workflow.title}")
            print(f"   Description: {current_workflow.description}")
            print(f"   ID: {current_workflow.id}")
            print(f"   Rev: {current_workflow.rev}")
            print(f"   Nodes ({len(current_workflow.nodes)}):")
            for i, node in enumerate(current_workflow.nodes):
                print(f"     {i+1}. {node.id} ({node.type}): {node.label}")
                print(f"        Config: {node.data.config}")
                print(f"        Ins: {node.data.ins}")
                print(f"        Outs: {node.data.outs}")
            print(f"   Edges ({len(current_workflow.edges)}):")
            for i, edge in enumerate(current_workflow.edges):
                print(f"     {i+1}. {edge.source} -> {edge.target}")
                print(f"        Condition: {edge.data.condition}")
                print(f"        Handles: {edge.sourceHandle} -> {edge.targetHandle}")
        
        # Broadcast update to connected clients
        print(f"📡 Broadcasting workflow update to {len(connections)} connections")
        await broadcast_workflow_update(current_workflow)
        
        workflow_data = current_workflow.model_dump()
        # Convert UUID to string for JSON serialization
        if 'id' in workflow_data:
            workflow_data['id'] = str(workflow_data['id'])
        print(f"📦 Returning workflow data: {len(str(workflow_data))} characters")
        
        return WorkflowResponse(
            success=True,
            workflow=workflow_data
        )
        
    except Exception as e:
        print(f"❌ Error generating workflow: {type(e).__name__}: {e}")
        import traceback
        print(f"🔍 Traceback: {traceback.format_exc()}")
        return WorkflowResponse(
            success=False,
            error=str(e)
        )


@app.post("/api/clear")
async def clear_workflow():
    """Clear the current workflow."""
    global current_workflow
    current_workflow = None
    
    # Broadcast update
    await broadcast_workflow_update(None)
    
    return {"success": True}


@app.get("/api/example")
async def create_example_workflow():
    """Create an example workflow."""
    global current_workflow
    
    print("📋 Creating example workflow")
    
    if not planner:
        print("❌ WorkflowPlanner not initialized")
        raise HTTPException(status_code=500, detail="WorkflowPlanner not initialized")
    
    current_workflow = planner.create_example_workflow("Web Example Workflow")
    print(f"✅ Example workflow created: '{current_workflow.title}' with {len(current_workflow.nodes)} nodes, {len(current_workflow.edges)} edges")
    
    # Broadcast update
    print(f"📡 Broadcasting example workflow to {len(connections)} connections")
    await broadcast_workflow_update(current_workflow)
    
    workflow_data = current_workflow.model_dump()
    # Convert UUID to string for JSON serialization
    if 'id' in workflow_data:
        workflow_data['id'] = str(workflow_data['id'])
    print(f"📦 Returning example workflow data: {len(str(workflow_data))} characters")
    
    return {"workflow": workflow_data}


# Serve static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page."""
    html_file = static_dir / "index.html"
    if html_file.exists():
        return FileResponse(html_file)
    
    # Return a simple default page if the file doesn't exist
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>WorkflowPlanner</title>
    </head>
    <body>
        <h1>WorkflowPlanner Web Interface</h1>
        <p>The static files are not yet created. Please run the setup script.</p>
        <p>API is available at <a href="/docs">/docs</a></p>
    </body>
    </html>
    """)


if __name__ == "__main__":
    uvicorn.run(
        "workflow_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )