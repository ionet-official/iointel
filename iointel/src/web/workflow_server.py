"""
FastAPI server for serving WorkflowSpecs to the web interface.
"""

import sys
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware
import uvicorn
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from iointel.src.agent_methods.agents.workflow_planner import WorkflowPlanner
from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec
from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env
from iointel.src.utilities.tool_registry_utils import create_tool_catalog
from iointel.src.memory import AsyncMemory
from iointel.src.workflow import Workflow
from iointel.src.utilities.decorators import register_custom_task
import uuid
import asyncio

# Import example tools to register them globally

# Import conditional gate tools to register them globally

# Import workflow storage
from .workflow_storage import WorkflowStorage
from .execution_feedback import (
    feedback_collector, 
    create_execution_feedback_prompt,
    ExecutionStatus,
    WorkflowExecutionSummary
)
from ..utilities.io_logger import workflow_logger, execution_logger, system_logger
from iointel.src.chainables import execute_tool_task, execute_agent_task
from iointel.src.agent_methods.tools.collection_manager import search_collections, create_collection


# Register executors for web interface
@register_custom_task("tool")
async def web_tool_executor(task_metadata, objective, agents, execution_metadata):
    """Tool executor for web interface with real-time updates, delegates to backend executor."""
    execution_id = execution_metadata.get("execution_id")
    tool_name = task_metadata.get("tool_name")
    print(f"🔧 [WEB] Executing tool: {tool_name} (execution: {execution_id})")
    # Broadcast task start if we have connections available
    if execution_id and 'connections' in globals() and len(connections) > 0:
        try:
            await broadcast_execution_update(
                execution_id, 
                "running", 
                results={"current_task": tool_name, "status": "started"}
            )
        except Exception as e:
            print(f"⚠️ Failed to broadcast task start: {e}")
    try:
        # Record node start in feedback collector
        if execution_id:
            node_id = execution_metadata.get('node_id', tool_name)
            feedback_collector.record_node_start(
                execution_id=execution_id,
                node_id=node_id,
                node_type="tool",
                node_label=tool_name
            )
        
        # Delegate to backend portable executor
        result = await execute_tool_task(task_metadata, objective, agents, execution_metadata)
        print(f"✅ Tool '{tool_name}' completed: {result}")
        
        # Record node completion in feedback collector
        if execution_id:
            feedback_collector.record_node_completion(
                execution_id=execution_id,
                node_id=node_id,
                status=ExecutionStatus.SUCCESS,
                result_preview=str(result)[:200] if result else None,
                tool_usage=[tool_name]
            )
        
        # Broadcast task completion
        if execution_id and 'connections' in globals() and len(connections) > 0:
            try:
                await broadcast_execution_update(
                    execution_id,
                    "running",
                    results={"current_task": tool_name, "status": "completed", "result": result}
                )
            except Exception as e:
                print(f"⚠️ Failed to broadcast completion: {e}")
        return result
    except Exception as e:
        error_msg = f"Tool '{tool_name}' failed: {str(e)}"
        print(f"❌ {error_msg}")
        
        # Record node failure in feedback collector
        if execution_id:
            feedback_collector.record_node_completion(
                execution_id=execution_id,
                node_id=node_id,
                status=ExecutionStatus.FAILED,
                error_message=str(e),
                tool_usage=[tool_name]
            )
        
        if execution_id and 'connections' in globals() and len(connections) > 0:
            try:
                await broadcast_execution_update(execution_id, "failed", error=error_msg)
            except Exception as e:
                print(f"⚠️ Failed to broadcast tool error: {e}")
        raise


@register_custom_task("agent")
async def web_agent_executor(task_metadata, objective, agents, execution_metadata):
    """Agent executor for web interface with real-time updates."""
    execution_id = execution_metadata.get("execution_id")
    
    print(f"🤖 [WEB] Executing agent task (execution: {execution_id})")
    
    # Use the chainables agent executor to avoid duplication
    # from iointel.src.chainables import execute_agent_task # This line is removed as per the edit hint
    
    # Broadcast task start if we have connections available
    if execution_id and len(connections) > 0:
        try:
            await broadcast_execution_update(
                execution_id, 
                "running", 
                results={"current_task": "agent", "status": "started"}
            )
        except Exception as e:
            print(f"⚠️ Failed to broadcast agent start: {e}")
    
    try:
        # Record node start in feedback collector
        node_id = execution_metadata.get('node_id', 'agent')
        agent_label = task_metadata.get('name', 'Agent Task')
        
        if execution_id:
            feedback_collector.record_node_start(
                execution_id=execution_id,
                node_id=node_id,
                node_type="agent",
                node_label=agent_label
            )
        
        result = await execute_agent_task(task_metadata, objective, agents, execution_metadata)
        print(f"✅ Agent task completed: {result}")
        
        # Record node completion in feedback collector
        if execution_id:
            # Extract tool usage from result if available
            tool_usage = []
            if isinstance(result, dict) and 'tool_usage_results' in result:
                # Handle both dict and ToolUsageResult objects
                tool_usage = []
                for tool in result['tool_usage_results']:
                    if hasattr(tool, 'tool_name'):
                        # It's a ToolUsageResult object
                        tool_usage.append(tool.tool_name)
                    elif isinstance(tool, dict):
                        # It's a dict
                        tool_usage.append(tool.get('tool_name', 'unknown'))
                    else:
                        tool_usage.append('unknown')
            
            feedback_collector.record_node_completion(
                execution_id=execution_id,
                node_id=node_id,
                status=ExecutionStatus.SUCCESS,
                result_preview=str(result.get('result', result))[:200] if result else None,
                tool_usage=tool_usage
            )
        
        # Broadcast task completion
        if execution_id and len(connections) > 0:
            try:
                # Serialize the result for proper JSON transmission
                serialized_result = serialize_execution_results({"result": result})
                await broadcast_execution_update(
                    execution_id, 
                    "running", 
                    results={"agent_completed": True, **serialized_result}
                )
            except Exception as e:
                print(f"⚠️ Failed to broadcast agent completion: {e}")
        
        return result
    except Exception as e:
        error_msg = f"Agent task failed: {str(e)}"
        print(f"❌ {error_msg}")
        
        # Record node failure in feedback collector
        if execution_id:
            feedback_collector.record_node_completion(
                execution_id=execution_id,
                node_id=node_id,
                status=ExecutionStatus.FAILED,
                error_message=str(e)
            )
        
        if execution_id and len(connections) > 0:
            try:
                await broadcast_execution_update(execution_id, "failed", error=error_msg)
            except Exception as e:
                print(f"⚠️ Failed to broadcast agent error: {e}")
        raise


@register_custom_task("decision")
async def web_decision_executor(task_metadata, objective, agents, execution_metadata):
    """Decision executor for web interface - delegates to tool executor or agent."""
    execution_id = execution_metadata.get("execution_id")
    tool_name = task_metadata.get('tool_name')
    
    print(f"🤔 [WEB] Executing decision task: {tool_name or 'agent-based'} (execution: {execution_id})")
    
    # If no tool_name specified, treat as agent-based decision
    if not tool_name:
        print("   📝 No tool_name specified, treating as agent-based decision")
        
        # Create a simple agent-based decision that returns a boolean result
        # This is a fallback for workflows with null tool_name in decision nodes
        try:
            # For now, return a default decision result
            # In a real implementation, you might analyze the available data
            result = {
                "result": True,  # Default decision
                "details": "Agent-based decision (fallback)",
                "confidence": 0.5
            }
            
            # Broadcast completion
            if execution_id and len(connections) > 0:
                try:
                    await broadcast_execution_update(
                        execution_id, 
                        "running", 
                        results={"decision_completed": True, "result": result}
                    )
                except Exception as e:
                    print(f"⚠️ Failed to broadcast decision completion: {e}")
            
            return result
            
        except Exception as e:
            error_msg = f"Agent-based decision failed: {str(e)}"
            print(f"❌ {error_msg}")
            if execution_id and len(connections) > 0:
                try:
                    await broadcast_execution_update(execution_id, "failed", error=error_msg)
                except Exception as e:
                    print(f"⚠️ Failed to broadcast decision error: {e}")
            raise
    
    # If tool_name is specified, use tool-based decision
    return await web_tool_executor(task_metadata, objective, agents, execution_metadata)


@register_custom_task("workflow_call")
async def web_workflow_call_executor(task_metadata, objective, agents, execution_metadata):
    """Workflow call executor for web interface."""
    execution_id = execution_metadata.get("execution_id")
    workflow_id = task_metadata.get("workflow_id", "unknown")
    
    print(f"📞 [WEB] Executing workflow call: {workflow_id} (execution: {execution_id})")
    
    # Broadcast task start
    if execution_id and len(connections) > 0:
        try:
            await broadcast_execution_update(
                execution_id, 
                "running", 
                results={"current_task": f"workflow_call:{workflow_id}", "status": "started"}
            )
        except Exception as e:
            print(f"⚠️ Failed to broadcast workflow_call start: {e}")
    
    try:
        # For now, return a success message (implement actual workflow calling later)
        result = f"Workflow '{workflow_id}' executed successfully (mock)"
        
        # Broadcast completion
        if execution_id and len(connections) > 0:
            try:
                await broadcast_execution_update(
                    execution_id, 
                    "running", 
                    results={"workflow_call_completed": True, "result": result}
                )
            except Exception as e:
                print(f"⚠️ Failed to broadcast workflow_call completion: {e}")
        
        return result
    except Exception as e:
        error_msg = f"Workflow call '{workflow_id}' failed: {str(e)}"
        print(f"❌ {error_msg}")
        if execution_id and len(connections) > 0:
            try:
                await broadcast_execution_update(execution_id, "failed", error=error_msg)
            except Exception as e:
                print(f"⚠️ Failed to broadcast workflow_call error: {e}")
        raise


# Initialize FastAPI app
app = FastAPI(title="WorkflowPlanner Web Interface", version="1.0.0")

# Add session middleware for conversation ID persistence
app.add_middleware(
    SessionMiddleware,
    secret_key="workflow-session-key-2025",  # In production, use environment variable
    max_age=86400 * 7  # 7 days
)

# Global state
planner: Optional[WorkflowPlanner] = None
current_workflow: Optional[WorkflowSpec] = None
workflow_history: List[WorkflowSpec] = []
tool_catalog: Dict[str, Any] = {}
active_executions: Dict[str, Dict[str, Any]] = {}  # execution_id -> execution_info
workflow_storage: Optional[WorkflowStorage] = None

# WebSocket connections for real-time updates
connections: List[WebSocket] = []


class WorkflowRequest(BaseModel):
    query: str
    refine: bool = False


class WorkflowResponse(BaseModel):
    success: bool
    workflow: Optional[WorkflowSpec] = None
    agent_response: Optional[str] = None
    error: Optional[str] = None


class ExecutionRequest(BaseModel):
    execute_current: bool = True
    workflow_data: Optional[WorkflowSpec] = None
    user_inputs: Optional[Dict] = None
    form_id: Optional[str] = None


class ExecutionResponse(BaseModel):
    success: bool
    execution_id: Optional[str] = None
    status: str = "started"  # started, running, completed, failed
    results: Optional[Dict] = None
    error: Optional[str] = None


class SaveWorkflowRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None


class SaveWorkflowResponse(BaseModel):
    success: bool
    workflow_id: Optional[str] = None
    error: Optional[str] = None


class SearchWorkflowsRequest(BaseModel):
    query: Optional[str] = None
    tags: Optional[List[str]] = None

class SearchCollectionsRequest(BaseModel):
    query: str
    tool_filter: Optional[str] = None

class SearchCollectionsResponse(BaseModel):
    success: bool
    results: Optional[List[Dict]] = None
    error: Optional[str] = None

class SaveToCollectionRequest(BaseModel):
    collection_name: str
    record: str
    tool_type: Optional[str] = None

class SaveToCollectionResponse(BaseModel):
    success: bool
    message: Optional[str] = None
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
        
        await broadcast_message(message)


def serialize_execution_results(results: Optional[Dict] = None) -> Optional[Dict]:
    """Serialize execution results for JSON transmission, handling all custom objects."""
    if not results:
        return results
        
    def serialize_value(value):
        """Recursively serialize a value for JSON transmission."""
        # Handle BaseModel objects (Pydantic models like ToolUsageResult, GateResult)
        if hasattr(value, 'model_dump'):
            try:
                # Use Pydantic's model_dump for clean serialization
                return value.model_dump()
            except Exception:
                # Fallback to dict conversion
                return dict(value)
        
        # Handle AgentRunResult objects specifically
        elif hasattr(value, '__class__') and 'AgentRunResult' in str(value.__class__):
            return {
                "result": getattr(value, 'output', None) or getattr(value, 'result', None),
                "tool_usage_results": [serialize_value(tur) for tur in getattr(value, 'tool_usage_results', [])],
                "conversation_id": getattr(value, 'conversation_id', None),
                "type": "AgentRunResult"
            }
        
        # Handle lists
        elif isinstance(value, list):
            return [serialize_value(item) for item in value]
        
        # Handle dictionaries
        elif isinstance(value, dict):
            # Check if this is an already-processed agent result dict
            if 'tool_usage_results' in value and 'full_result' in value:
                # This is already a processed agent result, preserve tool_usage_results
                result = value.copy()
                # Serialize tool_usage_results
                result['tool_usage_results'] = [serialize_value(tur) for tur in value.get('tool_usage_results', [])]
                # Only serialize the full_result if it's an object
                if hasattr(value.get('full_result'), '__class__'):
                    result['full_result'] = serialize_value(value['full_result'])
                return result
            else:
                # Recursively serialize nested dicts
                return {k: serialize_value(v) for k, v in value.items()}
        
        # Handle other objects that might not be JSON serializable
        elif hasattr(value, '__dict__'):
            try:
                # Try to convert to dict
                return {k: serialize_value(v) for k, v in value.__dict__.items()}
            except Exception:
                # If that fails, convert to string
                return str(value)
        
        # Return primitive values as-is
        else:
            return value
    
    # Apply serialization to all values in the results dict
    serialized = {}
    for key, value in results.items():
        serialized[key] = serialize_value(value)
    return serialized


async def broadcast_execution_update(execution_id: str, status: str, results: Optional[Dict] = None, error: Optional[str] = None):
    """Broadcast execution status updates to all connected clients."""
    # Only log significant status changes
    if status in ['started', 'completed', 'failed']:
        execution_logger.info(
            f"Execution status update: {status}", 
            data={"execution_id": execution_id[:8], "status": status},
            execution_id=execution_id
        )
    
    # Serialize results to handle AgentRunResult objects
    serialized_results = serialize_execution_results(results)
    
    message = {
        "type": "execution_update",
        "execution_id": execution_id,
        "status": status,
        "results": serialized_results,
        "error": error,
        "timestamp": datetime.now().isoformat()
    }
    await broadcast_message(message)


async def broadcast_message(message: Dict):
    """Broadcast a message to all connected clients."""
    if connections:
        disconnected = []
        sent_count = 0
        for connection in connections:
            try:
                await connection.send_json(message)
                sent_count += 1
            except Exception as e:
                print(f"❌ Failed to send to connection: {e}")
                disconnected.append(connection)
        
        # Only log meaningful updates
        if message['type'] == 'execution_update' and message.get('status') in ['started', 'completed', 'failed']:
            print(f"📡 Broadcast {message['type']} ({message['status']}) to {sent_count} clients")
        elif message['type'] == 'workflow_update':
            print(f"📡 Broadcast workflow update to {sent_count} clients")
        
        # Remove disconnected clients
        for conn in disconnected:
            connections.remove(conn)
            print("🗑️ Removed disconnected client")




@app.on_event("startup")
async def startup_event(conversation_id: str = "web_interface_session_01"):
    """Initialize the application on startup."""
    global planner, tool_catalog, workflow_storage
    
    print("🚀 Starting WorkflowPlanner web server...")
    
    # Initialize memory (SQLite)
    try:
        memory = AsyncMemory("sqlite+aiosqlite:///conversations.db")
        await memory.init_models()
        system_logger.success("Memory system initialized successfully")
    except Exception as e:
        system_logger.error("Memory initialization failed", data={"error": str(e), "error_type": type(e).__name__})
        raise
    
    # Load tools
    try:
        from dotenv import load_dotenv
        load_dotenv("creds.env")
        available_tools = load_tools_from_env("creds.env")
        tool_catalog = create_tool_catalog()
        system_logger.success(
            "Tools and catalog loaded successfully",
            data={
                "available_tools": len(available_tools),
                "catalog_tools": len(tool_catalog),
                "tools_loaded": available_tools[:5] + (["..."] if len(available_tools) > 5 else [])
            }
        )
    except Exception as e:
        system_logger.warning("Could not load tools from environment", data={"error": str(e), "fallback": "using empty catalog"})
        tool_catalog = {}
    
    # Initialize WorkflowPlanner
    try:
        planner = WorkflowPlanner(
            memory=memory,
            conversation_id=conversation_id,
            debug=False
        )
        system_logger.success("WorkflowPlanner initialized", data={"memory_enabled": True, "conversation_id": conversation_id})
    except Exception as e:
        system_logger.error("WorkflowPlanner initialization failed", data={"error": str(e), "error_type": type(e).__name__})
        raise
    
    # Initialize WorkflowStorage
    try:
        workflow_storage = WorkflowStorage()
        print("✅ WorkflowStorage initialized")
    except Exception as e:
        print(f"❌ WorkflowStorage initialization failed: {e}")
        raise
    
    print("🎉 WorkflowPlanner web server initialized successfully!")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    connections.append(websocket)
    system_logger.info("New WebSocket connection established", data={"total_connections": len(connections)})
    
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
            # Keep connection alive - handle any incoming messages or just wait
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                # Echo back any received messages for debugging
                print(f"📩 Received WebSocket message: {message}")
            except asyncio.TimeoutError:
                # No message received, send a keepalive ping
                await websocket.send_json({"type": "ping", "timestamp": datetime.now().isoformat()})
    except WebSocketDisconnect:
        if websocket in connections:
            connections.remove(websocket)
        system_logger.info("WebSocket disconnected", data={"remaining_connections": len(connections)})


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


@app.get("/api/executions")
async def get_active_executions():
    """Get all active executions."""
    return {"executions": active_executions}


@app.get("/api/executions/{execution_id}")
async def get_execution_status(execution_id: str):
    """Get status of a specific execution."""
    if execution_id in active_executions:
        return {"execution": active_executions[execution_id]}
    else:
        raise HTTPException(status_code=404, detail="Execution not found")


@app.get("/api/executions/{execution_id}/feedback")
async def get_execution_feedback(execution_id: str):
    """Get WorkflowPlanner analysis and feedback for an execution."""
    if execution_id not in active_executions:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    execution_info = active_executions[execution_id]
    execution_summary = execution_info.get("execution_summary")
    planner_feedback = execution_info.get("planner_feedback")
    
    if not execution_summary:
        raise HTTPException(status_code=404, detail="Execution feedback not available")
    
    # Import here to avoid circular imports
    from .execution_feedback import ExecutionResultCurator
    
    return {
        "execution_id": execution_id,
        "curated_summary": ExecutionResultCurator.curate_execution_summary(execution_summary),
        "planner_feedback": planner_feedback,
        "execution_metrics": {
            "total_duration": execution_summary.total_duration_seconds,
            "nodes_executed": len(execution_summary.nodes_executed),
            "nodes_skipped": len(execution_summary.nodes_skipped),
            "success_rate": len([n for n in execution_summary.nodes_executed if n.status.value == "success"]) / max(len(execution_summary.nodes_executed), 1) * 100
        },
        "execution_summary": {
            "workflow_title": execution_summary.workflow_title,
            "status": execution_summary.status.value,
            "total_duration_seconds": execution_summary.total_duration_seconds,
            "nodes_executed": [
                {
                    "node_label": node.node_label,
                    "node_type": node.node_type,
                    "status": node.status.value,
                    "duration_seconds": node.duration_seconds,
                    "result_preview": node.result_preview,
                    "error_message": node.error_message,
                    "tool_usage": node.tool_usage
                }
                for node in execution_summary.nodes_executed
            ],
            "nodes_skipped": execution_summary.nodes_skipped,
            "user_inputs": execution_summary.user_inputs,
            "error_summary": execution_summary.error_summary
        }
    }


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
async def generate_workflow(workflow_request: WorkflowRequest, request: Request) -> WorkflowResponse:
    """Generate or refine a workflow."""
    global current_workflow
    
    print(f"🔥 Generate workflow request: query='{workflow_request.query}', refine={workflow_request.refine}")
    
    # Get or create faux user for session management
    session_id = request.session.get("workflow_session_id")
    faux_user = get_or_create_faux_user(session_id) if session_id else None
    
    if not planner:
        print("❌ WorkflowPlanner not initialized")
        raise HTTPException(status_code=500, detail="WorkflowPlanner not initialized")
    
    try:
        print(f"✨ Generating workflow with {len(tool_catalog)} tools available")
        
        # Set current workflow context so planner can reference it
        if current_workflow:
            from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec
            if isinstance(current_workflow, WorkflowSpec):
                print(f"🔧 Building upon existing workflow (rev {current_workflow.rev}): '{current_workflow.title}'")
                planner.set_current_workflow(current_workflow)
            else:
                print(f"🔧 Building upon existing chat context: '{getattr(current_workflow, 'title', 'Chat')}'")
                # Don't set current workflow for chat-only responses
            
        # Generate workflow (could be new workflow or chat-only response)
        # Use faux user's interface conversation ID for WorkflowPlanner memory continuity
        conversation_id = faux_user.interface_conversation_id if faux_user else None
        
        result = await planner.generate_workflow(
            query=workflow_request.query,
            tool_catalog=tool_catalog,
            context={
                "timestamp": datetime.now().isoformat(),
                "is_refinement": workflow_request.refine,
                "previous_workflow_title": getattr(current_workflow, 'title', None) if current_workflow else None,
                "user_id": faux_user.user_id if faux_user else None
            },
            conversation_id=conversation_id
        )
        
        # Check if this is a chat-only response or normal workflow
        from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpecLLM
        if isinstance(result, WorkflowSpecLLM):
            if result.nodes is None or result.edges is None:
                print(f"💬 Chat-only response: {result.reasoning}")
                
                # Return chat response without updating current workflow
                return WorkflowResponse(
                    success=True,
                    workflow=None,  # No workflow update
                    agent_response=result.reasoning  # Use reasoning field for chat
                )
        
        # Handle normal workflow generation (WorkflowSpec)
        workflow_spec = result
        
        # Only update current_workflow if we have a valid WorkflowSpec (not a chat-only response)
        from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec
        if isinstance(workflow_spec, WorkflowSpec):
            if current_workflow:
                # Store current workflow in history before generating new one
                workflow_history.append(current_workflow)
                
            current_workflow = workflow_spec
            
            # Increment revision for continual improvement tracking
            current_workflow.rev = len(workflow_history) + 1
            workflow_logger.success(
                "Workflow evolved to new revision", 
                data={
                    "revision": current_workflow.rev,
                    "title": current_workflow.title,
                    "nodes_count": len(current_workflow.nodes),
                    "edges_count": len(current_workflow.edges)
                }
            )
            
            # Register workflow revision with faux user for stable conversation tracking
            if faux_user:
                faux_user.register_workflow_revision(current_workflow, "default")
                print(f"📝 Registered workflow revision with user {faux_user.user_id}")
            
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
                if hasattr(node.data, 'sla') and node.data.sla is not None:
                    print(f"        SLA: {node.data.sla.model_dump_json(indent=2) if hasattr(node.data.sla, 'model_dump_json') else node.data.sla}")
            print(f"   Edges ({len(current_workflow.edges)}):")
            for i, edge in enumerate(current_workflow.edges):
                print(f"     {i+1}. {edge.source} -> {edge.target}")
                print(f"        Condition: {edge.data.condition}")
                print(f"        Handles: {edge.sourceHandle} -> {edge.targetHandle}")

            # Build a pretty-printed workflow spec for the agent/planner, including SLA
            dag_pretty = f"\n---\nWORKFLOW DAG (full spec, including SLA):\nTitle: {current_workflow.title}\nDescription: {current_workflow.description}\nID: {current_workflow.id}\nRev: {current_workflow.rev}\nNodes ({len(current_workflow.nodes)}):\n"
            for i, node in enumerate(current_workflow.nodes):
                dag_pretty += f"  {i+1}. {node.id} ({node.type}): {node.label}\n"
                dag_pretty += f"     Config: {node.data.config}\n"
                dag_pretty += f"     Ins: {node.data.ins}\n"
                dag_pretty += f"     Outs: {node.data.outs}\n"
                if hasattr(node.data, 'sla') and node.data.sla is not None:
                    dag_pretty += f"     SLA: {node.data.sla.model_dump_json(indent=2) if hasattr(node.data.sla, 'model_dump_json') else node.data.sla}\n"
            dag_pretty += f"Edges ({len(current_workflow.edges)}):\n"
            for i, edge in enumerate(current_workflow.edges):
                dag_pretty += f"  {i+1}. {edge.source} -> {edge.target}\n"
                dag_pretty += f"     Condition: {edge.data.condition}\n"
                dag_pretty += f"     Handles: {edge.sourceHandle} -> {edge.targetHandle}\n"
            dag_pretty += "---\n"

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
                workflow=workflow_data,
                agent_response=f"{result.reasoning}\n\n{dag_pretty}"
            )
            
        else:
            # Check if this was a WorkflowSpecLLM that we already handled above
            if isinstance(result, WorkflowSpecLLM) and (result.nodes is None or result.edges is None):
                # This case should have been handled above, but just in case
                print("⚠️ Chat-only response reached else block")
                return WorkflowResponse(
                    success=True,
                    workflow=None,
                    agent_response=result.reasoning
                )
            else:
                # This shouldn't happen with current implementation, but handle gracefully
                print("❌ Unexpected result format from planner.generate_workflow")
                return WorkflowResponse(
                    success=False,
                    error="Unexpected result format from workflow planner"
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


@app.post("/api/execute", response_model=ExecutionResponse)
async def execute_workflow(execution_request: ExecutionRequest, request: Request):
    """Execute the current workflow or provided workflow data."""
    global current_workflow
    
    print(f"🚀 Execute workflow request: execute_current={execution_request.execute_current}")
    print(f"🔍 Current workflow exists: {current_workflow is not None}")
    if current_workflow:
        print(f"🔍 Current workflow title: '{current_workflow.title}'")
    print(f"🔍 Request has workflow_data: {execution_request.workflow_data is not None}")
    
    # Determine which workflow to execute
    workflow_to_execute = None
    if execution_request.execute_current and current_workflow:
        workflow_to_execute = current_workflow
        print(f"📋 Executing current workflow: '{workflow_to_execute.title}'")
    elif execution_request.workflow_data:
        # Reconstruct WorkflowSpec from provided data
        try:
            workflow_to_execute = WorkflowSpec.model_validate(execution_request.workflow_data)
            print(f"📋 Executing provided workflow: '{workflow_to_execute.title}'")
        except Exception as e:
            print(f"❌ Failed to validate workflow data: {e}")
            return ExecutionResponse(
                success=False,
                error=f"Invalid workflow data: {str(e)}"
            )
    
    if not workflow_to_execute:
        print(f"❌ No workflow to execute - current_workflow: {current_workflow is not None}, workflow_data: {execution_request.workflow_data is not None}")
        return ExecutionResponse(
            success=False,
            error="No workflow to execute. Generate a workflow first or provide workflow data."
        )
    
    # Create execution ID
    execution_id = str(uuid.uuid4())
    print(f"🏷️ Generated execution ID: {execution_id}")
    
    # Store execution info
    active_executions[execution_id] = {
        "id": execution_id,
        "workflow_title": workflow_to_execute.title,
        "workflow_id": str(workflow_to_execute.id),
        "status": "started",
        "start_time": datetime.now().isoformat(),
        "results": None,
        "error": None
    }
    
    # Broadcast execution start
    await broadcast_execution_update(execution_id, "started")
    
    # Start execution in background with user inputs
    asyncio.create_task(execute_workflow_background(
        workflow_to_execute, 
        execution_id, 
        user_inputs=execution_request.user_inputs,
        form_id=execution_request.form_id,
        session_info={
            "workflow_session_id": request.session.get("workflow_session_id"),
            "chat_mode": request.session.get("chat_mode", False)
        }
    ))
    
    return ExecutionResponse(
        success=True,
        execution_id=execution_id,
        status="started"
    )


async def execute_workflow_background(
    workflow_spec: WorkflowSpec, 
    execution_id: str, 
    user_inputs: Optional[Dict] = None,
    form_id: Optional[str] = None,
    session_info: Optional[Dict] = None
):
    """Execute workflow in background with real-time updates."""
    try:
        print(f"🛠️ Starting background execution: {execution_id}")
        if user_inputs:
            print(f"📝 User inputs provided: {list(user_inputs.keys())}")
        
        # Update status
        active_executions[execution_id]["status"] = "running"
        await broadcast_execution_update(execution_id, "running")
        
        # Convert WorkflowSpec to executable Workflow
        workflow_spec.to_workflow_definition()
        yaml_content = workflow_spec.to_yaml()
        
        # Create workflow from YAML with custom conversation ID
        workflow = Workflow.from_yaml(yaml_str=yaml_content)
        workflow.objective = workflow_spec.description
        
        print(f"📋 Executing workflow with {len(workflow.tasks)} tasks")
        
        # Start execution feedback tracking
        feedback_collector.start_execution_tracking(
            execution_id=execution_id,
            workflow_spec=current_workflow,
            user_inputs=user_inputs or {}
        )
        
        # Execute workflow with execution context using faux user model
        session_info = session_info or {}
        session_id = session_info.get("workflow_session_id")
        chat_mode = session_info.get("chat_mode", False)
        
        if chat_mode and session_id:
            # Get faux user and stable workflow conversation ID
            faux_user = get_or_create_faux_user(session_id)
            workflow_conversation_id = faux_user.get_or_create_workflow_conversation_id("default")
            conversation_id = workflow_conversation_id
            print(f"🗣️ Using chat mode with stable workflow conversation_id: {conversation_id}")
            print(f"👤 Faux user: {faux_user.user_id}, Active workflows: {len(faux_user.active_workflows)}")
        else:
            # Use unique execution ID for single-serve runs (no conversation memory)
            conversation_id = f"web_execution_{execution_id}"
            print(f"🔄 Using single-serve mode with execution conversation_id: {conversation_id}")
        
        # Add execution_id and user inputs to all task metadata for real-time updates
        for task in workflow.tasks:
            if "execution_metadata" not in task:
                task["execution_metadata"] = {}
            task["execution_metadata"]["execution_id"] = execution_id
            
            # Add user inputs to task metadata if provided
            if user_inputs:
                task["execution_metadata"]["user_inputs"] = user_inputs
                task["execution_metadata"]["form_id"] = form_id
                print(f"🔍 Adding user_inputs to task {task.get('task_id', 'unknown')}: {user_inputs}")
        
        # Execute the workflow
        results = await workflow.run_tasks(conversation_id=conversation_id)
        
        # Complete execution feedback tracking
        final_outputs = results.get('results', {}) if results else {}
        execution_summary = feedback_collector.complete_execution(
            execution_id=execution_id,
            final_outputs=final_outputs
        )
        
        # Update execution info with serialized results
        active_executions[execution_id].update({
            "status": "completed",
            "end_time": datetime.now().isoformat(),
            "results": serialize_execution_results(results),
            "error": None,
            "execution_summary": execution_summary
        })
        
        # Import logger
        from ..utilities.io_logger import execution_logger
        
        execution_logger.success(
            "Workflow execution completed",
            data={
                "execution_id": execution_id,
                "workflow_title": workflow_spec.title,
                "total_results": len(results.get('results', {})),
                "execution_time": execution_summary.total_duration_seconds,
                "nodes_executed": len(execution_summary.nodes_executed),
                "nodes_skipped": len(execution_summary.nodes_skipped),
                "status": execution_summary.status.value if hasattr(execution_summary.status, 'value') else str(execution_summary.status)
            },
            execution_id=execution_id
        )
        
        # Generate and send feedback to WorkflowPlanner using interface conversation ID
        interface_conversation_id = None
        if chat_mode and session_id:
            faux_user = get_or_create_faux_user(session_id)
            interface_conversation_id = faux_user.interface_conversation_id
        
        await send_execution_feedback_to_planner(execution_summary, interface_conversation_id, workflow_spec)
        
        # Broadcast completion
        await broadcast_execution_update(
            execution_id, 
            "completed", 
            results=results
        )
        
    except Exception as e:
        error_msg = str(e)
        execution_logger.error(
            "Workflow execution failed",
            data={
                "execution_id": execution_id,
                "error_message": error_msg,
                "workflow_title": getattr(workflow_spec, 'title', 'Unknown')
            },
            execution_id=execution_id
        )
        import traceback
        print(f"🔍 Traceback: {traceback.format_exc()}")
        
        # Complete execution feedback tracking with error
        execution_summary = feedback_collector.complete_execution(
            execution_id=execution_id,
            error_summary=error_msg
        )
        
        # Update execution info
        active_executions[execution_id].update({
            "status": "failed",
            "end_time": datetime.now().isoformat(),
            "error": error_msg,
            "execution_summary": execution_summary
        })
        
        # Generate and send feedback to WorkflowPlanner for error analysis
        # TODO: Pass interface_conversation_id for error feedback too
        await send_execution_feedback_to_planner(execution_summary, None, workflow_spec)
        
        # Broadcast failure
        await broadcast_execution_update(execution_id, "failed", error=error_msg)


async def send_execution_feedback_to_planner(execution_summary: WorkflowExecutionSummary, interface_conversation_id: Optional[str] = None, workflow_spec: Optional[WorkflowSpec] = None):
    """Send execution results back to WorkflowPlanner for analysis and suggestions."""
    try:
        # Import logger
        from ..utilities.io_logger import execution_logger
        
        # Generate feedback prompt
        feedback_prompt = create_execution_feedback_prompt(execution_summary, workflow_spec)
        
        # Log execution report with all relevant data
        execution_logger.execution_report(
            title=f"WorkflowPlanner Analysis for {execution_summary.workflow_title}",
            report_data={
                "execution_id": execution_summary.execution_id,
                "status": execution_summary.status.value if hasattr(execution_summary.status, 'value') else str(execution_summary.status),
                "duration": f"{execution_summary.total_duration_seconds:.2f}s",
                "nodes_executed": len(execution_summary.nodes_executed),
                "nodes_skipped": len(execution_summary.nodes_skipped),
                "workflow_spec": workflow_spec,
                "execution_summary": execution_summary,
                "feedback_prompt": feedback_prompt
            },
            execution_id=execution_summary.execution_id
        )
        
        # Initialize WorkflowPlanner for feedback analysis
        from ..agent_methods.agents.workflow_planner import WorkflowPlanner
        
        # Use interface conversation ID for feedback continuity, fallback to execution-specific
        feedback_conversation_id = interface_conversation_id or f"feedback_{execution_summary.execution_id}"
        print(f"🗣️ Using feedback conversation_id: {feedback_conversation_id}")
        
        planner = WorkflowPlanner(
            model="gpt-4o-mini",
            api_key=None,  # Use env var
            conversation_id=feedback_conversation_id
        )
        
        # Get the current tool catalog for analysis context
        try:
            feedback_tool_catalog = create_tool_catalog()
        except Exception as e:
            print(f"⚠️ Warning: Could not load tool catalog for feedback analysis: {e}")
            feedback_tool_catalog = {}
        
        # Send feedback as system input to planner with proper tool catalog
        response = await planner.generate_workflow(
            query=feedback_prompt,
            tool_catalog=feedback_tool_catalog,  # Use actual tool catalog for analysis context
            context={
                "timestamp": datetime.now().isoformat(),
                "execution_feedback": True,
                "execution_id": execution_summary.execution_id,
                "workflow_title": execution_summary.workflow_title
            }
        )
        
        # Log the analysis response
        if hasattr(response, 'reasoning') and response.reasoning:
            execution_logger.success(
                "WorkflowPlanner analysis completed",
                data={
                    "analysis_length": len(response.reasoning),
                    "analysis_preview": response.reasoning[:300] + "..." if len(response.reasoning) > 300 else response.reasoning,
                    "has_workflow_spec": workflow_spec is not None,
                    "feedback_conversation_id": feedback_conversation_id
                },
                execution_id=execution_summary.execution_id
            )
            
            # Store feedback for potential display to user
            if execution_summary.execution_id in active_executions:
                active_executions[execution_summary.execution_id]["planner_feedback"] = response.reasoning
        
        execution_logger.success("Execution feedback processing completed", execution_id=execution_summary.execution_id)
        
    except Exception as e:
        execution_logger.error(
            "Failed to send execution feedback", 
            data={"error": str(e), "error_type": type(e).__name__},
            execution_id=execution_summary.execution_id if execution_summary else None
        )
        # Don't fail the entire execution if feedback fails


def create_example_workflows():
    """Create multiple example workflows for the dropdown."""
    from ..test_workflows import create_workflow_examples
    return create_workflow_examples()


@app.get("/api/examples")
async def get_example_workflows():
    """Get list of available example workflows."""
    examples = create_example_workflows()
    
    # Return metadata only (title, description) for dropdown
    examples_metadata = {}
    for key, workflow in examples.items():
        examples_metadata[key] = {
            "title": workflow.title,
            "description": workflow.description,
            "node_count": len(workflow.nodes),
            "edge_count": len(workflow.edges)
        }
    
    return {"examples": examples_metadata}


@app.post("/api/example/{example_id}")
async def load_example_workflow(example_id: str):
    """Load a specific example workflow."""
    global current_workflow
    
    print(f"📋 Loading example workflow: {example_id}")
    
    examples = create_example_workflows()
    
    if example_id not in examples:
        available_examples = list(examples.keys())
        raise HTTPException(
            status_code=404, 
            detail=f"Example '{example_id}' not found. Available: {available_examples}"
        )
    
    current_workflow = examples[example_id]
    print(f"✅ Example workflow loaded: '{current_workflow.title}' with {len(current_workflow.nodes)} nodes, {len(current_workflow.edges)} edges")
    
    # Update planner context with loaded workflow
    if planner:
        planner.set_current_workflow(current_workflow)
        print("🧠 Set planner context to loaded example workflow")
    
    # Broadcast update
    print(f"📡 Broadcasting example workflow to {len(connections)} connections")
    await broadcast_workflow_update(current_workflow)
    
    workflow_data = current_workflow.model_dump()
    # Convert UUID to string for JSON serialization
    if 'id' in workflow_data:
        workflow_data['id'] = str(workflow_data['id'])
    print(f"📦 Returning example workflow data: {len(str(workflow_data))} characters")
    
    return {"workflow": workflow_data}


@app.delete("/api/executions/{execution_id}")
async def cancel_execution(execution_id: str):
    """Cancel/remove an execution."""
    if execution_id in active_executions:
        # Note: This doesn't actually cancel running tasks, just removes from tracking
        del active_executions[execution_id]
        await broadcast_execution_update(execution_id, "cancelled")
        return {"success": True, "message": "Execution removed"}
    else:
        raise HTTPException(status_code=404, detail="Execution not found")


# === Workflow Storage Endpoints ===

@app.post("/api/workflows/save", response_model=SaveWorkflowResponse)
async def save_current_workflow(request: SaveWorkflowRequest):
    """Save the current workflow to persistent storage."""
    global current_workflow, workflow_storage
    
    if not current_workflow:
        raise HTTPException(status_code=400, detail="No current workflow to save")
    
    if not workflow_storage:
        raise HTTPException(status_code=500, detail="Workflow storage not initialized")
    
    try:
        workflow_id = workflow_storage.save_workflow(
            workflow_spec=current_workflow,
            name=request.name,
            description=request.description,
            tags=request.tags
        )
        
        return SaveWorkflowResponse(
            success=True,
            workflow_id=workflow_id
        )
    except Exception as e:
        return SaveWorkflowResponse(
            success=False,
            error=str(e)
        )


@app.get("/api/workflows/saved")
async def get_saved_workflows():
    """Get list of all saved workflows."""
    global workflow_storage
    
    if not workflow_storage:
        raise HTTPException(status_code=500, detail="Workflow storage not initialized")
    
    try:
        workflows = workflow_storage.list_workflows()
        return {"workflows": workflows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving workflows: {str(e)}")


@app.post("/api/workflows/search")
async def search_saved_workflows(request: SearchWorkflowsRequest):
    """Search saved workflows by query string or tags."""
    global workflow_storage
    
    if not workflow_storage:
        raise HTTPException(status_code=500, detail="Workflow storage not initialized")
    
    try:
        if request.query:
            workflows = workflow_storage.search_workflows(request.query)
        elif request.tags:
            workflows = workflow_storage.list_workflows(tags=request.tags)
        else:
            workflows = workflow_storage.list_workflows()
        
        return {"workflows": workflows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching workflows: {str(e)}")


@app.post("/api/workflows/load/{workflow_id}")
async def load_saved_workflow(workflow_id: str):
    """Load a saved workflow by ID."""
    global current_workflow, workflow_storage
    
    if not workflow_storage:
        raise HTTPException(status_code=500, detail="Workflow storage not initialized")
    
    try:
        loaded_workflow = workflow_storage.load_workflow(workflow_id)
        
        if not loaded_workflow:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
        
        current_workflow = loaded_workflow
        print(f"✅ Loaded saved workflow: '{current_workflow.title}' (ID: {workflow_id[:8]})")
        
        # Update planner context with loaded workflow
        if planner:
            planner.set_current_workflow(current_workflow)
            print("🧠 Set planner context to loaded saved workflow")
        
        # Broadcast update
        await broadcast_workflow_update(current_workflow)
        
        workflow_data = current_workflow.model_dump()
        # Convert UUID to string for JSON serialization
        if 'id' in workflow_data:
            workflow_data['id'] = str(workflow_data['id'])
        
        return {"workflow": workflow_data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading workflow: {str(e)}")


@app.delete("/api/workflows/{workflow_id}")
async def delete_saved_workflow(workflow_id: str):
    """Delete a saved workflow."""
    global workflow_storage
    
    if not workflow_storage:
        raise HTTPException(status_code=500, detail="Workflow storage not initialized")
    
    try:
        success = workflow_storage.delete_workflow(workflow_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
        
        return {"success": True, "message": f"Workflow {workflow_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting workflow: {str(e)}")


@app.get("/api/workflows/{workflow_id}/yaml")
async def export_workflow_yaml(workflow_id: str):
    """Export a workflow as YAML."""
    global workflow_storage
    
    if not workflow_storage:
        raise HTTPException(status_code=500, detail="Workflow storage not initialized")
    
    try:
        yaml_content = workflow_storage.export_workflow_yaml(workflow_id)
        
        if not yaml_content:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
        
        return {"yaml": yaml_content}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting workflow: {str(e)}")


@app.get("/api/workflows/stats")
async def get_workflow_stats():
    """Get statistics about saved workflows."""
    global workflow_storage
    
    if not workflow_storage:
        raise HTTPException(status_code=500, detail="Workflow storage not initialized")
    
    try:
        stats = workflow_storage.get_workflow_stats()
        return {"stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@app.post("/api/search_collections", response_model=SearchCollectionsResponse)
async def search_collections_endpoint(request: SearchCollectionsRequest):
    """Search collections for matching prompts/inputs."""
    try:
        # Use the collection manager tool to search
        search_result = search_collections(
            query=request.query,
            tool_filter=request.tool_filter
        )
        
        if search_result.get("status") == "success":
            return SearchCollectionsResponse(
                success=True,
                results=search_result.get("results", [])
            )
        else:
            return SearchCollectionsResponse(
                success=False,
                error=search_result.get("message", "Search failed")
            )
            
    except Exception as e:
        return SearchCollectionsResponse(
            success=False,
            error=f"Search error: {str(e)}"
        )

@app.post("/api/save_to_collection", response_model=SaveToCollectionResponse)
async def save_to_collection_endpoint(request: SaveToCollectionRequest):
    """Save a record to a collection."""
    try:
        # Import here to avoid circular imports
        from iointel.src.agent_methods.data_models.prompt_collections import prompt_collection_manager
        
        # Try to find existing collection by name
        existing_collection = None
        for collection in prompt_collection_manager.list_collections():
            if collection.name == request.collection_name:
                existing_collection = collection
                break
        
        if existing_collection:
            # Add to existing collection
            existing_collection.add_record(request.record)
            prompt_collection_manager.save_collection(existing_collection)
            message = f"Added to existing collection '{request.collection_name}'"
        else:
            # Create new collection
            create_result = create_collection(
                name=request.collection_name,
                records=[request.record],
                description=f"Collection for {request.tool_type or 'prompts'}",
                tags=[request.tool_type or "prompt_tool", "web_interface"]
            )
            
            if create_result.get("status") == "success":
                message = f"Created new collection '{request.collection_name}'"
            else:
                return SaveToCollectionResponse(
                    success=False,
                    error=create_result.get("message", "Failed to create collection")
                )
        
        return SaveToCollectionResponse(
            success=True,
            message=message
        )
        
    except Exception as e:
        return SaveToCollectionResponse(
            success=False,
            error=f"Save error: {str(e)}"
        )

@app.get("/api/workflows/combined")
async def get_combined_workflows():
    """Get both example workflows and saved workflows for the dropdown."""
    global workflow_storage
    
    # Get centralized examples
    examples = create_example_workflows()
    examples_data = {}
    for key, workflow in examples.items():
        examples_data[key] = {
            "id": key,
            "title": workflow.title,
            "description": workflow.description,
            "source": "example",
            "node_count": len(workflow.nodes),
            "edge_count": len(workflow.edges),
            "complexity": _calculate_complexity_simple(workflow)
        }
    
    # Get saved workflows if storage is available
    saved_data = {}
    lineage_groups = {}
    if workflow_storage:
        try:
            saved_workflows = workflow_storage.list_workflows()
            
            # Group workflows by lineage (base_id)
            for workflow in saved_workflows:
                workflow_id = workflow["id"]
                
                # Extract base_id (first 8 chars before first hyphen)
                base_id = workflow_id.split('-')[0] if '-' in workflow_id else workflow_id[:8]
                
                workflow_data = {
                    "id": workflow_id,
                    "title": workflow.get("name", "Unnamed Workflow"),
                    "description": workflow.get("description", "No description"),
                    "source": "saved",
                    "node_count": workflow.get("node_count", 0),
                    "edge_count": workflow.get("edge_count", 0),
                    "complexity": workflow.get("complexity", "Unknown"),
                    "created_at": workflow.get("created_at"),
                    "tags": workflow.get("tags", []),
                    "rev": workflow.get("rev", 1),
                    "base_id": base_id
                }
                
                # Add to lineage groups
                if base_id not in lineage_groups:
                    lineage_groups[base_id] = {
                        "base_id": base_id,
                        "title": workflow_data["title"],
                        "revisions": [],
                        "latest_rev": 0,
                        "total_revisions": 0
                    }
                
                lineage_groups[base_id]["revisions"].append(workflow_data)
                lineage_groups[base_id]["latest_rev"] = max(lineage_groups[base_id]["latest_rev"], workflow_data["rev"])
                lineage_groups[base_id]["total_revisions"] = len(lineage_groups[base_id]["revisions"])
                
                # Keep individual workflows for backward compatibility
                saved_data[workflow_id] = workflow_data
                
            # Sort revisions within each group
            for group in lineage_groups.values():
                group["revisions"].sort(key=lambda x: x["rev"], reverse=True)  # Latest first
                
        except Exception as e:
            print(f"⚠️ Error loading saved workflows: {e}")
    
    return {
        "examples": examples_data,
        "saved": saved_data,
        "lineage_groups": lineage_groups
    }


def _calculate_complexity_simple(workflow: WorkflowSpec) -> str:
    """Simple complexity calculation for workflow display."""
    node_count = len(workflow.nodes)
    if node_count == 1:
        return "Simple"
    elif node_count <= 3:
        return "Basic"
    elif node_count <= 6:
        return "Intermediate"
    else:
        return "Advanced"


# Faux User Model - In production, replace with real user system
class FauxUser:
    def __init__(self, session_id: str):
        self.user_id = f"user_{session_id[:8]}"  # Faux user based on session
        self.session_id = session_id
        self.interface_conversation_id = f"interface_{session_id}"
        self.active_workflows = {}  # {stable_workflow_id: {current_rev, specs_history}}
    
    def get_or_create_workflow_conversation_id(self, workflow_context_name: str = "default") -> str:
        """Get or create a stable workflow conversation ID that persists across revisions."""
        if workflow_context_name not in self.active_workflows:
            stable_workflow_id = str(uuid.uuid4())[:8]  # Short stable ID
            self.active_workflows[workflow_context_name] = {
                "stable_id": stable_workflow_id,
                "current_rev": 0,
                "specs_history": []
            }
        
        stable_id = self.active_workflows[workflow_context_name]["stable_id"]
        return f"workflow_{self.user_id}_{stable_id}"
    
    def register_workflow_revision(self, workflow_spec: WorkflowSpec, workflow_context_name: str = "default"):
        """Register a new revision of the workflow."""
        if workflow_context_name not in self.active_workflows:
            self.get_or_create_workflow_conversation_id(workflow_context_name)
        
        self.active_workflows[workflow_context_name]["current_rev"] += 1
        self.active_workflows[workflow_context_name]["specs_history"].append({
            "rev": self.active_workflows[workflow_context_name]["current_rev"],
            "spec_id": str(workflow_spec.id),
            "title": workflow_spec.title
        })

# Global faux user registry - In production, replace with database
faux_users = {}

def get_or_create_faux_user(session_id: str) -> FauxUser:
    """Get or create a faux user for the session."""
    if session_id not in faux_users:
        faux_users[session_id] = FauxUser(session_id)
    return faux_users[session_id]

# Session management endpoints for conversation continuity
@app.post("/api/session/init")
async def initialize_session(request: Request):
    """Initialize a new workflow session with persistent conversation ID."""
    session_id = str(uuid.uuid4())
    request.session["workflow_session_id"] = session_id
    request.session["chat_mode"] = True  # Default to chat mode for new sessions
    
    # Initialize faux user
    faux_user = get_or_create_faux_user(session_id)
    
    return {
        "session_id": session_id,
        "chat_mode": True,
        "user_id": faux_user.user_id,
        "interface_conversation_id": faux_user.interface_conversation_id,
        "conversation_id": faux_user.interface_conversation_id  # For backwards compatibility
    }

@app.get("/api/session/status")
async def get_session_status(request: Request):
    """Get current session status and settings."""
    session_id = request.session.get("workflow_session_id")
    chat_mode = request.session.get("chat_mode", False)
    
    faux_user = None
    if session_id:
        faux_user = get_or_create_faux_user(session_id)
    
    return {
        "session_id": session_id,
        "chat_mode": chat_mode,
        "user_id": faux_user.user_id if faux_user else None,
        "interface_conversation_id": faux_user.interface_conversation_id if faux_user else None,
        "active_workflows": faux_user.active_workflows if faux_user else {},
        "conversation_id": faux_user.interface_conversation_id if faux_user and chat_mode else None,
        "has_session": session_id is not None
    }

@app.post("/api/session/chat-mode")
async def toggle_chat_mode(request: Request, data: dict):
    """Toggle chat mode on/off. When enabled, uses persistent conversation ID."""
    chat_mode = data.get("enabled", False)
    
    if chat_mode and not request.session.get("workflow_session_id"):
        # Initialize session if enabling chat mode without existing session
        session_id = str(uuid.uuid4())
        request.session["workflow_session_id"] = session_id
    
    request.session["chat_mode"] = chat_mode
    session_id = request.session.get("workflow_session_id")
    
    return {
        "chat_mode": chat_mode,
        "session_id": session_id,
        "conversation_id": f"workflow_session_{session_id}" if session_id and chat_mode else None
    }

@app.post("/api/session/reset")
async def reset_session(request: Request):
    """Reset the current session, creating a new conversation ID."""
    session_id = str(uuid.uuid4())
    chat_mode = request.session.get("chat_mode", True)  # Keep current chat mode preference
    
    request.session["workflow_session_id"] = session_id
    request.session["chat_mode"] = chat_mode
    
    return {
        "session_id": session_id,
        "chat_mode": chat_mode,
        "conversation_id": f"workflow_session_{session_id}" if chat_mode else None
    }

@app.get("/api/debug/user-model")
async def debug_user_model(request: Request):
    """Debug endpoint to show faux user model state."""
    session_id = request.session.get("workflow_session_id")
    if not session_id:
        return {"error": "No session found"}
    
    faux_user = get_or_create_faux_user(session_id)
    
    return {
        "user_id": faux_user.user_id,
        "session_id": faux_user.session_id,
        "interface_conversation_id": faux_user.interface_conversation_id,
        "active_workflows": faux_user.active_workflows,
        "global_faux_users_count": len(faux_users)
    }


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