"""
FastAPI server for serving WorkflowSpecs to the web interface.
"""

import os
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
from iointel.src.web.conversation_storage import get_conversation_storage
from iointel.src.utilities.constants import get_model_config
import uuid
import asyncio

# Import example tools to register them globally

# Import conditional gate tools to register them globally

# Import workflow storage
from .workflow_storage import WorkflowStorage
from .execution_feedback import (
    feedback_collector, 
    create_execution_feedback_prompt,
    WorkflowExecutionSummary
)
from ..utilities.io_logger import workflow_logger, execution_logger, system_logger
from iointel.src.agent_methods.data_models.execution_models import DataSourceResult, ExecutionStatus
from iointel.src.utilities.workflow_helpers import execute_workflow_with_metadata
from iointel.src.agent_methods.tools.collection_manager import search_collections, create_collection
from .test_analytics_api import test_analytics_router
from .workflow_rag_router import workflow_rag_router
from .unified_search_service import UnifiedSearchService


# Register executors for web interface
# REMOVED task registration - using typed execution instead
# @register_custom_task("data_source")
# @register_custom_task("tool")  # Backward compatibility
async def web_tool_executor(task_metadata, objective, agents, execution_metadata):
    """Tool executor for web interface with real-time updates, delegates to backend executor."""
    execution_id = execution_metadata.get("execution_id")
    tool_name = task_metadata.get("tool_name")
    source_name = task_metadata.get("source_name")
    task_name = source_name or tool_name
    task_type = "data_source" if source_name else "tool"
    workflow_logger.info(
        f"Executing {task_type}: {task_name}",
        data={"task_type": task_type, "task_name": task_name, "execution_id": execution_id}
    )
    # Broadcast task start if we have connections available
    if execution_id and 'connections' in globals() and len(connections) > 0:
        try:
            await broadcast_execution_update(
                execution_id, 
                "running", 
                results={"current_task": tool_name, "status": "started"}
            )
        except Exception as e:
            workflow_logger.warning(f"Failed to broadcast task start: {e}")
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
        
        # Delegate to appropriate backend executor based on task type
        # Check if this is a data_source task or a tool task
        source_name = task_metadata.get("source_name")
        if source_name:
            # This is a data_source task - use data source executor
            result: DataSourceResult = await execute_data_source_task(task_metadata, objective, agents, execution_metadata)
        else:
            # This is a tool task - use tool executor
            result: DataSourceResult = await execute_tool_task(task_metadata, objective, agents, execution_metadata)
        workflow_logger.success(f"{task_type.title()} '{task_name}' completed successfully")
        
        # Record node completion in feedback collector
        if execution_id:
            # For tools, the result is the direct tool output
            tool_usage_results = [{
                'tool_name': tool_name,
                'input': objective,  # The task objective was the input
                'result': result.result,
                'metadata': {'execution_type': 'direct_tool'}
            }]
            
            feedback_collector.record_node_completion(
                execution_id=execution_id,
                node_id=node_id,
                status=ExecutionStatus.SUCCESS if result.status.value == "completed" else ExecutionStatus.FAILED,
                result_preview=str(result.result)[:1000] if result.result else None,
                tool_usage=[tool_name],
                full_agent_output=None,  # Tools don't have agent output
                tool_usage_results=tool_usage_results,
                final_result=result
            )
        
        # Broadcast task completion
        if execution_id and 'connections' in globals() and len(connections) > 0:
            try:
                await broadcast_execution_update(
                    execution_id,
                    "running",
                    results={"current_task": tool_name, "status": "completed", "result": result.result}
                )
            except Exception as e:
                workflow_logger.warning(f"Failed to broadcast completion: {e}")
        return result.result if result.status.value == "completed" else None
    except Exception as e:
        error_msg = f"Tool '{tool_name}' failed: {str(e)}"
        workflow_logger.error(error_msg)
        
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
                workflow_logger.warning(f"Failed to broadcast tool error: {e}")
        raise


# REMOVED task registration - using typed execution instead
# @register_custom_task("agent")
async def web_agent_executor(task_metadata, objective, agents, execution_metadata):
    """Agent executor for web interface with real-time updates."""
    execution_id = execution_metadata.get("execution_id")
    
    workflow_logger.info(f"Executing agent task", data={"execution_id": execution_id})
    
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
            workflow_logger.warning(f"Failed to broadcast agent start: {e}")
    
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
        workflow_logger.success(f"Agent task completed", data={"result_preview": str(result)[:200]})
        
        # Record node completion in feedback collector
        if execution_id:
            # Extract tool usage from typed result
            tool_usage = []
            if result.agent_response and result.agent_response.tool_usage_results:
                for tool in result.agent_response.tool_usage_results:
                    tool_usage.append(tool.tool_name)
            
            # Extract data from typed result for feedback
            agent_output = None
            result_preview = None
            
            if result.agent_response:
                # Get the agent output from typed response
                agent_output = result.agent_response.result
                # Convert to string if needed
                if not isinstance(agent_output, str):
                    agent_output = str(agent_output) if agent_output else ''
                result_preview = agent_output[:1000] if agent_output else None
            
            # Pass typed models directly to feedback collector
            feedback_collector.record_node_completion(
                execution_id=execution_id,
                node_id=node_id,
                status=ExecutionStatus.SUCCESS,
                result_preview=result_preview,
                tool_usage=tool_usage,
                full_agent_output=agent_output,
                tool_usage_results=result.agent_response.tool_usage_results if result.agent_response else [],
                final_result=result  # Pass the typed AgentExecutionResult directly
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
                workflow_logger.warning(f"Failed to broadcast agent completion: {e}")
        
        return result
    except Exception as e:
        error_msg = f"Agent task failed: {str(e)}"
        workflow_logger.error(error_msg)
        
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
                workflow_logger.warning(f"Failed to broadcast agent error: {e}")
        raise


# REMOVED task registration - using typed execution instead
# @register_custom_task("decision")
async def web_decision_executor(task_metadata, objective, agents, execution_metadata):
    """Decision executor for web interface - delegates to tool executor or agent."""
    execution_id = execution_metadata.get("execution_id")
    tool_name = task_metadata.get('tool_name')
    
    workflow_logger.info(
        f"Executing decision task",
        data={"tool_name": tool_name or 'agent-based', "execution_id": execution_id}
    )
    
    # If no tool_name specified, treat as agent-based decision
    if not tool_name:
        workflow_logger.info("No tool_name specified, treating as agent-based decision")
        
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
                    workflow_logger.warning(f"Failed to broadcast decision completion: {e}")
            
            return result
            
        except Exception as e:
            error_msg = f"Agent-based decision failed: {str(e)}"
            workflow_logger.error(error_msg)
            if execution_id and len(connections) > 0:
                try:
                    await broadcast_execution_update(execution_id, "failed", error=error_msg)
                except Exception as e:
                    workflow_logger.warning(f"Failed to broadcast decision error: {e}")
            raise
    
    # If tool_name is specified, use tool-based decision
    return await web_tool_executor(task_metadata, objective, agents, execution_metadata)


# REMOVED task registration - using typed execution instead
# @register_custom_task("workflow_call")
async def web_workflow_call_executor(task_metadata, objective, agents, execution_metadata):
    """Workflow call executor for web interface."""
    execution_id = execution_metadata.get("execution_id")
    workflow_id = task_metadata.get("workflow_id", "unknown")
    
    workflow_logger.info(
        f"Executing workflow call",
        data={"workflow_id": workflow_id, "execution_id": execution_id}
    )
    
    # Broadcast task start
    if execution_id and len(connections) > 0:
        try:
            await broadcast_execution_update(
                execution_id, 
                "running", 
                results={"current_task": f"workflow_call:{workflow_id}", "status": "started"}
            )
        except Exception as e:
            workflow_logger.warning(f"Failed to broadcast workflow_call start: {e}")
    
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
                workflow_logger.warning(f"Failed to broadcast workflow_call completion: {e}")
        
        return result
    except Exception as e:
        error_msg = f"Workflow call '{workflow_id}' failed: {str(e)}"
        workflow_logger.error(error_msg)
        if execution_id and len(connections) > 0:
            try:
                await broadcast_execution_update(execution_id, "failed", error=error_msg)
            except Exception as e:
                workflow_logger.warning(f"Failed to broadcast workflow_call error: {e}")
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
unified_search_service: Optional[UnifiedSearchService] = None
tool_catalog: Dict[str, Any] = {}
active_executions: Dict[str, Dict[str, Any]] = {}  # execution_id -> execution_info
workflow_storage: Optional[WorkflowStorage] = None

# WebSocket connections for real-time updates
connections: List[WebSocket] = []

# Debug prompt history
# Import prompt logging from io_logger
from ..utilities.io_logger import log_prompt, get_prompt_history, clear_prompt_history


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
        # Import here to avoid circular imports
        from iointel.src.agent_methods.data_models.execution_models import (
            AgentExecutionResult, AgentRunResponse, DataSourceResult
        )
        
        # Handle AgentExecutionResult specifically (from typed execution)
        if isinstance(value, AgentExecutionResult):
            # Extract the inner agent response for frontend compatibility
            if value.agent_response and isinstance(value.agent_response, AgentRunResponse):
                return {
                    "result": value.agent_response.result,
                    "tool_usage_results": [serialize_value(tur) for tur in value.agent_response.tool_usage_results],
                    "conversation_id": value.agent_response.conversation_id,
                    "status": value.status.value,
                    "execution_time": value.execution_time,
                    "node_id": value.node_id,
                    "type": "AgentExecutionResult"
                }
            else:
                # No agent response, return minimal structure
                return {
                    "result": None,
                    "tool_usage_results": [],
                    "status": value.status.value,
                    "error": value.error,
                    "type": "AgentExecutionResult"
                }
        
        # Handle DataSourceResult specifically
        elif isinstance(value, DataSourceResult):
            return {
                "result": value.result,
                "status": value.status.value,
                "source": value.source,
                "type": "DataSourceResult"
            }
        
        # Handle other BaseModel objects (Pydantic models like ToolUsageResult, GateResult)
        elif hasattr(value, 'model_dump'):
            try:
                # Use Pydantic's model_dump for clean serialization
                return value.model_dump()
            except Exception:
                # Fallback to dict conversion
                return dict(value)
        
        # Handle AgentRunResult objects specifically (legacy, shouldn't exist anymore)
        elif hasattr(value, '__class__') and 'AgentRunResult' in str(value.__class__):
            return {
                "result": getattr(value, 'output', None) or getattr(value, 'result', None),
                "tool_usage_results": [serialize_value(tur) for tur in getattr(value, 'tool_usage_results', [])],
                "conversation_id": getattr(value, 'conversation_id', None),
                "type": "AgentRunResult"
            }
        
        # Handle datetime objects
        elif isinstance(value, datetime):
            return value.isoformat()
        
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
    
    # Debug logging for serialization
    if 'results' in serialized:
        workflow_logger.info(
            "🔍 Serialized results structure",
            data={
                "top_level_keys": list(serialized.keys()),
                "results_keys": list(serialized.get('results', {}).keys()) if isinstance(serialized.get('results'), dict) else "not a dict",
                "sample_node": next(iter(serialized.get('results', {}).keys())) if serialized.get('results') else None
            }
        )
        # Log a sample node result structure
        if serialized.get('results') and isinstance(serialized.get('results'), dict):
            sample_key = next(iter(serialized['results'].keys()))
            sample_value = serialized['results'][sample_key]
            workflow_logger.info(
                f"🔍 Sample node result structure for {sample_key}",
                data={
                    "type": sample_value.get('type') if isinstance(sample_value, dict) else type(sample_value).__name__,
                    "has_result": 'result' in sample_value if isinstance(sample_value, dict) else False,
                    "has_tool_usage": 'tool_usage_results' in sample_value if isinstance(sample_value, dict) else False,
                    "keys": list(sample_value.keys()) if isinstance(sample_value, dict) else None
                }
            )
    
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
                workflow_logger.error(f"Failed to send to connection: {e}")
                disconnected.append(connection)
        
        # Only log meaningful updates
        if message['type'] == 'execution_update' and message.get('status') in ['started', 'completed', 'failed']:
            workflow_logger.info(
                f"Broadcast {message['type']} ({message['status']}) to {sent_count} clients"
            )
        elif message['type'] == 'workflow_update':
            workflow_logger.info(f"Broadcast workflow update to {sent_count} clients")
        
        # Remove disconnected clients
        for conn in disconnected:
            connections.remove(conn)
            workflow_logger.info("Removed disconnected client")




@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    global planner, tool_catalog, workflow_storage, unified_search_service
    
    system_logger.info("Starting WorkflowPlanner web server...")
    
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
        tool_catalog = create_tool_catalog(filter_broken=True, verbose_format=False, use_working_filter=True)
        system_logger.success(
            "Tools and catalog loaded successfully",
            data={
                "available_tools": len(available_tools),
                "catalog_tools": len(tool_catalog),
            }
        )
    except Exception as e:
        system_logger.warning("Could not load tools from environment", data={"error": str(e), "fallback": "using empty catalog"})
        tool_catalog = {}
    
    # Initialize WorkflowPlanner with managed conversation
    try:
        conversation_storage = get_conversation_storage()
        startup_conversation_id = conversation_storage.get_active_web_conversation()
        
        # Use shared model configuration for main planner
        main_model = os.getenv("WORKFLOW_PLANNER_MODEL", "gpt-4o")
        model_config = get_model_config(model=main_model)
        system_logger.info(
            f"Using main model config",
            data={"model": model_config['model'], "base_url": model_config['base_url']}
        )
        
        planner = WorkflowPlanner(
            model=model_config["model"],
            api_key=model_config["api_key"],
            base_url=model_config["base_url"],
            memory=memory,
            conversation_id=startup_conversation_id,
            debug=False
        )
        system_logger.success("WorkflowPlanner initialized", data={"memory_enabled": True, "conversation_id": startup_conversation_id})
    except Exception as e:
        system_logger.error("WorkflowPlanner initialization failed", data={"error": str(e), "error_type": type(e).__name__})
        raise
    
    # Initialize WorkflowStorage
    try:
        workflow_storage = WorkflowStorage()
        system_logger.success("WorkflowStorage initialized")
    except Exception as e:
        system_logger.error(f"WorkflowStorage initialization failed: {e}")
        raise
    
    # Initialize UnifiedSearchService
    try:
        # Check environment variable for search mode
        use_fast_search = os.getenv("FAST_SEARCH_MODE", "true").lower() == "true"
        
        unified_search_service = UnifiedSearchService(
            storage_dir="saved_workflows",
            test_repository_dir="smart_test_repository",
            fast_mode=use_fast_search
        )
        
        search_mode = "fast hash encoding" if use_fast_search else "real semantic vectors"
        system_logger.success(
            f"UnifiedSearchService initialized with {search_mode}",
            data={"search_mode": search_mode, "tip": "Set FAST_SEARCH_MODE=false for real semantic vectors"}
        )
    except Exception as e:
        system_logger.warning(f"UnifiedSearchService initialization failed: {e}")
        # Don't raise - fallback to simple search
    
    # REMOVED: Task executor registration - no longer needed with typed execution
    # DAGExecutor with use_typed_execution=True handles all node types through typed_executors.py
    # This eliminates the need for chainables imports and duplicate execution paths
    system_logger.success(
        "Using typed execution system - no custom task executors needed",
        data={
            "execution_mode": "typed_execution",
            "handled_by": "DAGExecutor + typed_executors.py"
        }
    )
    
    system_logger.success("WorkflowPlanner web server initialized successfully!")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    connections.append(websocket)
    system_logger.info("New WebSocket connection established", data={"total_connections": len(connections)})
    
    # Don't automatically send current workflow - let client request it
    # This prevents session bleed between different users/tabs
    workflow_logger.info("New WebSocket connection established - waiting for client requests")
    
    try:
        while True:
            # Keep connection alive - handle any incoming messages or just wait
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                # Echo back any received messages for debugging
                workflow_logger.debug(f"Received WebSocket message: {message}")
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


# log_prompt function now imported from io_logger


@app.get("/api/prompts")
async def get_prompt_history_api():
    """Get the prompt history for debugging."""
    return {"prompts": get_prompt_history()}


@app.post("/api/prompts/clear")
async def clear_prompt_history_api():
    """Clear the prompt history."""
    count = clear_prompt_history()
    return {"success": True, "message": f"Cleared {count} prompts"}


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
            "execution_id": execution_summary.execution_id,
            "workflow_title": execution_summary.workflow_title,
            "status": execution_summary.status.value,
            "total_duration_seconds": execution_summary.total_duration_seconds,
            "nodes_executed": [
                {
                    "node_id": node.node_id,
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
    
    workflow_logger.info(
        "Generate workflow request",
        data={
            "query": workflow_request.query,
            "refine": workflow_request.refine
        }
    )
    
    # Get or create faux user for session management
    session_id = request.session.get("workflow_session_id")
    faux_user = get_or_create_faux_user(session_id) if session_id else None
    
    if not planner:
        workflow_logger.error("WorkflowPlanner not initialized")
        raise HTTPException(status_code=500, detail="WorkflowPlanner not initialized")
    
    try:
        workflow_logger.info(
            "Generating workflow",
            data={"tool_count": len(tool_catalog)}
        )
        
        # Set current workflow context so planner can reference it
        if current_workflow:
            from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec
            if isinstance(current_workflow, WorkflowSpec):
                workflow_logger.info(
                    "Building upon existing workflow",
                    data={
                        "revision": current_workflow.rev,
                        "title": current_workflow.title
                    }
                )
                planner.set_current_workflow(current_workflow)
            else:
                workflow_logger.info(f"Building upon existing chat context: '{getattr(current_workflow, 'title', 'Chat')}'")
                # Don't set current workflow for chat-only responses
            
        # Generate workflow (could be new workflow or chat-only response)
        # Use managed conversation storage for better conversation tracking
        conversation_storage = get_conversation_storage()
        conversation_id = conversation_storage.get_active_web_conversation()
        
        # Update conversation usage
        conversation_storage.update_conversation_usage(conversation_id, workflow_delta=1)
        
        # Workflow planner now logs the full prompt internally
        
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
                workflow_logger.info(
                    "Chat-only response detected",
                    data={"reasoning_preview": result.reasoning[:100] + "..." if len(result.reasoning) > 100 else result.reasoning}
                )
                
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
                workflow_logger.info(f"Registered workflow revision with user {faux_user.user_id}")
            
            # Debug: Log the actual workflow structure
            workflow_logger.debug(
                "Generated workflow details",
                data={
                    "title": current_workflow.title,
                    "description": current_workflow.description,
                    "reasoning": current_workflow.reasoning[:200] + "..." if len(current_workflow.reasoning) > 200 else current_workflow.reasoning,
                    "id": str(current_workflow.id),
                    "rev": current_workflow.rev,
                    "node_count": len(current_workflow.nodes),
                    "edge_count": len(current_workflow.edges),
                    "nodes": [
                        {
                            "id": node.id,
                            "type": node.type,
                            "label": node.label,
                            "has_sla": hasattr(node, 'sla') and node.sla is not None
                        }
                        for node in current_workflow.nodes
                    ]
                }
            )

            # Build a pretty-printed workflow spec for the agent/planner, including SLA
            dag_pretty = f"\n---\nWORKFLOW DAG (full spec, including SLA):\nTitle: {current_workflow.title}\nDescription: {current_workflow.description}\nID: {current_workflow.id}\nRev: {current_workflow.rev}\nNodes ({len(current_workflow.nodes)}):\n"
            for i, node in enumerate(current_workflow.nodes):
                dag_pretty += f"  {i+1}. {node.id} ({node.type}): {node.label}\n"
                dag_pretty += f"     Config: {node.data.config}\n"
                dag_pretty += f"     Ins: {node.data.ins}\n"
                dag_pretty += f"     Outs: {node.data.outs}\n"
                if hasattr(node, 'sla') and node.sla is not None:
                    dag_pretty += f"     SLA: {node.sla.model_dump_json(indent=2) if hasattr(node.sla, 'model_dump_json') else node.sla}\n"
            dag_pretty += f"Edges ({len(current_workflow.edges)}):\n"
            for i, edge in enumerate(current_workflow.edges):
                dag_pretty += f"  {i+1}. {edge.source} -> {edge.target}\n"
                dag_pretty += f"     Condition: {edge.data.condition}\n"
                dag_pretty += f"     Handles: {edge.sourceHandle} -> {edge.targetHandle}\n"
            dag_pretty += "---\n"

            # Broadcast update to connected clients
            workflow_logger.info(f"Broadcasting workflow update to {len(connections)} connections")
            await broadcast_workflow_update(current_workflow)
            
            workflow_data = current_workflow.model_dump()
            # Convert UUID to string for JSON serialization
            if 'id' in workflow_data:
                workflow_data['id'] = str(workflow_data['id'])
            workflow_logger.debug(f"Returning workflow data: {len(str(workflow_data))} characters")
            
            return WorkflowResponse(
                success=True,
                workflow=workflow_data,
                agent_response=result.reasoning  # Only return the planner's reasoning, not the raw DAG
            )
            
        else:
            # Check if this was a WorkflowSpecLLM that we already handled above
            if isinstance(result, WorkflowSpecLLM) and (result.nodes is None or result.edges is None):
                # This case should have been handled above, but just in case
                workflow_logger.warning("Chat-only response reached else block")
                return WorkflowResponse(
                    success=True,
                    workflow=None,
                    agent_response=result.reasoning
                )
            else:
                # This shouldn't happen with current implementation, but handle gracefully
                workflow_logger.error("Unexpected result format from planner.generate_workflow")
                return WorkflowResponse(
                    success=False,
                    error="Unexpected result format from workflow planner"
                )
        
    except Exception as e:
        workflow_logger.error(
            f"Error generating workflow: {type(e).__name__}: {e}",
            data={"error_type": type(e).__name__, "traceback": traceback.format_exc()}
        )
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
        
        # No need to convert to Workflow - DAGExecutor works directly with WorkflowSpec
        print(f"📋 Executing workflow with {len(workflow_spec.nodes)} nodes using DAGExecutor")
        
        # Start execution feedback tracking
        feedback_collector.start_execution_tracking(
            execution_id=execution_id,
            workflow_spec=workflow_spec,
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
        
        # Use the standardized workflow execution helper
        print(f"🔍 User inputs will be handled by execute_workflow_with_metadata: {user_inputs}")
        
        # Execute using the standard helper that matches test execution
        result = await execute_workflow_with_metadata(
            workflow_spec=workflow_spec,
            execution_id=execution_id,
            user_inputs=user_inputs,
            form_id=form_id,
            conversation_id=conversation_id,
            feedback_collector=feedback_collector,
            client_mode=True,
            debug=True
        )
        
        # Extract results in the expected format
        dag_results = result["results"]
        
        # Convert results to expected format
        results = {
            'results': dag_results,
            'status': result.get('status', 'completed')
        }
        
        # Get execution stats from result (not summary)
        execution_stats = result.get('execution_stats')
        
        # Create execution summary for feedback if we have stats
        execution_summary = None
        if execution_stats:
            # Create a summary-like object from stats for compatibility
            from iointel.src.web.execution_feedback import WorkflowExecutionSummary, NodeExecutionTracking
            from iointel.src.agent_methods.data_models.execution_models import (
                ExecutionStatus, 
                AgentExecutionResult,
                DataSourceResult
            )
            from datetime import datetime
            
            # Build list of executed nodes from results (nodes that actually ran)
            executed_node_results = []
            
            # Create lookups from workflow_spec
            node_type_lookup = {node.id: node.type for node in workflow_spec.nodes}
            node_label_lookup = {node.id: node.label for node in workflow_spec.nodes}
            
            for node_id in dag_results.keys():
                if node_id not in execution_stats.get('skipped_node_ids', []):
                    node_result = dag_results.get(node_id)
                    node_type = node_type_lookup.get(node_id, "agent")
                    node_label = node_label_lookup.get(node_id, node_id)
                    
                    # Extract result preview based on type
                    result_preview = None
                    tool_usage = []
                    
                    if isinstance(node_result, AgentExecutionResult):
                        if node_result.agent_response and node_result.agent_response.result:
                            result_preview = str(node_result.agent_response.result)[:200]
                        if node_result.agent_response and node_result.agent_response.tool_usage_results:
                            tool_usage = [t.tool_name for t in node_result.agent_response.tool_usage_results]
                    elif isinstance(node_result, dict) and "result" in node_result:
                        result_preview = str(node_result["result"])[:200]
                    else:
                        result_preview = str(node_result)[:200]
                    
                    executed_node_results.append(NodeExecutionTracking(
                        node_id=node_id,
                        node_type=node_type,
                        node_label=node_label,
                        status=ExecutionStatus.COMPLETED,
                        started_at=datetime.now().isoformat(),
                        finished_at=datetime.now().isoformat(),
                        duration_seconds=0,  # We don't have individual node timings from stats
                        result_preview=result_preview,
                        error_message=None,
                        tool_usage=tool_usage
                    ))
            
            execution_summary = WorkflowExecutionSummary(
                execution_id=execution_id,
                workflow_id=str(workflow_spec.id),
                workflow_title=workflow_spec.title,
                status=ExecutionStatus.COMPLETED if result.get('success') else ExecutionStatus.FAILED,
                started_at=datetime.now().isoformat(),
                finished_at=datetime.now().isoformat(),
                total_duration_seconds=execution_stats.get('total_duration', 0),
                nodes_executed=executed_node_results,
                nodes_skipped=execution_stats.get('skipped_node_ids', []),
                user_inputs=user_inputs or {},
                final_outputs=dag_results,
                workflow_spec=workflow_spec,
                error_summary=None,
                performance_metrics={
                    'node_durations': execution_stats.get('node_durations', {}),
                    'execution_order': execution_stats.get('execution_order', []),
                    'total_nodes': execution_stats.get('total_nodes', 0),
                    'executed_nodes_count': execution_stats.get('executed_nodes', 0),
                    'skipped_nodes_count': execution_stats.get('skipped_nodes', 0),
                    'failed_nodes': execution_stats.get('failed_nodes', 0)
                }
            )
        elif feedback_collector:
            # Fallback: complete execution tracking if not done
            final_outputs = dag_results if dag_results else {}
            execution_summary = feedback_collector.complete_execution(
                execution_id=execution_id,
                final_outputs=final_outputs
            )
        
        # Update execution info with serialized results
        serialized_results = serialize_execution_results(results)
        
        # Debug logging
        if serialized_results:
            workflow_logger.info(
                "Serialized results for broadcast",
                data={
                    "execution_id": execution_id[:8],
                    "num_task_results": len(serialized_results.get('results', {})),
                    "result_keys": list(serialized_results.get('results', {}).keys()) if serialized_results.get('results') else []
                }
            )
        
        active_executions[execution_id].update({
            "status": "completed",
            "end_time": datetime.now().isoformat(),
            "results": serialized_results,
            "error": None,
            "execution_summary": execution_summary
        })
        
        # Log execution completion
        log_data = {
            "execution_id": execution_id,
            "workflow_title": workflow_spec.title,
            "total_results": len(results.get('results', {}))
        }
        
        # Add execution summary data if available
        if execution_summary:
            log_data.update({
                "execution_time": execution_summary.total_duration_seconds,
                "nodes_executed": len(execution_summary.nodes_executed) if execution_summary.nodes_executed else 0,
                "nodes_skipped": len(execution_summary.nodes_skipped) if execution_summary.nodes_skipped else 0,
                "status": execution_summary.status.value if hasattr(execution_summary.status, 'value') else str(execution_summary.status)
            })
        elif execution_stats:
            # Use stats directly if no summary
            log_data.update({
                "execution_time": execution_stats.get('total_duration', 0),
                "nodes_executed": execution_stats.get('executed_nodes', 0),
                "nodes_skipped": execution_stats.get('skipped_nodes', 0),
                "status": "completed" if result.get('success') else "failed"
            })
        
        execution_logger.success(
            "Workflow execution completed",
            data=log_data,
            execution_id=execution_id
        )
        
        # Generate and send feedback to WorkflowPlanner using interface conversation ID
        interface_conversation_id = None
        if chat_mode and session_id:
            faux_user = get_or_create_faux_user(session_id)
            interface_conversation_id = faux_user.interface_conversation_id
        
        # Only send feedback if we have an execution summary
        if execution_summary:
            await send_execution_feedback_to_planner(execution_summary, interface_conversation_id, workflow_spec)
        
        # Broadcast completion with serialized results
        await broadcast_execution_update(
            execution_id, 
            "completed", 
            results=serialized_results
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
        
        # Complete execution feedback tracking with error if collector exists
        execution_summary = None
        if feedback_collector:
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
        # Extract session info for feedback continuity
        session_info = session_info or {}
        session_id = session_info.get("workflow_session_id")
        chat_mode = session_info.get("chat_mode", False)
        
        interface_conversation_id = None
        if chat_mode and session_id:
            faux_user = get_or_create_faux_user(session_id)
            interface_conversation_id = faux_user.interface_conversation_id
        
        # Only send feedback if we have an execution summary
        if execution_summary:
            await send_execution_feedback_to_planner(execution_summary, interface_conversation_id, workflow_spec)
        
        # Broadcast failure
        await broadcast_execution_update(execution_id, "failed", error=error_msg)



async def send_execution_feedback_to_planner(execution_summary: WorkflowExecutionSummary, interface_conversation_id: Optional[str] = None, workflow_spec: Optional[WorkflowSpec] = None):
    """Send execution results back to WorkflowPlanner for analysis and suggestions."""
    try:
        print(f"🔍 [FEEDBACK] Starting feedback to planner for execution: {execution_summary.execution_id}")
        print(f"🔍 [FEEDBACK] Interface conversation ID: {interface_conversation_id}")
        print(f"🔍 [FEEDBACK] Has workflow spec: {workflow_spec is not None}")
        
        # Import logger
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
        
        # CRITICAL: Use the EXACT SAME conversation ID as the main planner for continuity
        # This ensures the feedback planner sees its own workflow generation in conversation history
        conversation_storage = get_conversation_storage()
        feedback_conversation_id = conversation_storage.get_active_web_conversation()
        print(f"🗣️ Using SAME conversation_id as main planner for feedback: {feedback_conversation_id}")
        print(f"🔗 This ensures feedback planner sees its own workflow generation context")
        
        # Use shared model configuration - can be overridden via WORKFLOW_PLANNER_MODEL env var
        feedback_model = os.getenv("WORKFLOW_PLANNER_MODEL", "gpt-4o")  # Use lightweight model for feedback
        model_config = get_model_config(model=feedback_model)
        print(f"🤖 [FEEDBACK] Using model config: {model_config['model']} @ {model_config['base_url']}")
        
        planner = WorkflowPlanner(
            model=model_config["model"],
            api_key=model_config["api_key"],
            base_url=model_config["base_url"],
            conversation_id=feedback_conversation_id
        )
        
        # Get the current tool catalog for analysis context
        try:
            feedback_tool_catalog = create_tool_catalog(filter_broken=True, verbose_format=False, use_working_filter=True)
        except Exception as e:
            print(f"⚠️ Warning: Could not load tool catalog for feedback analysis: {e}")
            feedback_tool_catalog = {}
        
        # Send feedback as system input to planner with proper tool catalog
        print(f"🔍 [FEEDBACK] Sending feedback to planner.generate_workflow")
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
                
                # CRITICAL: Send the feedback as a chat message via WebSocket
                # This uses the same pattern as chat-only responses
                await broadcast_message({
                    "type": "chat_message",
                    "role": "assistant",
                    "content": response.reasoning,
                    "metadata": {
                        "is_execution_feedback": True,
                        "execution_id": execution_summary.execution_id
                    },
                    "timestamp": datetime.now().isoformat()
                })
                print(f"📡 [FEEDBACK] Broadcast planner feedback as chat message")
        else:
            print(f"⚠️ [FEEDBACK] No reasoning in response or response.reasoning is empty")
            if execution_summary.execution_id in active_executions:
                print(f"⚠️ [FEEDBACK] Not storing planner_feedback (would be None/empty)")
        
        execution_logger.success("Execution feedback processing completed", execution_id=execution_summary.execution_id)
        print(f"✅ [FEEDBACK] Successfully sent execution feedback to planner for {execution_summary.execution_id}")
        
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
@app.post("/api/executions/{execution_id}/cancel")
async def cancel_execution_endpoint(execution_id: str):
    """Cancel a running execution."""
    return await cancel_execution(execution_id)

async def cancel_execution(execution_id: str):
    """Cancel/remove an execution."""
    if execution_id in active_executions:
        # Note: This doesn't actually cancel running tasks, just removes from tracking
        print(f"🛑 Cancelling execution: {execution_id}")
        del active_executions[execution_id]
        await broadcast_execution_update(execution_id, "cancelled")
        return {"success": True, "message": "Execution cancelled"}
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


# Include test analytics router
app.include_router(test_analytics_router)

# Include workflow RAG router
app.include_router(workflow_rag_router)

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
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }
            .service { background: #f8f9fa; padding: 20px; margin: 10px 0; border-radius: 8px; }
            .service h3 { margin-top: 0; color: #2563eb; }
            a { color: #2563eb; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <h1>🚀 WorkflowPlanner Web Interface</h1>
        <p>All services are running in a unified server. Choose what you need:</p>
        
        <div class="service">
            <h3>📊 Test Analytics Panel</h3>
            <p>Search through 49 tests with RAG, view coverage metrics, and analyze test quality.</p>
            <p><a href="/test-analytics">Open Test Analytics Panel</a></p>
        </div>
        
        <div class="service">
            <h3>🔍 Workflow RAG Search</h3>
            <p>Semantic search over saved workflows using vector similarity.</p>
            <p><a href="/api/workflow-rag/">RAG Service Info</a> | <a href="/docs#/workflow-rag">API Docs</a></p>
        </div>
        
        <div class="service">
            <h3>🧪 Main Workflow Interface</h3>
            <p>The main workflow planning and execution interface.</p>
            <p><em>Static files not yet created. Run setup script to enable.</em></p>
        </div>
        
        <div class="service">
            <h3>📚 API Documentation</h3>
            <p>Complete API documentation for all services.</p>
            <p><a href="/docs">Open API Docs</a></p>
        </div>
        
        <p style="margin-top: 30px; color: #6b7280;">
            🎯 All services are now integrated! No need to run separate servers.
        </p>
    </body>
    </html>
    """)


@app.get("/test-analytics", response_class=HTMLResponse)
async def test_analytics_panel():
    """Serve the test analytics panel."""
    analytics_file = static_dir / "test_analytics_panel.html"
    if analytics_file.exists():
        return FileResponse(analytics_file)
    else:
        raise HTTPException(status_code=404, detail="Test analytics panel not found")


@app.get("/search/tools")
async def search_tools(query: str, top_k: int = 5):
    """Search available tools using unified semantic search."""
    try:
        # Use unified search service if available
        if unified_search_service:
            response = unified_search_service.search(
                query=query,
                search_types=["tools"],
                top_k=top_k
            )
            
            # Extract just the tool results
            tool_results = [r for r in response.results if r.result_type == "tool"]
            
            return {
                "query": query,
                "results": [r.dict() for r in tool_results],
                "total_found": len(tool_results)
            }
        
        # Fallback to simple text search
        print("⚠️ UnifiedSearchService not available, using simple text search")
        tool_catalog = create_tool_catalog(filter_broken=True, verbose_format=False, use_working_filter=True)
        
        # Simple text search through tools
        results = []
        query_lower = query.lower()
        
        for tool_name, tool_info in tool_catalog.items():
            score = 0.0
            
            # Check name match
            if query_lower in tool_name.lower():
                score += 10.0
            
            # Check description match
            description = tool_info.get("description", "")
            if query_lower in description.lower():
                score += 5.0
            
            # Check category match
            category = tool_info.get("category", "")
            if query_lower in category.lower():
                score += 3.0
            
            if score > 0:
                results.append({
                    "result_type": "tool",
                    "title": tool_name,
                    "description": description,
                    "similarity_score": min(score / 10.0, 1.0),  # Normalize to 0-1
                    "metadata": {
                        "parameter_count": len(tool_info.get("parameters", {})),
                        "category": category,
                        "required_parameters": tool_info.get("required_parameters", [])
                    }
                })
        
        # Sort by score and limit
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        results = results[:top_k]
        
        return {
            "query": query,
            "results": results,
            "total_found": len(results)
        }
        
    except Exception as e:
        print(f"Error searching tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search/workflows")
async def search_workflows_unified(query: str, top_k: int = 5):
    """Search saved workflows using unified semantic search."""
    try:
        # Use unified search service if available
        if unified_search_service:
            response = unified_search_service.search(
                query=query,
                search_types=["workflows"],
                top_k=top_k
            )
            
            # Extract workflow results and format for frontend
            results = []
            for r in response.results:
                if r.result_type == "workflow":
                    results.append({
                        "result_type": "workflow",
                        "title": r.title,
                        "description": r.description,
                        "similarity_score": r.similarity_score,
                        "workflow_spec": r.data,  # Full spec data
                        "metadata": r.metadata
                    })
            
            return {
                "query": query,
                "results": results,
                "total_found": len(results)
            }
        
        # Fallback to workflow RAG service
        from .workflow_rag_router import get_rag_service
        
        service = get_rag_service()
        search_response = service.search_workflows(query=query, top_k=top_k)
        
        # Convert to the format expected by the frontend
        results = []
        for result in search_response.results:
            workflow = result.workflow_spec
            results.append({
                "result_type": "workflow",
                "title": workflow.title,
                "description": workflow.description,
                "similarity_score": result.similarity_score,
                "workflow_spec": workflow.model_dump(),  # Include full spec for loading
                "metadata": {
                    "node_count": len(workflow.nodes),
                    "edge_count": len(workflow.edges),
                    "id": workflow.id
                }
            })
        
        return {
            "query": query,
            "results": results,
            "total_found": search_response.total_found
        }
        
    except Exception as e:
        print(f"Error searching workflows: {e}")
        # If search services are not available, return empty results
        return {
            "query": query,
            "results": [],
            "total_found": 0,
            "error": "Workflow search service initializing..."
        }


@app.get("/api/test-analytics/search")
async def search_test_analytics(query: str, top_k: int = 10):
    """Search test cases using unified semantic search."""
    try:
        # Use unified search service if available
        if unified_search_service:
            response = unified_search_service.search(
                query=query,
                search_types=["tests"],
                top_k=top_k
            )
            
            # Extract test results and format for frontend
            results = []
            for r in response.results:
                if r.result_type == "test":
                    test_data = r.data
                    results.append({
                        "test_id": test_data.get("id"),
                        "test_name": r.title,
                        "test_description": r.description,
                        "test_layer": test_data.get("layer", "unknown"),
                        "test_category": test_data.get("category", "uncategorized"),
                        "test_tags": test_data.get("tags", []),
                        "similarity_score": r.similarity_score,
                        "user_prompt": test_data.get("user_prompt", ""),
                        "expected_result": test_data.get("expected_result", {}),
                        "should_pass": test_data.get("should_pass", True)
                    })
            
            return results
        
        # Fallback - return empty results if no unified search
        print("⚠️ UnifiedSearchService not available for test analytics search")
        return []
        
    except Exception as e:
        print(f"Error searching test analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Conversation management endpoints
@app.get("/api/conversations")
async def list_conversations():
    """List all conversations with metadata."""
    storage = get_conversation_storage()
    conversations = storage.list_conversations()
    
    return {
        "conversations": [
            {
                "conversation_id": conv.conversation_id,
                "version": conv.version,
                "session_type": conv.session_type,
                "status": conv.status,
                "created_at": conv.created_at,
                "last_used_at": conv.last_used_at,
                "total_messages": conv.total_messages,
                "workflow_count": conv.workflow_count,
                "execution_count": conv.execution_count,
                "notes": conv.notes
            }
            for conv in conversations
        ]
    }

@app.post("/api/conversations/new")
async def create_new_conversation():
    """Create a new conversation session."""
    storage = get_conversation_storage()
    
    # Archive the current active conversation first
    current_conversations = [
        conv for conv in storage.conversations.values()
        if conv.session_type == "web_interface" and conv.status == "active"
    ]
    
    for conv in current_conversations:
        storage.archive_conversation(conv.conversation_id)
        print(f"📦 Auto-archived previous conversation: {conv.conversation_id} ({conv.version})")
    
    # Force create a new conversation 
    conversation_id = storage.create_conversation(
        session_type="web_interface",
        notes="Created via web interface - forced new session"
    )
    
    print(f"✨ Force created new conversation: {conversation_id}")
    
    return {
        "conversation_id": conversation_id,
        "status": "created"
    }

@app.post("/api/conversations/{conversation_id}/archive")
async def archive_conversation(conversation_id: str):
    """Archive a conversation."""
    storage = get_conversation_storage()
    storage.archive_conversation(conversation_id)
    
    return {
        "conversation_id": conversation_id,
        "status": "archived"
    }

@app.get("/api/conversations/active")
async def get_active_conversation():
    """Get the current active conversation."""
    storage = get_conversation_storage()
    conversation_id = storage.get_active_web_conversation()
    conversation = storage.get_conversation(conversation_id)
    
    if conversation:
        return {
            "conversation_id": conversation.conversation_id,
            "version": conversation.version,
            "session_type": conversation.session_type,
            "status": conversation.status,
            "created_at": conversation.created_at,
            "last_used_at": conversation.last_used_at,
            "total_messages": conversation.total_messages,
            "workflow_count": conversation.workflow_count,
            "execution_count": conversation.execution_count,
            "notes": conversation.notes
        }
    else:
        raise HTTPException(status_code=404, detail="Active conversation not found")


if __name__ == "__main__":
    uvicorn.run(
        "workflow_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )