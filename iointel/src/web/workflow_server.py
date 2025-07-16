"""
FastAPI server for serving WorkflowSpecs to the web interface.
"""

import os
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from iointel.src.agent_methods.agents.workflow_planner import WorkflowPlanner
from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec
from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env
from iointel.src.utilities.registries import TOOLS_REGISTRY
from iointel.src.memory import AsyncMemory
from iointel.src.workflow import Workflow
from iointel.src.utilities.decorators import register_custom_task
from iointel.src import chainables  # Import chainables to ensure agent executor is available
import uuid
import asyncio
import json

# Import example tools to register them globally
import iointel.src.RL.example_tools

# Import conditional gate tools to register them globally
import iointel.src.agent_methods.tools.conditional_gate

# Import workflow storage
from .workflow_storage import WorkflowStorage
from iointel.src.chainables import execute_tool_task, execute_agent_task


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
        # Delegate to backend portable executor
        result = await execute_tool_task(task_metadata, objective, agents, execution_metadata)
        print(f"✅ Tool '{tool_name}' completed: {result}")
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
        result = await execute_agent_task(task_metadata, objective, agents, execution_metadata)
        print(f"✅ Agent task completed: {result}")
        
        # Broadcast task completion
        if execution_id and len(connections) > 0:
            try:
                await broadcast_execution_update(
                    execution_id, 
                    "running", 
                    results={"agent_completed": True, "result": result}
                )
            except Exception as e:
                print(f"⚠️ Failed to broadcast agent completion: {e}")
        
        return result
    except Exception as e:
        error_msg = f"Agent task failed: {str(e)}"
        print(f"❌ {error_msg}")
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
        print(f"   📝 No tool_name specified, treating as agent-based decision")
        
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
    workflow: Optional[Dict] = None
    agent_response: Optional[str] = None
    error: Optional[str] = None


class ExecutionRequest(BaseModel):
    execute_current: bool = True
    workflow_data: Optional[Dict] = None
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
        print(f"📢 Execution {execution_id[:8]}: {status}")
    
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


def create_tool_catalog() -> Dict[str, Any]:
    """Create a tool catalog from available tools."""
    catalog = {}
    
    for tool_name, tool in TOOLS_REGISTRY.items():
        # Extract parameter names from JSON schema
        # tool.parameters is a JSON schema like:
        # {"properties": {"city": {"type": "string"}}, "required": ["city"], "title": "...", "type": "object"}
        # We need to extract the parameter names from the "properties" field
        parameters = {}
        required_params = []
        
        if isinstance(tool.parameters, dict) and "properties" in tool.parameters:
            properties = tool.parameters["properties"]
            required_params = tool.parameters.get("required", [])
            
            for param_name, param_info in properties.items():
                # Extract type from JSON schema type
                param_type = param_info.get("type", "any")
                # Handle complex types like anyOf (optional nullable types)
                if "anyOf" in param_info:
                    # Look for the non-null type in anyOf
                    for type_option in param_info["anyOf"]:
                        if type_option.get("type") != "null":
                            param_type = type_option.get("type", "any")
                            break
                
                # Map JSON schema types to Python types
                type_mapping = {
                    "string": "str",
                    "integer": "int", 
                    "number": "float",
                    "boolean": "bool",
                    "array": "list",
                    "object": "dict"
                }
                parameters[param_name] = type_mapping.get(param_type, param_type)
        
        catalog[tool_name] = {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,  # All parameter names and types
            "required_parameters": required_params,  # Only actually required parameters
            "is_async": tool.is_async
        }
    
    return catalog


@app.on_event("startup")
async def startup_event(conversation_id: str = "web_interface_session_01"):
    """Initialize the application on startup."""
    global planner, tool_catalog, workflow_storage
    
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
            # Set current workflow context so planner can reference it
            if current_workflow:
                planner.set_current_workflow(current_workflow)
            
            # Generate new workflow
            current_workflow, agent_response = await planner.generate_workflow(
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


@app.post("/api/execute", response_model=ExecutionResponse)
async def execute_workflow(request: ExecutionRequest):
    """Execute the current workflow or provided workflow data."""
    global current_workflow
    
    print(f"🚀 Execute workflow request: execute_current={request.execute_current}")
    print(f"🔍 Current workflow exists: {current_workflow is not None}")
    if current_workflow:
        print(f"🔍 Current workflow title: '{current_workflow.title}'")
    print(f"🔍 Request has workflow_data: {request.workflow_data is not None}")
    
    # Determine which workflow to execute
    workflow_to_execute = None
    if request.execute_current and current_workflow:
        workflow_to_execute = current_workflow
        print(f"📋 Executing current workflow: '{workflow_to_execute.title}'")
    elif request.workflow_data:
        # Reconstruct WorkflowSpec from provided data
        try:
            workflow_to_execute = WorkflowSpec.model_validate(request.workflow_data)
            print(f"📋 Executing provided workflow: '{workflow_to_execute.title}'")
        except Exception as e:
            print(f"❌ Failed to validate workflow data: {e}")
            return ExecutionResponse(
                success=False,
                error=f"Invalid workflow data: {str(e)}"
            )
    
    if not workflow_to_execute:
        print(f"❌ No workflow to execute - current_workflow: {current_workflow is not None}, workflow_data: {request.workflow_data is not None}")
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
        user_inputs=request.user_inputs,
        form_id=request.form_id
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
    form_id: Optional[str] = None
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
        workflow_def = workflow_spec.to_workflow_definition()
        yaml_content = workflow_spec.to_yaml()
        
        # Create workflow from YAML with custom conversation ID
        workflow = Workflow.from_yaml(yaml_str=yaml_content)
        workflow.objective = workflow_spec.description
        
        print(f"📋 Executing workflow with {len(workflow.tasks)} tasks")
        
        # Execute workflow with execution context
        conversation_id = f"web_execution_{execution_id}"
        
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
        
        # Update execution info with serialized results
        active_executions[execution_id].update({
            "status": "completed",
            "end_time": datetime.now().isoformat(),
            "results": serialize_execution_results(results),
            "error": None
        })
        
        print(f"✅ Workflow execution completed: {execution_id}")
        print(f"📈 Results: {len(results.get('results', {}))} task results")
        
        # Broadcast completion
        await broadcast_execution_update(
            execution_id, 
            "completed", 
            results=results
        )
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Workflow execution failed: {execution_id} - {error_msg}")
        import traceback
        print(f"🔍 Traceback: {traceback.format_exc()}")
        
        # Update execution info
        active_executions[execution_id].update({
            "status": "failed",
            "end_time": datetime.now().isoformat(),
            "error": error_msg
        })
        
        # Broadcast failure
        await broadcast_execution_update(execution_id, "failed", error=error_msg)


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
        print(f"🧠 Set planner context to loaded example workflow")
    
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
            print(f"🧠 Set planner context to loaded saved workflow")
        
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