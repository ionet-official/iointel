"""
Pydantic models for typed execution data flow.

This module provides strongly typed models for the entire execution pipeline,
replacing the current dict-based approach that makes debugging difficult.

We reuse existing models where possible:
- ToolUsageResult from datamodels.py
- pydantic-ai's response types
"""

from typing import Dict, List, Optional, Any, Union, Literal
from pydantic import BaseModel, Field
from enum import Enum

# Import existing models instead of redefining
from .datamodels import ToolUsageResult


class ExecutionStatus(str, Enum):
    """Status of execution at any level."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SUCCESS = "success"  # Alias for COMPLETED
    FAILED = "failed"
    SKIPPED = "skipped"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    AWAITING_INPUT = "awaiting_input"  # For user input nodes


class ExecutionMetadata(BaseModel):
    """
    Typed execution metadata that flows through the system.
    
    This replaces the untyped dict that currently gets passed around
    as 'execution_metadata' in various functions.
    """
    # Core identifiers
    node_id: Optional[str] = Field(None, description="Current node being executed")
    task_id: Optional[str] = Field(None, description="Task ID (often same as node_id)")
    conversation_id: Optional[str] = Field(None, description="Conversation/session ID")
    
    # User inputs from web interface
    user_inputs: Dict[str, str] = Field(
        default_factory=dict,
        description="User inputs keyed by node_id, e.g. {'user_input_1': 'Hello, I love bubble gum!'}"
    )
    
    # Execution configuration
    client_mode: bool = Field(False, description="Whether executing in client mode")
    agent_result_format: Literal["full", "chat", "chat_w_tools", "workflow", "minimal"] = Field(
        "full",
        description="Format for agent results"
    )
    
    # Web-specific fields
    form_id: Optional[str] = Field(None, description="Form ID for web UI matching")
    
    # Optional execution hints
    stages: Optional[Dict[str, Any]] = Field(None, description="Stage definitions for multi-stage tasks")
    execution_mode: Literal["sequential", "parallel"] = Field("sequential", description="Execution mode for stages")
    
    class Config:
        extra = "allow"  # Allow additional fields for backward compatibility


class AgentRunResponse(BaseModel):
    """
    Typed wrapper for the dict returned by Agent.run().
    
    This provides type safety for the generic agent response transport layer.
    It does NOT assume any specific structure in the result field - that's
    domain-specific and handled by higher-level models.
    """
    result: Any  # Could be string, dict, WorkflowSpec, anything - no assumptions
    conversation_id: Optional[str] = None
    tool_usage_results: List[ToolUsageResult] = Field(default_factory=list)
    full_result: Optional[Any] = None  # Original pydantic-ai result object
    
    @classmethod
    def from_dict(cls, response: Dict[str, Any]) -> "AgentRunResponse":
        """Create from the dict returned by Agent.run()."""
        return cls(
            result=response.get('result'),
            conversation_id=response.get('conversation_id'),
            tool_usage_results=[
                ToolUsageResult(**t) if isinstance(t, dict) else t
                for t in response.get('tool_usage_results', [])
            ],
            full_result=response.get('full_result')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dict for backward compatibility."""
        return {
            'result': self.result,
            'conversation_id': self.conversation_id,
            'tool_usage_results': [
                t.model_dump() if hasattr(t, 'model_dump') else t
                for t in self.tool_usage_results
            ],
            'full_result': self.full_result
        }


class AgentExecutionResult(BaseModel):
    """Result from execute_agent_task()."""
    status: ExecutionStatus
    agent_response: Optional[AgentRunResponse] = None
    error: Optional[str] = None  
    execution_time: Optional[float] = None
    node_id: Optional[str] = None
    node_type: Literal["agent"] = "agent"
    
    def to_legacy_dict(self) -> Dict[str, Any]:
        """Convert to legacy dict format for backward compatibility."""
        if self.agent_response:
            # Return the agent response dict with status metadata
            result = self.agent_response.to_dict()
            result.update({
                "status": self.status.value,
                "error": self.error,
                "execution_time": self.execution_time
            })
            return result
        else:
            # Error case
            return {
                "status": self.status.value,
                "error": self.error,
                "execution_time": self.execution_time,
                "result": None
            }


class DataSourceResult(BaseModel):
    """Result from data source execution."""
    tool_type: str
    status: ExecutionStatus
    result: Any = None
    error: Optional[str] = None
    message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NodeExecutionResult(BaseModel):
    """Generic result from any node execution."""
    node_id: str
    node_type: Literal["data_source", "agent", "tool", "decision", "workflow_call"]
    status: ExecutionStatus
    result: Union[AgentExecutionResult, DataSourceResult, Dict[str, Any]]
    error: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def get_agent_reasoning(self) -> Optional[str]:
        """Extract agent reasoning if this is an agent node."""
        if self.node_type == "agent" and isinstance(self.result, AgentExecutionResult):
            if self.result.agent_response:
                return self.result.agent_response.reasoning
        elif isinstance(self.result, dict) and "result" in self.result:
            # Handle legacy format
            return str(self.result.get("result", ""))
        return None


class WorkflowExecutionResult(BaseModel):
    """Complete workflow execution result."""
    workflow_id: str
    workflow_name: str
    status: ExecutionStatus
    node_results: Dict[str, NodeExecutionResult] = Field(default_factory=dict)
    final_output: Optional[Any] = None
    execution_time: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def get_all_agent_reasoning(self) -> List[str]:
        """Extract all agent reasoning from the workflow."""
        reasoning = []
        for node_result in self.node_results.values():
            if agent_reasoning := node_result.get_agent_reasoning():
                reasoning.append(agent_reasoning)
        return reasoning
    
    def to_feedback_summary(self) -> str:
        """Generate a meaningful feedback summary."""
        if self.status == ExecutionStatus.COMPLETED:
            # Extract actual agent outputs
            agent_outputs = self.get_all_agent_reasoning()
            if agent_outputs:
                # Return the actual valuable content
                return "\n\n".join(agent_outputs)
            else:
                return f"Workflow '{self.workflow_name}' completed successfully."
        elif self.status == ExecutionStatus.FAILED:
            return f"Workflow failed: {self.error}"
        else:
            return f"Workflow status: {self.status.value}"


class TaskExecutionResult(BaseModel):
    """Result from Task().run().execute()."""
    task_id: str
    status: ExecutionStatus
    output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
