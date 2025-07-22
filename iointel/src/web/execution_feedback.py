"""
Workflow Execution Result Feedback System

This module captures workflow execution results and feeds them back to the
WorkflowPlanner as system context for analysis and improvement suggestions.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec


class ExecutionStatus(Enum):
    """Execution status types."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class NodeExecutionResult:
    """Result of executing a single node."""
    node_id: str
    node_type: str
    node_label: str
    status: ExecutionStatus
    started_at: str
    finished_at: Optional[str]
    duration_seconds: Optional[float]
    result_preview: Optional[str] = None
    error_message: Optional[str] = None
    tool_usage: List[str] = None
    sla_enforcement_active: bool = False
    sla_validation_attempts: int = 0
    sla_validation_passed: bool = True
    sla_requirements: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.tool_usage is None:
            self.tool_usage = []


@dataclass
class WorkflowExecutionSummary:
    """Comprehensive summary of workflow execution."""
    execution_id: str
    workflow_id: str
    workflow_title: str
    status: ExecutionStatus
    started_at: str
    finished_at: Optional[str]
    total_duration_seconds: Optional[float]
    nodes_executed: List[NodeExecutionResult]
    nodes_skipped: List[str]
    user_inputs: Dict[str, Any]
    final_outputs: Dict[str, Any]
    workflow_spec: Optional[WorkflowSpec] = None  # Single source of truth for workflow representation
    error_summary: Optional[str] = None
    performance_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}


class ExecutionResultCurator:
    """Curates execution results into human-readable feedback."""
    
    @staticmethod
    def curate_execution_summary(summary: WorkflowExecutionSummary) -> str:
        """Generate a curated, human-readable execution summary."""
        
        # Header with key metrics
        header = f"""🤖 SYSTEM EXECUTION REPORT
=============================
Workflow: {summary.workflow_title}
Execution ID: {summary.execution_id}
Status: {summary.status.value.upper()}
Duration: {summary.total_duration_seconds:.2f}s
Timestamp: {summary.finished_at or summary.started_at}
"""
        
        # Execution overview
        total_nodes = len(summary.nodes_executed) + len(summary.nodes_skipped)
        executed_count = len(summary.nodes_executed)
        skipped_count = len(summary.nodes_skipped)
        
        overview = f"""
📊 EXECUTION OVERVIEW
--------------------
Total Nodes: {total_nodes}
Executed: {executed_count}
Skipped: {skipped_count}
Success Rate: {(len([n for n in summary.nodes_executed if n.status == ExecutionStatus.SUCCESS]) / max(executed_count, 1) * 100):.1f}%
"""
        
        # Node execution details
        node_details = "\n🔍 NODE EXECUTION DETAILS\n" + "-" * 1
        for node in summary.nodes_executed:
            status_emoji = "✅" if node.status == ExecutionStatus.SUCCESS else "❌"
            duration_str = f" ({node.duration_seconds:.2f}s)" if node.duration_seconds else ""
            
            node_details += f"""
{status_emoji} {node.node_label} ({node.node_type}){duration_str}"""
            
            if node.tool_usage:
                node_details += f"\n   Tools: {', '.join(node.tool_usage)}"
            
            if node.result_preview:
                preview = node.result_preview[:100] + "..." if len(node.result_preview) > 100 else node.result_preview
                node_details += f"\n   Result: {preview}"
            
            if node.error_message:
                error_preview = node.error_message[:150] + "..." if len(node.error_message) > 150 else node.error_message
                node_details += f"\n   Error: {error_preview}"
        
        # Skipped nodes
        skipped_details = ""
        if summary.nodes_skipped:
            skipped_details = "\n\n⏭️ SKIPPED NODES\n" + "-" * 15 + "\n"
            skipped_details += "\n".join(f"• {node_id}" for node_id in summary.nodes_skipped)
        
        # User inputs
        inputs_details = ""
        if summary.user_inputs:
            inputs_details = "\n\n📝 USER INPUTS\n" + "-" * 13 + "\n"
            for key, value in summary.user_inputs.items():
                value_str = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                inputs_details += f"• {key}: {value_str}\n"
        
        # Error analysis
        error_analysis = ""
        if summary.status != ExecutionStatus.SUCCESS:
            error_analysis = "\n\n🚨 ERROR ANALYSIS\n" + "-" * 16
            if summary.error_summary:
                error_analysis += f"\n{summary.error_summary}"
            
            # Common error patterns
            failed_nodes = [n for n in summary.nodes_executed if n.status != ExecutionStatus.SUCCESS]
            if failed_nodes:
                error_analysis += f"\n\nFailed Nodes: {len(failed_nodes)}"
                for node in failed_nodes:
                    if node.error_message:
                        error_analysis += f"\n• {node.node_label}: {node.error_message[:100]}"
        
        # Performance insights
        perf_insights = ""
        if summary.performance_metrics:
            perf_insights = "\n\n⚡ PERFORMANCE INSIGHTS\n" + "-" * 22
            for metric, value in summary.performance_metrics.items():
                perf_insights += f"\n• {metric}: {value}"
        
        return header + overview + node_details + skipped_details + inputs_details + error_analysis + perf_insights
    
    @staticmethod
    def generate_improvement_prompt(summary: WorkflowExecutionSummary, workflow_spec=None) -> str:
        """Generate a prompt for the WorkflowPlanner to analyze and suggest improvements."""
        
        curated_summary = ExecutionResultCurator.curate_execution_summary(summary)
        
        # Add readable execution path
        perf = summary.performance_metrics or {}
        path_lines = []
        executed_nodes = []
        skipped_nodes = []
        if "execution_path" in perf:
            path_lines.append("\nEXECUTION PATH:")
            for idx, node in enumerate(perf["execution_path"]):
                status = "✓" if node["status"] == "success" else "⏭️"
                label = node["node_label"]
                typ = node["node_type"]
                skip = f" (skipped: {node['skip_reason']})" if node["status"] == "skipped" else ""
                path_lines.append(f"  [{idx+1:02d}] {status} {label} [{typ}]{skip}")
                if node["status"] == "success":
                    executed_nodes.append(label)
                else:
                    skipped_nodes.append(f"{label} (reason: {node['skip_reason']})")
        path_str = "\n".join(path_lines)
        
        # Add explicit summary for agent/planner
        exec_summary = f"\nEXECUTION SUMMARY:\nExecuted nodes: {', '.join(executed_nodes) if executed_nodes else 'None'}\nSkipped nodes: {', '.join(skipped_nodes) if skipped_nodes else 'None'}\n\nNOTE: Nodes skipped with reason 'decision_gated' indicate correct conditional routing. Only one branch should execute; all others should be skipped. If multiple branches are executed, conditional routing failed unless using a conditional_multi_gate node which allows multiple branches to execute."
        
        # Use workflow spec from summary (single source of truth) or fallback to parameter
        active_workflow_spec = summary.workflow_spec or workflow_spec
        workflow_context = ""
        if active_workflow_spec:
            workflow_context = f"""

{active_workflow_spec.to_llm_prompt()}

"""
        
        return f"""SYSTEM: Workflow execution completed. Here's the complete analysis context:

{workflow_context}📊 EXECUTION RESULTS
{curated_summary}
{path_str}
{exec_summary}

CRITICAL: Compare the EXPECTED EXECUTION PATTERNS in the workflow spec above with actual results:
- For conditional workflows: verify only the correct path executed based on routing logic
- For SLA enforcement: verify required tools were used as specified in node SLAs  
- Skipped nodes are EXPECTED in conditional workflows when routing works correctly
- Multiple branch execution indicates conditional routing FAILURE (unless using conditional_multi_gate)

RESPONSE FORMAT: This is execution analysis, NOT a workflow generation request.
Use CHAT-ONLY mode: Set nodes: null, edges: null in your response.

Provide a BRIEF, CONVERSATIONAL analysis in the reasoning field:
1. What the workflow was SUPPOSED to do (based on spec) vs what it actually did
2. If conditional routing worked or failed (compare intended topology with execution path)
3. If SLA enforcement worked (verify tool usage against requirements)
4. Suggest specific fixes if anything didn't work as designed

Be precise - refer to the workflow specification when analyzing whether execution matched the design."""


class ExecutionFeedbackCollector:
    """Collects and processes execution results for feedback."""
    
    def __init__(self):
        self.active_executions: Dict[str, Dict[str, Any]] = {}
    
    def start_execution_tracking(
        self, 
        execution_id: str, 
        workflow_spec: WorkflowSpec,
        user_inputs: Dict[str, Any] = None
    ):
        """Start tracking a workflow execution."""
        self.active_executions[execution_id] = {
            "workflow_spec": workflow_spec,
            "user_inputs": user_inputs or {},
            "started_at": datetime.now().isoformat(),
            "node_results": {},
            "nodes_skipped": [],
            "status": ExecutionStatus.SUCCESS  # Assume success until failure
        }
    
    def record_node_start(self, execution_id: str, node_id: str, node_type: str, node_label: str):
        """Record that a node has started executing."""
        if execution_id not in self.active_executions:
            return
        
        self.active_executions[execution_id]["node_results"][node_id] = NodeExecutionResult(
            node_id=node_id,
            node_type=node_type,
            node_label=node_label,
            status=ExecutionStatus.SUCCESS,  # Will be updated
            started_at=datetime.now().isoformat(),
            finished_at=None,
            duration_seconds=None
        )
    
    def record_node_completion(
        self, 
        execution_id: str, 
        node_id: str, 
        status: ExecutionStatus,
        result_preview: str = None,
        error_message: str = None,
        tool_usage: List[str] = None
    ):
        """Record that a node has completed executing."""
        if execution_id not in self.active_executions:
            return
        
        if node_id not in self.active_executions[execution_id]["node_results"]:
            return
        
        node_result = self.active_executions[execution_id]["node_results"][node_id]
        node_result.status = status
        node_result.finished_at = datetime.now().isoformat()
        node_result.result_preview = result_preview
        node_result.error_message = error_message
        node_result.tool_usage = tool_usage or []
        
        # Calculate duration
        if node_result.started_at and node_result.finished_at:
            start_time = datetime.fromisoformat(node_result.started_at)
            end_time = datetime.fromisoformat(node_result.finished_at)
            node_result.duration_seconds = (end_time - start_time).total_seconds()
        
        # Update overall execution status if there's a failure
        if status != ExecutionStatus.SUCCESS:
            self.active_executions[execution_id]["status"] = ExecutionStatus.FAILED
    
    def record_node_skipped(self, execution_id: str, node_id: str):
        """Record that a node was skipped (e.g., due to conditional routing)."""
        if execution_id not in self.active_executions:
            return
        
        self.active_executions[execution_id]["nodes_skipped"].append(node_id)
    
    def complete_execution(
        self, 
        execution_id: str, 
        final_outputs: Dict[str, Any] = None,
        error_summary: str = None
    ) -> WorkflowExecutionSummary:
        """Complete execution tracking and generate summary."""
        if execution_id not in self.active_executions:
            return None
        
        execution_data = self.active_executions[execution_id]
        workflow_spec = execution_data["workflow_spec"]
        
        # Calculate total duration
        start_time = datetime.fromisoformat(execution_data["started_at"])
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # Determine final status
        final_status = execution_data["status"]
        if error_summary:
            final_status = ExecutionStatus.FAILED
        
        # Calculate decision-gated skips
        node_results = execution_data["node_results"]
        nodes_skipped = execution_data["nodes_skipped"]
        decision_gated_skips = [nid for nid in nodes_skipped if _is_decision_gated_skip(node_results.get(nid))]
        num_decision_gated_skips = len(decision_gated_skips)
        total_nodes = len(workflow_spec.nodes)
        effective_total_nodes = total_nodes - num_decision_gated_skips
        executed_nodes = len(node_results)
        efficiency = (executed_nodes / effective_total_nodes) if effective_total_nodes > 0 else 1.0

        # Build ordered execution path (executed + skipped, in workflow order)
        {n.id: n for n in workflow_spec.nodes}
        execution_path = []
        for n in workflow_spec.nodes:
            node_id = n.id
            node_result = node_results.get(node_id)
            status = "skipped" if node_id in nodes_skipped else "success"
            skip_reason = None
            result_preview = None
            tool_usage = []
            if node_result:
                result_preview = getattr(node_result, "result_preview", None) if hasattr(node_result, "result_preview") else node_result.get("result_preview") if isinstance(node_result, dict) else None
                tool_usage = getattr(node_result, "tool_usage", []) if hasattr(node_result, "tool_usage") else node_result.get("tool_usage", []) if isinstance(node_result, dict) else []
                if status == "skipped":
                    skip_reason = getattr(node_result, "error_message", None) if hasattr(node_result, "error_message") else node_result.get("reason") if isinstance(node_result, dict) else None
            execution_path.append({
                "node_id": node_id,
                "node_label": n.label,
                "node_type": n.type,
                "status": status,
                "skip_reason": skip_reason,
                "result_preview": result_preview,
                "tool_usage": tool_usage
            })

        # Create summary
        summary = WorkflowExecutionSummary(
            execution_id=execution_id,
            workflow_id=str(workflow_spec.id),
            workflow_title=workflow_spec.title,
            status=final_status,
            started_at=execution_data["started_at"],
            finished_at=end_time.isoformat(),
            total_duration_seconds=total_duration,
            nodes_executed=list(node_results.values()),
            nodes_skipped=nodes_skipped,
            user_inputs=execution_data["user_inputs"],
            final_outputs=final_outputs or {},
            workflow_spec=workflow_spec,  # Single source of truth
            error_summary=error_summary,
            performance_metrics={
                "nodes_total": total_nodes,
                "nodes_executed": executed_nodes,
                "nodes_skipped": len(nodes_skipped),
                "decision_gated_skips": decision_gated_skips,
                "execution_efficiency": f"{executed_nodes}/{effective_total_nodes} ({100*efficiency:.1f}%)" if effective_total_nodes > 0 else "N/A (all skips decision-gated)",
                "execution_path": execution_path
            }
        )
        
        # Clean up tracking
        del self.active_executions[execution_id]
        
        return summary

def _is_decision_gated_skip(node_result):
    # Returns True if the node was skipped due to decision gating
    if isinstance(node_result, dict):
        return node_result.get("status") == "skipped" and node_result.get("reason") == "decision_gated"
    return False


# Global feedback collector instance
feedback_collector = ExecutionFeedbackCollector()


def create_execution_feedback_prompt(summary: WorkflowExecutionSummary, workflow_spec=None) -> str:
    """Helper function to create feedback prompt for WorkflowPlanner."""
    # Note: workflow_spec parameter kept for backwards compatibility, but summary.workflow_spec is preferred
    return ExecutionResultCurator.generate_improvement_prompt(summary, workflow_spec)


if __name__ == "__main__":
    # Example usage
    from uuid import uuid4
    from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec, NodeSpec, EdgeSpec, NodeData, EdgeData
    
    # Create example workflow spec
    example_workflow = WorkflowSpec(
        id=uuid4(),
        rev=1,
        title="Test Email Processing Workflow",
        description="Test workflow for processing emails",
        nodes=[
            NodeSpec(id="agent_1", type="agent", label="Email Agent", data=NodeData()),
            NodeSpec(id="agent_2", type="agent", label="Context Tree Agent", data=NodeData())
        ],
        edges=[
            EdgeSpec(id="e1", source="agent_1", target="agent_2", data=EdgeData())
        ]
    )
    
    # Example execution tracking
    execution_id = "test-execution-123"
    
    feedback_collector.start_execution_tracking(
        execution_id=execution_id,
        workflow_spec=example_workflow,
        user_inputs={"user_input_1": "Test email content"}
    )
    
    # Simulate node execution
    feedback_collector.record_node_start(execution_id, "agent_1", "agent", "Email Agent")
    feedback_collector.record_node_completion(
        execution_id, "agent_1", ExecutionStatus.SUCCESS,
        result_preview="Email processed successfully"
    )
    
    feedback_collector.record_node_start(execution_id, "agent_2", "agent", "Context Tree Agent")
    feedback_collector.record_node_completion(
        execution_id, "agent_2", ExecutionStatus.FAILED,
        error_message="KeyError: 'self'"
    )
    
    # Complete execution and generate feedback
    summary = feedback_collector.complete_execution(
        execution_id=execution_id,
        error_summary="Agent initialization failed due to type hint issue"
    )
    
    # Generate feedback prompt
    # feedback_prompt = create_execution_feedback_prompt(summary)
    # print(feedback_prompt)

    print("--------------START OF EXECUTION SUMMARY------------------")
    # add ExecutionResultCurator.curate_execution_summary(summary) to the feedback_prompt
    feedback_prompt = ExecutionResultCurator.generate_improvement_prompt(summary, example_workflow)
    print(feedback_prompt)