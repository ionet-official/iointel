"""
Workflow Execution Result Feedback System

This module captures workflow execution results and feeds them back to the
WorkflowPlanner as system context for analysis and improvement suggestions.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec
from iointel.src.agent_methods.data_models.datamodels import ToolUsageResult
from iointel.src.agent_methods.data_models.execution_models import AgentExecutionResult, ExecutionStatus


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
    # NEW: Capture full outputs and tool results
    full_agent_output: Optional[str] = None  # Complete agent response/reasoning
    tool_usage_results: List[Dict[str, Any]] = None  # Detailed tool results with inputs/outputs  
    final_result: Optional[Any] = None  # Complete structured result from node
    
    def __post_init__(self):
        if self.tool_usage is None:
            self.tool_usage = []
        if self.tool_usage_results is None:
            self.tool_usage_results = []


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
        # Use centralized conversion utility for consistent formatting
        from iointel.src.utilities.conversion_utils import execution_summary_to_llm_prompt
        return execution_summary_to_llm_prompt(summary)
        # DEPRECATED: Formatting logic moved to centralized conversion utilities
        # This method now uses the single source of truth conversion utilities
    
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
            # Use centralized conversion utility instead of scattered methods
            from iointel.src.utilities.conversion_utils import workflow_spec_to_llm_structured
            workflow_context = f"""

{workflow_spec_to_llm_structured(active_workflow_spec)}

"""
        
        # Use unified prompt system for cleaner, more focused feedback
        try:
            from iointel.src.utilities.unified_prompt_system import unified_prompt_system, PromptType
        except ImportError:
            # Fallback: unified prompt system not available, use simple feedback
            unified_prompt_system = None
            PromptType = None
        
        if summary.status == ExecutionStatus.SUCCESS:
            # SUCCESS: Build concise results summary
            
            # Extract key results from successful nodes
            agent_results = []
            tool_results = []
            
            for node in summary.nodes_executed:
                if node.status == ExecutionStatus.SUCCESS:
                    # Agent outputs (the actual reasoning/responses) - NO TRUNCATION
                    if hasattr(node, 'full_agent_output') and node.full_agent_output:
                        agent_results.append(f"• **{node.node_label}**: {node.full_agent_output}")
                    elif node.result_preview:
                        agent_results.append(f"• **{node.node_label}**: {node.result_preview}")
                    
                    # Tool usage (what tools actually did)
                    if hasattr(node, 'tool_usage_results') and node.tool_usage_results:
                        for tool_result in node.tool_usage_results:
                            # Handle both dict and ToolUsageResult objects
                            if isinstance(tool_result, dict):
                                tool_name = tool_result.get('tool_name', 'unknown')
                                result = tool_result.get('result', 'No output')
                            else:
                                # It's a ToolUsageResult object
                                tool_name = tool_result.tool_name
                                result = tool_result.tool_result
                            
                            if result and result != 'None':
                                # Show key info but keep reasonable length
                                result_str = str(result)
                                if len(result_str) > 300:
                                    result_str = result_str[:300] + "..."
                                tool_results.append(f"• **{tool_name}**: {result_str}")
            
            # Create concise prompt using unified system
            agent_outputs_text = "\n".join(agent_results) if agent_results else "No agent outputs captured"
            tool_results_text = "\n".join(tool_results) if tool_results else "No tool results captured"
            
            # Check if we actually have meaningful results
            has_meaningful_results = bool(agent_results) or bool(tool_results)
            
            # If no meaningful results, treat as incomplete execution
            if not has_meaningful_results:
                return f"""SYSTEM: ⚠️ **{summary.workflow_title}** executed in {summary.total_duration_seconds:.1f}s but produced no meaningful results.

🔍 **EXECUTION ANALYSIS:**
- Workflow nodes executed: {len(summary.nodes_executed)}
- Agent outputs captured: 0
- Tool results captured: 0

❓ **POTENTIAL ISSUES:**
- Agents may not be producing output correctly
- Tool results may not be captured properly
- The workflow may be missing agent nodes that produce results

RESPONSE FORMAT: This is an execution analysis, NOT a workflow generation request.
Use CHAT-ONLY mode: Set nodes: null, edges: null in your response.

Please acknowledge that the workflow ran but didn't produce expected outputs, and suggest checking the workflow design."""
            
            # Try to use unified prompt system if available
            if unified_prompt_system and PromptType:
                try:
                    templates = unified_prompt_system.search_templates(
                        prompt_type=PromptType.EXECUTION_FEEDBACK,
                        tags=["success"]
                    )
                    
                    if templates:
                        instance = unified_prompt_system.render_prompt(
                            templates[0].id,
                            workflow_title=summary.workflow_title,
                            duration=f"{summary.total_duration_seconds:.1f}",
                            agent_outputs=agent_outputs_text,
                            tool_results=tool_results_text,
                            value_summary=f"Successfully processed user input through {len(summary.nodes_executed)} workflow steps"
                        )
                        
                        # Record usage for improvement
                        unified_prompt_system.record_prompt_result(instance.id, success=True)
                        
                        return f"""SYSTEM: {instance.content}

RESPONSE FORMAT: This is a results report, NOT a workflow generation request.
Use CHAT-ONLY mode: Set nodes: null, edges: null in your response.

Focus on celebrating what the workflow accomplished and the value delivered to the user."""
                    
                except Exception as e:
                    print(f"⚠️ Failed to use unified prompt template: {e}")
            
            # Fallback to simple success prompt with FULL RESULTS
            return f"""SYSTEM: ✅ **{summary.workflow_title}** completed in {summary.total_duration_seconds:.1f}s

🎯 **RESULTS:**
{agent_outputs_text}

🛠️ **TOOLS USED:**
{tool_results_text}

RESPONSE FORMAT: This is a results report, NOT a workflow generation request.
Use CHAT-ONLY mode: Set nodes: null, edges: null in your response.

Summarize the KEY RESULTS above - focus on the specific recommendations, decisions, or outcomes the user received."""
        else:
            # FAILURE: Do introspection and analysis
            return f"""SYSTEM: Workflow execution failed or had issues. Provide analysis and suggestions.

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
        tool_usage: List[str] = None,
        full_agent_output: str = None,
        tool_usage_results: Union[List[Dict[str, Any]], List[ToolUsageResult]] = None,
        final_result: Union[Any, AgentExecutionResult] = None
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
        node_result.full_agent_output = full_agent_output
        node_result.tool_usage_results = tool_usage_results or []
        node_result.final_result = final_result
        
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
        
        # Safety check for None workflow_spec
        if workflow_spec is None:
            print(f"⚠️ Warning: workflow_spec is None for execution {execution_id}")
            # Return minimal summary with error status
            return WorkflowExecutionSummary(
                execution_id=execution_id,
                status=ExecutionStatus.FAILED,
                total_duration=0,
                node_count=0,
                executed_count=0,
                skipped_count=0,
                efficiency=0.0,
                execution_path=[],
                error_summary=error_summary or "Workflow specification was not found"
            )
        
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
    
    # Test Case 1: FAILURE scenario (existing test)
    print("=" * 80)
    print("TEST CASE 1: FAILURE SCENARIO")
    print("=" * 80)
    
    execution_id_fail = "test-execution-fail"
    
    feedback_collector.start_execution_tracking(
        execution_id=execution_id_fail,
        workflow_spec=example_workflow,
        user_inputs={"user_input_1": "Test email content"}
    )
    
    # Simulate node execution with failure
    feedback_collector.record_node_start(execution_id_fail, "agent_1", "agent", "Email Agent")
    feedback_collector.record_node_completion(
        execution_id_fail, "agent_1", ExecutionStatus.SUCCESS,
        result_preview="Email processed successfully"
    )
    
    feedback_collector.record_node_start(execution_id_fail, "agent_2", "agent", "Context Tree Agent")
    feedback_collector.record_node_completion(
        execution_id_fail, "agent_2", ExecutionStatus.FAILED,
        error_message="KeyError: 'self'"
    )
    
    # Complete execution and generate feedback
    summary_fail = feedback_collector.complete_execution(
        execution_id=execution_id_fail,
        error_summary="Agent initialization failed due to type hint issue"
    )
    
    feedback_prompt_fail = ExecutionResultCurator.generate_improvement_prompt(summary_fail, example_workflow)
    print(feedback_prompt_fail)
    
    print("\n" + "=" * 80)
    print("TEST CASE 2: SUCCESS SCENARIO")
    print("=" * 80)
    
    # Test Case 2: SUCCESS scenario
    execution_id_success = "test-execution-success"
    
    feedback_collector.start_execution_tracking(
        execution_id=execution_id_success,
        workflow_spec=example_workflow,
        user_inputs={"user_input_1": "Process email about project updates"}
    )
    
    # Simulate successful node execution with rich data
    feedback_collector.record_node_start(execution_id_success, "agent_1", "agent", "Email Agent")
    feedback_collector.record_node_completion(
        execution_id_success, "agent_1", ExecutionStatus.SUCCESS,
        result_preview="Extracted 3 action items from email: 1) Review proposal 2) Schedule meeting 3) Update timeline",
        tool_usage=["email_parser", "text_analyzer"],
        full_agent_output="I've analyzed the email about project updates and identified three critical action items that need attention. The email contains important timeline information and resource requests that require immediate action.",
        tool_usage_results=[
            {
                "tool_name": "email_parser", 
                "input": "Process email about project updates",
                "result": {"subject": "Project Updates - Action Required", "sender": "project_manager@company.com", "action_items": ["Review proposal", "Schedule meeting", "Update timeline"]},
                "metadata": {"confidence": 0.95, "processed_at": "2025-07-27T14:31:48Z"}
            },
            {
                "tool_name": "text_analyzer",
                "input": "Analyze extracted action items for priority",
                "result": {"high_priority": ["Review proposal"], "medium_priority": ["Schedule meeting", "Update timeline"], "urgency_score": 8.2},
                "metadata": {"analysis_type": "priority_scoring"}
            }
        ],
        final_result={"action_items": 3, "high_priority_count": 1, "total_urgency_score": 8.2}
    )
    
    feedback_collector.record_node_start(execution_id_success, "agent_2", "agent", "Context Tree Agent") 
    feedback_collector.record_node_completion(
        execution_id_success, "agent_2", ExecutionStatus.SUCCESS,
        result_preview="Created context tree with 3 nodes, stored in memory for easy retrieval",
        tool_usage=["create_context_tree", "memory_store"],
        full_agent_output="Successfully organized the action items into a hierarchical context tree structure. Each item has been categorized by priority and assigned metadata for efficient retrieval and tracking.",
        tool_usage_results=[
            {
                "tool_name": "create_context_tree",
                "input": {"action_items": ["Review proposal", "Schedule meeting", "Update timeline"], "priority_data": {"high": 1, "medium": 2}},
                "result": {
                    "tree_id": "ctx_tree_001",
                    "nodes": [
                        {"id": "node_1", "content": "Review proposal", "priority": "high", "parent": None},
                        {"id": "node_2", "content": "Schedule meeting", "priority": "medium", "parent": "node_1"},
                        {"id": "node_3", "content": "Update timeline", "priority": "medium", "parent": "node_1"}
                    ],
                    "structure": "hierarchical"
                },
                "metadata": {"creation_time": "2025-07-27T14:31:48Z", "node_count": 3}
            },
            {
                "tool_name": "memory_store",
                "input": {"tree_id": "ctx_tree_001", "storage_type": "persistent"},
                "result": {"storage_id": "mem_store_abc123", "status": "stored", "retrieval_key": "project_updates_tree"},
                "metadata": {"storage_size": "2.4KB", "ttl": "30_days"}
            }
        ],
        final_result={"context_tree_id": "ctx_tree_001", "stored_location": "mem_store_abc123", "retrieval_key": "project_updates_tree", "node_count": 3}
    )
    
    # Complete successful execution
    summary_success = feedback_collector.complete_execution(
        execution_id=execution_id_success,
        final_outputs={"action_items": 3, "context_tree_nodes": 3, "stored_in_memory": True}
    )
    
    feedback_prompt_success = ExecutionResultCurator.generate_improvement_prompt(summary_success, example_workflow)
    print(feedback_prompt_success)