"""
Workflow Helper Utilities
========================
High-level helpers for generating and executing workflows from natural language prompts.
"""
from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Dict, Any, Optional, Mapping
from uuid import uuid4

from iointel.src.agent_methods.agents.workflow_agent import WorkflowPlanner
from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec
from iointel.src.agent_methods.data_models.execution_models import (
    WorkflowExecutionResult,
    ExecutionStatus,
    NodeExecutionResult,
    AgentExecutionResult,
    DataSourceResult,
)
from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env
from iointel.src.utilities.tool_registry_utils import create_tool_catalog
from iointel.src.utilities.dag_executor import DAGExecutor
from iointel.src.utilities.graph_nodes import WorkflowState
from iointel.src.utilities.io_logger import get_component_logger
from iointel.src.web.execution_feedback import ExecutionFeedbackCollector

logger = get_component_logger("workflow_helpers")


# --------- Cached config providers ---------

@lru_cache(maxsize=1)
def _get_model_config() -> Dict[str, Any]:
    """Get shared LLM model configuration (cached)."""
    from iointel.src.utilities.constants import get_model_config
    helper_model = os.getenv("WORKFLOW_PLANNER_MODEL", "gpt-4o")
    return get_model_config(model=helper_model)


@lru_cache(maxsize=1)
def _get_default_tool_catalog() -> Dict[str, Any]:
    """Get default tool catalog with standard settings (cached)."""
    load_tools_from_env()
    return create_tool_catalog(filter_broken=True, verbose_format=False, use_working_filter=True)


# --------- Small helpers ---------

def _create_workflow_planner(conversation_id: str, debug: bool = False) -> WorkflowPlanner:
    cfg = _get_model_config()
    return WorkflowPlanner(
        model=cfg["model"],
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        conversation_id=conversation_id,
        debug=debug,
    )


def _create_dag_executor(
    feedback_collector: Optional[ExecutionFeedbackCollector] = None,
    max_concurrent_agents: Optional[int] = None,
) -> DAGExecutor:
    if max_concurrent_agents is None:
        max_concurrent_agents = int(os.getenv("MAX_CONCURRENT_AGENTS", "3"))
    return DAGExecutor(
        feedback_collector=feedback_collector,
        max_concurrent_agents=max_concurrent_agents,
    )


def _create_initial_state(
    objective: str,
    conversation_id: str,
    user_inputs: Optional[Dict[str, Any]] = None,
    execution_id: Optional[str] = None,
) -> WorkflowState:
    return WorkflowState(
        initial_text=objective,
        conversation_id=conversation_id,
        user_inputs=user_inputs or {},
        results={},
        execution_id=execution_id or str(uuid4()),
    )


def _friendly_node_label(
    node: Any,
    node_type: str,
    fallback_id: str,
) -> str:
    """
    Generate a human-friendly label for a node based on its type and data.
    Keeps logic isolated for testability.
    """
    if getattr(node, "label", None) and not str(node.label).startswith(f"{node_type}_"):
        return node.label

    if node_type == "agent" and hasattr(node.data, "agent_instructions"):
        instructions = (node.data.agent_instructions or "").strip()
        if instructions:
            words = instructions.split()[:3]
            label = " ".join(words).title()
            return (label[:27] + "...") if len(label) > 30 else label
        return f"Agent {fallback_id.split('_')[-1]}"

    if node_type == "data_source" and hasattr(node.data, "source_name"):
        return f"{str(node.data.source_name).replace('_', ' ').title()} Input"

    if node_type == "decision":
        return f"Decision Point {fallback_id.split('_')[-1]}"

    return f"{node_type.replace('_', ' ').title()} {fallback_id.split('_')[-1]}"


def _build_node_results(final_state: WorkflowState, workflow_spec: WorkflowSpec) -> Dict[str, NodeExecutionResult]:
    node_type_lookup = {n.id: n.type for n in workflow_spec.nodes}
    node_lookup = {n.id: n for n in workflow_spec.nodes}

    node_results: Dict[str, NodeExecutionResult] = {}
    
    # Defensive check: ensure results is a dictionary
    if not isinstance(final_state.results, dict):
        logger.error(f"❌ final_state.results is not a dict, got {type(final_state.results)}: {final_state.results}")
        return node_results
    
    for node_id, raw_result in final_state.results.items():
        actual_node_type = node_type_lookup.get(node_id, "agent")
        node = node_lookup.get(node_id)
        node_label = _friendly_node_label(node, actual_node_type, node_id) if node else node_id

        # Normalize type to execution model categories
        if actual_node_type in ("data_source", "decision"):
            normalized_type = actual_node_type
        else:
            normalized_type = "agent"

        if isinstance(raw_result, (AgentExecutionResult, DataSourceResult)):
            result_payload = raw_result
        else:
            # Mark legacy shape explicitly
            result_payload = {"type": "legacy", "result": raw_result}

        node_results[node_id] = NodeExecutionResult(
            node_id=node_id,
            node_label=node_label,
            node_type=normalized_type,
            status=ExecutionStatus.COMPLETED,
            result=result_payload,
        )
    return node_results


def _determine_execution_status(stats: Mapping[str, Any]) -> ExecutionStatus:
    return ExecutionStatus.COMPLETED if int(stats.get("failed_nodes", 0)) == 0 else ExecutionStatus.PARTIAL


# --------- Public APIs ---------

async def execute_workflow(
    workflow_spec: WorkflowSpec,
    *,
    user_inputs: Optional[Dict[str, Any]] = None,
    objective: Optional[str] = None,
    conversation_id: Optional[str] = None,
    execution_id: Optional[str] = None,
    form_id: Optional[str] = None,
    client_mode: Optional[bool] = None,
    feedback_collector: Optional[ExecutionFeedbackCollector] = None,
    execution_metadata_by_node: Optional[Dict[str, Dict[str, Any]]] = None,
    max_concurrent_agents: Optional[int] = None,
    debug: bool = False,
    allow_objective_from_first_input: bool = False,
) -> WorkflowExecutionResult:
    """
    Execute a workflow spec and return results. Consolidates the two previous execution paths.

    Parameters
    ----------
    workflow_spec : WorkflowSpec
    user_inputs : Optional[Dict[str, Any]]
    objective : Optional[str]
    conversation_id : Optional[str]
    execution_id : Optional[str]
    form_id : Optional[str]
    client_mode : Optional[bool]
    feedback_collector : Optional[ExecutionFeedbackCollector]
    execution_metadata_by_node : Optional[Dict[str, Dict[str, Any]]]
        Per-node metadata to pass to the executor when building the graph.
    max_concurrent_agents : Optional[int]
    debug : bool
    allow_objective_from_first_input : bool
        If True and `objective` is None, uses the first user input value as objective.
    """
    # IDs
    conversation_id = conversation_id or str(uuid4())
    execution_id = execution_id or str(uuid4())

    # Objective
    derived_objective = objective or workflow_spec.description or workflow_spec.title
    if objective is None and allow_objective_from_first_input and user_inputs:
        # Opt-in legacy behavior
        try:
            derived_objective = next(iter(user_inputs.values()))
        except StopIteration:
            pass

    # State
    initial_state = _create_initial_state(
        objective=derived_objective,
        conversation_id=conversation_id,
        user_inputs=user_inputs,
        execution_id=execution_id,
    )

    if debug:
        logger.info(f"🚀 Executing workflow: {workflow_spec.title}")
        logger.debug(f"Objective: {derived_objective}")
        if user_inputs:
            logger.debug(f"User inputs keys: {list(user_inputs.keys())}")

    t0 = time.monotonic()
    try:
        executor = _create_dag_executor(
            feedback_collector=feedback_collector,
            max_concurrent_agents=max_concurrent_agents,
        )

        # If no explicit per-node metadata provided, synthesize one that mirrors prior web path.
        if execution_metadata_by_node is None:
            execution_metadata_by_node = {
                node.id: {
                    "execution_id": execution_id,
                    "user_inputs": user_inputs or {},
                    "form_id": form_id,
                    "client_mode": client_mode,
                }
                for node in workflow_spec.nodes
            }

        executor.build_execution_graph(
            workflow_spec=workflow_spec,
            objective=derived_objective,
            conversation_id=conversation_id,
            execution_metadata_by_node=execution_metadata_by_node,
        )

        dag_issues = executor.validate_dag()
        if dag_issues and debug:
            logger.warning(f"DAG validation issues: {dag_issues}")

        final_state = await executor.execute_dag(initial_state)
        stats = executor.get_execution_statistics()

        if debug:
            logger.info(
                f"✅ Execution completed: {stats.get('executed_nodes')}/{stats.get('total_nodes')} nodes (efficiency={stats.get('execution_efficiency')})"
            )
            logger.info(f"🔍 final_state.results type: {type(final_state.results)}")
            logger.info(f"🔍 final_state.results content: {final_state.results}")

        node_results = _build_node_results(final_state, workflow_spec)
        overall_status = _determine_execution_status(stats)
        execution_summary = getattr(final_state, "execution_summary", None)

        result = WorkflowExecutionResult(
            workflow_id=str(workflow_spec.id),
            workflow_name=workflow_spec.title,
            status=overall_status,
            node_results=node_results,
            final_output=final_state.results,
            execution_time=time.monotonic() - t0,
            metadata={
                "stats": stats,
                "conversation_id": conversation_id,
                "execution_id": execution_id,
                "execution_summary": execution_summary,
                "form_id": form_id,
                "client_mode": client_mode,
            },
        )

        if feedback_collector:
            feedback_collector.complete_execution(
                execution_id=execution_id,
                final_outputs=final_state.results,
                error_summary=None,
            )

        return result

    except Exception as exc:  # noqa: BLE001
        logger.error(f"❌ Failed to execute workflow: {exc}")
        if feedback_collector:
            feedback_collector.complete_execution(
                execution_id=execution_id,
                final_outputs={},
                error_summary=str(exc),
            )

        return WorkflowExecutionResult(
            workflow_id=str(workflow_spec.id),
            workflow_name=workflow_spec.title,
            status=ExecutionStatus.FAILED,
            node_results={},
            final_output=None,
            execution_time=time.monotonic() - t0,
            error=str(exc),
            metadata={
                "conversation_id": conversation_id,
                "execution_id": execution_id,
                "form_id": form_id,
                "client_mode": client_mode,
            },
        )


async def plan_and_execute(
    prompt: str,
    *,
    tool_catalog: Optional[Dict[str, Any]] = None,
    conversation_id: Optional[str] = None,
    max_retries: int = 3,
    user_inputs: Optional[Dict[str, Any]] = None,
    allow_objective_from_first_input: bool = False,
    debug: bool = False,
) -> WorkflowExecutionResult:
    """
    Generate a workflow from natural language and execute it immediately.
    """
    conversation_id = conversation_id or str(uuid4())
    t0 = time.monotonic()

    logger.info(f"📋 Generating workflow from prompt: {prompt}")
    planner = _create_workflow_planner(conversation_id, debug)
    tool_catalog = tool_catalog or _get_default_tool_catalog()

    try:
        workflow_spec = await planner.generate_workflow(
            query=prompt,
            tool_catalog=tool_catalog,
            max_retries=max_retries,
        )
        logger.info(f"✅ Generated workflow: {workflow_spec.title} (nodes={len(workflow_spec.nodes)}, edges={len(workflow_spec.edges)})")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"❌ Failed to generate workflow: {exc}")
        return WorkflowExecutionResult(
            workflow_id="generated_workflow",
            workflow_name="Generated Workflow",
            status=ExecutionStatus.FAILED,
            node_results={},
            final_output=None,
            execution_time=time.monotonic() - t0,
            error=f"Failed to generate workflow: {exc}",
            metadata={"conversation_id": conversation_id, "generation_failed": True},
        )

    try:
        result = await execute_workflow(
            workflow_spec=workflow_spec,
            user_inputs=user_inputs,
            objective=prompt,  # canonical: objective = prompt unless explicitly overridden
            conversation_id=conversation_id,
            allow_objective_from_first_input=allow_objective_from_first_input,
            debug=debug,
        )
        result.metadata["generation_prompt"] = prompt
        result.metadata["generation_successful"] = True
        return result

    except Exception as exc:  # noqa: BLE001
        logger.error(f"❌ Failed to execute generated workflow: {exc}")
        return WorkflowExecutionResult(
            workflow_id=str(workflow_spec.id),
            workflow_name=workflow_spec.title,
            status=ExecutionStatus.FAILED,
            node_results={},
            final_output=None,
            execution_time=time.monotonic() - t0,
            error=f"Failed to execute workflow: {exc}",
            metadata={
                "conversation_id": conversation_id,
                "generation_successful": True,
                "execution_failed": True,
                "generation_prompt": prompt,
            },
        )


async def generate_only(
    prompt: str,
    *,
    tool_catalog: Optional[Dict[str, Any]] = None,
    conversation_id: Optional[str] = None,
    max_retries: int = 3,
    debug: bool = False,
) -> Optional[WorkflowSpec]:
    """
    Generate a workflow without executing it. Useful for testing generation logic.
    """
    conversation_id = conversation_id or str(uuid4())
    planner = _create_workflow_planner(conversation_id, debug)
    tool_catalog = tool_catalog or _get_default_tool_catalog()
    try:
        return await planner.generate_workflow(
            query=prompt,
            tool_catalog=tool_catalog,
            max_retries=max_retries,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to generate workflow: {exc}")
        return None
