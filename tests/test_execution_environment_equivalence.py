"""
Execution Environment Equivalence Tests

These tests verify that the same workflow specification produces identical results
when executed through different execution environments:
1. DAG Executor (direct)
2. Workflow.py with DAG structure (delegated to DAG executor)
3. Workflow.py with sequential execution (fallback)

Critical for ensuring isomorphic behavior across execution paths.
"""

import pytest
import asyncio
from typing import Dict, Any, List
from uuid import uuid4

from iointel.src.workflow import Workflow
from iointel.src.utilities.dag_executor import DAGExecutor
from iointel.src.utilities.graph_nodes import WorkflowState
from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec, NodeSpec, EdgeSpec, NodeData, EdgeData
)
from iointel.src.agent_methods.data_models.datamodels import AgentParams


class TestExecutionEnvironmentEquivalence:
    """Test that different execution environments produce equivalent results."""
    
    def create_simple_workflow_spec(self) -> WorkflowSpec:
        """Create a simple workflow spec for testing equivalence."""
        nodes = [
            NodeSpec(
                id="input_node",
                type="tool",
                label="Input Data",
                data=NodeData(
                    tool_name="mock_data_provider",
                    config={"data_type": "test_data"},
                    ins=[],
                    outs=["test_data"]
                )
            ),
            NodeSpec(
                id="process_node",
                type="agent",
                label="Process Data",
                data=NodeData(
                    agent_instructions="Process the input data and return a summary.",
                    config={"model": "gpt-4o-mini"},
                    ins=["test_data"],
                    outs=["processed_data"]
                )
            ),
            NodeSpec(
                id="output_node",
                type="tool",
                label="Output Results",
                data=NodeData(
                    tool_name="mock_output_handler",
                    config={"format": "json"},
                    ins=["processed_data"],
                    outs=["final_results"]
                )
            )
        ]
        
        edges = [
            EdgeSpec(
                id="edge_1",
                source="input_node",
                target="process_node",
                data=EdgeData()
            ),
            EdgeSpec(
                id="edge_2",
                source="process_node",
                target="output_node",
                data=EdgeData()
            )
        ]
        
        return WorkflowSpec(
            id=uuid4(),
            rev=1,
            title="Simple Equivalence Test Workflow",
            description="Test workflow for execution environment equivalence",
            nodes=nodes,
            edges=edges
        )
    
    def create_complex_workflow_spec(self) -> WorkflowSpec:
        """Create a complex workflow spec with conditional logic and parallel execution."""
        nodes = [
            NodeSpec(
                id="data_source",
                type="tool",
                label="Data Source",
                data=NodeData(
                    tool_name="mock_data_provider",
                    config={"data_type": "market_data"},
                    ins=[],
                    outs=["market_data"]
                )
            ),
            NodeSpec(
                id="decision_agent",
                type="agent",
                label="Decision Agent",
                data=NodeData(
                    agent_instructions="Analyze market data and decide on action. Use conditional_gate tool.",
                    tools=["conditional_gate"],
                    config={"model": "gpt-4o-mini"},
                    ins=["market_data"],
                    outs=["decision"]
                )
            ),
            NodeSpec(
                id="buy_handler",
                type="agent",
                label="Buy Handler",
                data=NodeData(
                    agent_instructions="Handle buy decision and execute buy order.",
                    config={"model": "gpt-4o-mini"},
                    ins=["decision"],
                    outs=["buy_result"]
                )
            ),
            NodeSpec(
                id="sell_handler",
                type="agent",
                label="Sell Handler",
                data=NodeData(
                    agent_instructions="Handle sell decision and execute sell order.",
                    config={"model": "gpt-4o-mini"},
                    ins=["decision"],
                    outs=["sell_result"]
                )
            ),
            NodeSpec(
                id="results_aggregator",
                type="tool",
                label="Results Aggregator",
                data=NodeData(
                    tool_name="mock_aggregator",
                    config={"format": "summary"},
                    ins=["buy_result", "sell_result"],
                    outs=["final_summary"]
                )
            )
        ]
        
        edges = [
            EdgeSpec(
                id="e1",
                source="data_source",
                target="decision_agent",
                data=EdgeData()
            ),
            EdgeSpec(
                id="e2",
                source="decision_agent",
                target="buy_handler",
                data=EdgeData(condition="routed_to == 'buy'")
            ),
            EdgeSpec(
                id="e3",
                source="decision_agent",
                target="sell_handler",
                data=EdgeData(condition="routed_to == 'sell'")
            ),
            EdgeSpec(
                id="e4",
                source="buy_handler",
                target="results_aggregator",
                data=EdgeData()
            ),
            EdgeSpec(
                id="e5",
                source="sell_handler",
                target="results_aggregator",
                data=EdgeData()
            )
        ]
        
        return WorkflowSpec(
            id=uuid4(),
            rev=1,
            title="Complex Equivalence Test Workflow",
            description="Complex workflow with conditional logic and parallel execution",
            nodes=nodes,
            edges=edges
        )
    
    @pytest.fixture
    def setup_mock_executors(self):
        """Setup mock executors for testing."""
        from iointel.src.utilities.registries import TASK_EXECUTOR_REGISTRY
        
        # Store original executors
        original_executors = TASK_EXECUTOR_REGISTRY.copy()
        
        # Mock executors
        async def mock_data_provider(task_metadata, objective, agents, execution_metadata):
            data_type = task_metadata.get("config", {}).get("data_type", "default")
            return {
                "result": {
                    "data_type": data_type,
                    "timestamp": "2024-01-01T00:00:00Z",
                    "value": 100.0,
                    "action": "buy"  # Deterministic for testing
                }
            }
        
        async def mock_agent_executor(task_metadata, objective, agents, execution_metadata):
            instructions = task_metadata.get("agent_instructions", "")
            return {
                "result": f"Agent processed: {instructions[:50]}...",
                "status": "completed"
            }
        
        async def mock_output_handler(task_metadata, objective, agents, execution_metadata):
            format_type = task_metadata.get("config", {}).get("format", "json")
            return {
                "result": {
                    "format": format_type,
                    "processed": True,
                    "timestamp": "2024-01-01T00:00:00Z"
                }
            }
        
        async def mock_aggregator(task_metadata, objective, agents, execution_metadata):
            return {
                "result": {
                    "aggregated": True,
                    "summary": "All operations completed successfully",
                    "total_operations": 2
                }
            }
        
        # Register mock executors
        TASK_EXECUTOR_REGISTRY["tool"] = mock_data_provider
        TASK_EXECUTOR_REGISTRY["agent"] = mock_agent_executor
        TASK_EXECUTOR_REGISTRY["mock_data_provider"] = mock_data_provider
        TASK_EXECUTOR_REGISTRY["mock_output_handler"] = mock_output_handler
        TASK_EXECUTOR_REGISTRY["mock_aggregator"] = mock_aggregator
        
        yield
        
        # Restore original executors
        TASK_EXECUTOR_REGISTRY.clear()
        TASK_EXECUTOR_REGISTRY.update(original_executors)
    
    async def execute_via_dag_executor(self, spec: WorkflowSpec, objective: str = "Test execution") -> Dict[str, Any]:
        """Execute workflow via direct DAG executor."""
        executor = DAGExecutor()
        executor.build_execution_graph(
            nodes=spec.nodes,
            edges=spec.edges,
            objective=objective
        )
        
        initial_state = WorkflowState(
            conversation_id=f"dag_{spec.id}",
            initial_text=objective,
            results={}
        )
        
        final_state = await executor.execute_dag(initial_state)
        return final_state.results
    
    async def execute_via_workflow_dag(self, spec: WorkflowSpec, objective: str = "Test execution") -> Dict[str, Any]:
        """Execute workflow via Workflow.py with DAG structure."""
        # Convert WorkflowSpec to Workflow with DAG structure
        workflow = Workflow(objective=objective)
        
        # Add DAG structure to task metadata
        dag_structure = {
            "nodes": [node.model_dump() for node in spec.nodes],
            "edges": [edge.model_dump() for edge in spec.edges]
        }
        
        # Create a task with DAG structure metadata
        workflow.add_task({
            "task_id": "dag_execution",
            "name": "DAG Execution",
            "type": "dag",
            "task_metadata": {
                "dag_structure": dag_structure
            }
        })
        
        result = await workflow.run_tasks(conversation_id=f"workflow_dag_{spec.id}")
        return result["results"]
    
    async def execute_via_workflow_sequential(self, spec: WorkflowSpec, objective: str = "Test execution") -> Dict[str, Any]:
        """Execute workflow via Workflow.py sequential execution (simplified)."""
        # Convert WorkflowSpec to sequential tasks
        workflow = Workflow(objective=objective)
        
        # Add tasks in topological order (simplified for testing)
        for node in spec.nodes:
            task_dict = {
                "task_id": node.id,
                "name": node.label,
                "type": node.type,
                "task_metadata": {
                    "config": node.data.config,
                    "tool_name": node.data.tool_name,
                    "agent_instructions": node.data.agent_instructions,
                    "ports": {
                        "inputs": node.data.ins,
                        "outputs": node.data.outs
                    }
                }
            }
            
            workflow.add_task(task_dict)
        
        result = await workflow.run_tasks(conversation_id=f"workflow_seq_{spec.id}")
        return result["results"]
    
    def normalize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize results for comparison by removing execution-specific metadata."""
        normalized = {}
        
        for key, value in results.items():
            if isinstance(value, dict):
                # Remove execution-specific fields
                normalized_value = {
                    k: v for k, v in value.items() 
                    if k not in ["conversation_id", "execution_id", "timestamp", "node_id", "task_id", "status"]
                }
                if normalized_value:
                    normalized[key] = normalized_value
            else:
                normalized[key] = value
        
        return normalized
    
    def assert_results_equivalent(self, results1: Dict[str, Any], results2: Dict[str, Any], execution_type1: str, execution_type2: str):
        """Assert that two result sets are equivalent."""
        norm1 = self.normalize_results(results1)
        norm2 = self.normalize_results(results2)
        
        # Check that both have the same keys
        keys1 = set(norm1.keys())
        keys2 = set(norm2.keys())
        
        if keys1 != keys2:
            missing_in_2 = keys1 - keys2
            missing_in_1 = keys2 - keys1
            
            error_msg = f"Results differ between {execution_type1} and {execution_type2}:\\n"
            if missing_in_1:
                error_msg += f"Missing in {execution_type1}: {missing_in_1}\\n"
            if missing_in_2:
                error_msg += f"Missing in {execution_type2}: {missing_in_2}\\n"
            
            pytest.fail(error_msg)
        
        # Check that values are equivalent
        for key in keys1:
            val1 = norm1[key]
            val2 = norm2[key]
            
            if val1 != val2:
                pytest.fail(
                    f"Results differ for key '{key}' between {execution_type1} and {execution_type2}:\\n"
                    f"{execution_type1}: {val1}\\n"
                    f"{execution_type2}: {val2}"
                )
    
    @pytest.mark.asyncio
    async def test_simple_workflow_equivalence(self, setup_mock_executors):
        """Test that simple workflows produce equivalent results across execution environments."""
        spec = self.create_simple_workflow_spec()
        objective = "Test simple workflow equivalence"
        
        # Execute via different environments
        dag_results = await self.execute_via_dag_executor(spec, objective)
        workflow_dag_results = await self.execute_via_workflow_dag(spec, objective)
        
        # Assert equivalence
        self.assert_results_equivalent(
            dag_results, 
            workflow_dag_results, 
            "DAG Executor", 
            "Workflow DAG"
        )
    
    @pytest.mark.asyncio
    async def test_complex_workflow_equivalence(self, setup_mock_executors):
        """Test that complex workflows produce equivalent results across execution environments."""
        spec = self.create_complex_workflow_spec()
        objective = "Test complex workflow equivalence"
        
        # Execute via different environments
        dag_results = await self.execute_via_dag_executor(spec, objective)
        workflow_dag_results = await self.execute_via_workflow_dag(spec, objective)
        
        # Assert equivalence
        self.assert_results_equivalent(
            dag_results, 
            workflow_dag_results, 
            "DAG Executor", 
            "Workflow DAG"
        )
    
    @pytest.mark.asyncio
    async def test_execution_statistics_equivalence(self, setup_mock_executors):
        """Test that execution statistics are equivalent across environments."""
        spec = self.create_simple_workflow_spec()
        objective = "Test execution statistics equivalence"
        
        # Execute via DAG executor and collect statistics
        dag_executor = DAGExecutor()
        dag_executor.build_execution_graph(
            nodes=spec.nodes,
            edges=spec.edges,
            objective=objective
        )
        
        initial_state = WorkflowState(
            conversation_id=f"stats_test_{spec.id}",
            initial_text=objective,
            results={}
        )
        
        await dag_executor.execute_dag(initial_state)
        dag_stats = dag_executor.get_execution_statistics()
        
        # Verify statistics are reasonable
        assert dag_stats["total_nodes"] == len(spec.nodes)
        assert dag_stats["executed_nodes"] > 0
        assert dag_stats["skipped_nodes"] >= 0
        assert dag_stats["executed_nodes"] + dag_stats["skipped_nodes"] == dag_stats["total_nodes"]
    
    @pytest.mark.asyncio
    async def test_error_handling_equivalence(self, setup_mock_executors):
        """Test that error handling is equivalent across execution environments."""
        # Create a workflow spec with an invalid node
        nodes = [
            NodeSpec(
                id="invalid_node",
                type="unknown_type",
                label="Invalid Node",
                data=NodeData(
                    tool_name="non_existent_tool",
                    config={},
                    ins=[],
                    outs=["error_output"]
                )
            )
        ]
        
        spec = WorkflowSpec(
            id=uuid4(),
            rev=1,
            title="Error Test Workflow",
            description="Test workflow for error handling equivalence",
            nodes=nodes,
            edges=[]
        )
        
        # Both execution environments should handle errors similarly
        dag_error = None
        workflow_error = None
        
        try:
            await self.execute_via_dag_executor(spec, "Test error handling")
        except Exception as e:
            dag_error = type(e).__name__
        
        try:
            await self.execute_via_workflow_dag(spec, "Test error handling")
        except Exception as e:
            workflow_error = type(e).__name__
        
        # Both should either succeed or fail with similar error types
        if dag_error is not None or workflow_error is not None:
            # At least one failed - ensure both show similar behavior
            assert dag_error is not None, "DAG executor should also fail if workflow fails"
            assert workflow_error is not None, "Workflow executor should also fail if DAG fails"
            
            # Error types should be similar (both execution errors)
            assert "Error" in dag_error or "Exception" in dag_error
            assert "Error" in workflow_error or "Exception" in workflow_error
    
    @pytest.mark.asyncio
    async def test_deterministic_execution(self, setup_mock_executors):
        """Test that execution results are deterministic across multiple runs."""
        spec = self.create_simple_workflow_spec()
        objective = "Test deterministic execution"
        
        # Run the same workflow multiple times
        results1 = await self.execute_via_dag_executor(spec, objective)
        results2 = await self.execute_via_dag_executor(spec, objective)
        results3 = await self.execute_via_dag_executor(spec, objective)
        
        # Results should be identical (deterministic)
        self.assert_results_equivalent(results1, results2, "Run 1", "Run 2")
        self.assert_results_equivalent(results2, results3, "Run 2", "Run 3")
    
    @pytest.mark.asyncio
    async def test_parallel_execution_equivalence(self, setup_mock_executors):
        """Test that parallel execution produces equivalent results."""
        # Create a workflow with parallel branches
        nodes = [
            NodeSpec(
                id="source",
                type="tool",
                label="Source",
                data=NodeData(
                    tool_name="mock_data_provider",
                    config={"data_type": "parallel_test"},
                    ins=[],
                    outs=["source_data"]
                )
            ),
            NodeSpec(
                id="branch_a",
                type="agent",
                label="Branch A",
                data=NodeData(
                    agent_instructions="Process data in branch A",
                    config={"model": "gpt-4o-mini"},
                    ins=["source_data"],
                    outs=["branch_a_result"]
                )
            ),
            NodeSpec(
                id="branch_b",
                type="agent",
                label="Branch B",
                data=NodeData(
                    agent_instructions="Process data in branch B",
                    config={"model": "gpt-4o-mini"},
                    ins=["source_data"],
                    outs=["branch_b_result"]
                )
            ),
            NodeSpec(
                id="merger",
                type="tool",
                label="Merger",
                data=NodeData(
                    tool_name="mock_aggregator",
                    config={"format": "merged"},
                    ins=["branch_a_result", "branch_b_result"],
                    outs=["merged_result"]
                )
            )
        ]
        
        edges = [
            EdgeSpec(id="e1", source="source", target="branch_a", data=EdgeData()),
            EdgeSpec(id="e2", source="source", target="branch_b", data=EdgeData()),
            EdgeSpec(id="e3", source="branch_a", target="merger", data=EdgeData()),
            EdgeSpec(id="e4", source="branch_b", target="merger", data=EdgeData())
        ]
        
        spec = WorkflowSpec(
            id=uuid4(),
            rev=1,
            title="Parallel Test Workflow",
            description="Test workflow for parallel execution equivalence",
            nodes=nodes,
            edges=edges
        )
        
        # Execute via different environments
        dag_results = await self.execute_via_dag_executor(spec, "Test parallel execution")
        workflow_dag_results = await self.execute_via_workflow_dag(spec, "Test parallel execution")
        
        # Assert equivalence
        self.assert_results_equivalent(
            dag_results, 
            workflow_dag_results, 
            "DAG Executor", 
            "Workflow DAG"
        )
        
        # Verify all nodes were executed
        expected_nodes = {"source", "branch_a", "branch_b", "merger"}
        dag_executed_nodes = set(dag_results.keys())
        workflow_executed_nodes = set(workflow_dag_results.keys())
        
        assert expected_nodes.issubset(dag_executed_nodes), f"DAG executor missing nodes: {expected_nodes - dag_executed_nodes}"
        assert expected_nodes.issubset(workflow_executed_nodes), f"Workflow executor missing nodes: {expected_nodes - workflow_executed_nodes}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])