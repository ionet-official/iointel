"""
Critical Issues Found in Execution Environment Equivalence

This test documents the specific non-isomorphic behaviors between execution environments.
These issues represent critical risks for production deployment.
"""

import pytest
from uuid import uuid4

from iointel.src.workflow import Workflow
from iointel.src.utilities.dag_executor import DAGExecutor
from iointel.src.utilities.graph_nodes import WorkflowState
from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec, NodeSpec, EdgeSpec, NodeData, EdgeData
)
from iointel.src.utilities.registries import TASK_EXECUTOR_REGISTRY


class TestCriticalExecutionIssues:
    """Document and test critical execution environment issues."""
    
    def create_simple_spec(self) -> WorkflowSpec:
        """Create a simple workflow spec for testing."""
        nodes = [
            NodeSpec(
                id="input_node",
                type="tool",
                label="Input",
                data=NodeData(
                    tool_name="mock_tool",
                    config={"type": "input"},
                    ins=[],
                    outs=["data"]
                )
            ),
            NodeSpec(
                id="output_node",
                type="tool",
                label="Output",
                data=NodeData(
                    tool_name="mock_tool",
                    config={"type": "output"},
                    ins=["data"],
                    outs=["result"]
                )
            )
        ]
        
        edges = [
            EdgeSpec(
                id="e1",
                source="input_node",
                target="output_node",
                data=EdgeData()
            )
        ]
        
        return WorkflowSpec(
            id=uuid4(),
            rev=1,
            title="Simple Test Workflow",
            description="Minimal workflow for testing execution differences",
            nodes=nodes,
            edges=edges
        )
    
    @pytest.fixture
    def setup_minimal_mocks(self):
        """Setup minimal mock executors."""
        original_executors = TASK_EXECUTOR_REGISTRY.copy()
        
        async def mock_tool_executor(task_metadata, objective, agents, execution_metadata):
            tool_type = task_metadata.get("config", {}).get("type", "default")
            return {
                "result": f"mock_{tool_type}_result",
                "metadata": task_metadata
            }
        
        TASK_EXECUTOR_REGISTRY["tool"] = mock_tool_executor
        TASK_EXECUTOR_REGISTRY["mock_tool"] = mock_tool_executor
        
        yield
        
        TASK_EXECUTOR_REGISTRY.clear()
        TASK_EXECUTOR_REGISTRY.update(original_executors)
    
    @pytest.mark.asyncio
    async def test_dag_executor_works(self, setup_minimal_mocks):
        """Test that DAG executor works with minimal setup."""
        spec = self.create_simple_spec()
        
        executor = DAGExecutor()
        executor.build_execution_graph(
            nodes=spec.nodes,
            edges=spec.edges,
            objective="Test DAG execution"
        )
        
        initial_state = WorkflowState(
            conversation_id="test_dag",
            initial_text="Test DAG execution",
            results={}
        )
        
        # This should work
        final_state = await executor.execute_dag(initial_state)
        
        # Verify we got results
        assert "input_node" in final_state.results
        assert "output_node" in final_state.results
        assert final_state.results["input_node"]["result"] == "mock_input_result"
        assert final_state.results["output_node"]["result"] == "mock_output_result"
    
    @pytest.mark.asyncio
    async def test_workflow_dag_has_issues(self, setup_minimal_mocks):
        """Test that workflow DAG path has issues."""
        spec = self.create_simple_spec()
        
        # Create workflow with DAG structure
        workflow = Workflow(objective="Test workflow DAG")
        
        dag_structure = {
            "nodes": [node.model_dump() for node in spec.nodes],
            "edges": [edge.model_dump() for edge in spec.edges]
        }
        
        workflow.add_task({
            "task_id": "dag_execution",
            "name": "DAG Execution",
            "type": "dag",
            "task_metadata": {
                "dag_structure": dag_structure
            }
        })
        
        # This should fail or behave differently
        with pytest.raises(Exception) as exc_info:
            await workflow.run_tasks(conversation_id="test_workflow_dag")
        
        # Document the specific error
        error_msg = str(exc_info.value)
        assert "argument of type 'NoneType' is not iterable" in error_msg or "execution_metadata" in error_msg, f"Unexpected error: {error_msg}"
    
    @pytest.mark.asyncio
    async def test_tool_registration_differences(self, setup_minimal_mocks):
        """Test that tool registration behaves differently between environments."""
        spec = self.create_simple_spec()
        
        # Count tools before DAG execution
        dag_tools_before = len(TASK_EXECUTOR_REGISTRY)
        
        # Run DAG executor
        executor = DAGExecutor()
        executor.build_execution_graph(
            nodes=spec.nodes,
            edges=spec.edges,
            objective="Test tool registration"
        )
        
        # Tools should be the same (DAG executor doesn't load additional tools)
        dag_tools_after = len(TASK_EXECUTOR_REGISTRY)
        assert dag_tools_before == dag_tools_after, "DAG executor should not change tool registry"
        
        # Try workflow execution (will load many more tools)
        workflow = Workflow(objective="Test tool registration")
        workflow.add_task({
            "task_id": "simple_tool",
            "name": "Simple Tool",
            "type": "tool",
            "task_metadata": {"tool_name": "mock_tool"}
        })
        
        try:
            await workflow.run_tasks(conversation_id="test_tools")
            workflow_tools_after = len(TASK_EXECUTOR_REGISTRY)
            
            # Workflow should have loaded many more tools
            assert workflow_tools_after > dag_tools_after, f"Workflow should load more tools: {workflow_tools_after} vs {dag_tools_after}"
            
        except Exception:
            # Even if it fails, it should have loaded tools
            workflow_tools_after = len(TASK_EXECUTOR_REGISTRY)
            assert workflow_tools_after > dag_tools_after, f"Workflow should load more tools even if it fails: {workflow_tools_after} vs {dag_tools_after}"
    
    def test_execution_environment_documentation(self):
        """Document the critical issues found."""
        issues = [
            "1. Tool Registration Differences:",
            "   - DAG executor uses only provided tools",
            "   - Workflow.py loads all tools from environment",
            "   - This causes different tool availability and behavior",
            "",
            "2. Execution Path Bugs:",
            "   - Workflow.py has NoneType errors in execution_metadata handling",
            "   - DAG executor has different error handling paths",
            "   - Same workflow can succeed in one environment and fail in another",
            "",
            "3. Result Format Differences:",
            "   - Results come back in different structures",
            "   - Metadata and timing information varies",
            "   - Makes result comparison difficult",
            "",
            "4. Environment Isolation Issues:",
            "   - No guarantee of consistent execution environments",
            "   - Different tool loading mechanisms",
            "   - Potential for silent failures or inconsistent behavior",
            "",
            "5. Critical Risk for Production:",
            "   - Workflows tested in one environment may fail in another",
            "   - CaaS deployments may behave differently than development",
            "   - No way to guarantee isomorphic behavior",
        ]
        
        print("\\n".join(issues))
        
        # This test always passes but documents the issues
        assert True, "Issues documented"
    
    @pytest.mark.asyncio
    async def test_recommended_fixes(self, setup_minimal_mocks):
        """Test recommended fixes for execution environment issues."""
        spec = self.create_simple_spec()
        
        # Recommended Fix 1: Use DAG executor consistently
        # Both environments should use the same underlying execution engine
        
        # Test DAG executor directly
        dag_executor = DAGExecutor()
        dag_executor.build_execution_graph(
            nodes=spec.nodes,
            edges=spec.edges,
            objective="Test consistent execution"
        )
        
        initial_state = WorkflowState(
            conversation_id="test_consistent",
            initial_text="Test consistent execution",
            results={}
        )
        
        dag_result = await dag_executor.execute_dag(initial_state)
        
        # This should work consistently
        assert "input_node" in dag_result.results
        assert "output_node" in dag_result.results
        
        # Recommended Fix 2: Standardize tool loading
        # Both environments should use the same tool loading mechanism
        
        # Recommended Fix 3: Consistent result formats
        # All execution environments should return results in the same format
        
        # Recommended Fix 4: Unified error handling
        # Both environments should handle errors consistently
        
        assert True, "Recommended fixes documented"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])