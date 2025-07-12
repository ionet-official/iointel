"""
Tests for WorkflowConverter functionality.
"""

import pytest
import uuid
import yaml
from unittest.mock import patch, MagicMock

from iointel.src.agent_methods.workflow_converter import (
    WorkflowConverter,
    spec_to_yaml,
    spec_to_definition
)
from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec,
    NodeSpec,
    NodeData,
    EdgeSpec,
    EdgeData
)
from iointel.src.agent_methods.data_models.datamodels import (
    WorkflowDefinition,
    TaskDefinition,
    AgentParams
)


@pytest.fixture
def sample_workflow_spec():
    """Sample WorkflowSpec for testing conversion."""
    return WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Test Conversion Workflow",
        description="A workflow for testing conversion functionality",
        nodes=[
            NodeSpec(
                id="fetch_data",
                type="tool",
                label="Fetch Data",
                data=NodeData(
                    tool_name="api_client",
                    config={"url": "https://api.example.com/data", "timeout": 30},
                    ins=[],
                    outs=["raw_data", "status"]
                )
            ),
            NodeSpec(
                id="process_data",
                type="agent",
                label="Process Data",
                data=NodeData(
                    agent_instructions="Analyze the raw data and extract key insights",
                    config={"model": "gpt-4", "temperature": 0.3},
                    ins=["raw_data"],
                    outs=["processed_data", "insights"]
                )
            ),
            NodeSpec(
                id="store_results",
                type="tool",
                label="Store Results",
                data=NodeData(
                    tool_name="database_writer",
                    config={"table": "results", "batch_size": 100},
                    ins=["processed_data"],
                    outs=["success", "record_count"]
                )
            ),
            NodeSpec(
                id="sub_workflow",
                type="workflow_call",
                label="Run Sub-Workflow",
                data=NodeData(
                    workflow_id="cleanup_workflow_v1",
                    config={"mode": "full_cleanup"},
                    ins=["success"],
                    outs=["cleanup_status"]
                )
            )
        ],
        edges=[
            EdgeSpec(
                id="fetch_to_process",
                source="fetch_data",
                target="process_data",
                sourceHandle="raw_data",
                targetHandle="raw_data",
                data=EdgeData(condition="status == 'success'")
            ),
            EdgeSpec(
                id="process_to_store",
                source="process_data",
                target="store_results",
                sourceHandle="processed_data",
                targetHandle="processed_data"
            ),
            EdgeSpec(
                id="store_to_cleanup",
                source="store_results",
                target="sub_workflow",
                sourceHandle="success",
                targetHandle="success",
                data=EdgeData(condition="record_count > 0")
            )
        ]
    )


@pytest.fixture
def default_agents():
    """Sample default agents for testing."""
    return [
        AgentParams(
            name="DefaultAgent",
            instructions="A default agent for processing tasks"
        )
    ]


class TestWorkflowConverter:
    """Test cases for WorkflowConverter class."""

    def test_converter_initialization(self, default_agents):
        """Test WorkflowConverter initialization."""
        converter = WorkflowConverter(
            default_agents=default_agents,
            default_timeout=120,
            default_retries=5,
            default_client_mode=False
        )
        
        assert converter.default_agents == default_agents
        assert converter.default_timeout == 120
        assert converter.default_retries == 5
        assert not converter.default_client_mode

    def test_converter_default_initialization(self):
        """Test WorkflowConverter with default parameters."""
        converter = WorkflowConverter()
        
        assert converter.default_agents == []
        assert converter.default_timeout == 60
        assert converter.default_retries == 3
        assert converter.default_client_mode

    @patch('iointel.src.agent_methods.workflow_converter.TOOLS_REGISTRY')
    def test_convert_basic_workflow(self, mock_registry, sample_workflow_spec, default_agents):
        """Test converting a basic workflow."""
        # Mock tool registry
        mock_registry.__contains__ = MagicMock(side_effect=lambda x: x in ["api_client", "database_writer"])
        
        converter = WorkflowConverter(default_agents=default_agents)
        result = converter.convert(sample_workflow_spec)
        
        assert isinstance(result, WorkflowDefinition)
        assert result.name == "Test Conversion Workflow"
        assert result.objective == "A workflow for testing conversion functionality"
        assert len(result.tasks) == 4  # All 4 nodes converted to tasks

    def test_convert_node_to_task_tool_type(self):
        """Test converting a tool node to task."""
        converter = WorkflowConverter()
        
        node = NodeSpec(
            id="test_tool",
            type="tool",
            label="Test Tool",
            data=NodeData(
                tool_name="test_tool_name",
                config={"param1": "value1"},
                ins=["input"],
                outs=["output"]
            )
        )
        
        task = converter._convert_node_to_task(node, [])
        
        assert isinstance(task, TaskDefinition)
        assert task.task_id == "test_tool"
        assert task.name == "Test Tool"
        assert task.type == "tool"
        assert task.task_metadata["tool_name"] == "test_tool_name"
        assert task.task_metadata["config"] == {"param1": "value1"}
        assert task.task_metadata["ports"]["inputs"] == ["input"]
        assert task.task_metadata["ports"]["outputs"] == ["output"]

    def test_convert_node_to_task_agent_type(self, default_agents):
        """Test converting an agent node to task."""
        converter = WorkflowConverter(default_agents=default_agents)
        
        node = NodeSpec(
            id="test_agent",
            type="agent",
            label="Test Agent",
            data=NodeData(
                agent_instructions="Do something intelligent",
                model="gpt-4",
                config={},
                ins=["data"],
                outs=["result"]
            )
        )
        
        task = converter._convert_node_to_task(node, [])
        
        assert task.type == "agent"
        assert task.task_metadata["agent_instructions"] == "Do something intelligent"
        # Agent should be customized with node-specific instructions, not use defaults
        assert len(task.agents) == 1
        agent = task.agents[0]
        assert agent.name == "agent_test_agent"
        assert agent.instructions == "Do something intelligent"
        assert agent.model == "gpt-4"

    def test_convert_node_to_task_workflow_call_type(self):
        """Test converting a workflow_call node to task."""
        converter = WorkflowConverter()
        
        node = NodeSpec(
            id="test_workflow_call",
            type="workflow_call",
            label="Test Workflow Call",
            data=NodeData(
                workflow_id="sub_workflow_v2",
                config={"mode": "test"},
                ins=["trigger"],
                outs=["status"]
            )
        )
        
        task = converter._convert_node_to_task(node, [])
        
        assert task.type == "workflow_call"
        assert task.task_metadata["workflow_id"] == "sub_workflow_v2"

    def test_convert_with_edge_conditions(self, sample_workflow_spec):
        """Test conversion preserves edge conditions."""
        converter = WorkflowConverter()
        result = converter.convert(sample_workflow_spec)
        
        # Find the task that corresponds to process_data node (target of conditional edge)
        process_task = next(task for task in result.tasks if task.task_id == "process_data")
        
        # Should have preconditions from incoming edge
        assert "preconditions" in process_task.execution_metadata
        assert "status == 'success'" in process_task.execution_metadata["preconditions"]

    @patch('iointel.src.agent_methods.workflow_converter.TOOLS_REGISTRY')
    def test_convert_with_unknown_tool(self, mock_registry, caplog):
        """Test conversion handles unknown tools gracefully."""
        mock_registry.__contains__ = MagicMock(return_value=False)  # No tools exist
        
        converter = WorkflowConverter()
        
        workflow_spec = WorkflowSpec(
            id=uuid.uuid4(),
            rev=1,
            title="Unknown Tool Workflow",
            nodes=[
                NodeSpec(
                    id="unknown_tool",
                    type="tool",
                    label="Unknown Tool",
                    data=NodeData(tool_name="nonexistent_tool")
                )
            ],
            edges=[]
        )
        
        result = converter.convert(workflow_spec)
        
        # Should still convert successfully
        assert isinstance(result, WorkflowDefinition)
        assert len(result.tasks) == 1
        
        # Should log a warning
        assert "not found in registry" in caplog.text

    def test_build_edge_map(self):
        """Test building edge map for dependency tracking."""
        converter = WorkflowConverter()
        
        edges = [
            EdgeSpec(id="e1", source="a", target="b"),
            EdgeSpec(id="e2", source="a", target="c"),
            EdgeSpec(id="e3", source="b", target="c")
        ]
        
        edge_map = converter._build_edge_map(edges)
        
        assert "b" in edge_map
        assert "c" in edge_map
        assert len(edge_map["c"]) == 2  # Two edges target node c
        assert edge_map["b"][0].source == "a"

    def test_get_agents_for_node_tool(self, default_agents):
        """Test agent assignment for tool nodes."""
        converter = WorkflowConverter(default_agents=default_agents)
        
        tool_node = NodeSpec(
            id="tool_node",
            type="tool",
            label="Tool Node",
            data=NodeData()
        )
        
        agents = converter._get_agents_for_node(tool_node)
        assert agents is None  # Tool nodes don't need agents

    def test_get_agents_for_node_agent(self, default_agents):
        """Test agent assignment for agent nodes."""
        converter = WorkflowConverter(default_agents=default_agents)
        
        agent_node = NodeSpec(
            id="agent_node",
            type="agent",
            label="Agent Node",
            data=NodeData()
        )
        
        agents = converter._get_agents_for_node(agent_node)
        assert agents == default_agents

    def test_get_agents_for_node_workflow_call(self, default_agents):
        """Test agent assignment for workflow_call nodes."""
        converter = WorkflowConverter(default_agents=default_agents)
        
        workflow_node = NodeSpec(
            id="workflow_node",
            type="workflow_call",
            label="Workflow Node",
            data=NodeData()
        )
        
        agents = converter._get_agents_for_node(workflow_node)
        assert agents == default_agents


class TestWorkflowSpecConversionMethods:
    """Test conversion methods on WorkflowSpec itself."""

    @patch('iointel.src.agent_methods.workflow_converter.spec_to_definition')
    def test_to_workflow_definition(self, mock_spec_to_def, sample_workflow_spec):
        """Test WorkflowSpec.to_workflow_definition method."""
        mock_workflow_def = MagicMock()
        mock_spec_to_def.return_value = mock_workflow_def
        
        result = sample_workflow_spec.to_workflow_definition(
            agents=["test_agent"],
            timeout=90
        )
        
        mock_spec_to_def.assert_called_once_with(
            sample_workflow_spec,
            agents=["test_agent"],
            timeout=90
        )
        assert result == mock_workflow_def

    @patch('iointel.src.agent_methods.workflow_converter.spec_to_yaml')
    def test_to_yaml(self, mock_spec_to_yaml, sample_workflow_spec):
        """Test WorkflowSpec.to_yaml method."""
        mock_spec_to_yaml.return_value = "yaml_content"
        
        result = sample_workflow_spec.to_yaml(client_mode=False)
        
        mock_spec_to_yaml.assert_called_once_with(
            sample_workflow_spec,
            client_mode=False
        )
        assert result == "yaml_content"


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    @patch('iointel.src.agent_methods.workflow_converter.WorkflowConverter')
    def test_spec_to_definition_function(self, mock_converter_class, sample_workflow_spec):
        """Test spec_to_definition convenience function."""
        mock_converter = MagicMock()
        mock_converter_class.return_value = mock_converter
        mock_converter.convert.return_value = "converted_workflow"
        
        result = spec_to_definition(
            sample_workflow_spec,
            agents=["agent1"],
            timeout=120
        )
        
        mock_converter_class.assert_called_once_with(
            default_agents=["agent1"],
            timeout=120
        )
        mock_converter.convert.assert_called_once_with(sample_workflow_spec)
        assert result == "converted_workflow"

    @patch('iointel.src.agent_methods.workflow_converter.WorkflowConverter')
    @patch('yaml.safe_dump')
    def test_spec_to_yaml_function(self, mock_yaml_dump, mock_converter_class, sample_workflow_spec):
        """Test spec_to_yaml convenience function."""
        mock_converter = MagicMock()
        mock_converter_class.return_value = mock_converter
        
        mock_workflow_def = MagicMock()
        mock_workflow_def.model_dump.return_value = {"workflow": "data"}
        mock_converter.convert.return_value = mock_workflow_def
        
        mock_yaml_dump.return_value = "yaml_output"
        
        result = spec_to_yaml(sample_workflow_spec, retries=5)
        
        mock_converter_class.assert_called_once_with(retries=5)
        mock_converter.convert.assert_called_once_with(sample_workflow_spec)
        mock_workflow_def.model_dump.assert_called_once_with(mode="json")
        mock_yaml_dump.assert_called_once_with({"workflow": "data"}, sort_keys=False)
        assert result == "yaml_output"


class TestWorkflowConverterIntegration:
    """Integration tests for workflow conversion."""

    def test_full_conversion_pipeline(self, sample_workflow_spec, default_agents):
        """Test complete conversion from WorkflowSpec to YAML."""
        converter = WorkflowConverter(
            default_agents=default_agents,
            default_timeout=90,
            default_retries=2
        )
        
        # Convert to WorkflowDefinition
        workflow_def = converter.convert(sample_workflow_spec)
        
        # Verify structure
        assert isinstance(workflow_def, WorkflowDefinition)
        assert workflow_def.name == sample_workflow_spec.title
        assert workflow_def.objective == sample_workflow_spec.description
        assert len(workflow_def.tasks) == len(sample_workflow_spec.nodes)
        
        # Convert to YAML
        yaml_output = spec_to_yaml(sample_workflow_spec, default_agents=default_agents)
        
        # Parse YAML to verify structure
        yaml_data = yaml.safe_load(yaml_output)
        assert yaml_data["name"] == sample_workflow_spec.title
        assert "tasks" in yaml_data
        assert len(yaml_data["tasks"]) == len(sample_workflow_spec.nodes)

    def test_conversion_preserves_node_types(self, default_agents):
        """Test that conversion preserves different node types correctly."""
        workflow_spec = WorkflowSpec(
            id=uuid.uuid4(),
            rev=1,
            title="Multi-Type Workflow",
            nodes=[
                NodeSpec(
                    id="tool_node",
                    type="tool",
                    label="Tool Node",
                    data=NodeData(tool_name="test_tool")
                ),
                NodeSpec(
                    id="agent_node",
                    type="agent",
                    label="Agent Node",
                    data=NodeData(agent_instructions="Process data")
                ),
                NodeSpec(
                    id="workflow_node",
                    type="workflow_call",
                    label="Workflow Node",
                    data=NodeData(workflow_id="sub_workflow")
                )
            ],
            edges=[]
        )
        
        converter = WorkflowConverter(default_agents=default_agents)
        result = converter.convert(workflow_spec)
        
        # Verify each node type was converted correctly
        tasks_by_type = {task.type: task for task in result.tasks}
        
        assert "tool" in tasks_by_type
        assert "agent" in tasks_by_type
        assert "workflow_call" in tasks_by_type
        
        # Verify type-specific metadata
        assert "tool_name" in tasks_by_type["tool"].task_metadata
        assert "agent_instructions" in tasks_by_type["agent"].task_metadata
        assert "workflow_id" in tasks_by_type["workflow_call"].task_metadata

    def test_conversion_with_complex_edges(self):
        """Test conversion with complex edge conditions."""
        workflow_spec = WorkflowSpec(
            id=uuid.uuid4(),
            rev=1,
            title="Complex Edges Workflow",
            nodes=[
                NodeSpec(id="node1", type="tool", label="Node 1", data=NodeData()),
                NodeSpec(id="node2", type="tool", label="Node 2", data=NodeData()),
                NodeSpec(id="node3", type="tool", label="Node 3", data=NodeData()),
                NodeSpec(id="node4", type="tool", label="Node 4", data=NodeData())
            ],
            edges=[
                EdgeSpec(
                    id="conditional1",
                    source="node1",
                    target="node2",
                    data=EdgeData(condition="status == 'success' and count > 10")
                ),
                EdgeSpec(
                    id="conditional2",
                    source="node1",
                    target="node3",
                    data=EdgeData(condition="status == 'error'")
                ),
                EdgeSpec(
                    id="merge",
                    source="node2",
                    target="node4"
                ),
                EdgeSpec(
                    id="merge2",
                    source="node3",
                    target="node4"
                )
            ]
        )
        
        converter = WorkflowConverter()
        result = converter.convert(workflow_spec)
        
        # Check that conditions are preserved in execution metadata
        node2_task = next(task for task in result.tasks if task.task_id == "node2")
        node3_task = next(task for task in result.tasks if task.task_id == "node3")
        node4_task = next(task for task in result.tasks if task.task_id == "node4")
        
        assert "status == 'success' and count > 10" in node2_task.execution_metadata.get("preconditions", [])
        assert "status == 'error'" in node3_task.execution_metadata.get("preconditions", [])
        
        # Node4 should have no conditions (unconditional edges)
        node4_conditions = node4_task.execution_metadata.get("preconditions", [])
        assert len([c for c in node4_conditions if c]) == 0  # No non-empty conditions


class TestWorkflowConverterErrorHandling:
    """Test error handling in workflow conversion."""

    def test_conversion_with_empty_workflow(self):
        """Test converting an empty workflow."""
        empty_workflow = WorkflowSpec(
            id=uuid.uuid4(),
            rev=1,
            title="Empty Workflow",
            nodes=[],
            edges=[]
        )
        
        converter = WorkflowConverter()
        result = converter.convert(empty_workflow)
        
        assert isinstance(result, WorkflowDefinition)
        assert result.name == "Empty Workflow"
        assert len(result.tasks) == 0

    def test_conversion_with_malformed_node_data(self):
        """Test conversion handles node data with empty values gracefully."""
        workflow_spec = WorkflowSpec(
            id=uuid.uuid4(),
            rev=1,
            title="Minimal Data Workflow",
            nodes=[
                NodeSpec(
                    id="minimal_node",
                    type="tool",
                    label="Minimal Node",
                    data=NodeData()  # Using defaults: empty dicts and lists
                )
            ],
            edges=[]
        )
        
        converter = WorkflowConverter()
        
        # Should handle empty/default values gracefully
        result = converter.convert(workflow_spec)
        assert isinstance(result, WorkflowDefinition)
        assert len(result.tasks) == 1
        
        # Check the task was created with default empty values
        task = result.tasks[0]
        assert task.task_id == "minimal_node"
        assert task.task_metadata["config"] == {}
        assert task.task_metadata["ports"]["inputs"] == []
        assert task.task_metadata["ports"]["outputs"] == []


if __name__ == "__main__":
    pytest.main([__file__])