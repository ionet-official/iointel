"""
Tests for WorkflowSpec models and validation.
"""

import pytest
import json
import uuid
from uuid import UUID

from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec,
    NodeSpec,
    NodeData,
    EdgeSpec,
    EdgeData,
    WorkflowRunSummary,
    NodeRunSummary,
    ArtifactRef
)


class TestNodeData:
    """Test cases for NodeData model."""

    def test_node_data_creation(self):
        """Test basic NodeData creation."""
        node_data = NodeData(
            config={"param1": "value1", "param2": 42},
            ins=["input1", "input2"],
            outs=["output1"],
            tool_name="test_tool"
        )
        
        assert node_data.config == {"param1": "value1", "param2": 42}
        assert node_data.ins == ["input1", "input2"]
        assert node_data.outs == ["output1"]
        assert node_data.tool_name == "test_tool"

    def test_node_data_defaults(self):
        """Test NodeData with default values."""
        node_data = NodeData()
        
        assert node_data.config == {}
        assert node_data.ins == []
        assert node_data.outs == []
        assert node_data.tool_name is None
        assert node_data.agent_instructions is None
        assert node_data.workflow_id is None

    def test_node_data_serialization(self):
        """Test NodeData JSON serialization."""
        node_data = NodeData(
            config={"test": True},
            ins=["data"],
            outs=["result"],
            tool_name="example_tool"
        )
        
        json_data = node_data.model_dump()
        expected = {
            "config": {"test": True},
            "ins": ["data"],
            "outs": ["result"],
            "tool_name": "example_tool",
            "agent_instructions": None,
            "tools": None,
            "workflow_id": None
        }
        
        assert json_data == expected

    def test_node_data_agent_instructions(self):
        """Test NodeData with agent instructions."""
        instructions = "Process the input data and extract key insights"
        node_data = NodeData(
            agent_instructions=instructions,
            config={"model": "gpt-4"},
            ins=["data"],
            outs=["insights"]
        )
        
        assert node_data.agent_instructions == instructions
        assert node_data.tool_name is None


class TestNodeSpec:
    """Test cases for NodeSpec model."""

    def test_node_spec_creation(self):
        """Test basic NodeSpec creation."""
        node_spec = NodeSpec(
            id="test_node",
            type="tool",
            label="Test Node",
            data=NodeData(tool_name="test_tool")
        )
        
        assert node_spec.id == "test_node"
        assert node_spec.type == "tool"
        assert node_spec.label == "Test Node"
        assert node_spec.data.tool_name == "test_tool"

    def test_node_spec_types(self):
        """Test all valid node types."""
        valid_types = ["tool", "agent", "workflow_call"]
        
        for node_type in valid_types:
            node_spec = NodeSpec(
                id=f"node_{node_type}",
                type=node_type,
                label=f"Test {node_type}",
                data=NodeData()
            )
            assert node_spec.type == node_type

    def test_node_spec_with_position(self):
        """Test NodeSpec with position data."""
        position = {"x": 100, "y": 200}
        node_spec = NodeSpec(
            id="positioned_node",
            type="tool",
            label="Positioned Node",
            data=NodeData(),
            position=position
        )
        
        assert node_spec.position == position

    def test_node_spec_with_runtime(self):
        """Test NodeSpec with runtime configuration."""
        runtime = {"timeout": 30, "retries": 3}
        node_spec = NodeSpec(
            id="runtime_node",
            type="agent",
            label="Runtime Node",
            data=NodeData(),
            runtime=runtime
        )
        
        assert node_spec.runtime == runtime


class TestEdgeData:
    """Test cases for EdgeData model."""

    def test_edge_data_creation(self):
        """Test basic EdgeData creation."""
        edge_data = EdgeData(condition="status == 'success'")
        assert edge_data.condition == "status == 'success'"

    def test_edge_data_defaults(self):
        """Test EdgeData with default values."""
        edge_data = EdgeData()
        assert edge_data.condition is None

    def test_edge_data_serialization(self):
        """Test EdgeData JSON serialization."""
        edge_data = EdgeData(condition="count > 10")
        json_data = edge_data.model_dump()
        
        assert json_data == {"condition": "count > 10"}


class TestEdgeSpec:
    """Test cases for EdgeSpec model."""

    def test_edge_spec_creation(self):
        """Test basic EdgeSpec creation."""
        edge_spec = EdgeSpec(
            id="test_edge",
            source="node1",
            target="node2"
        )
        
        assert edge_spec.id == "test_edge"
        assert edge_spec.source == "node1"
        assert edge_spec.target == "node2"

    def test_edge_spec_with_handles(self):
        """Test EdgeSpec with handle specifications."""
        edge_spec = EdgeSpec(
            id="handled_edge",
            source="producer",
            target="consumer",
            sourceHandle="output",
            targetHandle="input"
        )
        
        assert edge_spec.sourceHandle == "output"
        assert edge_spec.targetHandle == "input"

    def test_edge_spec_with_condition(self):
        """Test EdgeSpec with conditional logic."""
        edge_spec = EdgeSpec(
            id="conditional_edge",
            source="checker",
            target="processor",
            data=EdgeData(condition="result.status == 'valid'")
        )
        
        assert edge_spec.data.condition == "result.status == 'valid'"


class TestWorkflowSpec:
    """Test cases for WorkflowSpec model."""

    def test_workflow_spec_creation(self):
        """Test basic WorkflowSpec creation."""
        workflow_id = uuid.uuid4()
        
        workflow_spec = WorkflowSpec(
            id=workflow_id,
            rev=1,
            title="Test Workflow",
            description="A test workflow",
            nodes=[
                NodeSpec(
                    id="node1",
                    type="tool",
                    label="Node 1",
                    data=NodeData()
                )
            ],
            edges=[]
        )
        
        assert workflow_spec.id == workflow_id
        assert workflow_spec.rev == 1
        assert workflow_spec.title == "Test Workflow"
        assert workflow_spec.description == "A test workflow"
        assert len(workflow_spec.nodes) == 1
        assert len(workflow_spec.edges) == 0

    def test_workflow_spec_with_metadata(self):
        """Test WorkflowSpec with metadata."""
        metadata = {
            "created_by": "test_user",
            "tags": ["test", "example"],
            "version": "1.0.0"
        }
        
        workflow_spec = WorkflowSpec(
            id=uuid.uuid4(),
            rev=1,
            title="Metadata Workflow",
            nodes=[],
            edges=[],
            metadata=metadata
        )
        
        assert workflow_spec.metadata == metadata

    def test_workflow_spec_serialization(self):
        """Test WorkflowSpec JSON serialization."""
        workflow_spec = WorkflowSpec(
            id=uuid.uuid4(),
            rev=1,
            title="Serialization Test",
            description="Test serialization",
            nodes=[
                NodeSpec(
                    id="test_node",
                    type="tool",
                    label="Test Node",
                    data=NodeData(tool_name="test_tool")
                )
            ],
            edges=[
                EdgeSpec(
                    id="test_edge",
                    source="node1",
                    target="test_node"
                )
            ]
        )
        
        json_str = workflow_spec.model_dump_json()
        parsed = json.loads(json_str)
        
        assert parsed["title"] == "Serialization Test"
        assert parsed["rev"] == 1
        assert len(parsed["nodes"]) == 1
        assert len(parsed["edges"]) == 1

    def test_workflow_spec_deserialization(self):
        """Test WorkflowSpec JSON deserialization."""
        workflow_data = {
            "id": str(uuid.uuid4()),
            "rev": 2,
            "title": "Deserialized Workflow",
            "description": "Test deserialization",
            "nodes": [
                {
                    "id": "deserial_node",
                    "type": "agent",
                    "label": "Deserialized Node",
                    "data": {
                        "config": {"param": "value"},
                        "ins": ["input"],
                        "outs": ["output"],
                        "tool_name": None,
                        "agent_instructions": "Do something",
                        "workflow_id": None
                    }
                }
            ],
            "edges": [],
            "metadata": {}
        }
        
        workflow_spec = WorkflowSpec.model_validate(workflow_data)
        
        assert workflow_spec.title == "Deserialized Workflow"
        assert workflow_spec.rev == 2
        assert len(workflow_spec.nodes) == 1
        assert workflow_spec.nodes[0].data.agent_instructions == "Do something"


class TestWorkflowSpecValidation:
    """Test WorkflowSpec validation methods."""

    def test_validate_structure_valid_workflow(self):
        """Test validation of a valid workflow structure."""
        workflow_spec = WorkflowSpec(
            id=uuid.uuid4(),
            rev=1,
            title="Valid Workflow",
            nodes=[
                NodeSpec(id="node1", type="tool", label="Node 1", data=NodeData(tool_name="test_tool")),
                NodeSpec(id="node2", type="agent", label="Node 2", data=NodeData(agent_instructions="Process the data"))
            ],
            edges=[
                EdgeSpec(id="edge1", source="node1", target="node2")
            ]
        )
        
        issues = workflow_spec.validate_structure()
        assert len(issues) == 0

    def test_validate_structure_duplicate_node_ids(self):
        """Test validation catches duplicate node IDs."""
        workflow_spec = WorkflowSpec(
            id=uuid.uuid4(),
            rev=1,
            title="Duplicate Nodes",
            nodes=[
                NodeSpec(id="duplicate", type="tool", label="Node 1", data=NodeData(tool_name="test_tool")),
                NodeSpec(id="duplicate", type="agent", label="Node 2", data=NodeData(agent_instructions="Process data"))
            ],
            edges=[]
        )
        
        issues = workflow_spec.validate_structure()
        assert len(issues) == 2  # Duplicate IDs and orphaned nodes
        assert any("Duplicate node IDs found" in issue for issue in issues)
        assert any("Orphaned nodes" in issue for issue in issues)

    def test_validate_structure_invalid_edge_source(self):
        """Test validation catches invalid edge sources."""
        workflow_spec = WorkflowSpec(
            id=uuid.uuid4(),
            rev=1,
            title="Invalid Edge Source",
            nodes=[
                NodeSpec(id="valid_node", type="tool", label="Valid Node", data=NodeData(tool_name="test_tool"))
            ],
            edges=[
                EdgeSpec(id="bad_edge", source="nonexistent", target="valid_node")
            ]
        )
        
        issues = workflow_spec.validate_structure()
        assert len(issues) == 1
        assert "unknown source: nonexistent" in issues[0]

    def test_validate_structure_invalid_edge_target(self):
        """Test validation catches invalid edge targets."""
        workflow_spec = WorkflowSpec(
            id=uuid.uuid4(),
            rev=1,
            title="Invalid Edge Target",
            nodes=[
                NodeSpec(id="valid_node", type="tool", label="Valid Node", data=NodeData(tool_name="test_tool"))
            ],
            edges=[
                EdgeSpec(id="bad_edge", source="valid_node", target="nonexistent")
            ]
        )
        
        issues = workflow_spec.validate_structure()
        assert len(issues) == 1
        assert "unknown target: nonexistent" in issues[0]

    def test_validate_structure_orphaned_nodes(self):
        """Test validation catches orphaned nodes."""
        workflow_spec = WorkflowSpec(
            id=uuid.uuid4(),
            rev=1,
            title="Orphaned Nodes",
            nodes=[
                NodeSpec(id="connected1", type="tool", label="Connected 1", data=NodeData(tool_name="test_tool1")),
                NodeSpec(id="connected2", type="agent", label="Connected 2", data=NodeData(agent_instructions="Process data")),
                NodeSpec(id="orphaned", type="tool", label="Orphaned", data=NodeData(tool_name="test_tool2"))
            ],
            edges=[
                EdgeSpec(id="connection", source="connected1", target="connected2")
            ]
        )
        
        issues = workflow_spec.validate_structure()
        assert len(issues) == 1
        assert "Orphaned nodes" in issues[0]
        assert "orphaned" in issues[0]

    def test_validate_structure_single_node_no_orphan_warning(self):
        """Test that single nodes don't trigger orphan warnings."""
        workflow_spec = WorkflowSpec(
            id=uuid.uuid4(),
            rev=1,
            title="Single Node",
            nodes=[
                NodeSpec(id="solo", type="tool", label="Solo Node", data=NodeData(tool_name="test_tool"))
            ],
            edges=[]
        )
        
        issues = workflow_spec.validate_structure()
        assert len(issues) == 0


class TestWorkflowRunModels:
    """Test cases for workflow execution models."""

    def test_artifact_ref_creation(self):
        """Test ArtifactRef creation."""
        artifact = ArtifactRef(
            artifact_id=uuid.uuid4(),
            uri="s3://bucket/artifact.json",
            mime="application/json"
        )
        
        assert isinstance(artifact.artifact_id, UUID)
        assert artifact.uri == "s3://bucket/artifact.json"
        assert artifact.mime == "application/json"

    def test_node_run_summary_creation(self):
        """Test NodeRunSummary creation."""
        summary = NodeRunSummary(
            node_id="test_node",
            status="success",
            started_at="2024-01-01T00:00:00Z",
            finished_at="2024-01-01T00:01:00Z",
            result_preview="Processed 100 records",
            artifacts=[
                ArtifactRef(
                    artifact_id=uuid.uuid4(),
                    uri="file://output.csv",
                    mime="text/csv"
                )
            ]
        )
        
        assert summary.node_id == "test_node"
        assert summary.status == "success"
        assert len(summary.artifacts) == 1

    def test_node_run_summary_with_error(self):
        """Test NodeRunSummary with error status."""
        summary = NodeRunSummary(
            node_id="failed_node",
            status="failed",
            started_at="2024-01-01T00:00:00Z",
            finished_at="2024-01-01T00:00:30Z",
            error_message="Connection timeout"
        )
        
        assert summary.status == "failed"
        assert summary.error_message == "Connection timeout"

    def test_workflow_run_summary_creation(self):
        """Test WorkflowRunSummary creation."""
        run_summary = WorkflowRunSummary(
            workflow_id=uuid.uuid4(),
            run_id=uuid.uuid4(),
            status="success",
            started_at="2024-01-01T00:00:00Z",
            finished_at="2024-01-01T00:05:00Z",
            node_summaries=[
                NodeRunSummary(
                    node_id="node1",
                    status="success",
                    started_at="2024-01-01T00:00:00Z",
                    finished_at="2024-01-01T00:02:00Z"
                ),
                NodeRunSummary(
                    node_id="node2",
                    status="success",
                    started_at="2024-01-01T00:02:00Z",
                    finished_at="2024-01-01T00:05:00Z"
                )
            ],
            total_duration_seconds=300.0
        )
        
        assert run_summary.status == "success"
        assert len(run_summary.node_summaries) == 2
        assert run_summary.total_duration_seconds == 300.0

    def test_workflow_run_summary_partial_failure(self):
        """Test WorkflowRunSummary with partial failure."""
        run_summary = WorkflowRunSummary(
            workflow_id=uuid.uuid4(),
            run_id=uuid.uuid4(),
            status="partial",
            started_at="2024-01-01T00:00:00Z",
            finished_at="2024-01-01T00:03:00Z",
            node_summaries=[
                NodeRunSummary(
                    node_id="success_node",
                    status="success",
                    started_at="2024-01-01T00:00:00Z",
                    finished_at="2024-01-01T00:01:00Z"
                ),
                NodeRunSummary(
                    node_id="failed_node",
                    status="failed",
                    started_at="2024-01-01T00:01:00Z",
                    finished_at="2024-01-01T00:03:00Z",
                    error_message="Processing failed"
                ),
                NodeRunSummary(
                    node_id="skipped_node",
                    status="skipped",
                    started_at="2024-01-01T00:03:00Z",
                    finished_at="2024-01-01T00:03:00Z"
                )
            ]
        )
        
        assert run_summary.status == "partial"
        
        # Check individual statuses
        statuses = [node.status for node in run_summary.node_summaries]
        assert "success" in statuses
        assert "failed" in statuses
        assert "skipped" in statuses


class TestWorkflowSpecEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_workflow(self):
        """Test workflow with no nodes or edges."""
        workflow_spec = WorkflowSpec(
            id=uuid.uuid4(),
            rev=1,
            title="Empty Workflow",
            nodes=[],
            edges=[]
        )
        
        issues = workflow_spec.validate_structure()
        assert len(issues) == 0  # Empty workflows are valid

    def test_workflow_with_complex_config(self):
        """Test workflow with complex nested configuration."""
        complex_config = {
            "database": {
                "connection": {
                    "host": "localhost",
                    "port": 5432,
                    "credentials": {
                        "username": "user",
                        "password_ref": "${SECRET_PASSWORD}"
                    }
                },
                "query": {
                    "sql": "SELECT * FROM users WHERE active = true",
                    "timeout": 30,
                    "retry_policy": {
                        "max_attempts": 3,
                        "backoff": "exponential"
                    }
                }
            },
            "processing": {
                "batch_size": 1000,
                "parallel_workers": 4,
                "output_format": ["json", "csv"]
            }
        }
        
        node_spec = NodeSpec(
            id="complex_node",
            type="tool",
            label="Complex Database Query",
            data=NodeData(
                tool_name="database_query",
                config=complex_config,
                ins=["trigger"],
                outs=["query_results", "metadata"]
            )
        )
        
        workflow_spec = WorkflowSpec(
            id=uuid.uuid4(),
            rev=1,
            title="Complex Config Workflow",
            nodes=[node_spec],
            edges=[]
        )
        
        # Should serialize and deserialize correctly
        json_data = workflow_spec.model_dump_json()
        restored = WorkflowSpec.model_validate_json(json_data)
        
        assert restored.nodes[0].data.config == complex_config

    def test_workflow_with_unicode_content(self):
        """Test workflow with Unicode characters."""
        workflow_spec = WorkflowSpec(
            id=uuid.uuid4(),
            rev=1,
            title="Workflow with émojis 🚀 and ünïcödé",
            description="Processing データ with spéciål characters: 中文, العربية, русский",
            nodes=[
                NodeSpec(
                    id="unicode_node",
                    type="agent",
                    label="Ünïcödé Prócéssör 📊",
                    data=NodeData(
                        agent_instructions="Process the 数据 and extract insights using ИИ",
                        config={"locale": "zh_CN", "encoding": "utf-8"}
                    )
                )
            ],
            edges=[]
        )
        
        # Should handle Unicode correctly
        json_data = workflow_spec.model_dump_json()
        restored = WorkflowSpec.model_validate_json(json_data)
        
        assert "émojis 🚀" in restored.title
        assert "中文" in restored.description
        assert "📊" in restored.nodes[0].label


if __name__ == "__main__":
    pytest.main([__file__])