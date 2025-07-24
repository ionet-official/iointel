"""
Tests for WorkflowPlanner CLI functionality.
"""

import pytest
import io
from unittest.mock import patch, MagicMock, AsyncMock
from contextlib import redirect_stdout

from iointel.src.cli.run_workflow_planner import (
    render_workflow_ascii,
    render_workflow_panel,
    interactive_workflow_planner
)
from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec,
    NodeSpec,
    NodeData,
    EdgeSpec,
    EdgeData
)


@pytest.fixture
def sample_cli_workflow():
    """Sample workflow for CLI testing."""
    return WorkflowSpec(
        id="test-uuid",
        rev=1,
        title="CLI Test Workflow",
        description="A workflow for testing CLI functionality",
        nodes=[
            NodeSpec(
                id="start_node",
                type="agent",
                label="Start Process",
                data=NodeData(
                    agent_instructions="Use the data_fetcher tool to complete this task",
                    tools=["data_fetcher"],
                    config={"source": "database", "limit": 100},
                    ins=[],
                    outs=["data", "status"]
                )
            ),
            NodeSpec(
                id="process_node",
                type="agent",
                label="Process Data",
                data=NodeData(
                    agent_instructions="Analyze the fetched data and extract insights",
                    config={"model": "gpt-4", "temperature": 0.5},
                    ins=["data"],
                    outs=["insights", "summary"]
                )
            ),
            NodeSpec(
                id="end_node",
                type="agent",
                label="Store Results",
                data=NodeData(
                    agent_instructions="Use the result_store tool to complete this task",
                    tools=["result_store"],
                    config={"destination": "s3://results/", "format": "json"},
                    ins=["insights"],
                    outs=["success"]
                )
            )
        ],
        edges=[
            EdgeSpec(
                id="start_to_process",
                source="start_node",
                target="process_node",
                sourceHandle="data",
                targetHandle="data",
                data=EdgeData(condition="status == 'success'")
            ),
            EdgeSpec(
                id="process_to_end",
                source="process_node",
                target="end_node",
                sourceHandle="insights",
                targetHandle="insights"
            )
        ]
    )


class TestWorkflowASCIIRendering:
    """Test ASCII rendering functionality."""

    def test_render_workflow_ascii_basic(self, sample_cli_workflow):
        """Test basic ASCII workflow rendering."""
        ascii_output = render_workflow_ascii(sample_cli_workflow)
        
        assert "CLI Test Workflow" in ascii_output
        assert "A workflow for testing CLI functionality" in ascii_output
        assert "🔧 Start Process" in ascii_output
        assert "🤖 Process Data" in ascii_output
        assert "🔧 Store Results" in ascii_output

    def test_render_workflow_ascii_with_conditions(self, sample_cli_workflow):
        """Test ASCII rendering includes edge conditions."""
        ascii_output = render_workflow_ascii(sample_cli_workflow)
        
        assert "[if: status == 'success']" in ascii_output

    def test_render_workflow_ascii_empty(self):
        """Test ASCII rendering of empty workflow."""
        empty_workflow = WorkflowSpec(
            id="empty-uuid",
            rev=1,
            title="Empty Workflow",
            nodes=[],
            edges=[]
        )
        
        ascii_output = render_workflow_ascii(empty_workflow)
        assert "📋 Empty workflow" in ascii_output

    def test_render_workflow_ascii_single_node(self):
        """Test ASCII rendering of single node workflow."""
        single_node_workflow = WorkflowSpec(
            id="single-uuid",
            rev=1,
            title="Single Node",
            nodes=[
                NodeSpec(
                    id="solo",
                    type="agent",
                    label="Solo Node",
                    data=NodeData(
                    agent_instructions="Use the solo_tool tool to complete this task",tools=["solo_tool"])
                )
            ],
            edges=[]
        )
        
        ascii_output = render_workflow_ascii(single_node_workflow)
        assert "🔧 Solo Node (solo)" in ascii_output

    def test_render_workflow_ascii_complex_branching(self):
        """Test ASCII rendering with complex branching."""
        branching_workflow = WorkflowSpec(
            id="branch-uuid",
            rev=1,
            title="Branching Workflow",
            nodes=[
                NodeSpec(id="root", type="agent", label="Root", data=NodeData()),
                NodeSpec(id="branch1", type="agent", label="Branch 1", data=NodeData()),
                NodeSpec(id="branch2", type="agent", label="Branch 2", data=NodeData()),
                NodeSpec(id="merge", type="agent", label="Merge", data=NodeData())
            ],
            edges=[
                EdgeSpec(id="e1", source="root", target="branch1"),
                EdgeSpec(id="e2", source="root", target="branch2"),
                EdgeSpec(id="e3", source="branch1", target="merge"),
                EdgeSpec(id="e4", source="branch2", target="merge")
            ]
        )
        
        ascii_output = render_workflow_ascii(branching_workflow)
        
        # Should show branching structure
        assert "Root" in ascii_output
        assert "Branch 1" in ascii_output
        assert "Branch 2" in ascii_output
        assert "Merge" in ascii_output

    def test_render_workflow_panel_with_rich(self, sample_cli_workflow):
        """Test Rich panel rendering."""
        
        panel = render_workflow_panel(sample_cli_workflow)
        
        # Should return a Rich Panel object
        assert hasattr(panel, 'renderable')
        assert "CLI Test Workflow" in str(panel)


class TestWorkflowCLIInteraction:
    """Test CLI interaction functionality."""

    @patch('iointel.src.cli.run_workflow_planner.input')
    @patch('iointel.src.cli.run_workflow_planner.load_tools_from_env')
    @patch('iointel.src.cli.run_workflow_planner.AsyncMemory')
    @patch('iointel.src.cli.run_workflow_planner.WorkflowPlanner')
    async def test_interactive_planner_help_command(
        self, 
        mock_planner_class,
        mock_memory_class,
        mock_load_tools,
        mock_input
    ):
        """Test help command in interactive planner."""
        # Setup mocks
        mock_input.side_effect = ["help", "exit"]
        mock_load_tools.return_value = []
        mock_memory_instance = AsyncMock()
        mock_memory_class.return_value = mock_memory_instance
        mock_planner_instance = MagicMock()
        mock_planner_class.return_value = mock_planner_instance
        
        # Capture output
        output = io.StringIO()
        
        with redirect_stdout(output):
            await interactive_workflow_planner()
        
        output_text = output.getvalue()
        assert "Available commands:" in output_text
        assert "help - Show this help message" in output_text

    @patch('iointel.src.cli.run_workflow_planner.input')
    @patch('iointel.src.cli.run_workflow_planner.load_tools_from_env')
    @patch('iointel.src.cli.run_workflow_planner.AsyncMemory')
    @patch('iointel.src.cli.run_workflow_planner.WorkflowPlanner')
    async def test_interactive_planner_tools_command(
        self,
        mock_planner_class,
        mock_memory_class,
        mock_load_tools,
        mock_input
    ):
        """Test tools command in interactive planner."""
        # Setup mocks
        mock_input.side_effect = ["tools", "exit"]
        mock_load_tools.return_value = ["tool1", "tool2"]
        
        # Mock TOOLS_REGISTRY
        with patch('iointel.src.cli.run_workflow_planner.TOOLS_REGISTRY', {
            "tool1": MagicMock(description="First tool"),
            "tool2": MagicMock(description="Second tool")
        }):
            mock_memory_instance = AsyncMock()
            mock_memory_class.return_value = mock_memory_instance
            mock_planner_instance = MagicMock()
            mock_planner_class.return_value = mock_planner_instance
            
            # Capture output
            output = io.StringIO()
            
            with redirect_stdout(output):
                await interactive_workflow_planner()
            
            output_text = output.getvalue()
            assert "Available Tools" in output_text

    @patch('iointel.src.cli.run_workflow_planner.input')
    @patch('iointel.src.cli.run_workflow_planner.load_tools_from_env')
    @patch('iointel.src.cli.run_workflow_planner.AsyncMemory')
    @patch('iointel.src.cli.run_workflow_planner.WorkflowPlanner')
    async def test_interactive_planner_workflow_generation(
        self,
        mock_planner_class,
        mock_memory_class,
        mock_load_tools,
        mock_input,
        sample_cli_workflow
    ):
        """Test workflow generation in interactive planner."""
        # Setup mocks
        mock_input.side_effect = ["create a simple workflow", "exit"]
        mock_load_tools.return_value = []
        mock_memory_instance = AsyncMock()
        mock_memory_class.return_value = mock_memory_instance
        
        mock_planner_instance = MagicMock()
        mock_planner_instance.generate_workflow = AsyncMock(return_value=sample_cli_workflow)
        mock_planner_class.return_value = mock_planner_instance
        
        # Capture output
        output = io.StringIO()
        
        with redirect_stdout(output):
            await interactive_workflow_planner()
        
        output_text = output.getvalue()
        assert "Workflow generated!" in output_text
        assert "CLI Test Workflow" in output_text

    @patch('iointel.src.cli.run_workflow_planner.input')
    @patch('iointel.src.cli.run_workflow_planner.load_tools_from_env')
    @patch('iointel.src.cli.run_workflow_planner.AsyncMemory')
    @patch('iointel.src.cli.run_workflow_planner.WorkflowPlanner')
    async def test_interactive_planner_show_command(
        self,
        mock_planner_class,
        mock_memory_class,
        mock_load_tools,
        mock_input,
        sample_cli_workflow
    ):
        """Test show command displays current workflow."""
        # Setup workflow generation first, then show
        mock_input.side_effect = ["create workflow", "show", "exit"]
        mock_load_tools.return_value = []
        mock_memory_instance = AsyncMock()
        mock_memory_class.return_value = mock_memory_instance
        
        mock_planner_instance = MagicMock()
        mock_planner_instance.generate_workflow = AsyncMock(return_value=sample_cli_workflow)
        mock_planner_class.return_value = mock_planner_instance
        
        # Capture output
        output = io.StringIO()
        
        with redirect_stdout(output):
            await interactive_workflow_planner()
        
        output_text = output.getvalue()
        # Should show workflow twice: once after generation, once after 'show' command
        assert output_text.count("CLI Test Workflow") >= 2

    @patch('iointel.src.cli.run_workflow_planner.input')
    @patch('iointel.src.cli.run_workflow_planner.load_tools_from_env')
    @patch('iointel.src.cli.run_workflow_planner.AsyncMemory')
    @patch('iointel.src.cli.run_workflow_planner.WorkflowPlanner')
    async def test_interactive_planner_clear_command(
        self,
        mock_planner_class,
        mock_memory_class,
        mock_load_tools,
        mock_input,
        sample_cli_workflow
    ):
        """Test clear command removes current workflow."""
        # Generate workflow, then clear it
        mock_input.side_effect = ["create workflow", "clear", "show", "exit"]
        mock_load_tools.return_value = []
        mock_memory_instance = AsyncMock()
        mock_memory_class.return_value = mock_memory_instance
        
        mock_planner_instance = MagicMock()
        mock_planner_instance.generate_workflow = AsyncMock(return_value=sample_cli_workflow)
        mock_planner_class.return_value = mock_planner_instance
        
        # Capture output
        output = io.StringIO()
        
        with redirect_stdout(output):
            await interactive_workflow_planner()
        
        output_text = output.getvalue()
        assert "Workflow cleared" in output_text
        assert "No workflow to display" in output_text

    @patch('iointel.src.cli.run_workflow_planner.input')
    @patch('iointel.src.cli.run_workflow_planner.load_tools_from_env')
    @patch('iointel.src.cli.run_workflow_planner.AsyncMemory')
    @patch('iointel.src.cli.run_workflow_planner.WorkflowPlanner')
    @patch('builtins.open', new_callable=MagicMock)
    async def test_interactive_planner_save_command(
        self,
        mock_open,
        mock_planner_class,
        mock_memory_class,
        mock_load_tools,
        mock_input,
        sample_cli_workflow
    ):
        """Test save command saves workflow to file."""
        # Generate workflow, then save it
        mock_input.side_effect = ["create workflow", "save test_workflow", "exit"]
        mock_load_tools.return_value = []
        mock_memory_instance = AsyncMock()
        mock_memory_class.return_value = mock_memory_instance
        
        mock_planner_instance = MagicMock()
        mock_planner_instance.generate_workflow = AsyncMock(return_value=sample_cli_workflow)
        mock_planner_class.return_value = mock_planner_instance
        
        # Mock file operations
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Capture output
        output = io.StringIO()
        
        with redirect_stdout(output):
            await interactive_workflow_planner()
        
        output_text = output.getvalue()
        assert "Workflow saved to test_workflow.json" in output_text
        mock_open.assert_called_with("test_workflow.json", "w")

    @patch('iointel.src.cli.run_workflow_planner.input')
    @patch('iointel.src.cli.run_workflow_planner.load_tools_from_env')
    @patch('iointel.src.cli.run_workflow_planner.AsyncMemory')
    @patch('iointel.src.cli.run_workflow_planner.WorkflowPlanner')
    async def test_interactive_planner_error_handling(
        self,
        mock_planner_class,
        mock_memory_class,
        mock_load_tools,
        mock_input
    ):
        """Test error handling in interactive planner."""
        # Simulate workflow generation error
        mock_input.side_effect = ["create invalid workflow", "exit"]
        mock_load_tools.return_value = []
        mock_memory_instance = AsyncMock()
        mock_memory_class.return_value = mock_memory_instance
        
        mock_planner_instance = MagicMock()
        mock_planner_instance.generate_workflow = AsyncMock(
            side_effect=Exception("Workflow generation failed")
        )
        mock_planner_class.return_value = mock_planner_instance
        
        # Capture output
        output = io.StringIO()
        
        with redirect_stdout(output):
            await interactive_workflow_planner()
        
        output_text = output.getvalue()
        assert "Error generating workflow" in output_text
        assert "Workflow generation failed" in output_text

    @patch('iointel.src.cli.run_workflow_planner.input')
    @patch('iointel.src.cli.run_workflow_planner.load_tools_from_env')
    @patch('iointel.src.cli.run_workflow_planner.AsyncMemory')
    @patch('iointel.src.cli.run_workflow_planner.WorkflowPlanner')
    async def test_interactive_planner_workflow_refinement(
        self,
        mock_planner_class,
        mock_memory_class,
        mock_load_tools,
        mock_input,
        sample_cli_workflow
    ):
        """Test workflow refinement in interactive planner."""
        # Create workflow, then refine it
        refined_workflow = sample_cli_workflow.model_copy()
        refined_workflow.rev = 2
        refined_workflow.title = "Refined CLI Test Workflow"
        
        mock_input.side_effect = [
            "create initial workflow",
            "add error handling and logging",
            "exit"
        ]
        mock_load_tools.return_value = []
        mock_memory_instance = AsyncMock()
        mock_memory_class.return_value = mock_memory_instance
        
        mock_planner_instance = MagicMock()
        mock_planner_instance.generate_workflow = AsyncMock(return_value=sample_cli_workflow)
        mock_planner_instance.refine_workflow = AsyncMock(return_value=refined_workflow)
        mock_planner_class.return_value = mock_planner_instance
        
        # Capture output
        output = io.StringIO()
        
        with redirect_stdout(output):
            await interactive_workflow_planner()
        
        output_text = output.getvalue()
        assert "Workflow refined!" in output_text
        assert "Refined CLI Test Workflow" in output_text


class TestWorkflowCLIUtilities:
    """Test CLI utility functions."""

    def test_ascii_rendering_node_icons(self):
        """Test that different node types get correct icons."""
        workflows = {
            "tool": WorkflowSpec(
                id="tool-test",
                rev=1,
                title="Tool Test",
                nodes=[NodeSpec(id="t", type="agent", label="Tool", data=NodeData())],
                edges=[]
            ),
            "agent": WorkflowSpec(
                id="agent-test",
                rev=1,
                title="Agent Test",
                nodes=[NodeSpec(id="a", type="agent", label="Agent", data=NodeData())],
                edges=[]
            ),
            "workflow_call": WorkflowSpec(
                id="wf-test",
                rev=1,
                title="Workflow Test",
                nodes=[NodeSpec(id="w", type="workflow_call", label="Workflow", data=NodeData())],
                edges=[]
            )
        }
        
        for node_type, workflow in workflows.items():
            ascii_output = render_workflow_ascii(workflow)
            
            if node_type == "tool":
                assert "🔧" in ascii_output
            elif node_type == "agent":
                assert "🤖" in ascii_output
            elif node_type == "workflow_call":
                assert "📞" in ascii_output

    def test_ascii_rendering_config_display(self):
        """Test that node configurations are displayed in ASCII."""
        workflow = WorkflowSpec(
            id="config-test",
            rev=1,
            title="Config Test",
            nodes=[
                NodeSpec(
                    id="configured_node",
                    type="agent",
                    label="Configured Node",
                    data=NodeData(
                        config={
                            "database_url": "postgresql://localhost:5432/db",
                            "batch_size": 1000,
                            "timeout": 30
                        }
                    )
                )
            ],
            edges=[]
        )
        
        ascii_output = render_workflow_ascii(workflow)
        
        # Should show some config parameters (truncated)
        assert "database_url" in ascii_output or "batch_size" in ascii_output

    def test_ascii_rendering_port_display(self):
        """Test that input/output ports are displayed."""
        workflow = WorkflowSpec(
            id="ports-test",
            rev=1,
            title="Ports Test",
            nodes=[
                NodeSpec(
                    id="port_node",
                    type="agent",
                    label="Port Node",
                    data=NodeData(
                        ins=["input_data", "config"],
                        outs=["result", "status", "metrics"]
                    )
                )
            ],
            edges=[]
        )
        
        ascii_output = render_workflow_ascii(workflow)
        
        # Should mention ports in some way
        assert "Port Node" in ascii_output


class TestWorkflowCLIEdgeCases:
    """Test edge cases in CLI functionality."""

    def test_render_empty_title_workflow(self):
        """Test rendering workflow with empty title."""
        workflow = WorkflowSpec(
            id="empty-title",
            rev=1,
            title="",
            nodes=[],
            edges=[]
        )
        
        ascii_output = render_workflow_ascii(workflow)
        # Should handle empty title gracefully
        assert ascii_output is not None

    def test_render_workflow_with_very_long_names(self):
        """Test rendering workflow with very long node names."""
        workflow = WorkflowSpec(
            id="long-names",
            rev=1,
            title="Long Names Test",
            nodes=[
                NodeSpec(
                    id="very_long_node_identifier_that_exceeds_normal_length",
                    type="agent",
                    label="This is a very long node label that might cause display issues",
                    data=NodeData()
                )
            ],
            edges=[]
        )
        
        ascii_output = render_workflow_ascii(workflow)
        
        # Should handle long names gracefully (possibly truncated)
        assert "very long node label" in ascii_output or "very_long_node" in ascii_output

    def test_render_workflow_with_special_characters(self):
        """Test rendering workflow with special characters."""
        workflow = WorkflowSpec(
            id="special-chars",
            rev=1,
            title="Special Characters: émojis 🚀 & ünïcödé",
            nodes=[
                NodeSpec(
                    id="special_node",
                    type="agent",
                    label="Spéciål Nödé with 中文 and العربية",
                    data=NodeData()
                )
            ],
            edges=[]
        )
        
        ascii_output = render_workflow_ascii(workflow)
        
        # Should handle Unicode characters
        assert "🚀" in ascii_output
        assert "Spéciål" in ascii_output or "special_node" in ascii_output


if __name__ == "__main__":
    pytest.main([__file__])