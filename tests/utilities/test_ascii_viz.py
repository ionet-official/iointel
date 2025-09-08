"""
Test the ASCII visualization for workflows.
"""

from iointel.src.agent_methods.agents.workflow_planner import WorkflowPlanner
from iointel.src.cli.run_workflow_planner import render_workflow_ascii, render_workflow_panel
from rich.console import Console

console = Console()

def test_ascii_visualization():
    """Test the ASCII workflow visualization."""
    print("Testing ASCII Workflow Visualization...")
    
    # Create a sample workflow
    planner = WorkflowPlanner()
    workflow = planner.create_example_workflow("Test ASCII Workflow")
    
    # Test ASCII rendering
    ascii_output = render_workflow_ascii(workflow)
    print("ASCII Representation:")
    print(ascii_output)
    
    # Test Rich panel rendering
    print("\nRich Panel Representation:")
    panel = render_workflow_panel(workflow)
    console.print(panel)
    
    return True

if __name__ == "__main__":
    test_ascii_visualization()