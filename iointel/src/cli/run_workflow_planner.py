"""
Interactive CLI for WorkflowPlanner with ASCII React Flow visualization.

This script provides an interactive interface for creating and refining workflows
using the WorkflowPlanner agent, with a simple ASCII visualization.
"""

import asyncio
import json
from datetime import datetime
from typing import List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
from dotenv import load_dotenv

from iointel import AsyncMemory
from iointel.src.agent_methods.agents.workflow_planner import WorkflowPlanner
from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec
from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env
from iointel.src.utilities.tool_registry_utils import create_tool_catalog

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)

load_dotenv("creds.env")

# Initialize rich console
console = Console()

# For SQLite (creates a local file)
memory = AsyncMemory("sqlite+aiosqlite:///conversations.db")

# Initialize the database tables
asyncio.run(memory.init_models())

# Create a fixed conversation ID for workflow planning
CONVERSATION_ID = "workflow_planner_session"



def render_workflow_ascii(workflow_spec: WorkflowSpec) -> str:
    """Render a workflow as ASCII art."""
    if not workflow_spec.nodes:
        return "📋 Empty workflow"
    
    # Create a simple ASCII representation
    ascii_lines = []
    ascii_lines.append(f"📋 {workflow_spec.title}")
    ascii_lines.append("=" * len(f"📋 {workflow_spec.title}"))
    
    if workflow_spec.description:
        ascii_lines.append(f"📝 {workflow_spec.description}")
        ascii_lines.append("")
    
    # Build a graph structure
    edges_by_source = {}
    edges_by_target = {}
    
    for edge in workflow_spec.edges:
        edges_by_source.setdefault(edge.source, []).append(edge)
        edges_by_target.setdefault(edge.target, []).append(edge)
    
    # Find start nodes (no incoming edges)
    start_nodes = [node for node in workflow_spec.nodes if node.id not in edges_by_target]
    
    # Render nodes in a simple flow
    rendered_nodes = set()
    
    def render_node_chain(node_id: str, level: int = 0) -> List[str]:
        if node_id in rendered_nodes:
            return []
        
        rendered_nodes.add(node_id)
        
        # Find the node
        node = next((n for n in workflow_spec.nodes if n.id == node_id), None)
        if not node:
            return []
        
        lines = []
        indent = "  " * level
        
        # Node representation
        icon = {"tool": "🔧", "agent": "🤖", "workflow_call": "📞"}.get(node.type, "⚙️")
        lines.append(f"{indent}{icon} {node.label} ({node.id})")
        
        # Show configuration if available
        if node.data.config:
            config_str = ", ".join(f"{k}={v}" for k, v in list(node.data.config.items())[:3])
            if len(node.data.config) > 3:
                config_str += "..."
            lines.append(f"{indent}   📄 {config_str}")
        
        # Show edges
        outgoing_edges = edges_by_source.get(node_id, [])
        for i, edge in enumerate(outgoing_edges):
            is_last = i == len(outgoing_edges) - 1
            connector = "└─" if is_last else "├─"
            condition = f" [if: {edge.data.condition}]" if edge.data.condition else ""
            lines.append(f"{indent}   {connector}➤{condition}")
            
            # Recursively render target node
            target_lines = render_node_chain(edge.target, level + 1)
            lines.extend(target_lines)
        
        return lines
    
    # Render starting from start nodes
    for start_node in start_nodes:
        chain_lines = render_node_chain(start_node.id)
        ascii_lines.extend(chain_lines)
        ascii_lines.append("")
    
    # Render any remaining nodes (in case of disconnected components)
    for node in workflow_spec.nodes:
        if node.id not in rendered_nodes:
            chain_lines = render_node_chain(node.id)
            ascii_lines.extend(chain_lines)
            ascii_lines.append("")
    
    return "\n".join(ascii_lines)


def render_workflow_panel(workflow_spec: WorkflowSpec) -> Panel:
    """Render a workflow as a rich panel."""
    ascii_flow = render_workflow_ascii(workflow_spec)
    
    # Create stats table
    stats_table = Table(show_header=False, box=None, padding=(0, 1))
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    
    stats_table.add_row("📊 Nodes", str(len(workflow_spec.nodes)))
    stats_table.add_row("🔗 Edges", str(len(workflow_spec.edges)))
    stats_table.add_row("📅 Rev", str(workflow_spec.rev))
    
    # Node type breakdown
    node_types = {}
    for node in workflow_spec.nodes:
        node_types[node.type] = node_types.get(node.type, 0) + 1
    
    for node_type, count in node_types.items():
        icon = {"tool": "🔧", "agent": "🤖", "workflow_call": "📞"}.get(node_type, "⚙️")
        stats_table.add_row(f"{icon} {node_type}", str(count))
    
    # Combine ASCII flow and stats
    layout = Layout()
    layout.split_column(
        Layout(Text(ascii_flow, style="white"), name="flow"),
        Layout(stats_table, name="stats", size=len(node_types) + 4)
    )
    
    return Panel(
        layout,
        title=f"🌊 {workflow_spec.title}",
        border_style="blue",
        padding=(1, 2)
    )


async def interactive_workflow_planner():
    """Run the interactive workflow planner."""
    console.print(
        Panel.fit(
            "🚀 Interactive Workflow Planner\n"
            "Create and refine workflows using natural language!\n"
            "Type 'help' for commands, 'exit' to quit",
            title="WorkflowPlanner CLI",
            border_style="green",
        )
    )
    
    # Load tools and create catalog
    console.print("[blue]Loading tools based on available credentials...[/blue]")
    available_tools = load_tools_from_env("creds.env")
    tool_catalog = create_tool_catalog()
    
    console.print(f"[green]Loaded {len(available_tools)} credential-based tools[/green]")
    console.print(f"[yellow]Total tools available: {len(tool_catalog)}[/yellow]")
    
    # Create the WorkflowPlanner with shared model configuration
    from iointel.src.utilities.constants import get_model_config
    import os
    
    cli_model = os.getenv("WORKFLOW_PLANNER_MODEL", "gpt-4o")
    model_config = get_model_config(model=cli_model)
    console.print(f"[blue]🤖 Using model config: {model_config['model']} @ {model_config['base_url']}[/blue]")
    
    planner = WorkflowPlanner(
        model=model_config["model"],
        api_key=model_config["api_key"],
        base_url=model_config["base_url"],
        memory=memory,
        conversation_id=CONVERSATION_ID,
        debug=False
    )
    
    # Track current workflow
    current_workflow = None
    workflow_history = []
    
    # Display available tools
    console.print("\n[cyan]Available tools:[/cyan]")
    for tool_name in sorted(list(tool_catalog.keys())[:10]):  # Show first 10
        tool = tool_catalog[tool_name]
        console.print(f"  • {tool_name}: {tool['description']}")
    
    if len(tool_catalog) > 10:
        console.print(f"  ... and {len(tool_catalog) - 10} more tools")
    console.print()
    
    while True:
        try:
            user_input = input("🌊 WorkflowPlanner> ").strip()
            
            if user_input.lower() == "exit":
                console.print("👋 Goodbye!")
                break
            
            elif user_input.lower() == "help":
                help_text = """
Available commands:
• help - Show this help message
• tools - List all available tools
• show - Display current workflow
• history - Show workflow history
• save <filename> - Save current workflow to file
• load <filename> - Load workflow from file
• clear - Clear current workflow
• exit - Exit the planner

Or describe a workflow in natural language to create/refine it!
"""
                console.print(Panel(help_text, title="Help", border_style="cyan"))
                continue
            
            elif user_input.lower() == "tools":
                tools_table = Table(title="Available Tools", show_header=True, header_style="bold magenta")
                tools_table.add_column("Tool", style="cyan")
                tools_table.add_column("Description", style="white")
                tools_table.add_column("Type", style="yellow")
                
                for tool_name, tool_info in list(tool_catalog.items())[:20]:  # Show first 20
                    tools_table.add_row(
                        tool_name,
                        tool_info["description"],
                        "async" if tool_info["is_async"] else "sync"
                    )
                
                console.print(tools_table)
                if len(tool_catalog) > 20:
                    console.print(f"... and {len(tool_catalog) - 20} more tools")
                continue
            
            elif user_input.lower() == "show":
                if current_workflow:
                    console.print(render_workflow_panel(current_workflow))
                else:
                    console.print("[yellow]No workflow to display[/yellow]")
                continue
            
            elif user_input.lower() == "history":
                if workflow_history:
                    console.print(f"[cyan]Workflow History ({len(workflow_history)} versions):[/cyan]")
                    for i, wf in enumerate(workflow_history[-5:], 1):  # Show last 5
                        console.print(f"  {i}. {wf.title} (rev {wf.rev}) - {len(wf.nodes)} nodes")
                else:
                    console.print("[yellow]No workflow history[/yellow]")
                continue
            
            elif user_input.lower().startswith("save "):
                if current_workflow:
                    filename = user_input[5:].strip()
                    if not filename.endswith('.json'):
                        filename += '.json'
                    
                    with open(filename, 'w') as f:
                        json.dump(current_workflow.model_dump(), f, indent=2, default=str)
                    
                    console.print(f"[green]Workflow saved to {filename}[/green]")
                else:
                    console.print("[yellow]No workflow to save[/yellow]")
                continue
            
            elif user_input.lower().startswith("load "):
                filename = user_input[5:].strip()
                try:
                    with open(filename) as f:
                        workflow_data = json.load(f)
                    
                    current_workflow = WorkflowSpec.model_validate(workflow_data)
                    console.print(f"[green]Workflow loaded from {filename}[/green]")
                    console.print(render_workflow_panel(current_workflow))
                except Exception as e:
                    console.print(f"[red]Error loading workflow: {e}[/red]")
                continue
            
            elif user_input.lower() == "clear":
                current_workflow = None
                console.print("[green]Workflow cleared[/green]")
                continue
            
            elif not user_input:
                continue
            
            # Generate or refine workflow
            console.print("[blue]🤖 Generating workflow...[/blue]")
            
            if current_workflow:
                # Refine existing workflow
                try:
                    new_workflow = await planner.refine_workflow(
                        workflow_spec=current_workflow,
                        feedback=user_input
                    )
                    
                    # Update revision
                    new_workflow.rev = current_workflow.rev + 1
                    
                    # Store in history
                    workflow_history.append(current_workflow)
                    current_workflow = new_workflow
                    
                    console.print("[green]✅ Workflow refined![/green]")
                    console.print(render_workflow_panel(current_workflow))
                    
                except Exception as e:
                    console.print(f"[red]❌ Error refining workflow: {e}[/red]")
            else:
                # Generate new workflow
                try:
                    current_workflow = await planner.generate_workflow(
                        query=user_input,
                        tool_catalog=tool_catalog,
                        context={"timestamp": datetime.now().isoformat()}
                    )
                    
                    console.print("[green]✅ Workflow generated![/green]")
                    console.print(render_workflow_panel(current_workflow))
                    
                except Exception as e:
                    console.print(f"[red]❌ Error generating workflow: {e}[/red]")
        
        except KeyboardInterrupt:
            console.print("\n👋 Goodbye!")
            break
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
            import traceback
            console.print(f"[red]Traceback:[/red] {traceback.format_exc()}")


if __name__ == "__main__":
    load_dotenv("creds.env")
    asyncio.run(interactive_workflow_planner())