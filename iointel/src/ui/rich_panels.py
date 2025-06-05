from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.columns import Columns
from rich.console import Group
from iointel.src.agent_methods.data_models.datamodels import ToolUsageResult
from typing import Literal


def render_agent_result_panel(
    result_output: str,
    query: str,
    agent_name: str,
    tool_usage_results: list[ToolUsageResult],
    show_tool_calls: bool = True,
    tool_pil_layout: Literal["horizontal", "vertical"] = "horizontal",
    console=None,
):
    """
    Render the agent result panel using Rich. If no console is provided, use the shared console.
    """
    from iointel.src.utilities.rich import console as default_console

    if console is None:
        console = default_console
    tool_usage_pils = tool_usage_results_to_panels(tool_usage_results)
    task_header = Text(f" Objective: {query} ", style="bold white on dark_green")
    agent_info = Text(f"Agent(s): {agent_name}", style="cyan bold")
    panel_content = Markdown(str(result_output), style="magenta")

    if show_tool_calls and tool_usage_pils:
        if tool_pil_layout == "horizontal":
            panel_content = Group(
                panel_content, Text("\n"), Columns(tool_usage_pils, expand=True)
            )
        else:
            panel_content = Group(panel_content, Text("\n"), *tool_usage_pils)

    panel = Panel(
        panel_content,
        title=task_header,
        subtitle=agent_info,
        border_style="electric_blue",
    )
    console.print(panel)


def tool_usage_results_to_panels(
    tool_usage_results: list[ToolUsageResult],
) -> list[Panel]:
    """
    Given a list of ToolUsageResult, return a list of Rich Panels for display.
    """
    tool_usage_pils: list[Panel] = []
    for tur in tool_usage_results:
        pil = Panel(
            f"[bold cyan]🛠️ Tool: [magenta]{tur.tool_name}[/magenta]\n[yellow]Args: {tur.tool_args}[/yellow]"
            + (
                f"\n\n[bold green]✅ Result: [white]{tur.tool_result}[/white]"
                if tur.tool_result is not None
                else ""
            ),
            border_style="cyan",
            title=f"== {tur.tool_name}({tur.tool_args}) ==",
            title_align="left",
            padding=(1, 2),
            style="on black",
        )
        tool_usage_pils.append(pil)
    return tool_usage_pils


def update_live_panel(live, markdown_content: str, query: str, agent_name: str):
    """
    Update a Rich Live panel with the current markdown content, query, and agent name.
    """
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.text import Text

    task_header = Text(f" Objective: {query} ", style="bold white on dark_green")
    agent_info = Text(f"Agent: {agent_name}", style="cyan bold")
    markdown_render = Markdown(markdown_content, style="magenta")
    panel = Panel(
        markdown_render,
        title=task_header,
        subtitle=agent_info,
        border_style="electric_blue",
    )
    live.update(panel)
