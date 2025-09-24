"""
formatting.py: Utilities for rendering agent and tool results as HTML for UI display.
"""

import json
from iointel.src.agent_methods.data_models.datamodels import AgentResult


def format_result_for_html(result: AgentResult) -> str:
    """
    Render the agent result and tool usage results as beautiful HTML for display in the UI.
    Args:
        result: The result from the agent, containing 'result' and 'tool_usage_results'.
    Returns:
        str: HTML string representing the agent output and tool usage pills.
    """
    html = []
    # Main agent result - convert to string if needed
    main_result = str(result.result) if result.result is not None else ""
    if main_result:
        html.append(
            f'<div style="margin-bottom:1em;"><b>Agent:</b> {main_result}</div>'
        )
    # Tool usage results as smart pills
    tool_usage_results = result.tool_usage_results
    for tur in tool_usage_results:
        tool_name = tur.tool_name
        tool_args = tur.tool_args
        tool_result = tur.tool_result or ""
        pill_html = f"""
<div class="tool-pill" style="margin-bottom:10px;">
    <div style="font-weight:bold;font-size:1.1em;">🛠️ {tool_name}</div>
    <div style="font-size:0.95em;"><b>Args:</b>
        <pre style="background:#23272f;color:#ffb300;padding:4px 8px;border-radius:6px;font-size:0.98em;box-shadow:0 2px 8px #0002;">{
            json.dumps(tool_args, indent=2)
        }</pre>
    </div>
    <div style="font-size:0.95em;"><b>Result:</b>
        {
            (
                f'<pre style="background:#23272f;color:#ffb300;padding:4px 8px;border-radius:6px;">{json.dumps(tool_result, indent=2)}</pre>'
                if not (
                    isinstance(tool_result, str)
                    and ("<" in tool_result and ">" in tool_result)
                )
                else tool_result
            )
        }
    </div>
</div>
"""
        html.append(pill_html)
    return "\n".join(html)
