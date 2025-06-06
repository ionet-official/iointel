"""
formatting.py: Utilities for rendering agent and tool results as HTML for UI display.
"""

import json


def format_result_for_html(result_dict):
    """
    Render the agent result and tool usage results as beautiful HTML for display in the UI.
    Args:
        result_dict (dict): The result dict from the agent, containing 'result' and 'tool_usage_results'.
    Returns:
        str: HTML string representing the agent output and tool usage pills.
    """
    html = []
    # Main agent result
    main_result = result_dict.get("result", "")
    if main_result:
        html.append(
            f'<div style="margin-bottom:1em;"><b>Agent:</b> {main_result}</div>'
        )
    # Tool usage results as smart pills
    tool_usage_results = result_dict.get("tool_usage_results", [])
    for tur in tool_usage_results:
        if hasattr(tur, "tool_name"):
            tool_name = getattr(tur, "tool_name", "(unknown tool)")
            tool_args = getattr(tur, "tool_args", {})
            tool_result = getattr(tur, "tool_result", "")
        else:
            tool_name = tur.get("tool_name", "(unknown tool)")
            tool_args = tur.get("tool_args", {})
            tool_result = tur.get("tool_result", "")
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
