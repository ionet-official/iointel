from iointel.src.agents import ToolUsageResult
from typing import List


def tool_usage_results_to_string(
    tool_usage_results: List[ToolUsageResult], prefix: str = "TOOL(s) Called"
) -> str:
    return f"{prefix}--\n" + "\n".join(
        [
            f"{i + 1}: {tool.tool_name}({tool.tool_args}) --[tool result]-> {tool.tool_result}"
            for i, tool in enumerate(tool_usage_results)
        ]
    )
