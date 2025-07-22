"""
Agent Ontology and Tool Usage Patterns
======================================

Defines the base ontology of agent effects and tool usage patterns that the
WorkflowPlanner understands and can use to create agents with explicit enforcement.

This makes tool usage enforcement first-order - decided at planning time, not inferred at runtime.
"""

from typing import List, Set, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class AgentEffect(str, Enum):
    """Base effects that agents can have in a workflow."""
    ROUTING = "routing"  # Routes/gates flow to different paths
    FETCHING = "fetching"  # Fetches external data
    TRANSFORMING = "transforming"  # Transforms/processes data
    ANALYZING = "analyzing"  # Analyzes data (may or may not use tools)
    CHATTING = "chatting"  # General conversation
    EXECUTING = "executing"  # Executes actions/side-effects


class ToolUsageRequirement(BaseModel):
    """
    Explicit tool usage requirements for an agent.
    """
    available_tools: List[str] = Field(
        default_factory=list,
        description="All tools available to the agent"
    )
    required_tools: List[str] = Field(
        default_factory=list,
        description="Tools that MUST be used (SLA enforcement)"
    )
    final_tool_must_be: Optional[str] = Field(
        None,
        description="The final tool call must be this tool (e.g., conditional_gate for routing)"
    )
    min_tool_calls: int = Field(
        0,
        description="Minimum number of tool calls required"
    )
    enforce_usage: bool = Field(
        True,
        description="Whether to enforce tool usage at runtime"
    )


class AgentPattern(BaseModel):
    """
    Agent pattern with explicit tool usage requirements and prompting strategies.
    """
    name: str = Field(..., description="Pattern name (e.g., 'decision_router', 'data_fetcher')")
    effect: AgentEffect = Field(..., description="Primary effect this agent has")
    description: str = Field(..., description="What this pattern does")
    
    # Tool usage requirements
    tool_requirements: ToolUsageRequirement = Field(
        default_factory=ToolUsageRequirement,
        description="Explicit tool usage requirements"
    )
    
    # Prompting strategies
    prompt_prefix: Optional[str] = Field(
        None,
        description="Text to prepend to agent instructions"
    )
    prompt_suffix: Optional[str] = Field(
        None,
        description="Text to append to agent instructions"
    )
    prompt_guardrails: List[str] = Field(
        default_factory=list,
        description="Additional guardrail prompts to inject"
    )
    
    # Example tools for this pattern
    typical_tools: List[str] = Field(
        default_factory=list,
        description="Tools typically used with this pattern"
    )


# Pre-defined agent patterns that WorkflowPlanner can use
AGENT_PATTERNS = {
    "decision_router": AgentPattern(
        name="decision_router",
        effect=AgentEffect.ROUTING,
        description="Makes decisions and routes to different workflow paths",
        tool_requirements=ToolUsageRequirement(
            required_tools=["conditional_gate"],  # At minimum
            final_tool_must_be="conditional_gate",  # Must end with routing
            min_tool_calls=1,
            enforce_usage=True
        ),
        prompt_prefix="You are a decision agent that MUST use tools to route the workflow.",
        prompt_suffix="\n\nIMPORTANT: You must use conditional_gate as your final tool to route the workflow to the appropriate path.",
        typical_tools=["conditional_gate", "boolean_gate"]
    ),
    
    "data_fetcher": AgentPattern(
        name="data_fetcher",
        effect=AgentEffect.FETCHING,
        description="Fetches data from external sources",
        tool_requirements=ToolUsageRequirement(
            min_tool_calls=1,
            enforce_usage=True
        ),
        prompt_prefix="You are a data agent that MUST fetch current data using your tools.",
        typical_tools=["get_current_stock_price", "get_weather", "fetch_data"]
    ),
    
    "analyzer": AgentPattern(
        name="analyzer",
        effect=AgentEffect.ANALYZING,
        description="Analyzes data and provides insights",
        tool_requirements=ToolUsageRequirement(
            min_tool_calls=0,  # Tools optional
            enforce_usage=False
        ),
        prompt_prefix="You are an analysis agent. Use tools if they help your analysis.",
        typical_tools=["calculate", "compare", "statistical_analysis"]
    ),
    
    "executor": AgentPattern(
        name="executor",
        effect=AgentEffect.EXECUTING,
        description="Executes actions with side effects",
        tool_requirements=ToolUsageRequirement(
            min_tool_calls=1,
            enforce_usage=True
        ),
        prompt_prefix="You are an action agent that MUST use tools to execute the requested actions.",
        typical_tools=["send_email", "create_order", "update_database"]
    ),
    
    "conversationalist": AgentPattern(
        name="conversationalist",
        effect=AgentEffect.CHATTING,
        description="General conversation and help",
        tool_requirements=ToolUsageRequirement(
            min_tool_calls=0,
            enforce_usage=False
        ),
        typical_tools=[]
    )
}


class EnhancedNodeData(BaseModel):
    """
    Enhanced NodeData that includes explicit agent pattern and tool requirements.
    This extends the basic NodeData with first-order tool usage semantics.
    """
    # Standard fields (from workflow_spec.NodeData)
    config: Dict = Field(default_factory=dict)
    ins: List[str] = Field(default_factory=list)
    outs: List[str] = Field(default_factory=list)
    tool_name: Optional[str] = None
    agent_instructions: Optional[str] = None
    tools: Optional[List[str]] = None
    workflow_id: Optional[str] = None
    model: Optional[str] = "gpt-4o"
    
    # Enhanced fields for first-order tool usage
    agent_pattern: Optional[str] = Field(
        None,
        description="Explicit agent pattern (e.g., 'decision_router', 'data_fetcher')"
    )
    tool_requirements: Optional[ToolUsageRequirement] = Field(
        None,
        description="Explicit tool usage requirements"
    )


def create_workflow_planner_examples() -> List[Dict[str, Any]]:
    """
    Create few-shot examples for the WorkflowPlanner showing how to use agent patterns.
    """
    return [
        {
            "user_query": "Create a workflow that fetches stock prices and decides whether to buy or sell",
            "nodes": [
                {
                    "type": "agent",
                    "label": "Fetch Stock Data",
                    "data": {
                        "agent_pattern": "data_fetcher",
                        "agent_instructions": "Fetch current and historical stock prices for analysis",
                        "tools": ["get_current_stock_price", "get_historical_stock_prices"],
                        "tool_requirements": {
                            "available_tools": ["get_current_stock_price", "get_historical_stock_prices"],
                            "required_tools": ["get_current_stock_price"],  # Must at least get current
                            "min_tool_calls": 1,
                            "enforce_usage": True
                        }
                    }
                },
                {
                    "type": "agent", 
                    "label": "Stock Decision Router",
                    "data": {
                        "agent_pattern": "decision_router",
                        "agent_instructions": "Analyze the stock data and decide whether to buy or sell based on price changes",
                        "tools": ["get_current_stock_price", "conditional_gate"],
                        "tool_requirements": {
                            "available_tools": ["get_current_stock_price", "conditional_gate"],
                            "required_tools": ["conditional_gate"],
                            "final_tool_must_be": "conditional_gate",
                            "min_tool_calls": 1,
                            "enforce_usage": True
                        }
                    }
                }
            ]
        }
    ]


def apply_agent_pattern(node_data: Dict[str, Any], pattern_name: str) -> Dict[str, Any]:
    """
    Apply an agent pattern to node data, adding tool requirements and prompts.
    
    Args:
        node_data: The node's data dictionary
        pattern_name: Name of the pattern to apply
        
    Returns:
        Enhanced node data with pattern applied
    """
    if pattern_name not in AGENT_PATTERNS:
        return node_data
        
    pattern = AGENT_PATTERNS[pattern_name]
    
    # Add tool requirements
    if "tool_requirements" not in node_data:
        node_data["tool_requirements"] = pattern.tool_requirements.model_dump()
    
    # Enhance agent instructions with pattern prompts
    instructions = node_data.get("agent_instructions", "")
    if pattern.prompt_prefix:
        instructions = pattern.prompt_prefix + "\n\n" + instructions
    if pattern.prompt_suffix:
        instructions = instructions + "\n\n" + pattern.prompt_suffix
    node_data["agent_instructions"] = instructions
    
    # Add pattern name for runtime reference
    node_data["agent_pattern"] = pattern_name
    
    return node_data