"""
Decision Tools Catalog for Agentic Reasoning
===========================================

This catalog provides the WorkflowPlanner with explicit knowledge about tools
and their SLA requirements. It enables the planner to reason about tool usage
enforcement during workflow creation.

The catalog makes tool→SLA relationships first-class knowledge for LLM planning.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from .workflow_spec import SLARequirements


class ToolEffect(str, Enum):
    """Primary effects that tools can have in workflows."""
    ROUTING = "routing"           # Routes workflow to different paths  
    DATA_FETCHING = "data_fetching"  # Fetches external data
    CALCULATION = "calculation"   # Performs calculations
    VALIDATION = "validation"     # Validates conditions
    ACTION = "action"            # Executes side effects
    TRANSFORMATION = "transformation"  # Transforms data


class SLARequirement(str, Enum):
    """Types of SLA requirements for tools."""
    MUST_USE = "must_use"                    # Tool usage is mandatory
    MUST_BE_FINAL = "must_be_final"         # Must be the last tool called
    MUST_PRODUCE_ROUTING = "must_produce_routing"  # Must produce routing decision
    OPTIONAL = "optional"                    # Tool usage is optional



class ToolSpec(BaseModel):
    """
    Specification for a tool that the WorkflowPlanner can reason about.
    """
    name: str = Field(..., description="Tool name")
    effect: ToolEffect = Field(..., description="Primary effect of this tool")
    description: str = Field(..., description="What this tool does")
    
    # SLA Requirements
    sla_requirement: SLARequirement = Field(
        SLARequirement.OPTIONAL,
        description="SLA requirement level for this tool"
    )
    must_be_final_tool: bool = Field(
        False,
        description="Whether this tool must be the final tool called"
    )
    
    # Usage Patterns
    typical_patterns: List[str] = Field(
        default_factory=list,
        description="Common usage patterns for this tool"
    )
    triggers_enforcement: bool = Field(
        False,
        description="Whether using this tool triggers SLA enforcement"
    )
    
    # Agent Integration
    requires_agent_type: Optional[str] = Field(
        None,
        description="Type of agent that should use this tool"
    )
    prompt_hints: List[str] = Field(
        default_factory=list,
        description="Prompt hints for agents using this tool"
    )


# Decision Tools Catalog
DECISION_TOOLS_CATALOG: Dict[str, ToolSpec] = {
    "conditional_gate": ToolSpec(
        name="conditional_gate",
        effect=ToolEffect.ROUTING,
        description="Routes workflow based on boolean condition evaluation",
        sla_requirement=SLARequirement.MUST_BE_FINAL,
        must_be_final_tool=True,
        typical_patterns=[
            "if price > threshold then 'buy_path' else 'sell_path'",
            "if sentiment == 'positive' then 'approve_path' else 'reject_path'",
            "if percentage_change > 0.03 then 'growth_path' else 'decline_path'"
        ],
        triggers_enforcement=True,
        requires_agent_type="decision",
        prompt_hints=[
            "You MUST use conditional_gate as your final tool to route the workflow",
            "Evaluate the condition and route to the appropriate path",
            "The routing decision determines which part of the workflow executes next"
        ]
    ),
    
    # "percentage_change_gate": ToolSpec(
    #     name="percentage_change_gate",
    #     effect=ToolEffect.ROUTING,
    #     description="Routes workflow based on percentage change calculations",
    #     sla_requirement=SLARequirement.MUST_BE_FINAL,
    #     must_be_final_tool=True,
    #     typical_patterns=[
    #         "if percentage_change(old, new) > threshold",
    #         "route based on price movement percentage",
    #         "compare growth rates and route accordingly"
    #     ],
    #     triggers_enforcement=True,
    #     requires_agent_type="decision_agent",
    #     prompt_hints=[
    #         "Calculate percentage change between values",
    #         "Use the result to make routing decisions",
    #         "This must be your final tool call"
    #     ]
    # ),
    
    "boolean_gate": ToolSpec(
        name="boolean_gate", 
        effect=ToolEffect.ROUTING,
        description="Simple true/false routing gate",
        sla_requirement=SLARequirement.MUST_BE_FINAL,
        must_be_final_tool=True,
        typical_patterns=[
            "route based on simple boolean condition",
            "if condition then path_a else path_b"
        ],
        triggers_enforcement=True,
        requires_agent_type="decision_agent"
    ),
    
    "get_current_stock_price": ToolSpec(
        name="get_current_stock_price",
        effect=ToolEffect.DATA_FETCHING,
        description="Fetches current stock price from financial data source",
        sla_requirement=SLARequirement.MUST_USE,
        typical_patterns=[
            "get latest price for analysis",
            "fetch current market data",
            "retrieve real-time stock information"
        ],
        triggers_enforcement=True,
        requires_agent_type="data_agent",
        prompt_hints=[
            "Always fetch current data before analysis",
            "Use real-time prices for accurate decisions"
        ]
    ),
    
    "get_historical_stock_prices": ToolSpec(
        name="get_historical_stock_prices", 
        effect=ToolEffect.DATA_FETCHING,
        description="Fetches historical stock price data",
        sla_requirement=SLARequirement.OPTIONAL,
        typical_patterns=[
            "analyze price trends over time",
            "compare historical performance",
            "calculate moving averages"
        ],
        triggers_enforcement=False,
        requires_agent_type="data_agent"
    ),
    
    "add": ToolSpec(
        name="add",
        effect=ToolEffect.CALCULATION,
        description="Adds two numbers together",
        sla_requirement=SLARequirement.OPTIONAL,
        typical_patterns=["basic arithmetic", "sum calculations"],
        triggers_enforcement=False
    ),
    
    "multiply": ToolSpec(
        name="multiply",
        effect=ToolEffect.CALCULATION, 
        description="Multiplies two numbers",
        sla_requirement=SLARequirement.OPTIONAL,
        typical_patterns=["scaling values", "compound calculations"],
        triggers_enforcement=False
    )
}


def get_tools_by_effect(effect: ToolEffect) -> List[ToolSpec]:
    """Get all tools that have a specific effect."""
    return [tool for tool in DECISION_TOOLS_CATALOG.values() if tool.effect == effect]


def get_enforcement_tools() -> List[ToolSpec]:
    """Get all tools that trigger SLA enforcement."""
    return [tool for tool in DECISION_TOOLS_CATALOG.values() if tool.triggers_enforcement]


def get_routing_tools() -> List[ToolSpec]:
    """Get all routing/decision tools."""
    return get_tools_by_effect(ToolEffect.ROUTING)


def get_data_fetching_tools() -> List[ToolSpec]:
    """Get all data fetching tools."""
    return get_tools_by_effect(ToolEffect.DATA_FETCHING)


def create_workflow_planner_tool_knowledge() -> str:
    """
    Create tool knowledge section for WorkflowPlanner system prompt.
    
    Returns:
        Formatted string with tool knowledge for LLM reasoning
    """
    knowledge_sections = []
    
    # Decision/Routing Tools
    routing_tools = get_routing_tools()
    if routing_tools:
        knowledge_sections.append("🎯 **DECISION/ROUTING TOOLS** (Require SLA Enforcement):")
        for tool in routing_tools:
            knowledge_sections.append(f"- **{tool.name}**: {tool.description}")
            knowledge_sections.append(f"  - Effect: {tool.effect}")
            knowledge_sections.append(f"  - SLA: {tool.sla_requirement}")
            if tool.must_be_final_tool:
                knowledge_sections.append("  - ⚠️ MUST be final tool call")
            knowledge_sections.append(f"  - Patterns: {', '.join(tool.typical_patterns[:2])}")
            knowledge_sections.append("")
    
    # Data Fetching Tools  
    data_tools = get_data_fetching_tools()
    if data_tools:
        knowledge_sections.append("📊 **DATA FETCHING TOOLS**:")
        for tool in data_tools:
            knowledge_sections.append(f"- **{tool.name}**: {tool.description}")
            if tool.triggers_enforcement:
                knowledge_sections.append(f"  - ⚠️ SLA: {tool.sla_requirement}")
            knowledge_sections.append("")
    
    # Other Tools
    other_tools = [t for t in DECISION_TOOLS_CATALOG.values() 
                   if t.effect not in [ToolEffect.ROUTING, ToolEffect.DATA_FETCHING]]
    if other_tools:
        knowledge_sections.append("🔧 **OTHER TOOLS**:")
        for tool in other_tools:
            knowledge_sections.append(f"- **{tool.name}**: {tool.description}")
            knowledge_sections.append("")
    
    # SLA Rules
    knowledge_sections.extend([
        "📋 **SLA ENFORCEMENT RULES**:",
        "1. If you include routing tools (conditional_gate, etc.) → SLA enforcement is automatic",
        "2. Routing tools MUST be the final tool called by the agent",
        "3. Data fetching tools with MUST_USE requirement trigger enforcement",
        "4. Agents will be retried if they don't meet SLA requirements",
        "",
        "💡 **Planning Guidelines**:",
        "- Need workflow routing? → Include conditional_gate → Creates decision agent with SLA",
        "- Need current data? → Include get_current_stock_price → Creates data agent with SLA", 
        "- Need analysis only? → No routing tools → No SLA enforcement",
        ""
    ])
    
    return "\n".join(knowledge_sections)


def get_sla_requirements_for_tools(tool_names: List[str]) -> SLARequirements:
    """
    Generate SLA requirements based on the tools an agent has.
    
    Args:
        tool_names: List of tool names the agent will have
        
    Returns:
        SLARequirements model with SLA requirements for the agent
    """
    requirements = SLARequirements()
    
    for tool_name in tool_names:
        if tool_name in DECISION_TOOLS_CATALOG:
            tool_spec = DECISION_TOOLS_CATALOG[tool_name]
            
            if tool_spec.triggers_enforcement:
                requirements.enforce_usage = True
                requirements.tool_usage_required = True
                requirements.min_tool_calls = max(requirements.min_tool_calls, 1)
                
                if tool_spec.sla_requirement in [SLARequirement.MUST_USE, SLARequirement.MUST_BE_FINAL]:
                    requirements.required_tools.append(tool_name)
                
                if tool_spec.must_be_final_tool:
                    requirements.final_tool_must_be = tool_name
    
    return requirements


# Example usage for WorkflowPlanner
WORKFLOW_PLANNER_TOOL_KNOWLEDGE = create_workflow_planner_tool_knowledge()