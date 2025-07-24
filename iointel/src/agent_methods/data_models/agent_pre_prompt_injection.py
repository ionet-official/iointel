"""
Agent Pre-Prompt Injection System
=================================

This module provides agent type classification and pre-prompt injection to ensure
agents have the correct instructions based on their intended role and available tools.

Instead of retrofitting SLA enforcement, we modify the agent's core instructions
at construction time based on its detected type.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass

from .decision_tools_catalog import DECISION_TOOLS_CATALOG, get_sla_requirements_for_tools


class AgentType(Enum):
    """Classification of agent types based on their intended role."""
    DECISION = "decision"      # Routes workflow based on conditions
    DATA = "data"             # Fetches and analyzes data  
    ACTION = "action"         # Performs actions/side effects
    ANALYSIS = "analysis"     # Analyzes data (tools optional)
    CHAT = "chat"            # Conversational (no specific tools needed)


@dataclass
class AgentTypeClassification:
    """Result of agent type classification."""
    agent_type: AgentType
    confidence: float
    reasoning: str
    required_pre_prompts: List[str]
    sla_enforcement: bool = False


class AgentTypeClassifier:
    """
    Classifies agents into types based on tools, instructions, and context.
    """
    
    def __init__(self):
        # Get only routing tools from catalog as decision tools
        from .decision_tools_catalog import ToolEffect
        self.decision_tools = {
            name for name, spec in DECISION_TOOLS_CATALOG.items() 
            if spec.effect == ToolEffect.ROUTING
        }
        self.data_tools = {
            name for name, spec in DECISION_TOOLS_CATALOG.items() 
            if spec.effect == ToolEffect.DATA_FETCHING
        }
        # Add other common data tools not in catalog
        self.data_tools.update({
            "search_the_web", "fetch_data", "query_database"
        })
        self.action_tools = {
            "execute", "send", "create", "delete", "update", 
            "trade", "buy", "sell"
        }
    
    def classify_agent(
        self, 
        tools: List[str], 
        instructions: str = "",
        agent_name: str = "",
        context: Optional[Dict[str, Any]] = None
    ) -> AgentTypeClassification:
        """
        Classify an agent based on its configuration.
        
        Args:
            tools: List of available tool names
            instructions: Agent's instructions
            agent_name: Name of the agent
            context: Additional context
            
        Returns:
            AgentTypeClassification with type and required pre-prompts
        """
        tools_set = set(tools)
        instructions_lower = instructions.lower()
        name_lower = agent_name.lower()
        
        # Decision Agent Classification (Highest Priority) - Only true decision tools
        decision_tools_found = list(tools_set.intersection(self.decision_tools))
        if decision_tools_found:
            return AgentTypeClassification(
                agent_type=AgentType.DECISION,
                confidence=0.95,
                reasoning=f"Has decision tools: {decision_tools_found}",
                required_pre_prompts=self._get_decision_agent_pre_prompts(decision_tools_found),
                sla_enforcement=True
            )
        
        # Check name/instruction patterns for decision agents
        decision_keywords = ["decision", "routing", "route", "gate", "choose", "decide", "conditional"]
        if any(keyword in name_lower for keyword in decision_keywords):
            return AgentTypeClassification(
                agent_type=AgentType.DECISION,
                confidence=0.8,
                reasoning=f"Agent name '{agent_name}' suggests decision-making role",
                required_pre_prompts=self._get_decision_agent_pre_prompts([]),
                sla_enforcement=False  # No decision tools available
            )
        
        if any(keyword in instructions_lower for keyword in decision_keywords):
            return AgentTypeClassification(
                agent_type=AgentType.DECISION,
                confidence=0.7,
                reasoning="Instructions suggest decision-making role",
                required_pre_prompts=self._get_decision_agent_pre_prompts([]),
                sla_enforcement=False  # No decision tools available
            )
        
        # Data Agent Classification
        if tools_set.intersection(self.data_tools):
            data_tools_found = list(tools_set.intersection(self.data_tools))
            return AgentTypeClassification(
                agent_type=AgentType.DATA,
                confidence=0.9,
                reasoning=f"Has data tools: {data_tools_found}",
                required_pre_prompts=self._get_data_agent_pre_prompts(data_tools_found),
                sla_enforcement=any(tool in DECISION_TOOLS_CATALOG for tool in data_tools_found)
            )
        
        # Action Agent Classification
        if tools_set.intersection(self.action_tools):
            action_tools_found = list(tools_set.intersection(self.action_tools))
            return AgentTypeClassification(
                agent_type=AgentType.ACTION,
                confidence=0.85,
                reasoning=f"Has action tools: {action_tools_found}",
                required_pre_prompts=self._get_action_agent_pre_prompts(action_tools_found),
                sla_enforcement=False
            )
        
        # Analysis Agent Classification
        analysis_keywords = ["analyze", "analysis", "review", "examine", "evaluate", "summarize"]
        if any(keyword in name_lower for keyword in analysis_keywords):
            return AgentTypeClassification(
                agent_type=AgentType.ANALYSIS,
                confidence=0.7,
                reasoning=f"Agent name '{agent_name}' suggests analysis role",
                required_pre_prompts=self._get_analysis_agent_pre_prompts([]),
                sla_enforcement=False
            )
        
        # Default to Chat Agent
        return AgentTypeClassification(
            agent_type=AgentType.CHAT,
            confidence=0.6,
            reasoning="No specific patterns detected, defaulting to chat agent",
            required_pre_prompts=self._get_chat_agent_pre_prompts([]),
            sla_enforcement=False
        )
    
    def _get_decision_agent_pre_prompts(self, decision_tools: List[str]) -> List[str]:
        """Generate pre-prompts for decision agents."""
        if not decision_tools:
            return [
                "You are a DECISION AGENT responsible for making routing decisions in workflows.",
                "When you have decision-making tools available, you MUST use them to determine the proper path.",
                "Your role is to analyze the situation and route data to the appropriate downstream nodes."
            ]
        
        # Get SLA requirements for the specific tools
        sla_requirements = get_sla_requirements_for_tools(decision_tools)
        
        prompts = [
            "🎯 You are a DECISION AGENT with MANDATORY tool usage requirements.",
            f"🔧 REQUIRED TOOLS: You MUST use these tools: {', '.join(decision_tools)}",
            "📊 DECISION PROCESS: 1) Use your tools to evaluate conditions, 2) Make routing decision to downstream nodes, 3) Provide your analysis",
        ]
        
        if sla_requirements.final_tool_must_be:
            prompts.append(f"⚡ CRITICAL: '{sla_requirements.final_tool_must_be}' must be your FINAL tool call")
        
        prompts.extend([
            "🚫 TOOL USAGE IS MANDATORY - Do not provide analysis without using your decision tools",
            "✅ This ensures proper workflow routing and data flow control"
        ])
        
        return prompts
    
    def _get_data_agent_pre_prompts(self, data_tools: List[str]) -> List[str]:
        """Generate pre-prompts for data agents."""
        prompts = [
            "📊 You are a DATA AGENT responsible for fetching and analyzing current information.",
        ]
        
        if data_tools:
            prompts.extend([
                f"🔧 AVAILABLE DATA TOOLS: {', '.join(data_tools)}",
                "📈 BEST PRACTICE: Use your data tools to get current information before analysis",
                "🔍 Always prefer real-time data over assumptions when tools are available"
            ])
        
        return prompts
    
    def _get_action_agent_pre_prompts(self, action_tools: List[str]) -> List[str]:
        """Generate pre-prompts for action agents."""
        return [
            "⚡ You are an ACTION AGENT responsible for executing operations.",
            f"🛠️ AVAILABLE ACTIONS: {', '.join(action_tools) if action_tools else 'Various action tools'}",
            "✅ Use your action tools to perform the requested operations effectively"
        ]
    
    def _get_analysis_agent_pre_prompts(self, tools: List[str]) -> List[str]:
        """Generate pre-prompts for analysis agents."""
        return [
            "🔍 You are an ANALYSIS AGENT focused on examining and evaluating information.",
            "📋 Provide thorough analysis based on available data and context",
            "💡 Use tools if they help with your analysis, but focus on insights and conclusions"
        ]
    
    def _get_chat_agent_pre_prompts(self, tools: List[str]) -> List[str]:
        """Generate pre-prompts for chat agents."""
        return [
            "💬 You are a CHAT AGENT providing helpful conversational assistance.",
            "🤝 Focus on understanding the user's needs and providing clear, helpful responses"
        ]


def inject_pre_prompts(
    original_instructions: str,
    tools: List[str],
    agent_name: str = "",
    context: Optional[Dict[str, Any]] = None,
    sla_requirements: Optional[Any] = None  # Accept SLARequirements or dict
) -> tuple[str, AgentTypeClassification]:
    """
    Injects pre-prompts into the agent's instructions based on its type and SLA requirements.
    If sla_requirements is provided, it overrides catalog/classifier defaults.
    Returns (enhanced_instructions, classification)
    """
    classifier = AgentTypeClassifier()
    # If explicit SLA requirements are provided, use them to guide classification
    if sla_requirements is not None:
        # If tool_usage_required or enforce_usage is True, treat as decision agent
        enforce = getattr(sla_requirements, 'enforce_usage', False) or getattr(sla_requirements, 'tool_usage_required', False)
        required_tools = getattr(sla_requirements, 'required_tools', [])
        if enforce and required_tools:
            classification = AgentTypeClassification(
                agent_type=AgentType.DECISION,
                confidence=1.0,
                reasoning=f"Explicit SLA requirements: must use {required_tools}",
                required_pre_prompts=classifier._get_decision_agent_pre_prompts(required_tools),
                sla_enforcement=True
            )
        else:
            # Fallback to normal classification if not a decision agent
            classification = classifier.classify_agent(tools, original_instructions, agent_name, context)
    else:
        classification = classifier.classify_agent(tools, original_instructions, agent_name, context)
    # Inject pre-prompts
    pre_prompts = classification.required_pre_prompts
    enhanced_instructions = "\n".join(pre_prompts + [original_instructions])
    return enhanced_instructions, classification


# Global classifier instance
agent_type_classifier = AgentTypeClassifier()