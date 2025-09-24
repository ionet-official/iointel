"""
Agent Pre-Prompt Injection System
=================================

This module provides agent type classification and pre-prompt injection to ensure
agents have the correct instructions based on their intended role and available tools.

Instead of retrofitting SLA enforcement, we modify the agent's core instructions
at construction time based on its detected type.
"""

from iointel.src.agent_methods.data_models.workflow_spec import SLARequirements
from typing import Iterable, Tuple, Optional


def inject_prompts_enforcement_from_sla(
    original_instructions: str,
    sla_requirements: SLARequirements,  # SLARequirements from WorkflowSpec
    *,
    route_mappings: Optional[Iterable[Tuple[int, str]]] = None,
) -> str:
    """
    Inject pre-prompts based on SLA requirements from WorkflowSpec - SINGLE SOURCE OF TRUTH.
    
    REFACTORED APPROACH:
    - SLA requirements are the authoritative source for agent behavior
    - No guessing or pattern matching - WorkflowPlanner sets the policy
    - Simple, consistent prompt enhancement based on actual requirements
    
    Args:
        original_instructions: Agent's base instructions
        sla_requirements: SLARequirements from WorkflowSpec NodeSpec
        agent_name: Agent name for context
        tools: Available tools (for logging/context only)
        
    Returns:
        Enhanced instructions string
    """
    if not sla_requirements or not sla_requirements.enforce_usage:
        # No SLA enforcement - return original instructions
        return original_instructions
    
    # Build SLA-specific pre-prompts based on actual requirements
    pre_prompts = []
    
    # Determine agent type from SLA requirements
    if sla_requirements.final_tool_must_be:
        # Has final tool requirement - likely a decision agent
        final_tool = sla_requirements.final_tool_must_be
        
        # Special handling for routing tools
        if final_tool in ['routing_gate']:
            pre_prompts.extend([
                "🎯 You are a ROUTING DECISION AGENT with MANDATORY routing requirements.",
                f"⚡ CRITICAL: '{final_tool}' must be your FINAL tool call to route the workflow",
                "🚫 ROUTING IS MANDATORY - You MUST route to one of the available paths"
            ])
            
            # Add routing_gate specific guidance
            if final_tool == 'routing_gate':
                pre_prompts.append("📍 Use routing_gate(data=<input to next agent>, route_index=<int-based index>, route_name=<optional short name>)")
                # If route mappings are provided, include a clear, machine-readable block
                if route_mappings:
                    pre_prompts.append("\n### ROUTING CONFIGURATION ###")
                    pre_prompts.append("When calling routing_gate, use these exact route configurations:")
                    for idx, label in route_mappings:
                        pre_prompts.append(f"  - route_index: {idx} → '{label}'")
                    pre_prompts.append("IMPORTANT: Call routing_gate ONLY ONCE with the appropriate route_index based on your analysis.")
                    pre_prompts.append("### END ROUTING CONFIGURATION ###\n")
        else:
            pre_prompts.extend([
                "🎯 You are a DECISION AGENT with MANDATORY tool usage requirements.",
                f"⚡ CRITICAL: '{final_tool}' must be your FINAL tool call",
                "🚫 TOOL USAGE IS MANDATORY - Do not provide analysis without using your required tools"
            ])
    else:
        # General agent with tool requirements
        pre_prompts.append("📊 You are an AGENT with specific tool usage requirements.")
    
    # Add required tools information
    required_tools = sla_requirements.required_tools
    if required_tools:
        pre_prompts.append(f"🔧 REQUIRED TOOLS: You MUST use these tools: {', '.join(required_tools)}")
    
    # Add minimum tool calls requirement
    min_calls = sla_requirements.min_tool_calls
    if min_calls > 1:
        pre_prompts.append(f"📈 MINIMUM USAGE: You must make at least {min_calls} tool calls")
    
    # Add enforcement reminder
    pre_prompts.append("✅ This ensures proper workflow execution and data flow control")
    
    # Add CRITICAL section for maximum clarity
    critical_section = [
        "",
        "### 🚨 CRITICAL TOOL USAGE REQUIREMENTS 🚨 ###"
    ]
    
    if sla_requirements.final_tool_must_be:
        critical_section.append(f"1. YOU MUST call '{sla_requirements.final_tool_must_be}' as your FINAL action")
        critical_section.append(f"2. DO NOT end your response without calling '{sla_requirements.final_tool_must_be}'")
        critical_section.append(f"3. Your analysis is NOT complete until '{sla_requirements.final_tool_must_be}' is called")
    
    if sla_requirements.required_tools:
        critical_section.append(f"4. YOU MUST use ALL of these tools: {', '.join(sla_requirements.required_tools)}")
    
    if sla_requirements.min_tool_calls > 0:
        critical_section.append(f"5. YOU MUST make at least {sla_requirements.min_tool_calls} tool calls total")
    
    critical_section.extend([
        "",
        "⚠️  FAILURE TO USE REQUIRED TOOLS WILL RESULT IN WORKFLOW FAILURE",
        "⚠️  DO NOT PROVIDE ANALYSIS WITHOUT USING YOUR TOOLS FIRST",
        "",
        "Now create the perfect workflow execution by following these requirements.",
        "### END CRITICAL REQUIREMENTS ###",
        ""
    ])
    
    # Combine all sections: pre-prompts + critical section + original instructions
    enhanced_instructions = "\n".join(pre_prompts + [original_instructions] + critical_section)
    
    return enhanced_instructions
