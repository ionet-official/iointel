"""
Tool Usage Enforcement V2 - First-Order Enforcement
==================================================

This module provides tool usage enforcement based on explicit requirements
from the WorkflowPlanner, rather than inferring intent from agent names/tools.

Key improvements:
- Tool requirements are explicit in the workflow spec
- No pattern inference needed - the planner decides
- Support for "final tool must be X" requirements
- More flexible enforcement strategies
"""

from typing import Dict, List, Optional, Any, Callable

from ..agent_methods.data_models.datamodels import ToolUsageResult
from ..agent_methods.data_models.agent_ontology import ToolUsageRequirement
from ..utilities.helpers import make_logger

logger = make_logger(__name__)


class ToolUsageEnforcer:
    """
    Enforces explicit tool usage requirements from workflow specifications.
    """
    
    def __init__(self):
        self.retry_prompts = self._create_retry_prompts()
    
    def _create_retry_prompts(self) -> List[str]:
        """Create progressive retry prompts."""
        return [
            "\n\nIMPORTANT: You have not used the required tools. Please use the tools specified in your requirements before providing analysis.",
            "\n\nCRITICAL: Tool usage is mandatory for this task. You MUST use the required tools: {required_tools}. This is your final attempt."
        ]
    
    def validate_tool_usage(
        self, 
        requirements: ToolUsageRequirement,
        tool_usage_results: List[ToolUsageResult]
    ) -> tuple[bool, str]:
        """
        Validate tool usage against explicit requirements.
        
        Returns:
            (is_valid, reason): Whether valid and why not if invalid
        """
        if not requirements.enforce_usage:
            return True, "No enforcement required"
        
        # Check minimum tool calls
        if len(tool_usage_results) < requirements.min_tool_calls:
            return False, f"Used {len(tool_usage_results)} tools, minimum {requirements.min_tool_calls} required"
        
        # Check required tools were used
        if requirements.required_tools:
            used_tools = {result.tool_name for result in tool_usage_results}
            required_set = set(requirements.required_tools)
            missing = required_set - used_tools
            
            if missing:
                return False, f"Missing required tools: {missing}"
        
        # Check final tool requirement
        if requirements.final_tool_must_be and tool_usage_results:
            last_tool = tool_usage_results[-1].tool_name
            if last_tool != requirements.final_tool_must_be:
                return False, f"Final tool must be '{requirements.final_tool_must_be}', but was '{last_tool}'"
        
        return True, "All requirements met"
    
    def create_enhanced_prompt(
        self,
        original_prompt: str,
        requirements: ToolUsageRequirement,
        retry_attempt: int,
        validation_reason: str
    ) -> str:
        """
        Create enhanced prompt based on specific requirements.
        """
        if retry_attempt > len(self.retry_prompts):
            retry_prompt = self.retry_prompts[-1]
        else:
            retry_prompt = self.retry_prompts[retry_attempt - 1]
        
        # Format with specific requirements
        retry_prompt = retry_prompt.format(
            required_tools=", ".join(requirements.required_tools) if requirements.required_tools else "your tools"
        )
        
        # Add specific guidance based on validation failure
        specific_guidance = f"\n\nValidation failed: {validation_reason}"
        
        if requirements.final_tool_must_be:
            specific_guidance += f"\nREMEMBER: Your final tool call must be '{requirements.final_tool_must_be}'"
        
        # Injection-resistant enforcement block
        enforcement_block = f"""

===== TOOL USAGE ENFORCEMENT =====
REQUIREMENTS:
- Available tools: {', '.join(requirements.available_tools)}
- Required tools: {', '.join(requirements.required_tools) if requirements.required_tools else 'Any of the available tools'}
- Minimum calls: {requirements.min_tool_calls}
{f"- Final tool must be: {requirements.final_tool_must_be}" if requirements.final_tool_must_be else ""}

ATTEMPT: {retry_attempt}/2
REASON: {validation_reason}

You MUST satisfy these requirements before providing your response.
===== END ENFORCEMENT =====
"""
        
        return original_prompt + retry_prompt + specific_guidance + enforcement_block
    
    async def enforce_tool_usage(
        self,
        agent_executor_func: Callable,
        requirements: Optional[ToolUsageRequirement],
        agent_name: str = "Agent",
        max_retries: int = 2,
        *args,
        **kwargs
    ) -> Any:
        """
        Enforce tool usage based on explicit requirements.
        
        Args:
            agent_executor_func: The agent execution function
            requirements: Explicit tool usage requirements (None = no enforcement)
            agent_name: Name for logging
            max_retries: Maximum retry attempts
            *args, **kwargs: Arguments for agent execution
            
        Returns:
            Agent execution result
        """
        # No requirements = no enforcement
        if not requirements or not requirements.enforce_usage:
            logger.debug(f"No enforcement for agent '{agent_name}'")
            return await agent_executor_func(*args, **kwargs)
        
        logger.info(f"Enforcing tool usage for '{agent_name}': {requirements.min_tool_calls} min calls, required: {requirements.required_tools}")
        
        original_query = args[0] if args else kwargs.get('query', '')
        validation_reason = ""  # Initialize for first retry attempts
        
        for attempt in range(max_retries + 1):
            # Enhance prompt on retries
            if attempt > 0:
                enhanced_query = self.create_enhanced_prompt(
                    original_query,
                    requirements,
                    attempt,
                    validation_reason
                )
                
                if args:
                    args = (enhanced_query,) + args[1:]
                else:
                    kwargs['query'] = enhanced_query
                
                logger.info(f"Retry {attempt} for '{agent_name}'")
            
            # Execute agent
            result = await agent_executor_func(*args, **kwargs)
            
            # Extract tool usage
            tool_usage_results = self._extract_tool_usage(result)
            
            # Validate
            is_valid, validation_reason = self.validate_tool_usage(requirements, tool_usage_results)
            
            if is_valid:
                logger.info(f"'{agent_name}' passed validation: {[r.tool_name for r in tool_usage_results]}")
                return result
            
            logger.warning(f"'{agent_name}' failed validation: {validation_reason}")
            
            # Last attempt - return with warning
            if attempt >= max_retries:
                logger.error(
                    f"'{agent_name}' failed after {max_retries + 1} attempts. "
                    f"Requirements not met: {validation_reason}"
                )
                return result
        
        return result
    
    def _extract_tool_usage(self, result: Any) -> List[ToolUsageResult]:
        """Extract tool usage from agent result."""
        if isinstance(result, dict) and "tool_usage_results" in result:
            return result["tool_usage_results"] or []
        return ['should_not_happen']


# Global instance
tool_usage_enforcer_v2 = ToolUsageEnforcer()


def create_requirements_from_node_data(node_data: Dict[str, Any]) -> Optional[ToolUsageRequirement]:
    """
    Create ToolUsageRequirement from node data.
    
    Handles both:
    1. Explicit tool_requirements field
    2. Legacy inference from tools list
    """
    # Explicit requirements take precedence
    if "tool_requirements" in node_data:
        return ToolUsageRequirement(**node_data["tool_requirements"])
    
    # Legacy: If agent has conditional_gate, make it required
    tools = node_data.get("tools", [])
    if "conditional_gate" in tools:
        return ToolUsageRequirement(
            available_tools=tools,
            required_tools=["conditional_gate"],
            final_tool_must_be="conditional_gate",
            min_tool_calls=1,
            enforce_usage=True
        )
    
    # No special requirements
    return None