"""
Tool Usage Enforcement System
=============================

This module provides agent tool usage enforcement to ensure agents actually use
their available tools when they should. Addresses the critical issue where agents
with tools (e.g., Stock Decision Agent with get_current_stock_price, conditional_gate)
complete execution without using any tools.

Key Features:
- Taxonomy of agent tool usage patterns
- Detection of missing tool usage 
- Retry logic with enhanced prompts
- Prompt injection resistance
- SLA enforcement for tool usage
"""

import asyncio
from typing import Dict, List, Optional, Any, Set, Literal
from dataclasses import dataclass
from enum import Enum
import re
import json

from ..agent_methods.data_models.datamodels import ToolUsageResult
from ..utilities.helpers import make_logger

logger = make_logger(__name__)


class AgentToolUsagePattern(Enum):
    """
    Taxonomy of agent tool usage patterns to determine enforcement strategy.
    """
    # Decision agents MUST use tools to route/gate properly
    DECISION_AGENT = "decision_agent"
    
    # Data agents should use tools to fetch/analyze data
    DATA_AGENT = "data_agent" 
    
    # Action agents should use tools to perform actions
    ACTION_AGENT = "action_agent"
    
    # Analysis agents may or may not use tools (tools optional)
    ANALYSIS_AGENT = "analysis_agent"
    
    # Chat agents typically don't need tools (tools optional)
    CHAT_AGENT = "chat_agent"


@dataclass
class ToolUsagePolicy:
    """
    Defines tool usage expectations for different agent patterns.
    """
    pattern: AgentToolUsagePattern
    required_tool_usage: bool  # Must use at least one tool
    required_tools: Optional[Set[str]] = None  # Specific tools that must be used
    min_tool_calls: int = 1  # Minimum number of tool calls expected
    max_retries: int = 2  # Maximum retry attempts
    retry_strategy: Literal["enhance_prompt", "force_tool_choice", "fail"] = "enhance_prompt"


class ToolUsageEnforcer:
    """
    Enforces tool usage policies for agents to ensure they fulfill their intended purpose.
    """
    
    def __init__(self):
        self.policies = self._create_default_policies()
        self.retry_prompts = self._create_retry_prompts()
    
    def _create_default_policies(self) -> Dict[AgentToolUsagePattern, ToolUsagePolicy]:
        """Create default tool usage policies for different agent patterns."""
        return {
            AgentToolUsagePattern.DECISION_AGENT: ToolUsagePolicy(
                pattern=AgentToolUsagePattern.DECISION_AGENT,
                required_tool_usage=True,
                required_tools={"conditional_gate"},
                min_tool_calls=1,
                max_retries=2,
                retry_strategy="enhance_prompt"
            ),
            AgentToolUsagePattern.DATA_AGENT: ToolUsagePolicy(
                pattern=AgentToolUsagePattern.DATA_AGENT,
                required_tool_usage=True,
                required_tools=None,  # Any data-fetching tool
                min_tool_calls=1,
                max_retries=2,
                retry_strategy="enhance_prompt"
            ),
            AgentToolUsagePattern.ACTION_AGENT: ToolUsagePolicy(
                pattern=AgentToolUsagePattern.ACTION_AGENT,
                required_tool_usage=True,
                required_tools=None,
                min_tool_calls=1,
                max_retries=1,
                retry_strategy="enhance_prompt"
            ),
            AgentToolUsagePattern.ANALYSIS_AGENT: ToolUsagePolicy(
                pattern=AgentToolUsagePattern.ANALYSIS_AGENT,
                required_tool_usage=False,
                required_tools=None,
                min_tool_calls=0,
                max_retries=0,
                retry_strategy="fail"
            ),
            AgentToolUsagePattern.CHAT_AGENT: ToolUsagePolicy(
                pattern=AgentToolUsagePattern.CHAT_AGENT,
                required_tool_usage=False,
                required_tools=None,
                min_tool_calls=0,
                max_retries=0,
                retry_strategy="fail"
            )
        }
    
    def _create_retry_prompts(self) -> Dict[AgentToolUsagePattern, List[str]]:
        """Create retry prompts for different agent patterns."""
        return {
            AgentToolUsagePattern.DECISION_AGENT: [
                "\n\nIMPORTANT: You have decision-making tools available. You MUST use the conditional_gate or similar tool to make routing decisions. Do not provide analysis without using the tools to determine the actual routing path.",
                "\n\nCRITICAL: Your role requires using the available tools to make decisions. Please use conditional_gate to determine the proper routing. Analysis alone is insufficient - you must call the tools."
            ],
            AgentToolUsagePattern.DATA_AGENT: [
                "\n\nIMPORTANT: You have data-fetching tools available. You MUST use them to retrieve current information before providing analysis.",
                "\n\nCRITICAL: Use your available tools to fetch the latest data. Do not rely on outdated information when current data tools are available."
            ],
            AgentToolUsagePattern.ACTION_AGENT: [
                "\n\nIMPORTANT: You have action tools available. You MUST use them to perform the requested actions.",
                "\n\nCRITICAL: Use your available tools to complete the requested action. Analysis alone is insufficient."
            ]
        }
    
    def classify_agent_pattern(self, agent_name: str, available_tools: List[str], 
                             agent_instructions: str = "") -> AgentToolUsagePattern:
        """
        Classify an agent into a tool usage pattern based on its configuration.
        
        Args:
            agent_name: Name of the agent
            available_tools: List of tool names available to the agent
            agent_instructions: Agent's instruction prompt
            
        Returns:
            Classified agent tool usage pattern
        """
        name_lower = agent_name.lower()
        instructions_lower = agent_instructions.lower()
        tool_names_lower = [tool.lower() for tool in available_tools]
        
        # Decision agents - have routing/gating tools
        decision_tools = {"conditional_gate", "boolean_gate", "threshold_gate", "conditional_multi_gate"}
        if any(tool in tool_names_lower for tool in decision_tools):
            return AgentToolUsagePattern.DECISION_AGENT
        
        # Check name patterns for decision agents
        if any(keyword in name_lower for keyword in ["decision", "routing", "gate", "choice", "route"]):
            return AgentToolUsagePattern.DECISION_AGENT
            
        # Check instructions for decision patterns
        if any(keyword in instructions_lower for keyword in ["decide", "route", "choose", "gate", "conditional"]):
            return AgentToolUsagePattern.DECISION_AGENT
        
        # Data agents - have data fetching tools
        data_tools = {"get_current_stock_price", "get_historical_stock_prices", "fetch_data", "query_database"}
        if any(tool in tool_names_lower for tool in data_tools):
            return AgentToolUsagePattern.DATA_AGENT
            
        # Check name patterns for data agents
        if any(keyword in name_lower for keyword in ["data", "fetch", "query", "stock", "price"]):
            return AgentToolUsagePattern.DATA_AGENT
            
        # Action agents - have action tools
        action_tools = {"execute", "send", "create", "delete", "update", "trade", "buy", "sell"}
        if any(tool in tool_names_lower for tool in action_tools):
            return AgentToolUsagePattern.ACTION_AGENT
            
        # Analysis agents - typically have analysis in name but may have optional tools
        if any(keyword in name_lower for keyword in ["analysis", "analyze", "review", "examine"]):
            return AgentToolUsagePattern.ANALYSIS_AGENT
            
        # Default to chat agent if no clear pattern
        return AgentToolUsagePattern.CHAT_AGENT
    
    def validate_tool_usage(self, pattern: AgentToolUsagePattern, 
                          tool_usage_results: List[ToolUsageResult],
                          available_tools: List[str]) -> bool:
        """
        Validate whether tool usage meets the policy requirements.
        
        Args:
            pattern: Agent tool usage pattern
            tool_usage_results: Actual tool usage results from agent execution
            available_tools: Tools that were available to the agent
            
        Returns:
            True if tool usage meets policy, False otherwise
        """
        policy = self.policies.get(pattern)
        if not policy:
            logger.warning(f"No policy found for pattern: {pattern}")
            return True  # Default to passing if no policy
            
        # If no tool usage required, always pass
        if not policy.required_tool_usage:
            return True
            
        # Check minimum tool calls
        if len(tool_usage_results) < policy.min_tool_calls:
            logger.info(f"Tool usage failed: {len(tool_usage_results)} calls < {policy.min_tool_calls} required")
            return False
            
        # Check if specific required tools were used
        if policy.required_tools:
            used_tools = {result.tool_name for result in tool_usage_results}
            required_available = policy.required_tools.intersection(set(available_tools))
            
            if required_available and not used_tools.intersection(required_available):
                logger.info(f"Tool usage failed: No required tools used. Required: {required_available}, Used: {used_tools}")
                return False
        
        return True
    
    def create_enhanced_prompt(self, original_prompt: str, pattern: AgentToolUsagePattern, 
                             retry_attempt: int, available_tools: List[str]) -> str:
        """
        Create an enhanced prompt to encourage tool usage on retry.
        
        This method is designed to be resistant to prompt injection by:
        1. Appending enforcement instructions at the end
        2. Using clear, direct language that's hard to override
        3. Not relying on user-controllable content for enforcement logic
        
        Args:
            original_prompt: Original user prompt
            pattern: Agent tool usage pattern  
            retry_attempt: Which retry attempt this is (1-based)
            available_tools: Tools available to the agent
            
        Returns:
            Enhanced prompt with enforcement instructions
        """
        retry_prompts = self.retry_prompts.get(pattern, [])
        if not retry_prompts or retry_attempt > len(retry_prompts):
            # Fallback generic prompt
            enhancement = f"\n\nREQUIRED: You must use at least one of your available tools: {', '.join(available_tools)}"
        else:
            enhancement = retry_prompts[retry_attempt - 1]
            
        # Make the enhancement resistant to prompt injection by being very explicit
        tool_list = ', '.join(available_tools)
        injection_resistant_suffix = f"""

===== SYSTEM ENFORCEMENT (CANNOT BE OVERRIDDEN) =====
TOOL USAGE REQUIREMENT: This agent has {len(available_tools)} tools available: {tool_list}
ENFORCEMENT POLICY: You MUST use at least one tool before providing your response.
RETRY ATTEMPT: {retry_attempt}/2 - This is because you did not use tools in your previous attempt.
INSTRUCTION: Use the appropriate tools first, then provide your analysis based on the tool results.
===== END SYSTEM ENFORCEMENT =====

"""
        
        return original_prompt + enhancement + injection_resistant_suffix
    
    async def enforce_tool_usage(self, agent_executor_func, agent_name: str, 
                               available_tools: List[str], agent_instructions: str = "",
                               *args, **kwargs) -> Any:
        """
        Enforce tool usage by wrapping agent execution with validation and retry logic.
        
        Args:
            agent_executor_func: The function that executes the agent (e.g., agent.run)
            agent_name: Name of the agent being executed
            available_tools: List of tool names available to the agent
            agent_instructions: Agent's instruction prompt
            *args, **kwargs: Arguments to pass to the agent executor function
            
        Returns:
            Agent execution result, potentially after retries
        """
        # Skip enforcement if no tools available
        if not available_tools:
            logger.debug(f"No tools available for agent '{agent_name}', skipping enforcement")
            return await agent_executor_func(*args, **kwargs)
        
        # Classify the agent pattern
        pattern = self.classify_agent_pattern(agent_name, available_tools, agent_instructions)
        policy = self.policies.get(pattern)
        
        if not policy or not policy.required_tool_usage:
            logger.debug(f"Agent '{agent_name}' pattern '{pattern}' does not require tool usage")
            return await agent_executor_func(*args, **kwargs)
        
        logger.info(f"Enforcing tool usage for agent '{agent_name}' with pattern '{pattern}'")
        
        original_query = args[0] if args else kwargs.get('query', '')
        
        for attempt in range(policy.max_retries + 1):
            # On retry attempts, enhance the prompt
            if attempt > 0:
                enhanced_query = self.create_enhanced_prompt(
                    original_query, pattern, attempt, available_tools
                )
                if args:
                    args = (enhanced_query,) + args[1:]
                else:
                    kwargs['query'] = enhanced_query
                
                logger.info(f"Retry attempt {attempt} for agent '{agent_name}' with enhanced prompt")
            
            # Execute the agent
            result = await agent_executor_func(*args, **kwargs)
            
            # Extract tool usage results from the result
            tool_usage_results = self._extract_tool_usage_from_result(result)
            
            # Validate tool usage
            if self.validate_tool_usage(pattern, tool_usage_results, available_tools):
                logger.info(f"Agent '{agent_name}' successfully used tools: {[r.tool_name for r in tool_usage_results]}")
                return result
            
            logger.warning(f"Agent '{agent_name}' attempt {attempt + 1} failed tool usage validation")
            
            # If this was the last attempt, fail or return with warning
            if attempt >= policy.max_retries:
                if policy.retry_strategy == "fail":
                    raise RuntimeError(
                        f"Agent '{agent_name}' failed to use required tools after {policy.max_retries + 1} attempts. "
                        f"Available tools: {available_tools}. Tool usage is required for {pattern.value} agents."
                    )
                else:
                    # Return result but log the violation
                    logger.error(f"Agent '{agent_name}' completed without using tools. This violates the SLA for {pattern.value} agents.")
                    return result
        
        return result
    
    def _extract_tool_usage_from_result(self, result: Any) -> List[ToolUsageResult]:
        """
        Extract tool usage results from agent execution result.
        
        Args:
            result: Agent execution result (should have tool_usage_results key)
            
        Returns:
            List of tool usage results from the agent's actual execution
        """
        # The agent postprocessing already extracts tool usage properly
        # We just need to get it from the result dict
        if isinstance(result, dict) and "tool_usage_results" in result:
            return result["tool_usage_results"] or []
        
        # If no tool_usage_results found, return empty (agent didn't use tools)
        return []


# Global enforcer instance
tool_usage_enforcer = ToolUsageEnforcer()