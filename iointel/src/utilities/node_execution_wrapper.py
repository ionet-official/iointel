"""
Simple SLA Validation for Workflow Nodes
========================================

This module provides lightweight SLA validation for the typed execution system.
It focuses on validation rather than complex retry logic, trusting that good
pre-prompt injection will lead to compliance on the first attempt.

Key features:
- Simple validation of tool usage against SLA requirements
- Single retry support if needed
- Clean integration with typed execution
- Minimal complexity
"""

from typing import List, Optional, Tuple, Any, Callable
from enum import Enum
from iointel.src.agent_methods.data_models.datamodels import ToolUsageResult
from iointel.src.agent_methods.data_models.workflow_spec import SLARequirements, NodeSpec
from iointel.src.agent_methods.data_models.execution_models import AgentExecutionResult
from iointel.src.utilities.io_logger import system_logger

logger = system_logger


class SLAValidationResult(Enum):
    """Result of SLA validation."""
    PASS = "PASS"
    FAIL_NO_TOOLS = "FAIL_NO_TOOLS"
    FAIL_INSUFFICIENT_CALLS = "FAIL_INSUFFICIENT_CALLS"
    FAIL_WRONG_TOOLS = "FAIL_WRONG_TOOLS"
    FAIL_WRONG_FINAL_TOOL = "FAIL_WRONG_FINAL_TOOL"


class SLAValidator:
    """
    Simple SLA validator for typed execution system.
    
    This replaces the complex wrapper with a straightforward validator
    that checks if agents met their tool usage requirements.
    """
    
    def extract_sla_requirements(self, node_spec: NodeSpec) -> Optional[SLARequirements]:
        """
        Extract SLA requirements from WorkflowSpec NodeSpec - SINGLE SOURCE OF TRUTH.
        
        REFACTORED APPROACH:
        1. Use NodeSpec.sla field as authoritative source (from WorkflowPlanner)
        2. NO GUESSING - if WorkflowPlanner didn't set SLA, no enforcement
        
        Args:
            node_spec: NodeSpec object with .sla field
        """
        # AUTHORITATIVE SOURCE: NodeSpec.sla field from WorkflowPlanner
        if node_spec.sla is not None:
            logger.debug(f"Using SLA from WorkflowSpec: {node_spec.sla}")
            return node_spec.sla
        
        # NO ENFORCEMENT - WorkflowPlanner is responsible for setting SLA
        logger.debug("No SLA found in NodeSpec - no enforcement")
        return None
    
    def validate_sla_compliance(
        self, 
        requirements: SLARequirements,
        tool_usage_results: List[ToolUsageResult]
    ) -> Tuple[SLAValidationResult, str]:
        """
        Validate tool usage against SLA requirements.
        
        Returns:
            (validation_result, detailed_reason)
        """
        if not requirements.enforce_usage:
            return SLAValidationResult.PASS, "No SLA enforcement required"
        
        # Check if no tools used when required (specific case)
        if requirements.tool_usage_required and len(tool_usage_results) == 0:
            return (
                SLAValidationResult.FAIL_NO_TOOLS,
                f"Tool usage required but no tools were used. Available: {requirements.required_tools}"
            )
        
        # Check minimum tool calls (general case)
        if len(tool_usage_results) < requirements.min_tool_calls:
            return (
                SLAValidationResult.FAIL_INSUFFICIENT_CALLS,
                f"Used {len(tool_usage_results)} tools, minimum {requirements.min_tool_calls} required"
            )
        
        # Check required tools were used
        if requirements.required_tools:
            used_tools = {result.tool_name for result in tool_usage_results}
            required_set = set(requirements.required_tools)
            missing = required_set - used_tools
            
            if missing:
                return (
                    SLAValidationResult.FAIL_WRONG_TOOLS,
                    f"Missing required tools: {missing}. Used: {used_tools}"
                )
        
        # Check final tool requirement
        if requirements.final_tool_must_be and tool_usage_results:
            last_tool = tool_usage_results[-1].tool_name
            if last_tool != requirements.final_tool_must_be:
                return (
                    SLAValidationResult.FAIL_WRONG_FINAL_TOOL,
                    f"Final tool must be '{requirements.final_tool_must_be}', but was '{last_tool}'"
                )
        
        return SLAValidationResult.PASS, "All SLA requirements satisfied"
    
    def validate_and_execute(
        self,
        execute_fn: Callable,
        node_spec: NodeSpec,
        allow_retry: bool = True
    ) -> Any:
        """
        Validate SLA compliance after execution, with optional single retry.
        
        This is a simple validation approach that:
        1. Executes the node
        2. Checks if SLA requirements were met
        3. Optionally retries ONCE if validation failed
        
        Args:
            execute_fn: Function that executes the node
            node_spec: NodeSpec from WorkflowSpec
            allow_retry: Whether to allow a single retry on failure
            
        Returns:
            Execution result (passes through from execute_fn)
        """
        # Extract SLA requirements
        sla_requirements = self.extract_sla_requirements(node_spec)
        
        # No SLA enforcement needed - just execute normally
        if not sla_requirements or not sla_requirements.enforce_usage:
            logger.debug(f"Node {node_spec.id} has no SLA requirements")
            return execute_fn()
        
        logger.info(f"🔒 SLA validation active for node {node_spec.id}")
        
        # Execute the node
        result = execute_fn()
        
        # Extract tool usage from result
        tool_usage_results = self._extract_tool_usage_from_result(result)
        
        # Validate SLA compliance
        validation_result, validation_reason = self.validate_sla_compliance(
            sla_requirements, tool_usage_results
        )
        
        if validation_result == SLAValidationResult.PASS:
            logger.info(f"✅ Node {node_spec.id} passed SLA validation")
            return result
        
        # Validation failed
        logger.warning(f"❌ Node {node_spec.id} failed SLA validation: {validation_reason}")
        
        # Try once more if allowed
        if allow_retry and sla_requirements.max_retries > 0:
            logger.info(f"🔄 Retrying node {node_spec.id} with enhanced prompts")
            
            # Execute again (the pre-prompt injection should help)
            result = execute_fn()
            
            # Check again
            tool_usage_results = self._extract_tool_usage_from_result(result)
            validation_result, validation_reason = self.validate_sla_compliance(
                sla_requirements, tool_usage_results
            )
            
            if validation_result == SLAValidationResult.PASS:
                logger.info(f"✅ Node {node_spec.id} passed SLA validation on retry")
                return result
            else:
                logger.error(f"🚫 Node {node_spec.id} failed SLA after retry: {validation_reason}")
        
        # Return result anyway but log the violation
        logger.error(f"⚠️  Returning result despite SLA violation for node {node_spec.id}")
        return result
    
    async def validate_async(
        self,
        execute_fn: Callable,
        node_spec: NodeSpec,
        allow_retry: bool = True
    ) -> Any:
        """
        Async version of validate_and_execute.
        
        Args:
            execute_fn: Async function that executes the node
            node_spec: NodeSpec from WorkflowSpec
            allow_retry: Whether to allow a single retry on failure
            
        Returns:
            Execution result (passes through from execute_fn)
        """
        # Extract SLA requirements
        sla_requirements = self.extract_sla_requirements(node_spec)
        
        # No SLA enforcement needed - just execute normally
        if not sla_requirements or not sla_requirements.enforce_usage:
            logger.debug(f"Node {node_spec.id} has no SLA requirements")
            return await execute_fn()
        
        logger.info(f"🔒 SLA validation active for node {node_spec.id}")
        
        # Execute the node
        result = await execute_fn()
        
        # Extract tool usage from result
        tool_usage_results = self._extract_tool_usage_from_result(result)
        
        # Validate SLA compliance
        validation_result, validation_reason = self.validate_sla_compliance(
            sla_requirements, tool_usage_results
        )
        
        if validation_result == SLAValidationResult.PASS:
            logger.info(f"✅ Node {node_spec.id} passed SLA validation")
            return result
        
        # Validation failed
        logger.warning(f"❌ Node {node_spec.id} failed SLA validation: {validation_reason}")
        
        # Try once more if allowed
        if allow_retry and sla_requirements.max_retries > 0:
            logger.info(f"🔄 Retrying node {node_spec.id} with enhanced prompts")
            
            # Execute again (the pre-prompt injection should help)
            result = await execute_fn()
            
            # Check again
            tool_usage_results = self._extract_tool_usage_from_result(result)
            validation_result, validation_reason = self.validate_sla_compliance(
                sla_requirements, tool_usage_results
            )
            
            if validation_result == SLAValidationResult.PASS:
                logger.info(f"✅ Node {node_spec.id} passed SLA validation on retry")
                return result
            else:
                logger.error(f"🚫 Node {node_spec.id} failed SLA after retry: {validation_reason}")
        
        # Return result anyway but log the violation
        logger.error(f"⚠️  Returning result despite SLA violation for node {node_spec.id}")
        return result
    
    def _extract_tool_usage_from_result(self, result: Any) -> List[ToolUsageResult]:
        """Extract tool usage from node execution result.
        
        Handles both typed AgentExecutionResult and legacy dict results.
        """
        # Handle typed AgentExecutionResult
        if isinstance(result, AgentExecutionResult):
            if result.agent_response and hasattr(result.agent_response, 'tool_usage_results'):
                return result.agent_response.tool_usage_results or []
            return []
        
        # Handle legacy dict result
        if isinstance(result, dict) and "tool_usage_results" in result:
            return result["tool_usage_results"] or []
        
        return []
    


# Global validator instance
sla_validator = SLAValidator()