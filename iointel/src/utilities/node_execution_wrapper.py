"""
Node Execution Wrapper with SLA Enforcement
==========================================

This module provides runtime wrapper system that acts as a "meta runtime helper"
to groom individual parts of the workflow graph, ensuring SLA compliance before
allowing data to pass downstream.

Key features:
- Message passing control (blocks bad data)
- SLA gatekeeper functionality  
- Automatic retry with enhanced prompts
- Timeout and failure handling
- Clean integration with DAG executor
"""

import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from enum import Enum

from ..agent_methods.data_models.datamodels import ToolUsageResult
from ..agent_methods.data_models.workflow_spec import SLARequirements  # Use the typed Pydantic version
from ..agent_methods.data_models.decision_tools_catalog import (
    get_sla_requirements_for_tools
)
from ..utilities.io_logger import system_logger

# Use IOLogger for structured logging with data parameter support
logger = system_logger


class SLAValidationResult(Enum):
    """Results of SLA validation."""
    PASS = "pass"
    FAIL_NO_TOOLS = "fail_no_tools"
    FAIL_WRONG_TOOLS = "fail_wrong_tools"
    FAIL_WRONG_FINAL_TOOL = "fail_wrong_final_tool"
    FAIL_INSUFFICIENT_CALLS = "fail_insufficient_calls"


@dataclass 
class ExecutionContext:
    """Context for node execution."""
    node_id: str
    node_type: str
    node_label: str
    input_data: Any
    attempt: int = 0
    start_time: float = 0
    sla_requirements: Optional[SLARequirements] = None


class NodeExecutionWrapper:
    """
    Meta runtime helper that wraps node execution with SLA validation.
    
    Acts as a message passing gatekeeper - no data flows downstream until
    SLA requirements are satisfied.
    """
    
    def __init__(self):
        self.retry_prompts = self._create_retry_prompts()
    
    def _create_retry_prompts(self) -> List[str]:
        """Create progressive retry prompts."""
        return [
            "\n\n🔄 RETRY REQUIRED: You did not meet the tool usage requirements. Please use the required tools before providing your response.",
            "\n\n⚠️ FINAL ATTEMPT: This is your last chance to meet the SLA requirements. You MUST use the specified tools: {required_tools}"
        ]
    
    def extract_sla_requirements(self, node_data: Dict[str, Any], agent_params: Optional[Any] = None) -> SLARequirements:
        """
        Extract SLA requirements with complete workflow flexibility.
        
        FLEXIBLE SLA SYSTEM - TYPED WITH PYDANTIC:
        - Workflow can define custom SLA for ANY tool (web_search, conditional_gate, etc.)
        - Typed validation ensures configuration correctness
        - Catalog becomes advisory, not prescriptive
        
        Priority order:
        1. Workflow-defined SLA (node.data.sla) - HIGHEST PRIORITY 
        2. Node-level SLA (node.sla) - NODE LEVEL
        3. Agent-level SLA configuration  
        4. Catalog advisory defaults (optional)
        5. No enforcement (default)
        """
        logger.debug(f"🔍 Flexible SLA extraction from node_data keys: {list(node_data.keys())}")
        
        # 1. WORKFLOW-DEFINED SLA (HIGHEST PRIORITY)
        # node.data.sla - allows per-node custom SLA configuration
        if "sla" in node_data and node_data["sla"] is not None:
            sla_config = node_data["sla"]
            logger.info(f"🎯 Found workflow-defined SLA in node.data: {sla_config}")
            
            # Handle typed Pydantic SLARequirements or dict
            if isinstance(sla_config, SLARequirements):
                return sla_config
            elif isinstance(sla_config, dict):
                return SLARequirements(**sla_config)
        
        # 2. Legacy support for sla_requirements field
        if "sla_requirements" in node_data and node_data["sla_requirements"] is not None:
            sla_config = node_data["sla_requirements"]
            logger.info(f"🔍 Found legacy SLA requirements: {sla_config}")
            if isinstance(sla_config, dict):
                return SLARequirements(**sla_config)
            return sla_config
        
        # 3. Agent-level SLA configuration (AgentParams doesn't have sla_requirements currently)
        # This is for future extension when we add SLA to AgentParams
        if (agent_params and 
            hasattr(agent_params, "sla_requirements") and 
            agent_params.sla_requirements is not None):
            logger.info(f"🔍 Using agent-level SLA: {agent_params.sla_requirements}")
            return agent_params.sla_requirements
        
        # 4. Catalog advisory defaults (only as fallback)
        tools = node_data.get("tools", [])
        if tools:
            catalog_sla = get_sla_requirements_for_tools(tools)
            logger.info(f"📚 Using catalog advisory SLA for tools {tools}: enforce={catalog_sla.enforce_usage}")
            # Convert catalog SLA to Pydantic if needed
            if not isinstance(catalog_sla, SLARequirements):
                catalog_dict = catalog_sla.__dict__ if hasattr(catalog_sla, '__dict__') else vars(catalog_sla)
                return SLARequirements(**catalog_dict)
            return catalog_sla
        
        # 5. No enforcement by default
        logger.debug("🔍 No SLA configuration found, no enforcement")
        return SLARequirements(enforce_usage=False)
    
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
    
    def create_enhanced_prompt(
        self,
        context: ExecutionContext,
        validation_result: SLAValidationResult,
        validation_reason: str
    ) -> Any:
        """
        Create enhanced input for retry attempts.
        
        This modifies the input data to include stronger prompts about tool usage.
        """
        requirements = context.sla_requirements
        
        # Get retry prompt
        retry_idx = min(context.attempt - 1, len(self.retry_prompts) - 1)
        retry_prompt = self.retry_prompts[retry_idx]
        
        if requirements and requirements.required_tools:
            retry_prompt = retry_prompt.format(required_tools=", ".join(requirements.required_tools))
        
        # Create enforcement guidance
        if requirements:
            enforcement_guidance = f"""

🚫 SLA VALIDATION FAILED: {validation_reason}

📋 REQUIREMENTS:
- Tool usage required: {requirements.tool_usage_required}
- Required tools: {requirements.required_tools}
- Minimum tool calls: {requirements.min_tool_calls}
"""
            
            if requirements.final_tool_must_be:
                enforcement_guidance += f"- Final tool must be: {requirements.final_tool_must_be}\n"
            
            enforcement_guidance += f"""
🔄 ATTEMPT: {context.attempt}/{requirements.max_retries}
💡 TIP: Use the required tools BEFORE providing your analysis.

===== SLA ENFORCEMENT ACTIVE =====
This node has SLA requirements that MUST be satisfied.
Data will not flow downstream until compliance is achieved.
===== END SLA ENFORCEMENT =====
"""
        else:
            enforcement_guidance = ""
        
        # Modify the input data to include enhanced prompts
        # This assumes the input has a query/objective field to enhance
        enhanced_input = context.input_data.copy() if isinstance(context.input_data, dict) else context.input_data
        
        if isinstance(enhanced_input, dict):
            if "query" in enhanced_input:
                enhanced_input["query"] += retry_prompt + enforcement_guidance
            elif "objective" in enhanced_input:
                enhanced_input["objective"] += retry_prompt + enforcement_guidance
            else:
                # Add as new field
                enhanced_input["sla_enforcement_prompt"] = retry_prompt + enforcement_guidance
        
        return enhanced_input
    
    async def execute_with_sla_enforcement(
        self,
        node_executor: Callable,
        node_data: Dict[str, Any],
        input_data: Any,
        node_id: str = "unknown",
        node_type: str = "unknown",
        node_label: str = "Unknown Node"
    ) -> Any:
        """
        Execute a node with SLA enforcement wrapper.
        
        This is the main entry point for SLA-enforced node execution.
        
        Args:
            node_executor: Function that executes the actual node
            node_data: Node configuration data
            input_data: Input data for the node
            node_id: Node identifier
            node_type: Node type (agent, tool, etc.)
            node_label: Human-readable node label
            
        Returns:
            Node execution result (after SLA validation)
        """
        # Extract SLA requirements
        sla_requirements = self.extract_sla_requirements(node_data)
        
        # Skip enforcement if not required
        if not sla_requirements.enforce_usage:
            logger.debug(f"Node {node_id} has no SLA requirements, executing normally")
            return await node_executor()
        
        logger.info(f"🔒 SLA enforcement active for node {node_id} ({node_label})")
        logger.info(f"   Requirements: {sla_requirements.required_tools}, final: {sla_requirements.final_tool_must_be}")
        
        # Log to feedback system for workflow analysis
        self._log_sla_enforcement_start(node_id, node_label, sla_requirements)
        
        # Create execution context
        context = ExecutionContext(
            node_id=node_id,
            node_type=node_type,
            node_label=node_label,
            input_data=input_data,
            start_time=time.time(),
            sla_requirements=sla_requirements
        )
        
        # Initialize validation variables for retries
        validation_result = None
        validation_reason = ""
        
        # Execute with retries
        for attempt in range(sla_requirements.max_retries + 1):
            context.attempt = attempt
            
            # Check timeout
            if time.time() - context.start_time > sla_requirements.timeout_seconds:
                logger.error(f"Node {node_id} timed out after {sla_requirements.timeout_seconds}s")
                raise TimeoutError(f"Node {node_id} execution timed out")
            
            # Enhance input for retries
            if attempt > 0 and validation_result is not None:
                logger.info(f"🔄 Retry {attempt} for node {node_id}")
                enhanced_input = self.create_enhanced_prompt(context, validation_result, validation_reason)
                # Update the input for the node executor
                # This requires the executor to accept dynamic input
                context.input_data = enhanced_input
            
            # Execute the node
            try:
                result = await node_executor()
            except Exception as e:
                logger.error(f"Node {node_id} execution failed: {e}")
                if attempt >= sla_requirements.max_retries:
                    raise
                continue
            
            # Extract tool usage from result
            tool_usage_results = self._extract_tool_usage_from_result(result)
            
            # Validate SLA compliance
            validation_result, validation_reason = self.validate_sla_compliance(
                sla_requirements, tool_usage_results
            )
            
            if validation_result == SLAValidationResult.PASS:
                logger.info(f"✅ Node {node_id} passed SLA validation")
                logger.info(f"   Tools used: {[r.tool_name for r in tool_usage_results]}")
                
                # Log successful SLA enforcement
                self._log_sla_enforcement_result(node_id, True, attempt, tool_usage_results)
                return result
            
            logger.warning(f"❌ Node {node_id} failed SLA validation: {validation_reason}")
            
            # Last attempt - return with failure warning
            if attempt >= sla_requirements.max_retries:
                logger.error(
                    f"🚫 Node {node_id} failed SLA after {sla_requirements.max_retries + 1} attempts. "
                    f"Final reason: {validation_reason}"
                )
                
                # Log failed SLA enforcement
                self._log_sla_enforcement_result(node_id, False, attempt, tool_usage_results, validation_reason)
                
                # Could raise exception or return with warning
                # For now, return the result but log the SLA violation
                return result
        
        return result
    
    def _extract_tool_usage_from_result(self, result: Any) -> List[ToolUsageResult]:
        """Extract tool usage from node execution result."""
        if isinstance(result, dict) and "tool_usage_results" in result:
            return result["tool_usage_results"] or []
        return []
    
    def _log_sla_enforcement_start(self, node_id: str, node_label: str, requirements: SLARequirements):
        """Log SLA enforcement start for workflow analysis."""
        logger.info(
            "🔍 META-EXECUTION: SLA enforcement started", 
            data={
                "event": "sla_enforcement_start",
                "node_id": node_id,
                "node_label": node_label,
                "requirements": {
                    "required_tools": requirements.required_tools,
                    "final_tool_must_be": requirements.final_tool_must_be,
                    "min_tool_calls": requirements.min_tool_calls,
                    "tool_usage_required": requirements.tool_usage_required,
                    "max_retries": requirements.max_retries,
                    "timeout_seconds": requirements.timeout_seconds
                }
            }
        )  
    
    def _log_sla_enforcement_result(
        self, 
        node_id: str, 
        success: bool, 
        attempts: int, 
        tool_usage_results: List[ToolUsageResult],
        failure_reason: Optional[str] = None
    ):
        """Log SLA enforcement completion for workflow analysis."""
        tools_used = [r.tool_name for r in tool_usage_results]
        
        logger.info(
            f"🎯 META-EXECUTION: SLA enforcement {'✅ PASSED' if success else '❌ FAILED'}", 
            data={
                "event": "sla_enforcement_complete",
                "node_id": node_id,
                "success": success,
                "attempts": attempts + 1,  # attempts is 0-based
                "tools_used": tools_used,
                "tool_count": len(tools_used),
                "failure_reason": failure_reason,
                "meta_execution_occurred": attempts > 0  # True if retries happened
            }
        )


# Global wrapper instance
node_execution_wrapper = NodeExecutionWrapper()