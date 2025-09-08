"""
Test Suite for Node Execution Wrapper
=====================================

Tests the runtime SLA enforcement system that acts as a message passing
gatekeeper for workflow nodes.
"""

import pytest
import asyncio

from iointel.src.utilities.node_execution_wrapper import (
    NodeExecutionWrapper,
    SLARequirements,
    SLAValidationResult,
    ExecutionContext
)
from iointel.src.agent_methods.data_models.datamodels import ToolUsageResult
from iointel.src.agent_methods.data_models.decision_tools_catalog import (
    get_sla_requirements_for_tools
)


class TestSLARequirementsExtraction:
    """Test SLA requirements extraction from node data."""
    
    def test_explicit_sla_requirements(self):
        """Should use explicit SLA requirements when provided."""
        wrapper = NodeExecutionWrapper()
        
        node_data = {
            "sla_requirements": {
                "tool_usage_required": True,
                "required_tools": ["conditional_gate"],
                "final_tool_must_be": "conditional_gate",
                "min_tool_calls": 1,
                "enforce_usage": True
            }
        }
        
        requirements = wrapper.extract_sla_requirements(node_data)
        assert requirements.tool_usage_required is True
        assert requirements.required_tools == ["conditional_gate"]
        assert requirements.final_tool_must_be == "conditional_gate"
        assert requirements.enforce_usage is True
    
    def test_generated_from_tools_catalog(self):
        """Should generate SLA requirements from tools using catalog."""
        wrapper = NodeExecutionWrapper()
        
        node_data = {
            "tools": ["conditional_gate", "get_current_stock_price"]
        }
        
        requirements = wrapper.extract_sla_requirements(node_data)
        assert requirements.enforce_usage is True
        assert "conditional_gate" in requirements.required_tools
        assert requirements.final_tool_must_be == "conditional_gate"
    
    def test_no_sla_requirements(self):
        """Should return no enforcement for nodes without special tools."""
        wrapper = NodeExecutionWrapper()
        
        node_data = {
            "tools": ["add", "multiply"]  # No SLA tools
        }
        
        requirements = wrapper.extract_sla_requirements(node_data)
        assert requirements.enforce_usage is False


class TestSLAValidation:
    """Test SLA validation logic."""
    
    def test_validation_pass_with_required_tools(self):
        """Should pass when required tools are used."""
        wrapper = NodeExecutionWrapper()
        
        requirements = SLARequirements(
            enforce_usage=True,
            tool_usage_required=True,
            required_tools=["conditional_gate"],
            min_tool_calls=1
        )
        
        tool_usage = [
            ToolUsageResult(
                tool_name="conditional_gate",
                tool_args={"condition": "price > 100"},
                tool_result={"routed_to": "buy_path"}
            )
        ]
        
        result, reason = wrapper.validate_sla_compliance(requirements, tool_usage)
        assert result == SLAValidationResult.PASS
        assert "satisfied" in reason.lower()
    
    def test_validation_fail_no_tools(self):
        """Should fail when tools required but none used."""
        wrapper = NodeExecutionWrapper()
        
        requirements = SLARequirements(
            enforce_usage=True,
            tool_usage_required=True,
            min_tool_calls=1
        )
        
        result, reason = wrapper.validate_sla_compliance(requirements, [])
        assert result == SLAValidationResult.FAIL_NO_TOOLS
        assert "no tools were used" in reason.lower()
    
    def test_validation_fail_wrong_final_tool(self):
        """Should fail when final tool requirement not met."""
        wrapper = NodeExecutionWrapper()
        
        requirements = SLARequirements(
            enforce_usage=True,
            final_tool_must_be="conditional_gate"
        )
        
        tool_usage = [
            ToolUsageResult(
                tool_name="get_current_stock_price",
                tool_args={},
                tool_result=150.0
            )
        ]
        
        result, reason = wrapper.validate_sla_compliance(requirements, tool_usage)
        assert result == SLAValidationResult.FAIL_WRONG_FINAL_TOOL
        assert "final tool must be" in reason.lower()
    
    def test_validation_skip_when_no_enforcement(self):
        """Should always pass when enforcement disabled."""
        wrapper = NodeExecutionWrapper()
        
        requirements = SLARequirements(enforce_usage=False)
        
        result, reason = wrapper.validate_sla_compliance(requirements, [])
        assert result == SLAValidationResult.PASS
        assert "no sla enforcement" in reason.lower()


class TestPromptEnhancement:
    """Test prompt enhancement for retries."""
    
    def test_enhanced_prompt_includes_requirements(self):
        """Enhanced prompts should include SLA requirements."""
        wrapper = NodeExecutionWrapper()
        
        context = ExecutionContext(
            node_id="test_node",
            node_type="agent",
            node_label="Test Agent",
            input_data={"query": "Original query"},
            attempt=1,
            sla_requirements=SLARequirements(
                required_tools=["conditional_gate"],
                final_tool_must_be="conditional_gate"
            )
        )
        
        enhanced_input = wrapper.create_enhanced_prompt(
            context,
            SLAValidationResult.FAIL_NO_TOOLS,
            "No tools were used"
        )
        
        assert isinstance(enhanced_input, dict)
        assert "conditional_gate" in enhanced_input["query"]
        assert "SLA VALIDATION FAILED" in enhanced_input["query"]
        assert "REQUIREMENTS:" in enhanced_input["query"]
    
    def test_progressive_retry_prompts(self):
        """Retry prompts should get stronger with each attempt."""
        wrapper = NodeExecutionWrapper()
        
        context = ExecutionContext(
            node_id="test_node",
            node_type="agent", 
            node_label="Test Agent",
            input_data={"query": "Original query"},
            sla_requirements=SLARequirements(required_tools=["conditional_gate"])
        )
        
        # First retry
        context.attempt = 1
        enhanced_1 = wrapper.create_enhanced_prompt(
            context, SLAValidationResult.FAIL_NO_TOOLS, "No tools"
        )
        
        # Second retry  
        context.attempt = 2
        enhanced_2 = wrapper.create_enhanced_prompt(
            context, SLAValidationResult.FAIL_NO_TOOLS, "No tools"
        )
        
        # Second attempt should be stronger
        assert "FINAL ATTEMPT" in enhanced_2["query"]
        assert "RETRY REQUIRED" in enhanced_1["query"]


class TestFullExecutionWorkflow:
    """Test the complete execution workflow with SLA enforcement."""
    
    @pytest.mark.asyncio
    async def test_execution_pass_first_attempt(self):
        """Should pass on first attempt when SLA met."""
        wrapper = NodeExecutionWrapper()
        
        # Mock executor that returns proper tool usage
        async def mock_executor():
            return {
                "result": "Decision made", 
                "tool_usage_results": [
                    ToolUsageResult(
                        tool_name="conditional_gate",
                        tool_args={},
                        tool_result={"routed_to": "buy_path"}
                    )
                ]
            }
        
        node_data = {
            "tools": ["conditional_gate"]
        }
        
        result = await wrapper.execute_with_sla_enforcement(
            node_executor=mock_executor,
            node_data=node_data,
            input_data={"query": "Should I buy?"},
            node_id="decision_node"
        )
        
        assert result["result"] == "Decision made"
        assert len(result["tool_usage_results"]) == 1
    
    @pytest.mark.asyncio
    async def test_execution_retry_then_pass(self):
        """Should retry and eventually pass."""
        wrapper = NodeExecutionWrapper()
        
        call_count = 0
        
        async def mock_executor():
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                # First attempt - no tools
                return {"result": "Just analysis"}
            else:
                # Second attempt - with tools
                return {
                    "result": "Decision with tools",
                    "tool_usage_results": [
                        ToolUsageResult(
                            tool_name="conditional_gate",
                            tool_args={},
                            tool_result={"routed_to": "buy_path"}
                        )
                    ]
                }
        
        node_data = {
            "sla_requirements": {
                "enforce_usage": True,
                "tool_usage_required": True,
                "required_tools": ["conditional_gate"],
                "max_retries": 2
            }
        }
        
        result = await wrapper.execute_with_sla_enforcement(
            node_executor=mock_executor,
            node_data=node_data,
            input_data={"query": "Should I buy?"},
            node_id="decision_node"
        )
        
        assert call_count == 2  # Should have retried once
        assert result["result"] == "Decision with tools"
    
    @pytest.mark.asyncio
    async def test_execution_no_enforcement_skip(self):
        """Should skip enforcement when not required."""
        wrapper = NodeExecutionWrapper()
        
        async def mock_executor():
            return {"result": "Analysis without tools"}
        
        node_data = {
            "tools": ["add", "multiply"]  # No SLA-triggering tools
        }
        
        result = await wrapper.execute_with_sla_enforcement(
            node_executor=mock_executor,
            node_data=node_data,
            input_data={"query": "Calculate something"},
            node_id="calc_node"
        )
        
        assert result["result"] == "Analysis without tools"


class TestCatalogIntegration:
    """Test integration with the decision tools catalog."""
    
    def test_conditional_gate_triggers_enforcement(self):
        """conditional_gate should trigger SLA enforcement."""
        requirements = get_sla_requirements_for_tools(["conditional_gate"])
        
        assert requirements.enforce_usage is True
        assert requirements.final_tool_must_be == "conditional_gate"
        assert "conditional_gate" in requirements.required_tools
    
    def test_mixed_tools_with_enforcement(self):
        """Mixed tools with some triggering enforcement."""
        requirements = get_sla_requirements_for_tools([
            "get_current_stock_price",
            "conditional_gate", 
            "add"
        ])
        
        assert requirements.enforce_usage is True
        assert requirements.final_tool_must_be == "conditional_gate"
    
    def test_no_enforcement_tools(self):
        """Tools that don't trigger enforcement."""
        requirements = get_sla_requirements_for_tools(["add", "multiply"])
        
        assert requirements.enforce_usage is False


if __name__ == "__main__":
    # Run a simple test
    async def test_basic():
        wrapper = NodeExecutionWrapper()
        
        # Test SLA extraction
        node_data = {"tools": ["conditional_gate"]}
        requirements = wrapper.extract_sla_requirements(node_data)
        print(f"Requirements: {requirements}")
        
        # Test validation
        tool_usage = [
            ToolUsageResult(
                tool_name="conditional_gate",
                tool_args={},
                tool_result={"routed_to": "test"}
            )
        ]
        result, reason = wrapper.validate_sla_compliance(requirements, tool_usage)
        print(f"Validation: {result}, {reason}")
    
    asyncio.run(test_basic())