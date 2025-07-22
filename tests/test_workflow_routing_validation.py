#!/usr/bin/env python3
"""
Comprehensive test suite for workflow routing validation.

Tests the routing consistency validation that ensures:
1. Decision/routing nodes have proper conditional outgoing edges
2. No dangling conditional edges exist
3. All conditional edges point to valid targets
4. No isolated nodes with missing connections

This validation helps catch edge condition mismatches at workflow generation time,
providing structured feedback to agents for correction.
"""

import pytest
from uuid import uuid4

from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec, NodeSpec, EdgeSpec, NodeData, EdgeData
)


class TestWorkflowRoutingValidation:
    """Test suite for workflow routing validation."""
    
    def test_valid_routing_workflow_passes(self):
        """Test that properly configured routing workflow passes validation."""
        nodes = [
            NodeSpec(
                id="decision_1",
                type="decision",
                label="Trading Decision",
                data=NodeData(
                    tool_name="conditional_gate",
                    config={
                        "conditions": [
                            {"field": "action", "operator": "==", "value": "buy", "route": "buy_signal"},
                            {"field": "action", "operator": "==", "value": "sell", "route": "sell_signal"}
                        ],
                        "default_route": "hold"
                    },
                    ins=["market_data"],
                    outs=["decision_result"]
                )
            ),
            NodeSpec(
                id="buy_agent",
                type="agent",
                label="Buy Agent",
                data=NodeData(
                    agent_instructions="Execute buy order",
                    ins=["decision_result"],
                    outs=["buy_result"]
                )
            ),
            NodeSpec(
                id="sell_agent",
                type="agent",
                label="Sell Agent",
                data=NodeData(
                    agent_instructions="Execute sell order",
                    ins=["decision_result"],
                    outs=["sell_result"]
                )
            )
        ]
        
        edges = [
            EdgeSpec(
                id="edge_1",
                source="decision_1",
                target="buy_agent",
                data=EdgeData(condition="routed_to == 'buy_signal'")
            ),
            EdgeSpec(
                id="edge_2",
                source="decision_1", 
                target="sell_agent",
                data=EdgeData(condition="routed_to == 'sell_signal'")
            )
        ]
        
        spec = WorkflowSpec(
            id=uuid4(),
            rev=1,
            title="Valid Routing Workflow",
            description="Properly configured routing workflow",
            nodes=nodes,
            edges=edges
        )
        
        issues = spec.validate_structure()
        
        # Should have no routing validation issues
        routing_issues = [issue for issue in issues if any(keyword in issue for keyword in [
            "DANGLING ROUTING NODE", "MISSING CONDITIONS", "BROKEN EDGE", "ORPHANED CONDITION", "ISOLATED NODE"
        ])]
        
        assert len(routing_issues) == 0, f"Valid workflow should not have routing issues: {routing_issues}"
    
    def test_routing_node_with_no_edges_fails(self):
        """Test that routing node with no outgoing edges fails validation."""
        nodes = [
            NodeSpec(
                id="decision_1",
                type="decision",
                label="Isolated Decision",
                data=NodeData(
                    tool_name="conditional_gate",
                    config={},
                    ins=["data"],
                    outs=["result"]
                )
            ),
            NodeSpec(
                id="agent_1",
                type="agent",
                label="Unreachable Agent",
                data=NodeData(
                    agent_instructions="This agent is unreachable",
                    ins=["result"],
                    outs=["final"]
                )
            )
        ]
        
        # No edges connecting the decision node
        edges = []
        
        spec = WorkflowSpec(
            id=uuid4(),
            rev=1,
            title="Dangling Routing Node",
            description="Decision node with no outgoing edges",
            nodes=nodes,
            edges=edges
        )
        
        issues = spec.validate_structure()
        
        # Should detect dangling routing node
        dangling_issues = [issue for issue in issues if "DANGLING ROUTING NODE" in issue]
        assert len(dangling_issues) >= 1, f"Should detect dangling routing node: {issues}"
        
        # Should also detect isolated node
        isolated_issues = [issue for issue in issues if "ISOLATED NODE" in issue]
        assert len(isolated_issues) >= 1, f"Should detect isolated nodes: {issues}"
    
    def test_routing_node_with_unconditional_edges_fails(self):
        """Test that routing node with only unconditional edges fails validation."""
        nodes = [
            NodeSpec(
                id="decision_1", 
                type="decision",
                label="Bad Decision Node",
                data=NodeData(
                    tool_name="conditional_gate",
                    config={},
                    ins=["data"],
                    outs=["result"]
                )
            ),
            NodeSpec(
                id="agent_1",
                type="agent",
                label="Target Agent",
                data=NodeData(
                    agent_instructions="Process result",
                    ins=["result"],
                    outs=["final"]
                )
            )
        ]
        
        edges = [
            EdgeSpec(
                id="edge_1",
                source="decision_1",
                target="agent_1",
                data=EdgeData()  # No condition - this is invalid for routing nodes
            )
        ]
        
        spec = WorkflowSpec(
            id=uuid4(),
            rev=1,
            title="Unconditional Routing",
            description="Routing node with unconditional edges",
            nodes=nodes,
            edges=edges
        )
        
        issues = spec.validate_structure()
        
        # Should detect missing conditions
        condition_issues = [issue for issue in issues if "MISSING CONDITIONS" in issue]
        assert len(condition_issues) >= 1, f"Should detect missing conditions: {issues}"
    
    def test_broken_edge_target_fails(self):
        """Test that edge pointing to non-existent target fails validation."""
        nodes = [
            NodeSpec(
                id="decision_1",
                type="decision", 
                label="Decision Node",
                data=NodeData(
                    tool_name="conditional_gate",
                    config={},
                    ins=["data"],
                    outs=["result"]
                )
            )
        ]
        
        edges = [
            EdgeSpec(
                id="edge_1",
                source="decision_1",
                target="non_existent_node",  # This node doesn't exist
                data=EdgeData(condition="routed_to == 'test'")
            )
        ]
        
        spec = WorkflowSpec(
            id=uuid4(),
            rev=1,
            title="Broken Edge",
            description="Edge pointing to non-existent target", 
            nodes=nodes,
            edges=edges
        )
        
        issues = spec.validate_structure()
        
        # Should detect broken edge (from both routing validation and basic structure validation)
        broken_issues = [issue for issue in issues if any(keyword in issue for keyword in [
            "BROKEN EDGE", "unknown target", "references unknown target"
        ])]
        assert len(broken_issues) >= 1, f"Should detect broken edge: {issues}"
    
    def test_orphaned_conditional_edge_fails(self):
        """Test that conditional edge from non-routing node fails validation."""
        nodes = [
            NodeSpec(
                id="agent_1",
                type="agent",
                label="Regular Agent", 
                data=NodeData(
                    agent_instructions="Regular processing",
                    tools=["some_regular_tool"],  # Not a routing tool
                    ins=["data"],
                    outs=["result"]
                )
            ),
            NodeSpec(
                id="agent_2", 
                type="agent",
                label="Target Agent",
                data=NodeData(
                    agent_instructions="Target processing",
                    ins=["result"],
                    outs=["final"]
                )
            )
        ]
        
        edges = [
            EdgeSpec(
                id="edge_1",
                source="agent_1",
                target="agent_2", 
                data=EdgeData(condition="status == 'success'")  # Conditional edge from non-routing node
            )
        ]
        
        spec = WorkflowSpec(
            id=uuid4(),
            rev=1,
            title="Orphaned Condition",
            description="Conditional edge from non-routing node",
            nodes=nodes,
            edges=edges
        )
        
        issues = spec.validate_structure()
        
        # Should detect orphaned condition
        orphaned_issues = [issue for issue in issues if "ORPHANED CONDITION" in issue]
        assert len(orphaned_issues) >= 1, f"Should detect orphaned condition: {issues}"
    
    def test_agent_with_routing_tools_validates_correctly(self):
        """Test that agent nodes with routing tools are properly validated."""
        nodes = [
            NodeSpec(
                id="routing_agent",
                type="agent",
                label="Routing Agent",
                data=NodeData(
                    agent_instructions="Make routing decisions",
                    tools=["conditional_gate", "some_other_tool"],  # Has routing tool
                    ins=["data"],
                    outs=["decision"]
                )
            ),
            NodeSpec(
                id="target_1",
                type="agent",
                label="Path 1",
                data=NodeData(
                    agent_instructions="Handle path 1",
                    ins=["decision"],
                    outs=["result_1"]
                )
            ),
            NodeSpec(
                id="target_2",
                type="agent", 
                label="Path 2",
                data=NodeData(
                    agent_instructions="Handle path 2",
                    ins=["decision"],
                    outs=["result_2"]
                )
            )
        ]
        
        edges = [
            EdgeSpec(
                id="edge_1",
                source="routing_agent",
                target="target_1",
                data=EdgeData(condition="routed_to == 'path_1'")
            ),
            EdgeSpec(
                id="edge_2",
                source="routing_agent",
                target="target_2", 
                data=EdgeData(condition="routed_to == 'path_2'")
            )
        ]
        
        spec = WorkflowSpec(
            id=uuid4(),
            rev=1,
            title="Agent Routing",
            description="Agent with routing tools",
            nodes=nodes,
            edges=edges
        )
        
        issues = spec.validate_structure()
        
        # Should have no routing issues since agent has routing tools and proper edges
        routing_issues = [issue for issue in issues if any(keyword in issue for keyword in [
            "DANGLING ROUTING NODE", "MISSING CONDITIONS", "ORPHANED CONDITION"
        ])]
        
        assert len(routing_issues) == 0, f"Agent with routing tools should validate correctly: {routing_issues}"
    
    def test_multiple_routing_tools_validation(self):
        """Test validation of workflows with multiple routing tools."""
        nodes = [
            NodeSpec(
                id="threshold_gate",
                type="decision",
                label="Threshold Decision",
                data=NodeData(
                    tool_name="threshold_gate",
                    config={"thresholds": {"high": 80, "medium": 50, "low": 20}},
                    ins=["value"],
                    outs=["threshold_result"]
                )
            ),
            NodeSpec(
                id="percentage_gate",
                type="decision", 
                label="Conditional Decision",
                data=NodeData(
                    tool_name="conditional_gate",
                    config={"routes": {"buy_path": "Buy signal", "sell_path": "Sell signal"}},
                    ins=["prices"],
                    outs=["percentage_result"]
                )
            ),
            NodeSpec(
                id="multi_gate",
                type="decision",
                label="Multi Decision", 
                data=NodeData(
                    tool_name="conditional_multi_gate",
                    config={"conditions": []},
                    ins=["data"],
                    outs=["multi_result"]
                )
            ),
            NodeSpec(
                id="final_agent",
                type="agent",
                label="Final Processor",
                data=NodeData(
                    agent_instructions="Process final results",
                    ins=["threshold_result", "percentage_result", "multi_result"],
                    outs=["final"]
                )
            )
        ]
        
        edges = [
            # Threshold gate edges
            EdgeSpec(
                id="edge_1",
                source="threshold_gate",
                target="final_agent",
                data=EdgeData(condition="routed_to == 'high'")
            ),
            # Percentage gate edges  
            EdgeSpec(
                id="edge_2",
                source="percentage_gate",
                target="final_agent",
                data=EdgeData(condition="routed_to == 'buy_path'")
            ),
            # Multi gate edges
            EdgeSpec(
                id="edge_3",
                source="multi_gate",
                target="final_agent",
                data=EdgeData(condition="routed_to == 'approved'")
            )
        ]
        
        spec = WorkflowSpec(
            id=uuid4(),
            rev=1,
            title="Multiple Routing Tools",
            description="Workflow with multiple routing tools",
            nodes=nodes,
            edges=edges
        )
        
        issues = spec.validate_structure()
        
        # Should validate all routing tools correctly
        routing_issues = [issue for issue in issues if any(keyword in issue for keyword in [
            "DANGLING ROUTING NODE", "MISSING CONDITIONS", "BROKEN EDGE", "ORPHANED CONDITION"
        ])]
        
        # All routing nodes have conditional edges, so should be valid
        assert len(routing_issues) == 0, f"Multiple routing tools should validate correctly: {routing_issues}"
    
    def test_complex_routing_workflow_validation(self):
        """Test validation of complex workflow with multiple decision points."""
        nodes = [
            NodeSpec(
                id="input_node",
                type="tool",
                label="Data Input",
                data=NodeData(
                    tool_name="data_source",
                    config={},
                    outs=["raw_data"]
                )
            ),
            NodeSpec(
                id="primary_decision",
                type="decision",
                label="Primary Router",
                data=NodeData(
                    tool_name="conditional_gate",
                    config={
                        "conditions": [
                            {"field": "category", "operator": "==", "value": "urgent", "route": "urgent_path"},
                            {"field": "category", "operator": "==", "value": "normal", "route": "normal_path"}
                        ],
                        "default_route": "default_path"
                    },
                    ins=["raw_data"],
                    outs=["primary_decision"]
                )
            ),
            NodeSpec(
                id="urgent_processor",
                type="agent",
                label="Urgent Processor",
                data=NodeData(
                    agent_instructions="Handle urgent items with priority routing",
                    tools=["conditional_gate"],  # Agent can also route
                    ins=["primary_decision"],
                    outs=["urgent_result"]
                )
            ),
            NodeSpec(
                id="normal_processor",
                type="agent",
                label="Normal Processor", 
                data=NodeData(
                    agent_instructions="Handle normal items",
                    ins=["primary_decision"],
                    outs=["normal_result"]
                )
            ),
            NodeSpec(
                id="escalation_handler",
                type="agent",
                label="Escalation Handler",
                data=NodeData(
                    agent_instructions="Handle escalated urgent items",
                    ins=["urgent_result"],
                    outs=["escalation_result"]
                )
            ),
            NodeSpec(
                id="final_output",
                type="tool",
                label="Output Collector",
                data=NodeData(
                    tool_name="output_collector",
                    config={},
                    ins=["urgent_result", "normal_result", "escalation_result"],
                    outs=["final_output"]
                )
            )
        ]
        
        edges = [
            # Input to primary decision
            EdgeSpec(
                id="e1",
                source="input_node",
                target="primary_decision"
            ),
            # Primary decision routing
            EdgeSpec(
                id="e2",
                source="primary_decision",
                target="urgent_processor",
                data=EdgeData(condition="routed_to == 'urgent_path'")
            ),
            EdgeSpec(
                id="e3", 
                source="primary_decision",
                target="normal_processor",
                data=EdgeData(condition="routed_to == 'normal_path'")
            ),
            # Urgent processor sub-routing (agent with routing tools)
            EdgeSpec(
                id="e4",
                source="urgent_processor", 
                target="escalation_handler",
                data=EdgeData(condition="routed_to == 'escalate'")
            ),
            # Final collection (unconditional edges to output)
            EdgeSpec(
                id="e5",
                source="urgent_processor",
                target="final_output"
            ),
            EdgeSpec(
                id="e6",
                source="normal_processor", 
                target="final_output"
            ),
            EdgeSpec(
                id="e7",
                source="escalation_handler",
                target="final_output"
            )
        ]
        
        spec = WorkflowSpec(
            id=uuid4(),
            rev=1,
            title="Complex Routing Workflow",
            description="Multi-level routing with agents and decision nodes",
            nodes=nodes,
            edges=edges
        )
        
        issues = spec.validate_structure()
        
        # Should validate complex routing correctly
        routing_issues = [issue for issue in issues if any(keyword in issue for keyword in [
            "DANGLING ROUTING NODE", "MISSING CONDITIONS", "BROKEN EDGE", "ORPHANED CONDITION", "ISOLATED NODE"
        ])]
        
        assert len(routing_issues) == 0, f"Complex routing workflow should validate correctly: {routing_issues}"
    
    def test_validation_provides_structured_feedback(self):
        """Test that validation provides clear, structured feedback for routing issues."""
        nodes = [
            NodeSpec(
                id="bad_decision",
                type="decision",
                label="Problematic Decision",
                data=NodeData(
                    tool_name="conditional_gate",
                    config={},
                    ins=["data"], 
                    outs=["result"]
                )
            )
        ]
        
        # No edges - will cause multiple validation issues
        edges = []
        
        spec = WorkflowSpec(
            id=uuid4(),
            rev=1, 
            title="Validation Feedback Test",
            description="Test validation feedback quality",
            nodes=nodes,
            edges=edges
        )
        
        issues = spec.validate_structure()
        
        # Should provide clear, structured feedback
        assert len(issues) > 0, "Should detect validation issues"
        
        # Check that issues are well-formatted with emojis and clear descriptions
        for issue in issues:
            assert isinstance(issue, str), "Issues should be strings"
            # Should have emoji indicators for severity
            has_emoji = any(emoji in issue for emoji in ["🚨", "⚠️", "🔍"])
            assert has_emoji or "DANGLING" in issue or "ISOLATED" in issue, f"Issue should have clear formatting: {issue}"
    
    def test_real_world_routing_mismatch_scenario(self):
        """Test scenario from real trading workflow where edge conditions don't match tool outputs."""
        # This tests the original issue: conditional_gate outputs 'buy_signal'/'sell_signal'
        # but workflow edges expect 'buy'/'sell' 
        nodes = [
            NodeSpec(
                id="decision_1",
                type="decision",
                label="Trading Decision",
                data=NodeData(
                    tool_name="conditional_gate",
                    config={
                        "conditions": [
                            {"field": "action", "operator": "==", "value": "buy", "route": "buy_signal"},
                            {"field": "action", "operator": "==", "value": "sell", "route": "sell_signal"}
                        ],
                        "default_route": "hold"
                    },
                    ins=["market_data"],
                    outs=["decision_result"]
                )
            ),
            NodeSpec(
                id="buy_agent",
                type="agent",
                label="Buy Agent",
                data=NodeData(
                    agent_instructions="Execute buy order",
                    ins=["decision_result"],
                    outs=["buy_result"]
                )
            ),
            NodeSpec(
                id="sell_agent",
                type="agent",
                label="Sell Agent",
                data=NodeData(
                    agent_instructions="Execute sell order",
                    ins=["decision_result"],
                    outs=["sell_result"]
                )
            )
        ]
        
        # Edge conditions that DON'T match the tool's configured route outputs
        # Tool outputs: 'buy_signal', 'sell_signal' 
        # Edge conditions: 'buy', 'sell' <- This mismatch causes runtime failures
        edges = [
            EdgeSpec(
                id="edge_1",
                source="decision_1",
                target="buy_agent",
                data=EdgeData(condition="routed_to == 'buy'")  # MISMATCH: should be 'buy_signal'
            ),
            EdgeSpec(
                id="edge_2",
                source="decision_1",
                target="sell_agent",
                data=EdgeData(condition="routed_to == 'sell'")  # MISMATCH: should be 'sell_signal'  
            )
        ]
        
        spec = WorkflowSpec(
            id=uuid4(),
            rev=1,
            title="Real World Routing Mismatch",
            description="Trading workflow with route name mismatches",
            nodes=nodes,
            edges=edges
        )
        
        issues = spec.validate_structure()
        
        # Current validation focuses on graph structure, not semantic route name matching
        # The workflow is structurally valid: decision node has conditional outgoing edges
        routing_issues = [issue for issue in issues if any(keyword in issue for keyword in [
            "DANGLING ROUTING NODE", "MISSING CONDITIONS", "BROKEN EDGE", "ORPHANED CONDITION"
        ])]
        
        # Should pass structural validation despite semantic mismatch
        assert len(routing_issues) == 0, f"Workflow is structurally valid: {routing_issues}"
        
        # NOTE: This test documents current behavior - we validate graph structure
        # Semantic validation of route names vs tool outputs could be future enhancement
        print(f"✅ Structural validation passed. Found {len(issues)} other issues: {issues}")
    


if __name__ == "__main__":
    pytest.main([__file__, "-v"])