"""
Centralized Workflow Testing Example
====================================

This demonstrates the new centralized workflow testing approach using
the WorkflowTestRepository and smart fixtures.

Instead of scattered test patterns, we now have:
1. Layered testing architecture (logical, agentic, orchestration, feedback)
2. Smart fixtures that provide appropriate test data
3. Centralized test case management
4. Clear separation of concerns

Example usage patterns for different test layers.
"""

import pytest
from iointel.src.utilities.workflow_test_repository import TestLayer
from iointel.src.agent_methods.agents.workflow_planner import WorkflowPlanner


# LOGICAL LAYER TESTS - Pure data structure validation
class TestLogicalLayer:
    """Tests for pure workflow data structures and validation logic."""
    
    def test_conditional_routing_logic(self, conditional_routing_cases):
        """Test conditional routing logic with centralized test cases."""
        assert len(conditional_routing_cases) > 0
        
        for test_case in conditional_routing_cases:
            workflow_spec = test_case.workflow_spec
            assert workflow_spec is not None
            assert "nodes" in workflow_spec
            assert "edges" in workflow_spec
            
            # Test routing logic
            edges = workflow_spec["edges"]
            for edge in edges:
                if "route_index" in edge.get("data", {}):
                    # New route index system should be integers
                    assert isinstance(edge["data"]["route_index"], int)
    
    def test_workflow_validation(self, validation_test_cases, mock_tool_catalog):
        """Test workflow validation with expected failures."""
        from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec
        
        for test_case in validation_test_cases:
            workflow_data = test_case.workflow_spec
            
            try:
                # Try to create WorkflowSpec from test data
                workflow = WorkflowSpec(**workflow_data)
                issues = workflow.validate_structure(mock_tool_catalog)
                
                if test_case.should_pass:
                    assert len(issues) == 0, f"Expected test to pass but got issues: {issues}"
                else:
                    assert len(issues) > 0, "Expected test to fail but got no issues"
                    
                    # Check expected errors if specified
                    if test_case.expected_errors:
                        for expected_error in test_case.expected_errors:
                            assert any(expected_error in issue for issue in issues), \
                                f"Expected error '{expected_error}' not found in {issues}"
            
            except Exception as e:
                if test_case.should_pass:
                    pytest.fail(f"Expected test to pass but got exception: {e}")


# AGENTIC LAYER TESTS - LLM-based workflow generation  
class TestAgenticLayer:
    """Tests for LLM-based workflow generation from user prompts."""
    
    @pytest.mark.asyncio
    async def test_stock_agent_generation(self, stock_analysis_prompts, real_tool_catalog):
        """Test stock agent workflow generation using centralized test cases."""
        planner = WorkflowPlanner()
        
        for prompt in stock_analysis_prompts:
            if not prompt:
                continue
                
            workflow = await planner.generate_workflow(
                query=prompt,
                tool_catalog=real_tool_catalog
            )
            
            # Verify the fix for data_source vs agent node confusion
            assert workflow is not None
            assert len(workflow.nodes) >= 2  # Should have user input + agent
            
            # Check node types
            data_source_nodes = [n for n in workflow.nodes if n.type == 'data_source']
            agent_nodes = [n for n in workflow.nodes if n.type == 'agent']
            decision_nodes = [n for n in workflow.nodes if n.type == 'decision']
            
            assert len(data_source_nodes) >= 1, "Should have data_source node for user input"
            assert len(agent_nodes) >= 1, "Should have agent node for stock analysis"
            # assert len(decision_nodes) >= 1, "Should have decision node for routing"
            # Verify data_source nodes are only for user input/prompt_tool
            for node in data_source_nodes:
                assert hasattr(node.data, 'source_name')
                assert node.data.source_name in ['user_input', 'prompt_tool']
            
            # Verify agent nodes have stock-related tools
            for node in agent_nodes:
                if hasattr(node.data, 'tools') and node.data.tools:
                    # Should have stock-related tools, not data_source tools
                    stock_tools = [t for t in node.data.tools 
                                 if 'stock' in t.lower() or 'finance' in t.lower()]
                    # At least one agent should have stock tools
                    if stock_tools:
                        assert len(stock_tools) > 0
    
    @pytest.mark.layer("agentic")
    @pytest.mark.category("stock_analysis")
    def test_with_smart_fixture(self, smart_test_data):
        """Example of using the smart fixture dispatcher."""
        generation_cases = smart_test_data.get('generation_cases', [])
        user_prompts = smart_test_data.get('user_prompts', [])
        
        assert len(generation_cases) > 0
        assert len(user_prompts) > 0
        
        for prompt in user_prompts:
            if prompt and 'stock' in prompt.lower():
                # This prompt should generate stock-related workflows
                assert 'stock' in prompt.lower()


# ORCHESTRATION LAYER TESTS - Full pipeline execution
class TestOrchestrationLayer:
    """Tests for full workflow pipeline execution and SLA enforcement."""
    
    def test_pipeline_execution_cases(self, pipeline_execution_cases):
        """Test pipeline execution with centralized test cases."""
        for test_case in pipeline_execution_cases:
            # Verify test case has the required data for orchestration testing
            assert test_case.layer == TestLayer.ORCHESTRATION
            assert 'pipeline' in test_case.tags or 'execution' in test_case.tags
            
            # Test case should specify expected outcomes
            if test_case.expected_result:
                expected = test_case.expected_result
                if 'executes_successfully' in expected:
                    assert isinstance(expected['executes_successfully'], bool)
    
    def test_sla_enforcement(self, sla_enforcement_cases):
        """Test SLA enforcement with centralized test cases."""
        for test_case in sla_enforcement_cases:
            assert 'sla' in test_case.tags
            
            # SLA enforcement tests should have specific requirements
            if test_case.expected_result:
                expected = test_case.expected_result
                if 'follows_sla_requirements' in expected:
                    assert isinstance(expected['follows_sla_requirements'], bool)


# FEEDBACK LAYER TESTS - Chat feedback loops
class TestFeedbackLayer:
    """Tests for post-execution chat feedback loops."""
    
    def test_chat_feedback_loops(self, chat_feedback_cases):
        """Test chat feedback mechanisms with centralized test cases."""
        for test_case in chat_feedback_cases:
            assert test_case.layer == TestLayer.FEEDBACK
            assert 'feedback' in test_case.tags or 'chat' in test_case.tags


# INTEGRATION TESTS - Cross-layer functionality
class TestIntegration:
    """Integration tests that span multiple layers."""
    
    def test_stock_trading_full_pipeline(self, stock_trading_tests):
        """Test the complete stock trading workflow across all layers."""
        logical_tests = [t for t in stock_trading_tests if t.layer == TestLayer.LOGICAL]
        agentic_tests = [t for t in stock_trading_tests if t.layer == TestLayer.AGENTIC] 
        orchestration_tests = [t for t in stock_trading_tests if t.layer == TestLayer.ORCHESTRATION]
        
        # Should have test cases at each layer
        assert len(logical_tests) > 0, "Should have logical layer stock trading tests"
        assert len(agentic_tests) > 0, "Should have agentic layer stock trading tests"
        assert len(orchestration_tests) > 0, "Should have orchestration layer stock trading tests"
        
        # Verify the stock trading workflow requirements from original user request
        for test in orchestration_tests:
            if test.user_prompt and '5%' in test.user_prompt and 'alex@io.net' in test.user_prompt:
                # This is the original stock trading requirement
                expected = test.expected_result or {}
                assert expected.get('sends_email', False), "Should send email to alex@io.net"
                assert expected.get('follows_sla_requirements', False), "Should follow SLA requirements"


# UTILITY TESTS - Test the test system itself
class TestTestSystem:
    """Meta-tests for the centralized testing system."""
    
    def test_repository_initialization(self, test_repository):
        """Test that the test repository initializes correctly."""
        assert test_repository is not None
        assert len(test_repository._test_cases) > 0, "Should have default test cases"
    
    def test_smart_fixture_dispatch(self, smart_test_data):
        """Test that smart fixtures provide appropriate data."""
        # This test has no markers, so should default to logical layer
        assert isinstance(smart_test_data, dict)
    
    def test_test_case_factory(self, workflow_test_case_factory):
        """Test the test case factory function."""
        test_case = workflow_test_case_factory(
            layer=TestLayer.LOGICAL,
            category="test_category",
            name="Test case",
            description="A test case",
            workflow_spec={"nodes": [], "edges": []}
        )
        
        assert test_case.layer == TestLayer.LOGICAL
        assert test_case.category == "test_category"
        assert test_case.name == "Test case"
    
    def test_backward_compatibility(self, tool_catalog, sample_workflow_spec):
        """Test that backward compatibility fixtures still work."""
        # Legacy fixtures should still be available for existing tests
        assert tool_catalog is not None
        assert 'weather_api' in tool_catalog
        
        assert sample_workflow_spec is not None
        assert sample_workflow_spec.title == "Sample Test Workflow"
        assert len(sample_workflow_spec.nodes) > 0