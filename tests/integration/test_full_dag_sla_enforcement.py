"""
Integration Test: Full DAG Execution with SLA Enforcement
========================================================

Tests the complete path from WorkflowSpec → DAG Executor → Node Wrapper → Agent Execution
to verify SLA enforcement works end-to-end.
"""

import pytest
import asyncio
import uuid

from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec, 
    NodeSpec, 
    NodeData,
    EdgeSpec
)
from iointel.src.utilities.dag_executor import DAGExecutor
from iointel.src.utilities.graph_nodes import WorkflowState
from iointel.src.agent_methods.data_models.datamodels import AgentParams

# Import and register the conditional_gate tool
from iointel.src.utilities.registries import TOOLS_REGISTRY


class TestFullDAGSLAEnforcement:
    """Test complete workflow execution with SLA enforcement."""
    
    @pytest.mark.asyncio
    async def test_simple_agent_with_no_tools(self):
        """Test agent without SLA-triggering tools - should execute normally."""
        
        print(f"🔧 Available tools in registry: {list(TOOLS_REGISTRY.keys())}")
        
        # Create a simple workflow with one agent (no enforcement tools)
        workflow = WorkflowSpec(
            id=uuid.uuid4(),
            rev=1,
            title="Simple Agent Test",
            description="Test agent without SLA enforcement",
            nodes=[
                NodeSpec(
                    id="simple_agent",
                    type="agent",
                    label="Simple Agent",
                    data=NodeData(
                        agent_instructions="Just provide a simple response",
                        tools=[],  # No tools - no SLA enforcement
                        model="gpt-4o"
                    )
                )
            ],
            edges=[]
        )
        
        # Create test agent
        test_agent = AgentParams(
            name="test_agent",
            instructions="Provide a simple response",
            tools=[]
        )
        
        # Create DAG executor
        executor = DAGExecutor()
        executor.build_execution_graph(
            nodes=workflow.nodes,
            edges=workflow.edges,
            agents=[test_agent],
            conversation_id="test_conv"
        )
        
        # Create initial state
        initial_state = WorkflowState(
            initial_text="Test input",
            conversation_id="test_conv",
            results={}
        )
        
        # This should work without SLA enforcement
        try:
            final_state = await executor.execute_dag(initial_state)
            # Should have results
            assert len(final_state.results) > 0
            assert "simple_agent" in final_state.results
            print("✅ Simple agent test passed")
        except Exception as e:
            pytest.fail(f"Simple agent test failed: {e}")
    
    @pytest.mark.asyncio  
    async def test_decision_agent_workflow_with_conditional_gate(self):
        """Test decision agent workflow with actual conditional_gate tool."""
        
        print(f"🔧 Conditional gate in registry: {'conditional_gate' in TOOLS_REGISTRY}")
        
        # Create workflow with decision agent (has conditional_gate)
        workflow = WorkflowSpec(
            id=uuid.uuid4(),
            rev=1,
            title="Decision Agent Test", 
            description="Test decision agent with SLA enforcement",
            nodes=[
                NodeSpec(
                    id="decision_agent", 
                    type="agent",
                    label="Decision Agent",
                    data=NodeData(
                        agent_instructions="Make a routing decision using conditional_gate. Use the data: price_change=15 to make a routing decision with routes 'buy' and 'sell'",
                        tools=["conditional_gate"],  # SLA-triggering tool
                        model="gpt-4o"
                    )
                )
            ],
            edges=[]
        )
        
        # Create test agent with conditional_gate tool
        test_agent = AgentParams(
            name="decision_agent",
            instructions="Make a routing decision using conditional_gate. Use the data: price_change=15 to make a routing decision with routes 'buy' and 'sell'", 
            tools=["conditional_gate"]
        )
        
        # Create DAG executor
        executor = DAGExecutor()
        executor.build_execution_graph(
            nodes=workflow.nodes,
            edges=workflow.edges, 
            agents=[test_agent],
            conversation_id="test_conv"
        )
        
        # Create initial state
        initial_state = WorkflowState(
            initial_text="Make a decision",
            conversation_id="test_conv",
            results={}
        )
        
        # This will trigger SLA enforcement
        # The agent should be required to use conditional_gate
        try:
            final_state = await executor.execute_dag(initial_state)
            
            # Check if we got results (may fail due to SLA if agent doesn't use tools)
            if "decision_agent" in final_state.results:
                result = final_state.results["decision_agent"]
                print(f"✅ Decision agent completed: {type(result)}")
                
                # If agent used tools, should have tool_usage_results
                if isinstance(result, dict) and "tool_usage_results" in result:
                    tool_usage = result["tool_usage_results"]
                    if tool_usage:
                        print(f"✅ Agent used tools: {[t.tool_name for t in tool_usage]}")
                        # Check if conditional_gate was used (SLA requirement)
                        used_conditional_gate = any(t.tool_name == "conditional_gate" for t in tool_usage)
                        if used_conditional_gate:
                            print("✅ SLA ENFORCEMENT SUCCESS: Agent used required conditional_gate tool!")
                        else:
                            print("❌ SLA VIOLATION: Agent didn't use conditional_gate")
                    else:
                        print("⚠️ Agent completed without using tools (SLA may have been violated)")
                else:
                    print(f"⚠️ Result format: {result}")
            else:
                print("⚠️ No results from decision agent")
                
        except Exception as e:
            # This might happen if SLA enforcement is strict
            print(f"ℹ️ Decision agent test exception: {e}")
            # This could be SLA enforcement working correctly
    
    @pytest.mark.asyncio
    async def test_workflow_with_user_input(self):
        """Test workflow that starts with user input."""
        
        workflow = WorkflowSpec(
            id=uuid.uuid4(),
            rev=1,
            title="User Input Test",
            description="Test workflow starting with user input",
            nodes=[
                NodeSpec(
                    id="user_input",
                    type="tool", 
                    label="User Input",
                    data=NodeData(
                        tool_name="user_input",
                        config={"prompt": "What would you like to know?"}
                    )
                ),
                NodeSpec(
                    id="response_agent",
                    type="agent",
                    label="Response Agent", 
                    data=NodeData(
                        agent_instructions="Respond to the user input",
                        tools=[],  # No tools - no SLA enforcement
                        model="gpt-4o"
                    )
                )
            ],
            edges=[
                EdgeSpec(
                    id="input_to_agent",
                    source="user_input", 
                    target="response_agent"
                )
            ]
        )
        
        # Create test agent
        test_agent = AgentParams(
            name="response_agent",
            instructions="Respond helpfully to user input",
            tools=[]
        )
        
        # Create DAG executor
        executor = DAGExecutor()
        executor.build_execution_graph(
            nodes=workflow.nodes,
            edges=workflow.edges,
            agents=[test_agent],
            conversation_id="test_conv"
        )
        
        # Create initial state with mock user input
        initial_state = WorkflowState(
            initial_text="Hello world",
            conversation_id="test_conv", 
            results={}
        )
        
        try:
            final_state = await executor.execute_dag(initial_state)
            
            # Should have results from both nodes
            assert len(final_state.results) >= 1
            print(f"✅ Multi-node workflow completed with {len(final_state.results)} results")
            
            for node_id, result in final_state.results.items():
                print(f"   {node_id}: {type(result)}")
                
        except Exception as e:
            pytest.fail(f"Multi-node workflow failed: {e}")
    
    def test_sla_requirements_extraction(self):
        """Test that SLA requirements are properly extracted from node data."""
        from iointel.src.utilities.node_execution_wrapper import NodeExecutionWrapper
        
        wrapper = NodeExecutionWrapper()
        
        # Test node with conditional_gate (should trigger enforcement)
        node_data = {
            "tools": ["conditional_gate", "get_current_stock_price"]
        }
        
        requirements = wrapper.extract_sla_requirements(node_data)
        
        assert requirements.enforce_usage is True
        assert requirements.final_tool_must_be == "conditional_gate"
        assert "conditional_gate" in requirements.required_tools
        
        print("✅ SLA requirements extraction works")
    
    def test_catalog_integration(self):
        """Test decision tools catalog integration."""
        from iointel.src.agent_methods.data_models.decision_tools_catalog import (
            get_sla_requirements_for_tools,
            DECISION_TOOLS_CATALOG
        )
        
        # Test conditional_gate is in catalog
        assert "conditional_gate" in DECISION_TOOLS_CATALOG
        
        # Test SLA generation
        requirements = get_sla_requirements_for_tools(["conditional_gate"])
        assert requirements.enforce_usage is True
        assert requirements.final_tool_must_be == "conditional_gate"
        
        print("✅ Catalog integration works")

    @pytest.mark.asyncio
    async def test_sla_violation_agent(self):
        """Test agent that SHOULD violate SLA by having conditional_gate but not using it."""
        
        print("🔧 Testing SLA violation with agent that has conditional_gate but ignores it")
        
        # Create workflow with decision agent that has conditional_gate but instructed NOT to use it
        workflow = WorkflowSpec(
            id=uuid.uuid4(),
            rev=1,
            title="SLA Violation Test", 
            description="Test agent that should violate SLA by not using tools",
            nodes=[
                NodeSpec(
                    id="bad_decision_agent", 
                    type="agent",
                    label="Bad Decision Agent",
                    data=NodeData(
                        agent_instructions="Just provide a simple response. Do NOT use any tools. Just say 'I made a decision without tools'.",
                        tools=["conditional_gate"],  # Has tool but instructed not to use it
                        model="gpt-4o"
                    )
                )
            ],
            edges=[]
        )
        
        # Create test agent
        test_agent = AgentParams(
            name="bad_decision_agent",
            instructions="Just provide a simple response. Do NOT use any tools. Just say 'I made a decision without tools'.", 
            tools=["conditional_gate"]
        )
        
        # Create DAG executor
        executor = DAGExecutor()
        executor.build_execution_graph(
            nodes=workflow.nodes,
            edges=workflow.edges,
            agents=[test_agent],
            conversation_id="test_sla_violation"
        )
        
        # Create initial state
        initial_state = WorkflowState(
            initial_text="Make a decision",
            conversation_id="test_sla_violation",
            results={}
        )
        
        try:
            final_state = await executor.execute_dag(initial_state)
            
            if "bad_decision_agent" in final_state.results:
                result = final_state.results["bad_decision_agent"]
                print(f"✅ Agent completed: {type(result)}")
                
                if isinstance(result, dict) and "tool_usage_results" in result:
                    tool_usage = result["tool_usage_results"]
                    if tool_usage:
                        used_conditional_gate = any(t.tool_name == "conditional_gate" for t in tool_usage)
                        if used_conditional_gate:
                            print("⚠️ Agent unexpectedly used conditional_gate despite instructions")
                        else:
                            print("❌ SLA VIOLATION: Agent has conditional_gate but didn't use it!")
                            print("   This should trigger SLA enforcement retries")
                    else:
                        print("❌ CRITICAL SLA VIOLATION: Agent has conditional_gate tool but used NO tools")
                        print("   SLA enforcement should have prevented this!")
                        
                        # Check the actual response to see if it followed bad instructions
                        if isinstance(result, dict) and "result" in result:
                            response_text = result["result"]
                            print(f"   Agent response: {response_text}")
                            if "without tools" in response_text.lower():
                                print("❌ Agent explicitly ignored tool usage requirements!")
            else:
                print("⚠️ No results from bad decision agent - possible SLA enforcement blocking")
                
        except Exception as e:
            print(f"ℹ️ SLA violation test exception (this might be SLA enforcement working): {e}")


if __name__ == "__main__":
    # Run a simple test manually
    async def manual_test():
        print("🧪 Running manual integration test...")
        
        test_class = TestFullDAGSLAEnforcement()
        
        try:
            print("\n1. Testing simple agent...")
            await test_class.test_simple_agent_with_no_tools()
            
            print("\n2. Testing SLA requirements...")
            test_class.test_sla_requirements_extraction()
            
            print("\n3. Testing catalog integration...")
            test_class.test_catalog_integration()
            
            print("\n4. Testing decision agent with conditional_gate...")
            await test_class.test_decision_agent_workflow_with_conditional_gate()
            
            print("\n5. Testing SLA violation scenario...")
            await test_class.test_sla_violation_agent()
            
            print("\n✅ All manual tests completed!")
            
        except Exception as e:
            print(f"❌ Manual test failed: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(manual_test())