#!/usr/bin/env python3
"""
Test Agent Creation from WorkflowSpec
=====================================

This test verifies that agents are created directly from the WorkflowSpec
without needing to pass pre-created agents. The WorkflowSpec is the single
source of truth for agent configuration.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec, NodeSpec, NodeData
from iointel.src.utilities.dag_executor import DAGExecutor
from iointel.src.utilities.graph_nodes import WorkflowState
import uuid

# Import tools to ensure they're registered


async def test_agent_creation():
    """Test that agents are created from WorkflowSpec without passing agents."""
    
    print("🧪 Testing Agent Creation from WorkflowSpec")
    print("=" * 60)
    
    # Create a workflow with an agent node that has instructions and tools
    workflow = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title='Test Agent Creation',
        description='Test that agents are created from WorkflowSpec',
        reasoning='Testing agent creation from WorkflowSpec without passing agents parameter',
        nodes=[
            NodeSpec(
                id='math_agent',
                type='agent',
                label='Math Agent',
                data=NodeData(
                    agent_instructions=(
                        "You are a math assistant. "
                        "Use the add tool to calculate 15 + 27. "
                        "You MUST use the tool, do not calculate mentally."
                    ),
                    tools=['add'],  # Agent has access to the add tool
                    model='gpt-4o'
                )
            )
        ],
        edges=[]
    )
    
    print("📋 Created WorkflowSpec with:")
    print("  - 1 agent node with instructions")
    print("  - Tools: ['add']")
    print("  - Model: gpt-4o")
    
    # Create executor with typed execution
    executor = DAGExecutor(use_typed_execution=True)
    
    # Build graph WITHOUT passing agents - they should be created from spec
    print("\n🔨 Building execution graph (no agents passed)...")
    executor.build_execution_graph(
        workflow_spec=workflow,
        objective='Calculate 15 + 27',
        conversation_id='test_agent_creation'
    )
    
    # Execute
    print("\n🚀 Executing workflow...")
    initial_state = WorkflowState(
        initial_text='Calculate 15 + 27',
        conversation_id='test_agent_creation',
        results={}
    )
    
    try:
        final_state = await executor.execute_dag(initial_state)
        
        # Check result
        if 'math_agent' in final_state.results:
            result = final_state.results['math_agent']
            print("\n✅ Agent executed successfully!")
            
            # Check if it's an AgentExecutionResult
            if hasattr(result, 'agent_response') and result.agent_response:
                print(f"  Agent response: {result.agent_response.result}")
                
                # Check tool usage
                if result.agent_response.tool_usage_results:
                    print(f"  Tools used: {[t.tool_name for t in result.agent_response.tool_usage_results]}")
                    for tool in result.agent_response.tool_usage_results:
                        print(f"    - {tool.tool_name}({tool.tool_args}) = {tool.tool_result}")
                else:
                    print("  ⚠️  No tools were used (agent should have used 'add' tool)")
            else:
                print(f"  Result type: {type(result)}")
                print(f"  Result: {result}")
        else:
            print("\n❌ Agent node did not execute")
            print(f"Available results: {list(final_state.results.keys())}")
            
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()


async def test_agent_with_decision():
    """Test a decision agent that uses conditional_gate tool."""
    
    print("\n\n🧪 Testing Decision Agent with conditional_gate")
    print("=" * 60)
    
    workflow = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title='Test Decision Agent',
        description='Test decision agent creation and routing',
        reasoning='Testing that decision agents are created with proper tools',
        nodes=[
            NodeSpec(
                id='decision_agent',
                type='decision',  # Decision type
                label='Stock Decision',
                data=NodeData(
                    agent_instructions=(
                        "Analyze the stock price change and make a routing decision. "
                        "If the price increased, route to index 0 (buy). "
                        "If the price decreased, route to index 1 (sell). "
                        "You MUST use the conditional_gate tool to make the routing decision."
                    ),
                    tools=['conditional_gate'],  # Decision tool
                    model='gpt-4o'
                )
            )
        ],
        edges=[]
    )
    
    print("📋 Created WorkflowSpec with:")
    print("  - 1 decision node")
    print("  - Tools: ['conditional_gate']")
    print("  - Should route based on stock price")
    
    executor = DAGExecutor(use_typed_execution=True)
    
    print("\n🔨 Building execution graph (no agents passed)...")
    executor.build_execution_graph(
        workflow_spec=workflow,
        objective='Stock price increased by 5%',
        conversation_id='test_decision'
    )
    
    print("\n🚀 Executing workflow...")
    initial_state = WorkflowState(
        initial_text='Stock price increased by 5%',
        conversation_id='test_decision',
        results={}
    )
    
    try:
        final_state = await executor.execute_dag(initial_state)
        
        if 'decision_agent' in final_state.results:
            result = final_state.results['decision_agent']
            print("\n✅ Decision agent executed successfully!")
            
            if hasattr(result, 'agent_response') and result.agent_response:
                print(f"  Agent analysis: {result.agent_response.result}")
                
                # Check if conditional_gate was used
                if result.agent_response.tool_usage_results:
                    tools_used = [t.tool_name for t in result.agent_response.tool_usage_results]
                    print(f"  Tools used: {tools_used}")
                    
                    if 'conditional_gate' in tools_used:
                        gate_result = result.agent_response.tool_usage_results[0].tool_result
                        if hasattr(gate_result, 'route_index'):
                            print(f"  ✅ Routing decision: index {gate_result.route_index}")
                            print(f"     Routed to: {gate_result.routed_to}")
                        else:
                            print(f"  Gate result: {gate_result}")
                else:
                    print("  ⚠️  No tools used (should have used conditional_gate)")
        else:
            print("\n❌ Decision agent did not execute")
            
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()


async def test_multi_agent_workflow():
    """Test a workflow with multiple agent nodes."""
    
    print("\n\n🧪 Testing Multi-Agent Workflow")
    print("=" * 60)
    
    workflow = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title='Multi-Agent Math Pipeline',
        description='Test multiple agents in sequence',
        reasoning='Testing that multiple agents are created from spec',
        nodes=[
            NodeSpec(
                id='agent_1',
                type='agent',
                label='First Calculator',
                data=NodeData(
                    agent_instructions="Calculate 10 + 5 using the add tool.",
                    tools=['add'],
                    model='gpt-4o',
                    outs=['result']
                )
            ),
            NodeSpec(
                id='agent_2',
                type='agent',
                label='Second Calculator',
                data=NodeData(
                    agent_instructions="Take the result from agent_1 and multiply it by 3 using the multiply tool.",
                    tools=['multiply'],
                    model='gpt-4o',
                    ins=['result']
                )
            )
        ],
        edges=[
            {'id': 'e1', 'source': 'agent_1', 'target': 'agent_2'}
        ]
    )
    
    print("📋 Created WorkflowSpec with:")
    print("  - 2 agent nodes in sequence")
    print("  - agent_1 uses 'add' tool")
    print("  - agent_2 uses 'multiply' tool")
    
    executor = DAGExecutor(use_typed_execution=True)
    
    print("\n🔨 Building execution graph (no agents passed)...")
    executor.build_execution_graph(
        workflow_spec=workflow,
        objective='Perform calculations',
        conversation_id='test_multi'
    )
    
    print("\n🚀 Executing workflow...")
    initial_state = WorkflowState(
        initial_text='Perform calculations',
        conversation_id='test_multi',
        results={}
    )
    
    try:
        final_state = await executor.execute_dag(initial_state)
        
        # Check both agents executed
        for agent_id in ['agent_1', 'agent_2']:
            if agent_id in final_state.results:
                result = final_state.results[agent_id]
                print(f"\n✅ {agent_id} executed successfully!")
                
                if hasattr(result, 'agent_response') and result.agent_response:
                    if result.agent_response.tool_usage_results:
                        for tool in result.agent_response.tool_usage_results:
                            print(f"  Used {tool.tool_name}: {tool.tool_result}")
            else:
                print(f"\n❌ {agent_id} did not execute")
                
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all tests."""
    
    print("=" * 80)
    print("TESTING AGENT CREATION FROM WORKFLOWSPEC")
    print("=" * 80)
    print("\nThis suite tests that agents are created from WorkflowSpec")
    print("without needing to pass pre-created agents.")
    print("\nKey points being tested:")
    print("1. Agents are created from NodeSpec.data (instructions, tools, model)")
    print("2. No agents parameter needed in build_execution_graph()")
    print("3. Decision agents get proper tool enforcement")
    print("4. Multi-agent workflows work correctly")
    
    await test_agent_creation()
    await test_agent_with_decision()
    await test_multi_agent_workflow()
    
    print("\n" + "=" * 80)
    print("TEST SUITE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())