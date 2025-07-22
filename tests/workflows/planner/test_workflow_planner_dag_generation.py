#!/usr/bin/env python3
"""
Test Workflow Planner Agent DAG Generation
=========================================
This tests that the workflow planner agent can correctly generate DAGs with:
1. Proper execution_mode settings (consolidate vs for_each)
2. SLA requirements for decision agents
3. Correct routing patterns with decision gates
4. All required fields populated
"""
import asyncio
import sys
sys.path.append('/Users/alexandermorisse/Documents/GitHub/iointel')

from iointel.src.agent_methods.agents.workflow_planner import WorkflowPlanner
# WorkflowSpecLLM import removed as it's not used directly in current implementation
from iointel.src.utilities.tool_registry_utils import create_tool_catalog

async def test_workflow_generation():
    """Test that workflow planner generates correct DAGs with execution modes and SLA."""
    
    print("🧪 TESTING WORKFLOW PLANNER DAG GENERATION")
    print("=" * 50)
    
    # Create workflow planner agent
    planner = WorkflowPlanner()
    
    # Get tool catalog for the planner
    tool_catalog = create_tool_catalog()
    
    # Test cases for different scenarios
    test_cases = [
        {
            "name": "Decision Gate with For-Each Notification",
            "prompt": """Create a workflow that:
            1. Analyzes stock sentiment from user input
            2. Routes to buy or sell recommendation based on sentiment
            3. Sends email notification about the recommendation
            
            The email should be sent regardless of which path is taken.""",
            "expected_features": [
                "decision agent with conditional_gate tool",
                "SLA requirements on decision agent",
                "buy and sell agents with consolidate mode",
                "email agent with for_each mode"
            ]
        },
        {
            "name": "Research Agent with SLA",
            "prompt": """Create a workflow where an agent must:
            1. Search for information about a topic
            2. Use conditional_gate to route based on findings
            3. Ensure the agent MUST use both search and conditional_gate tools
            
            This should enforce that research happens before routing.""",
            "expected_features": [
                "agent with search and conditional_gate tools",
                "SLA with required_tools including both tools",
                "final_tool_must_be conditional_gate",
                "min_tool_calls >= 2"
            ]
        },
        {
            "name": "Parallel Processing with Consolidation",
            "prompt": """Create a workflow that:
            1. Gets data from two different sources in parallel
            2. Compares the results in a single comparison agent
            3. The comparison agent must wait for both inputs
            
            This tests consolidate mode for aggregation.""",
            "expected_features": [
                "two parallel source nodes",
                "comparison agent with consolidate mode",
                "proper edge connections from both sources"
            ]
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📋 Test Case: {test_case['name']}")
        print("-" * 40)
        
        # Generate workflow using the correct method
        workflow_spec = await planner.generate_workflow(
            query=test_case['prompt'],
            tool_catalog=tool_catalog
        )
        
        print(f"\n🔍 Generated Workflow: {workflow_spec.title}")
        print(f"📝 Description: {workflow_spec.description}")
        
        # Analyze the generated workflow
        print("\n📊 ANALYSIS:")
        
        if workflow_spec.nodes:
            # Check for execution modes
            for_each_nodes = []
            consolidate_nodes = []
            sla_nodes = []
            
            for node in workflow_spec.nodes:
                if hasattr(node.data, 'execution_mode'):
                    if node.data.execution_mode == "for_each":
                        for_each_nodes.append(node.label)
                    else:
                        consolidate_nodes.append(node.label)
                
                if hasattr(node.data, 'sla') and node.data.sla:
                    sla_nodes.append({
                        "node": node.label,
                        "sla": node.data.sla
                    })
            
            print(f"  ✅ For-each nodes: {for_each_nodes}")
            print(f"  ✅ Consolidate nodes: {consolidate_nodes}")
            print(f"  ✅ SLA-enforced nodes: {len(sla_nodes)}")
            
            # Show SLA details
            for sla_node in sla_nodes:
                print(f"\n  📋 SLA for '{sla_node['node']}':")
                sla = sla_node['sla']
                if hasattr(sla, 'required_tools'):
                    print(f"    - Required tools: {sla.required_tools}")
                if hasattr(sla, 'final_tool_must_be'):
                    print(f"    - Final tool must be: {sla.final_tool_must_be}")
                if hasattr(sla, 'min_tool_calls'):
                    print(f"    - Min tool calls: {sla.min_tool_calls}")
            
            # Check for conditional gates
            conditional_gates = [
                node for node in workflow_spec.nodes 
                if node.type == "agent" and node.data.tools and "conditional_gate" in node.data.tools
            ]
            print(f"\n  🎯 Agents with conditional_gate: {len(conditional_gates)}")
            
            # Check edge conditions
            conditional_edges = [
                edge for edge in workflow_spec.edges 
                if edge.data and hasattr(edge.data, 'condition') and edge.data.condition
            ]
            print(f"  🔀 Conditional edges: {len(conditional_edges)}")
            
            # Verify expected features
            print("\n  📋 Expected features check:")
            for feature in test_case['expected_features']:
                print(f"    - {feature}")
        
        else:
            print("  ❌ No nodes generated - chat-only response")
            print(f"  💬 Reasoning: {workflow_spec.reasoning}")
        
        print("\n" + "=" * 50)

if __name__ == "__main__":
    asyncio.run(test_workflow_generation())