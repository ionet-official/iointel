#!/usr/bin/env python3
"""Test the unified fixes: validation catalog, retry context, and SLA enforcement."""

import asyncio
from iointel.src.agent_methods.agents.workflow_planner import WorkflowPlanner
from dotenv import load_dotenv

# Load environment variables
load_dotenv("creds.env")

# Load tools properly using discovery pattern
from iointel.src.agent_methods.tools.discovery import load_tools_from_env
tool_names = load_tools_from_env("creds.env")

async def test_unified_fixes():
    """Test that our fixes work together properly."""
    
    # Create tool catalog from registered tools
    from iointel.src.utilities.tool_registry_utils import create_validation_catalog
    from iointel.src.utilities.working_tools_filter import filter_available_tools
    
    # Get available tools and filter to working ones
    from iointel.src.utilities.registries import TOOLS_REGISTRY
    available_tool_names = list(TOOLS_REGISTRY.keys())
    working_tools = filter_available_tools(available_tool_names)
    
    print(f"Loaded {len(tool_names)} tools from environment")
    print(f"Registry has {len(available_tool_names)} tools")
    print(f"Filtered to {len(working_tools)} working tools")
    
    # Create tool catalog using shared helper
    from iointel.src.utilities.workflow_helpers import create_default_tool_catalog
    tool_catalog = create_default_tool_catalog()
    
    # Test 1: Unified validation catalog
    print("\n=== Testing Unified Validation Catalog ===")
    
    catalog = create_validation_catalog()
    print(f"Unified catalog size: {len(catalog)}")
    print(f"Has user_input: {'user_input' in catalog}")
    print(f"Has prompt_tool: {'prompt_tool' in catalog}")
    print(f"Has conditional_gate: {'conditional_gate' in catalog}")
    
    # Test 2: Workflow generation with retries (should pass failed workflow as context)
    print("\n=== Testing Workflow Generation with Retry Context ===")
    
    planner = WorkflowPlanner()
    
    # Direct request as you specified
    test_query = "lets make a simple stock analysis agent using tools that routes information to either a buy or sell agent, which both route to an email agent. The last three agents don't need tools. But the first one does need SLA, as decision agent. Also don't forget user input to decision agent."
    
    try:
        workflow = await planner.generate_workflow(
            query=test_query,
            tool_catalog=tool_catalog  # Pass the tool catalog!
        )
        print("\nWorkflow generated successfully!")
        print(f"Title: {workflow.title}")
        print(f"Nodes: {len(workflow.nodes)}")
        
        # Check if data source has proper config
        for node in workflow.nodes:
            if node.type == "data_source":
                print(f"\nData source node: {node.id}")
                print(f"  Has config: {node.data.config is not None}")
                if node.data.config:
                    print(f"  Config keys: {list(node.data.config.keys())}")
            elif node.type == "decision":
                print(f"\nDecision node: {node.id}")
                print(f"  Tools: {node.data.tools}")
                print(f"  SLA: {node.sla}")
                if node.sla:
                    print(f"    enforce_usage: {node.sla.enforce_usage}")
        
        # Test 4: Execute the generated workflow using shared helper
        print("\n=== Testing Workflow Execution ===")
        
        # Use shared workflow execution helper
        from iointel.src.utilities.workflow_helpers import execute_workflow
        
        # Execute workflow using the shared function
        result = await execute_workflow(
            workflow_spec=workflow,
            user_inputs={"user_input_1": "TSLA"},  # Provide actual stock symbol
            objective="Analyze TSLA stock",
            conversation_id="test-execution",
            debug=True
        )
        
        # Now result is typed WorkflowExecutionResult!
        from iointel.src.agent_methods.data_models.execution_models import ExecutionStatus, AgentExecutionResult
        
        if result.status == ExecutionStatus.COMPLETED:
            stats = result.metadata.get("stats", {})
            print("\nExecution completed!")
            print(f"Executed nodes: {stats['executed_nodes']}/{stats['total_nodes']}")
            print(f"Efficiency: {stats['execution_efficiency']}")
            print(f"Execution time: {result.execution_time}s")
            
            # Check results
            print("\nExecution results:")
            for node_id, node_result in result.node_results.items():
                print(f"  {node_id}: {node_result.node_type}")
                
                # Get the actual result based on node type
                if node_result.node_type == "agent" and isinstance(node_result.result, AgentExecutionResult):
                    agent_result = node_result.result
                    if agent_result.agent_response:
                        print("    Has agent response: Yes")
                        if agent_result.agent_response.tool_usage_results:
                            print(f"    Tool calls: {len(agent_result.agent_response.tool_usage_results)}")
            
            # Verify decision node used tools
            decision_node = result.node_results.get('decision_1')
            if decision_node and decision_node.node_type == "decision":
                agent_result = decision_node.result
                
                # Check if it's a typed AgentExecutionResult
                if isinstance(agent_result, AgentExecutionResult):
                    # Check if agent response has tool usage
                    if agent_result.agent_response and agent_result.agent_response.tool_usage_results:
                        tools_used = [tool.tool_name for tool in agent_result.agent_response.tool_usage_results]
                        print("\n✅ SUCCESS: Decision node used tools as required by SLA!")
                        print(f"   Tools used: {tools_used}")
                        print(f"   Number of tool calls: {len(agent_result.agent_response.tool_usage_results)}")
                        # Show tool results
                        for tool_result in agent_result.agent_response.tool_usage_results[:3]:  # Show first 3
                            result_preview = str(tool_result.tool_result)[:100]
                            print(f"   - {tool_result.tool_name}: {result_preview}...")
                    else:
                        print("\n❌ FAILURE: Decision node did not use tools despite SLA enforcement")
                        print(f"   Agent response type: {type(agent_result.agent_response)}")
                else:
                    print(f"\n⚠️  WARNING: Decision result is not AgentExecutionResult, got {type(agent_result)}")
            else:
                print("\n⚠️  WARNING: Could not find decision_1 in results")
        else:
            print(f"\nExecution failed: {result.error}")
            if result.status == ExecutionStatus.PARTIAL:
                print("Some nodes completed, but there were failures")
            import traceback
            traceback.print_exc()
                    
    except Exception as e:
        print(f"Workflow generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Check SLA enforcement is properly configured
    print("\n=== Testing SLA Enforcement Configuration ===")
    
    # Create a mock workflow with decision node
    from iointel.src.agent_methods.data_models.workflow_spec import (
        WorkflowSpec, NodeSpec, EdgeSpec, NodeData, SLARequirements
    )
    import uuid
    
    test_workflow = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        reasoning="Test workflow",
        title="Test Decision Workflow",
        description="Testing SLA enforcement",
        nodes=[
            NodeSpec(
                id="user_input_1",
                type="data_source",
                label="User Input",
                data=NodeData(
                    source_name="user_input",
                    config={
                        "message": "Enter stock symbol to analyze (e.g., AAPL, TSLA)",
                        "default_value": "AAPL"
                    }
                )
            ),
            NodeSpec(
                id="decision_1",
                type="decision",
                label="Stock Analysis Decision",
                data=NodeData(
                    agent_instructions="Analyze {user_input_1} stock and decide whether to buy or sell based on current market conditions",
                    tools=["conditional_gate"]
                ),
                sla=SLARequirements(
                    tool_usage_required=True,
                    required_tools=["conditional_gate"],
                    enforce_usage=True
                )
            )
        ],
        edges=[
            EdgeSpec(
                id="edge_1",
                source="user_input_1",
                target="decision_1"
            )
        ]
    )
    
    # Check if our typed executor would enhance the decision node
    from iointel.src.utilities.typed_execution import ExecutionContext
    from iointel.src.utilities.graph_nodes import WorkflowState
    
    context = ExecutionContext(
        workflow_spec=test_workflow,
        current_node_id="decision_1",
        state=WorkflowState()
    )
    
    print(f"Decision node type: {context.node_type}")
    print(f"Has SLA: {context.current_node.sla is not None}")
    print(f"Would get enhanced prompts: {context.node_type == 'decision' or context.current_node.sla is not None}")

if __name__ == "__main__":
    # Enable debug logging for execution
    import logging
    logging.basicConfig(level=logging.INFO)
    
    asyncio.run(test_unified_fixes())