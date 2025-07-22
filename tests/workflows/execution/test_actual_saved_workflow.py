#!/usr/bin/env python3
"""
Test the actual saved workflow file to see if conditional_gate tool is being called properly.

This test loads the exact workflow file that the app uses and runs it through the same
DAG executor to verify the agent actually calls the conditional_gate tool.
"""

import sys
from pathlib import Path
import asyncio

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from iointel.src.web.workflow_storage import WorkflowStorage
from iointel.src.utilities.registries import TASK_EXECUTOR_REGISTRY
from iointel.src.web.workflow_server import web_tool_executor, web_agent_executor

# Import tools to register them


async def test_actual_saved_workflow():
    """Test the actual saved workflow file exactly like the app does."""
    
    print("🔍 Testing actual saved workflow file...")
    
    # Load the workflow storage
    storage = WorkflowStorage()
    
    # Find the Simple Conditional Gate Demo workflow
    workflows = storage.list_workflows()
    target_workflow = None
    
    for workflow in workflows:
        if "Fixed Conditional Gate Demo" in workflow["name"]:
            target_workflow = workflow
            break
    
    if not target_workflow:
        print("❌ Could not find Simple Conditional Gate Demo workflow")
        return
    
    print(f"✅ Found workflow: {target_workflow['name']}")
    print(f"   ID: {target_workflow['id']}")
    print(f"   Description: {target_workflow['description']}")
    
    # Load the actual workflow spec
    workflow_spec = storage.load_workflow(target_workflow['id'])
    
    print("📋 Workflow loaded:")
    print(f"   Nodes: {len(workflow_spec.nodes)}")
    print(f"   Edges: {len(workflow_spec.edges)}")
    
    # Show the decision agent configuration
    decision_agent = None
    for node in workflow_spec.nodes:
        if node.id == "decision_agent":
            decision_agent = node
            break
    
    if decision_agent:
        print("🤖 Decision Agent found:")
        print(f"   Tools: {decision_agent.data.tools}")
        print(f"   Instructions (first 200 chars): {decision_agent.data.agent_instructions[:200]}...")
    
    # Register the same executors that the web app uses
    TASK_EXECUTOR_REGISTRY["tool"] = web_tool_executor
    TASK_EXECUTOR_REGISTRY["agent"] = web_agent_executor
    
    # Execute workflow exactly like the web app does
    print("\n🚀 Executing workflow using proper conversion...")
    
    # Convert WorkflowSpec to executable Workflow (this creates the agents!)
    workflow_spec.to_workflow_definition()
    yaml_content = workflow_spec.to_yaml()
    
    # Import Workflow class
    from iointel.src.workflow import Workflow
    
    # Create workflow from YAML with custom conversation ID
    workflow = Workflow.from_yaml(yaml_str=yaml_content)
    workflow.objective = workflow_spec.description
    
    print(f"📋 Executing workflow with {len(workflow.tasks)} tasks")
    
    # Execute workflow with execution context
    conversation_id = "test_actual_workflow"
    
    try:
        results = await workflow.run_tasks(conversation_id=conversation_id)
        
        print("\n✅ Workflow execution completed!")
        
        # Results format from workflow.run_tasks()
        task_results = results.get("results", {})
        print(f"   Results: {len(task_results)} tasks")
        
        # Analyze the results
        for task_id, result in task_results.items():
            print(f"\n📋 Task: {task_id}")
            
            if isinstance(result, dict):
                if result.get("status") == "skipped":
                    print("   ❌ Status: SKIPPED (this is good for conditional routing)")
                else:
                    print("   ✅ Status: EXECUTED")
                    
                    # Check if this is the decision agent
                    if task_id == "decision_agent":
                        print("   🔍 Checking if agent used conditional_gate tool...")
                        
                        # Check for tool usage
                        if "tool_usage_results" in result:
                            tool_usage = result["tool_usage_results"]
                            if tool_usage:
                                print(f"   ✅ Agent used {len(tool_usage)} tools:")
                                for usage in tool_usage:
                                    print(f"     - {usage.tool_name}: {usage.tool_result}")
                                    
                                    if usage.tool_name == "conditional_gate":
                                        print("     🎯 SUCCESS: Agent called conditional_gate tool!")
                                        gate_result = usage.tool_result
                                        print(f"     📊 Gate result: {gate_result}")
                                        
                                        if hasattr(gate_result, 'routed_to'):
                                            print(f"     🛤️  Routed to: {gate_result.routed_to}")
                                        else:
                                            print(f"     ❓ Gate result format: {type(gate_result)}")
                            else:
                                print("   ❌ Agent did not use any tools!")
                        else:
                            print("   ❌ No tool_usage_results found in agent result")
                            
                    # Show result preview
                    if "result" in result:
                        result_text = str(result["result"])
                        print(f"   📝 Result preview: {result_text[:150]}...")
            else:
                print("   ✅ Status: EXECUTED")
                print(f"   📝 Result: {str(result)[:150]}...")
        
        # Show execution summary
        executed_count = len([r for r in task_results.values() if isinstance(r, dict) and r.get("status") != "skipped"])
        skipped_count = len([r for r in task_results.values() if isinstance(r, dict) and r.get("status") == "skipped"])
        
        print("\n📊 Execution Summary:")
        print(f"   Total tasks: {len(task_results)}")
        print(f"   Executed: {executed_count}")
        print(f"   Skipped: {skipped_count}")
        
        if skipped_count > 0:
            print(f"   🎯 Conditional routing worked! {skipped_count} tasks were skipped")
        else:
            print("   ❌ Conditional routing failed - all tasks executed")
            
    except Exception as e:
        print(f"❌ Workflow execution failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_actual_saved_workflow())