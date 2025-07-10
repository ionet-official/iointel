#!/usr/bin/env python3
"""
End-to-End Workflow Demonstration
==================================

This demo shows the complete pipeline:
1. Generate a WorkflowSpec using WorkflowPlanner (AI agent)
2. Convert WorkflowSpec to WorkflowDefinition 
3. Execute the WorkflowDefinition using real tools
4. Show results from actual execution

This demonstrates the full workflow lifecycle from AI planning to execution.
"""

import os
import uuid
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load environment
env_path = Path(__file__).parent / "creds.env"
load_dotenv(env_path)

from iointel.src.agent_methods.agents.workflow_planner import WorkflowPlanner
from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env
from iointel.src.utilities.registries import TOOLS_REGISTRY
from iointel.src.utilities.decorators import register_custom_task
from iointel.src.workflow import Workflow

# Import the example tools to register them globally
import iointel.src.RL.example_tools


# Register tool executor for "tool" type tasks
@register_custom_task("tool")
async def execute_tool_task(task_metadata, objective, agents, execution_metadata):
    """Execute a tool task using the TOOLS_REGISTRY."""
    print(f"   🔧 Executing tool task: {task_metadata.get('tool_name', 'unknown')}")
    
    # Get tool name from metadata
    tool_name = task_metadata.get("tool_name")
    if not tool_name:
        raise ValueError("Tool task missing 'tool_name' in metadata")
    
    # Get tool from registry
    tool = TOOLS_REGISTRY.get(tool_name)
    if not tool:
        raise ValueError(f"Tool '{tool_name}' not found in registry")
    
    # Get tool configuration (variables should already be resolved by TaskNode)
    config = task_metadata.get("config", {})
    
    # Execute the tool
    try:
        result = await tool.run(config)
        print(f"   ✅ Tool '{tool_name}' executed successfully")
        return result
    except Exception as e:
        print(f"   ❌ Tool '{tool_name}' failed: {e}")
        raise


# Register agent executor for "agent" type tasks
@register_custom_task("agent")
async def execute_agent_task(task_metadata, objective, agents, execution_metadata):
    """Execute an agent task using the provided agents."""
    print(f"   🤖 Executing agent task")
    
    # Get agent instructions
    instructions = task_metadata.get("agent_instructions", objective)
    
    # Use the provided agents or create a default one
    if not agents:
        from iointel.src.agents import Agent
        agents = [Agent(instructions=instructions)]
    
    # Run the agents
    from iointel.src.utilities.runners import run_agents
    try:
        result = await run_agents(
            objective=instructions,
            agents=agents,
            output_type=str
        )
        print(f"   ✅ Agent task executed successfully")
        return result
    except Exception as e:
        print(f"   ❌ Agent task failed: {e}")
        raise


# Register decision executor for "decision" type tasks
@register_custom_task("decision")
async def execute_decision_task(task_metadata, objective, agents, execution_metadata):
    """Execute a decision task using decision tools."""
    print(f"   🤔 Executing decision task: {task_metadata.get('tool_name', 'unknown')}")
    
    # Decision tasks are actually tool tasks with decision logic
    return await execute_tool_task(task_metadata, objective, agents, execution_metadata)


# Register workflow_call executor for "workflow_call" type tasks
@register_custom_task("workflow_call")
async def execute_workflow_call_task(task_metadata, objective, agents, execution_metadata):
    """Execute a workflow call task."""
    print(f"   📞 Executing workflow call task: {task_metadata.get('workflow_id', 'unknown')}")
    
    # For demo purposes, just return a success message
    workflow_id = task_metadata.get("workflow_id", "unknown")
    return f"Workflow '{workflow_id}' executed successfully (mock)"


def load_available_tools():
    """Load available tools and create a tool catalog."""
    print("🔧 Loading available tools...")
    
    # Load tools from environment
    try:
        available_tools = load_tools_from_env("creds.env")
        print(f"   ✅ Loaded {len(available_tools)} tools from environment")
    except Exception as e:
        print(f"   ⚠️  Could not load tools from env: {e}")
    
    # Create tool catalog
    tool_catalog = {}
    for tool_name, tool in TOOLS_REGISTRY.items():
        tool_catalog[tool_name] = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "is_async": tool.is_async
        }
    
    print(f"   ✅ Tool catalog contains {len(tool_catalog)} tools:")
    for name in sorted(tool_catalog.keys())[:10]:  # Show first 10
        print(f"      - {name}: {tool_catalog[name]['description'][:50]}...")
    if len(tool_catalog) > 10:
        print(f"      ... and {len(tool_catalog) - 10} more tools")
    
    return tool_catalog


async def generate_workflow_with_ai(query: str, tool_catalog: dict):
    """Use WorkflowPlanner to generate a workflow from natural language."""
    print(f"\n🤖 Generating workflow with AI...")
    print(f"   Query: {query}")
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("   ❌ No OPENAI_API_KEY found - cannot generate workflow with AI")
        return None
    
    # Create workflow planner
    planner = WorkflowPlanner(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",  # Use cheaper model for demo
        debug=True
    )
    
    try:
        # Generate workflow
        workflow_spec = await planner.generate_workflow(
            query=query,
            tool_catalog=tool_catalog
        )
        
        print(f"   ✅ Generated workflow: {workflow_spec.title}")
        print(f"   - Description: {workflow_spec.description}")
        print(f"   - Nodes: {len(workflow_spec.nodes)}")
        print(f"   - Edges: {len(workflow_spec.edges)}")
        
        # Show LLM reasoning
        if workflow_spec.reasoning:
            print(f"   🤔 LLM Reasoning: {workflow_spec.reasoning}")
        
        # Validate structure with tool catalog
        issues = workflow_spec.validate_structure(tool_catalog)
        if issues:
            print(f"   ⚠️  Validation issues: {issues}")
        else:
            print("   ✅ Workflow structure and tools are valid")
        
        return workflow_spec
        
    except Exception as e:
        print(f"   ❌ Failed to generate workflow: {e}")
        return None


def convert_to_executable(workflow_spec):
    """Convert WorkflowSpec to executable format."""
    print(f"\n🔄 Converting to executable format...")
    
    try:
        # Convert to WorkflowDefinition
        workflow_def = workflow_spec.to_workflow_definition()
        print(f"   ✅ Converted to WorkflowDefinition")
        print(f"   - Name: {workflow_def.name}")
        print(f"   - Tasks: {len(workflow_def.tasks)}")
        
        # Convert to YAML for execution
        yaml_content = workflow_spec.to_yaml()
        print(f"   ✅ Generated YAML ({len(yaml_content)} characters)")
        
        return workflow_def, yaml_content
        
    except Exception as e:
        print(f"   ❌ Conversion failed: {e}")
        return None, None


async def execute_workflow(yaml_content: str):
    """Execute the workflow using the Workflow engine."""
    print(f"\n🚀 Executing workflow...")
    
    try:
        # Create workflow from YAML
        workflow = Workflow.from_yaml(yaml_str=yaml_content)
        print(f"   ✅ Created executable workflow")
        print(f"   - Objective: {workflow.objective}")
        print(f"   - Tasks: {len(workflow.tasks)}")
        print(f"   - Client mode: {workflow.client_mode}")
        
        # Execute the workflow
        print("   🔄 Running workflow tasks...")
        results = await workflow.run_tasks()
        
        print(f"   ✅ Workflow execution completed!")
        print(f"   - Conversation ID: {results['conversation_id']}")
        print(f"   - Results: {len(results['results'])} task results")
        
        return results
        
    except Exception as e:
        print(f"   ❌ Execution failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return None


def create_fallback_workflow():
    """Create a simple fallback workflow if AI generation fails."""
    print("\n📋 Creating fallback workflow...")
    
    from iointel.src.agent_methods.data_models.workflow_spec import (
        WorkflowSpec, NodeSpec, NodeData, EdgeSpec
    )
    
    # Create a simple math workflow using available tools
    workflow_spec = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Simple Math Calculation Workflow",
        description="Perform mathematical calculations using available simple math tools",
        nodes=[
            NodeSpec(
                id="check_time",
                type="tool",
                label="Check Current Time",
                data=NodeData(
                    tool_name="get_current_datetime",
                    config={},
                    ins=[],
                    outs=["current_time"]
                )
            ),
            NodeSpec(
                id="add_numbers",
                type="tool",
                label="Add Numbers (10 + 5)",
                data=NodeData(
                    tool_name="add",
                    config={"a": 10, "b": 5},
                    ins=[],
                    outs=["result"]
                )
            ),
            NodeSpec(
                id="multiply_result",
                type="tool", 
                label="Multiply Result by 3",
                data=NodeData(
                    tool_name="multiply",
                    config={"a": "{add_numbers}", "b": 3},
                    ins=["result"],
                    outs=["result"]
                )
            ),
            NodeSpec(
                id="square_root_final",
                type="tool",
                label="Square Root of Final Result",
                data=NodeData(
                    tool_name="square_root",
                    config={"x": "{multiply_result}"},
                    ins=["result"],
                    outs=["result"]
                )
            )
        ],
        edges=[
            EdgeSpec(
                id="add_to_multiply",
                source="add_numbers",
                target="multiply_result",
                sourceHandle="result",
                targetHandle="result"
            ),
            EdgeSpec(
                id="multiply_to_sqrt",
                source="multiply_result",
                target="square_root_final",
                sourceHandle="result",
                targetHandle="result"
            )
        ],
        reasoning="This workflow demonstrates data flow resolution: add 10+5=15, multiply 15*3=45, then square root of 45≈6.71. Uses variable references {add_numbers} and {multiply_result} to pass data between tasks."
    )
    
    print(f"   ✅ Created fallback workflow: {workflow_spec.title}")
    return workflow_spec


async def main():
    """Main demonstration function."""
    print("=" * 60)
    print("🌊 End-to-End Workflow Demonstration")
    print("=" * 60)
    
    # Step 1: Load available tools
    tool_catalog = load_available_tools()
    
    # Step 2: Generate workflow with AI (or use fallback)
    query = """
    Create a workflow that:
    1. Gets the current time using get_current_datetime
    2. Performs mathematical calculations: add 10 and 5, then multiply the result by 3
    3. Calculates the square root of the final result
    4. Shows the current time and all calculation results
    
    Use only the simple math tools available in the catalog: get_current_datetime, add, multiply, square_root.
    """
    
    # For testing data flow resolution, use fallback workflow
    print("\n   🔄 Using fallback workflow to test data flow resolution...")
    workflow_spec = create_fallback_workflow()
    
    # Uncomment below to test AI generation with data flow:
    # workflow_spec = await generate_workflow_with_ai(query, tool_catalog)
    # if not workflow_spec:
    #     print("\n   🔄 Using fallback workflow instead...")
    #     workflow_spec = create_fallback_workflow()
    
    # Step 3: Convert to executable format
    workflow_def, yaml_content = convert_to_executable(workflow_spec)
    
    if not yaml_content:
        print("\n❌ Cannot proceed without valid workflow definition")
        return
    
    # Step 4: Execute the workflow
    results = await execute_workflow(yaml_content)
    
    # Step 5: Display results
    if results:
        print(f"\n📊 Execution Results:")
        print(f"   Conversation ID: {results['conversation_id']}")
        print(f"\n   Task Results:")
        for task_name, result in results['results'].items():
            print(f"   - {task_name}: {result}")
    
    # Step 6: Save artifacts
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Save YAML
    yaml_file = output_dir / f"{workflow_spec.title.replace(' ', '_')}_executable.yaml"
    yaml_file.write_text(yaml_content)
    
    # Save results
    if results:
        import json
        results_file = output_dir / f"{workflow_spec.title.replace(' ', '_')}_results.json"
        results_file.write_text(json.dumps(results, indent=2, default=str))
        print(f"\n💾 Saved artifacts:")
        print(f"   - YAML: {yaml_file}")
        print(f"   - Results: {results_file}")
    
    print("\n✅ End-to-end demonstration completed!")


if __name__ == "__main__":
    asyncio.run(main())