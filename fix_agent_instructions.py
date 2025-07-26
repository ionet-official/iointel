#!/usr/bin/env python3
"""
Fix the agent instructions to ensure the agent actually calls the conditional_gate tool
instead of just describing what it would do.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from iointel.src.web.workflow_storage import WorkflowStorage


def fix_agent_instructions():
    """Fix the agent instructions to ensure tool calling works properly."""
    
    # Load the workflow storage
    storage = WorkflowStorage()
    
    # Find the Simple Conditional Gate Demo workflow
    workflows = storage.list_workflows()
    target_workflow = None
    
    for workflow in workflows:
        if "Simple Conditional Gate Demo" in workflow["name"]:
            target_workflow = workflow
            break
    
    if not target_workflow:
        print("❌ Could not find Simple Conditional Gate Demo workflow")
        return
    
    # Load the actual workflow spec
    workflow_spec = storage.load_workflow(target_workflow['id'])
    
    # Find the decision agent and fix its instructions
    for node in workflow_spec.nodes:
        if node.id == "decision_agent":
            print("🔧 Fixing decision agent instructions...")
            
            # Create simple, clear instructions that force tool calling
            node.data.agent_instructions = """You are a trading decision agent. You have access to the conditional_gate tool.

Your task:
1. Analyze the market data provided
2. Use the conditional_gate tool to make a routing decision
3. Return the tool result

Market data format: "Market analysis: BTC price increased 2.5% in last 24h with high volume. Sentiment: bullish, confidence: 0.8"

Extract sentiment and confidence from the data, then call:
conditional_gate with data and appropriate gate_config for buy/sell/hold routing.

You MUST call the tool - do not just describe what you would do."""
            
            print("✅ Updated agent instructions")
            break
    
    # Save the updated workflow (this will overwrite the existing one)
    # First, let's create a new workflow with fixed instructions
    from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec
    from uuid import uuid4
    
    # Create a new workflow spec with the same structure but fixed instructions
    new_workflow = WorkflowSpec(
        id=uuid4(),
        rev=1,
        title="Fixed Conditional Gate Demo",
        description="Fixed version that actually calls the conditional_gate tool",
        nodes=workflow_spec.nodes,
        edges=workflow_spec.edges,
        metadata={
            "tags": ["conditional", "fixed", "routing", "working"],
            "complexity": "intermediate",
            "use_case": "demo",
            "features": ["actual_tool_calling", "conditional_routing"]
        }
    )
    
    # Save the new workflow
    saved_id = storage.save_workflow(new_workflow)
    
    print("✅ Saved fixed workflow:")
    print(f"   ID: {saved_id}")
    print(f"   Title: {new_workflow.title}")
    print(f"   Description: {new_workflow.description}")
    
    return saved_id


if __name__ == "__main__":
    fix_agent_instructions()