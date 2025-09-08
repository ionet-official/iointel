#!/usr/bin/env python3
"""
Test script to manually test Llama model for WorkflowPlanner generation.
Uses the generate_only function with WORKFLOW_PLANNER_MODEL environment variable.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv("creds.env")

async def test_llama_workflow_planner():
    """Test Llama model for workflow planning."""
    
    # Import after loading env vars
    from iointel.src.utilities.workflow_helpers import generate_only
    
    print("🦙 Testing Llama WorkflowPlanner")
    print("=" * 50)
    
    # Check current model configuration
    current_model = os.getenv("WORKFLOW_PLANNER_MODEL", "gpt-4o")
    print(f"📋 Current WORKFLOW_PLANNER_MODEL: {current_model}")
    
    # Test prompts - start simple
    test_prompts = [
        "Create a simple agent that greets users",
        "Make a workflow with user input and a conversational agent",
        "Create a stock analysis agent that gets user input for a ticker symbol"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🔍 Test {i}: {prompt}")
        print("-" * 30)
        
        try:
            workflow_spec = await generate_only(
                prompt=prompt,
                debug=True  # Enable debug logging
            )
            
            if workflow_spec:
                print(f"✅ SUCCESS: Generated workflow '{workflow_spec.title}'")
                print(f"   Nodes: {len(workflow_spec.nodes)}")
                print(f"   Edges: {len(workflow_spec.edges)}")
                print(f"   Reasoning: {workflow_spec.reasoning[:200]}...")
                
                # Show node types
                node_types = {}
                for node in workflow_spec.nodes:
                    node_types[node.type] = node_types.get(node.type, 0) + 1
                print(f"   Node types: {dict(node_types)}")
                
            else:
                print("❌ FAILED: No workflow generated")
                
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print()

def main():
    """Main test function."""
    print("🚀 Starting Llama WorkflowPlanner Test")
    
    # Check if Llama model is configured
    current_model = os.getenv("WORKFLOW_PLANNER_MODEL", "gpt-4o")
    if "llama" not in current_model.lower():
        print("⚠️  Warning: WORKFLOW_PLANNER_MODEL is not set to a Llama model")
        print(f"   Current: {current_model}")
        print("   To test Llama, set: WORKFLOW_PLANNER_MODEL='meta-llama/Llama-3.3-70B-Instruct'")
        print()
    
    # Run the test
    asyncio.run(test_llama_workflow_planner())

if __name__ == "__main__":
    main()