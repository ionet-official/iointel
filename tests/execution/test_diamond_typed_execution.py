#!/usr/bin/env python3
"""
Test diamond workflow with typed execution to verify agent-to-agent data passing.
"""

import uuid
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load environment
env_path = Path(__file__).parent / "creds.env"
if env_path.exists():
    load_dotenv(env_path)

from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec, NodeSpec, NodeData, EdgeSpec
)
from iointel.src.utilities.dag_executor import DAGExecutor
from iointel.src.utilities.graph_nodes import WorkflowState
from iointel.src.utilities.io_logger import get_component_logger

# CRITICAL: Import tools so they appear in TOOLS_REGISTRY!
from iointel.src.agent_methods.tools.basic_math import *

logger = get_component_logger("TEST_DIAMOND_TYPED")


def create_diamond_workflow():
    """Create a diamond-shaped workflow to test parallel math operations."""
    
    nodes = [
        # Agent 1 - Generates a random number
        NodeSpec(
            id="random_generator",
            type="agent",
            label="Random Number Generator",
            data=NodeData(
                agent_instructions="You are a random number generator. Generate a random integer between 50 and 100 using the random_int tool. Return ONLY the number.",
                tools=["random_int"],
                config={},
                ins=[],
                outs=["random_number"]
            )
        ),
        
        # Agent 2 - Squares the number
        NodeSpec(
            id="squarer",
            type="agent",
            label="Number Squarer",
            data=NodeData(
                agent_instructions="You square numbers. Take the input number and calculate its square using the power tool with exponent 2. Return ONLY the result.",
                tools=["power"],
                config={},
                ins=["number_to_square"],
                outs=["squared_result"]
            )
        ),
        
        # Agent 3 - Multiplies by 10 and subtracts 237
        NodeSpec(
            id="transformer",
            type="agent",
            label="Number Transformer",
            data=NodeData(
                agent_instructions="You transform numbers by multiplying by 10 and subtracting 237. Use the multiply and subtract tools. Return ONLY the final result.",
                tools=["multiply", "subtract"],
                config={},
                ins=["number_to_transform"],
                outs=["transformed_result"]
            )
        ),
        
        # Agent 4 - Adds the two results together
        NodeSpec(
            id="adder",
            type="agent",
            label="Result Adder",
            data=NodeData(
                agent_instructions="You add two numbers together. Take the two input numbers and add them using the add tool. Return ONLY the sum.",
                tools=["add"],
                config={},
                ins=["first_number", "second_number"],
                outs=["final_sum"]
            )
        )
    ]
    
    edges = [
        # From random generator to both processing agents
        EdgeSpec(id="e1", source="random_generator", target="squarer", 
                sourceHandle="random_number", targetHandle="number_to_square"),
        EdgeSpec(id="e2", source="random_generator", target="transformer",
                sourceHandle="random_number", targetHandle="number_to_transform"),
        # From both processing agents to the final adder
        EdgeSpec(id="e3", source="squarer", target="adder",
                sourceHandle="squared_result", targetHandle="first_number"),
        EdgeSpec(id="e4", source="transformer", target="adder",
                sourceHandle="transformed_result", targetHandle="second_number")
    ]
    
    return WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Diamond Workflow Test",
        description="Testing agent-to-agent data flow in diamond topology",
        nodes=nodes,
        edges=edges,
        metadata={}
    )


async def run_test():
    """Run the diamond workflow test with typed execution."""
    print("🧪 DIAMOND WORKFLOW WITH TYPED EXECUTION")
    print("=" * 60)
    
    # Create workflow
    workflow = create_diamond_workflow()
    
    # Create DAG executor with typed execution enabled
    executor = DAGExecutor(use_typed_execution=True)
    
    # Build execution graph - no user inputs needed since we generate a random number
    executor.build_execution_graph(
        workflow_spec=workflow,
        objective="Test diamond workflow with parallel math operations",
        conversation_id="test_diamond_typed"
    )
    
    print("\n📊 DAG Structure:")
    print(f"   Nodes: {list(executor.nodes.keys())}")
    print(f"   Execution order: {executor.execution_order}")
    
    # Execute
    initial_state = WorkflowState(
        conversation_id="test_diamond_typed",
        initial_text="",
        results={}
    )
    
    print("\n🚀 Starting typed execution...")
    print("-" * 60)
    
    try:
        final_state = await executor.execute_dag(initial_state)
        
        # Display results
        print("\n✅ EXECUTION COMPLETED:")
        print("-" * 60)
        
        for node_id, result in final_state.results.items():
            print(f"\n🔸 {node_id}:")
            
            # Extract the actual result value
            result_value = None
            if hasattr(result, 'result'):
                result_value = result.result
            elif isinstance(result, dict):
                if 'result' in result:
                    result_value = result['result']
                elif 'agent_response' in result and isinstance(result['agent_response'], dict):
                    result_value = result['agent_response'].get('result')
            
            if result_value:
                print(f"   Result: {str(result_value)[:200]}...")
            else:
                print(f"   Raw: {str(result)[:200]}...")
            
            # Extract tool usage if available
            tool_usage = None
            if hasattr(result, 'agent_response') and hasattr(result.agent_response, 'tool_usage'):
                tool_usage = result.agent_response.tool_usage
            elif isinstance(result, dict):
                if 'agent_response' in result and isinstance(result['agent_response'], dict):
                    tool_usage = result['agent_response'].get('tool_usage')
                elif 'tool_usage' in result:
                    tool_usage = result['tool_usage']
            
            if tool_usage:
                print(f"   🔧 Tools used: {', '.join([t.get('tool_name', 'unknown') for t in tool_usage])}")
        
        # Verify data flow
        print("\n🔍 DATA FLOW VERIFICATION:")
        print("-" * 60)
        
        # Extract numeric results for verification
        random_num = None
        squared_result = None
        transformed_result = None
        final_sum = None
        
        # Get the random number
        if 'random_generator' in final_state.results:
            result = final_state.results['random_generator']
            try:
                if hasattr(result, 'result'):
                    random_num = int(str(result.result).strip())
                elif isinstance(result, dict) and 'result' in result:
                    random_num = int(str(result['result']).strip())
                print(f"✅ Random generator produced: {random_num}")
            except:
                print("❌ Could not extract random number")
        
        # Check if squarer received and processed the number
        if 'squarer' in final_state.results:
            result = final_state.results['squarer']
            try:
                if hasattr(result, 'result'):
                    squared_result = int(str(result.result).strip())
                elif isinstance(result, dict) and 'result' in result:
                    squared_result = int(str(result['result']).strip())
                
                if random_num and squared_result == random_num ** 2:
                    print(f"✅ Squarer correctly computed: {random_num}² = {squared_result}")
                else:
                    print(f"⚠️  Squarer result: {squared_result} (expected {random_num ** 2 if random_num else '?'})")
            except:
                print("❌ Could not extract squared result")
        
        # Check if transformer received and processed the number
        if 'transformer' in final_state.results:
            result = final_state.results['transformer']
            try:
                if hasattr(result, 'result'):
                    transformed_result = int(str(result.result).strip())
                elif isinstance(result, dict) and 'result' in result:
                    transformed_result = int(str(result['result']).strip())
                
                expected = (random_num * 10 - 237) if random_num else None
                if random_num and transformed_result == expected:
                    print(f"✅ Transformer correctly computed: {random_num} * 10 - 237 = {transformed_result}")
                else:
                    print(f"⚠️  Transformer result: {transformed_result} (expected {expected if expected else '?'})")
            except:
                print("❌ Could not extract transformed result")
        
        # Check if adder received both results and computed the sum
        if 'adder' in final_state.results:
            result = final_state.results['adder']
            try:
                if hasattr(result, 'result'):
                    final_sum = int(str(result.result).strip())
                elif isinstance(result, dict) and 'result' in result:
                    final_sum = int(str(result['result']).strip())
                
                expected = (squared_result + transformed_result) if squared_result and transformed_result else None
                if squared_result and transformed_result and final_sum == expected:
                    print(f"✅ Adder correctly computed: {squared_result} + {transformed_result} = {final_sum}")
                else:
                    print(f"⚠️  Adder result: {final_sum} (expected {expected if expected else '?'})")
            except:
                print("❌ Could not extract final sum")
        
        print("\n" + "=" * 60)
        print("🎉 Diamond workflow test complete!")
        
    except Exception as e:
        print(f"\n❌ EXECUTION FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_test())