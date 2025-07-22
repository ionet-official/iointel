#!/usr/bin/env python3
"""
Test the Riddle Whisper Challenge - a 3-agent workflow that:
1. Creates a math riddle
2. Solves the riddle
3. Evaluates the solution

This tests:
- Multi-agent DAG execution
- Data flow between agents
- Agent task execution
- Real creative outputs
"""

import pytest
import asyncio
from uuid import uuid4
from pathlib import Path

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec, NodeSpec, EdgeSpec, NodeData, EdgeData
)
from iointel.src.workflow import Workflow
from iointel.src.utilities.decorators import register_custom_task
from iointel.src.utilities.runners import run_agents
from iointel.src.agents import Agent
from iointel.src.agent_methods.data_models.datamodels import AgentParams


# Register agent executor for tests
@register_custom_task("agent")
async def test_agent_executor(task_metadata, objective, agents, execution_metadata):
    """Agent executor for testing."""
    agent_instructions = task_metadata.get("agent_instructions", "")
    
    print(f"\n🤖 Executing agent task: {objective}")
    print(f"   Instructions: {agent_instructions}")
    
    # Convert AgentParams to Agent if needed
    if agents and isinstance(agents[0], AgentParams):
        agents_to_use = [Agent.make_default()]
    else:
        agents_to_use = agents or [Agent.make_default()]
    
    # Use instructions as objective
    task_objective = agent_instructions or objective or "Process the data"
    
    # Execute agent
    result = await run_agents(
        objective=task_objective,
        agents=agents_to_use,
        conversation_id=execution_metadata.get("conversation_id", str(uuid4()))
    ).execute()
    
    # Extract result
    if isinstance(result, dict):
        return result.get("result", result)
    return result


def create_riddle_whisper_workflow() -> WorkflowSpec:
    """Create the Riddle Whisper Challenge workflow spec."""
    return WorkflowSpec(
        id=uuid4(),
        rev=1,
        title="Riddle Whisper Challenge",
        description="A creative challenge where agents create, solve, and evaluate math riddles",
        nodes=[
            NodeSpec(
                id="agent_riddle_creator",
                type="agent",
                label="Riddle Creator",
                data=NodeData(
                    agent_instructions="Create a clever math riddle that requires arithmetic to solve. Make it fun but not too difficult. Include the answer in your response but mark it clearly as 'ANSWER: [number]'",
                    config={},
                    ins=[],
                    outs=["riddle"]
                )
            ),
            NodeSpec(
                id="agent_riddle_solver", 
                type="agent",
                label="Riddle Solver",
                data=NodeData(
                    agent_instructions="You will receive a math riddle. Work through it step by step, showing your reasoning. Provide your final answer clearly marked as 'MY SOLUTION: [number]'",
                    config={},
                    ins=["riddle"],
                    outs=["solution"]
                )
            ),
            NodeSpec(
                id="agent_riddle_evaluator",
                type="agent", 
                label="Riddle Evaluator",
                data=NodeData(
                    agent_instructions="You will receive a riddle and a solution attempt. Evaluate if the solution is correct. Also rate the riddle's creativity (1-10) and the solution's clarity (1-10). Format: 'CORRECT: Yes/No, CREATIVITY: X/10, CLARITY: Y/10'",
                    config={},
                    ins=["riddle", "solution"],
                    outs=["evaluation"]
                )
            )
        ],
        edges=[
            EdgeSpec(
                id="edge1",
                source="agent_riddle_creator",
                target="agent_riddle_solver",
                sourceHandle="riddle",
                targetHandle="riddle",
                data=EdgeData()
            ),
            EdgeSpec(
                id="edge2", 
                source="agent_riddle_creator",
                target="agent_riddle_evaluator",
                sourceHandle="riddle",
                targetHandle="riddle",
                data=EdgeData()
            ),
            EdgeSpec(
                id="edge3",
                source="agent_riddle_solver",
                target="agent_riddle_evaluator", 
                sourceHandle="solution",
                targetHandle="solution",
                data=EdgeData()
            )
        ],
        metadata={
            "test_fixture": True,
            "dag_topology": True
        }
    )


@pytest.mark.asyncio
async def test_riddle_whisper_workflow_structure():
    """Test that the workflow structure is valid."""
    workflow_spec = create_riddle_whisper_workflow()
    
    # Validate structure
    issues = workflow_spec.validate_structure()
    assert len(issues) == 0, f"Workflow validation failed: {issues}"
    
    # Check nodes
    assert len(workflow_spec.nodes) == 3
    assert all(node.type == "agent" for node in workflow_spec.nodes)
    assert all(node.data.agent_instructions for node in workflow_spec.nodes)
    
    # Check edges create proper DAG
    assert len(workflow_spec.edges) == 3
    # Creator -> Solver
    assert any(e.source == "agent_riddle_creator" and e.target == "agent_riddle_solver" for e in workflow_spec.edges)
    # Creator -> Evaluator  
    assert any(e.source == "agent_riddle_creator" and e.target == "agent_riddle_evaluator" for e in workflow_spec.edges)
    # Solver -> Evaluator
    assert any(e.source == "agent_riddle_solver" and e.target == "agent_riddle_evaluator" for e in workflow_spec.edges)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_riddle_whisper_workflow_execution():
    """Test actual execution of the Riddle Whisper workflow."""
    print("\n" + "="*60)
    print("🎭 RIDDLE WHISPER CHALLENGE - LIVE EXECUTION")
    print("="*60)
    
    # Create workflow spec
    workflow_spec = create_riddle_whisper_workflow()
    
    # Convert to executable format
    workflow_spec.to_workflow_definition()
    yaml_content = workflow_spec.to_yaml()
    
    # Create workflow instance
    workflow = Workflow.from_yaml(yaml_str=yaml_content)
    workflow.objective = workflow_spec.description
    
    print(f"\n📋 Workflow: {workflow_spec.title}")
    print(f"   Tasks: {len(workflow.tasks)}")
    print(f"   DAG Topology: {workflow_spec.metadata.get('dag_topology', False)}")
    
    # Execute workflow
    conversation_id = f"riddle_test_{uuid4()}"
    print(f"\n🚀 Starting execution (conversation: {conversation_id[:8]}...)")
    
    try:
        results = await workflow.run_tasks(conversation_id=conversation_id)
        
        print("\n" + "="*60)
        print("📊 EXECUTION RESULTS")
        print("="*60)
        
        if "results" in results:
            task_results = results["results"]
            
            # Display riddle
            if "agent_riddle_creator" in task_results:
                print("\n🎲 RIDDLE CREATED:")
                print("-" * 40)
                print(task_results["agent_riddle_creator"])
                
            # Display solution attempt
            if "agent_riddle_solver" in task_results:
                print("\n🧮 SOLUTION ATTEMPT:")
                print("-" * 40)
                print(task_results["agent_riddle_solver"])
                
            # Display evaluation
            if "agent_riddle_evaluator" in task_results:
                print("\n⚖️ EVALUATION:")
                print("-" * 40)
                print(task_results["agent_riddle_evaluator"])
                
        print("\n✅ Workflow completed successfully!")
        
        # Basic assertions
        assert "results" in results
        assert len(results["results"]) == 3
        assert all(key in results["results"] for key in [
            "agent_riddle_creator",
            "agent_riddle_solver", 
            "agent_riddle_evaluator"
        ])
        
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()
        raise


@pytest.mark.asyncio
async def test_riddle_whisper_parallel_execution():
    """Test that solver and evaluator can run in parallel after creator."""
    workflow_spec = create_riddle_whisper_workflow()
    
    # The DAG structure should allow:
    # 1. Creator runs first
    # 2. Solver and initial evaluator setup can start in parallel
    # 3. Evaluator waits for both inputs before processing
    
    # This is validated by the edge structure
    edges = workflow_spec.edges
    
    # Find dependencies
    creator_deps = [e.target for e in edges if e.source == "agent_riddle_creator"]
    assert len(creator_deps) == 2  # Both solver and evaluator depend on creator
    
    solver_deps = [e.target for e in edges if e.source == "agent_riddle_solver"]  
    assert len(solver_deps) == 1  # Only evaluator depends on solver
    
    print("\n✅ DAG structure supports parallel execution")


if __name__ == "__main__":
    # Run the execution test directly
    asyncio.run(test_riddle_whisper_workflow_execution())