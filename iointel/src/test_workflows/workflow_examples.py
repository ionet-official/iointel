"""
Central repository for workflow examples used across tests and UI examples.

This module provides a single source of truth for workflow examples that can be:
1. Used in the web UI dropdown
2. Imported by test suites
3. Extended with new examples as the system grows
"""

import uuid
from typing import Dict, Any
from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec, EdgeSpec, AgentConfig, AgentNode


def create_workflow_examples() -> Dict[str, WorkflowSpec]:
    """
    Create a dictionary of workflow examples.
    
    Returns:
        Dict[str, WorkflowSpec]: Dictionary mapping example_id to WorkflowSpec
    """
    examples = {}
    
    # Example 1: Simple Addition
    examples["simple_addition"] = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Simple Addition",
        description="Add two numbers together",
        nodes=[
            AgentNode(
                id="add_numbers",
                type="agent",
                label="Add 10 + 5",
                data=AgentConfig(
                    agent_instructions="Add the numbers 10 and 5 together using the add tool",
                    tools=["add"],
                    config={"a": 10, "b": 5},
                )
            )
        ],
        edges=[],
        reasoning="Simple single-node workflow to test basic tool execution"
    )
    
    # Example 2: Linear Chain
    examples["linear_chain"] = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Linear Chain Calculation",
        description="Chain calculations: (10+5)*2 = 30, √30 ≈ 5.477, +1 ≈ 6.477",
        nodes=[
            AgentNode(
                id="step_a",
                type="agent",
                label="Step A: Add 10+5",
                data=AgentConfig(
                    agent_instructions="Use the add tool to complete this task",
                    tools=["add"],
                    config={"a": 10, "b": 5},
                )
            ),
            AgentNode(
                id="step_b",
                type="agent",
                label="Step B: Multiply by 2",
                data=AgentConfig(
                    agent_instructions="Use the multiply tool to complete this task",
                    tools=["multiply"],
                    config={"a": "{step_a}", "b": 2},
                )
            ),
            AgentNode(
                id="step_c",
                type="agent",
                label="Step C: Square root",
                data=AgentConfig(
                    agent_instructions="Use the square_root tool to complete this task",
                    tools=["square_root"],
                    config={"x": "{step_b}"},
                )
            ),
            AgentNode(
                id="step_d",
                type="agent",
                label="Step D: Add 1",
                data=AgentConfig(
                    agent_instructions="Use the add tool to complete this task",
                    tools=["add"],
                    config={"a": "{step_c}", "b": 1},
                )
            )
        ],
        edges=[
            EdgeSpec(id="a_to_b", source="step_a", target="step_b"),
            EdgeSpec(id="b_to_c", source="step_b", target="step_c"),
            EdgeSpec(id="c_to_d", source="step_c", target="step_d"),
        ],
        reasoning="Linear chain workflow testing data flow resolution through multiple steps"
    )
    
    # Example 3: Parallel Branches
    examples["parallel_branches"] = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Parallel Branches",
        description="Test parallel execution: 15 → (15²=225, 15*2=30) → 225+30=255",
        nodes=[
            AgentNode(
                id="source",
                type="agent",
                label="Source: Add 10+5",
                data=AgentConfig(
                    agent_instructions="Use the add tool to complete this task",
                    tools=["add"],
                    config={"a": 10, "b": 5},
                )
            ),
            AgentNode(
                id="branch_1",
                type="agent",
                label="Branch 1: Square",
                data=AgentConfig(
                    agent_instructions="Use the multiply tool to complete this task",
                    tools=["multiply"],
                    config={"a": "{source}", "b": "{source}"},
                )
            ),
            AgentNode(
                id="branch_2",
                type="agent",
                label="Branch 2: Double",
                data=AgentConfig(
                    agent_instructions="Use the multiply tool to complete this task",
                    tools=["multiply"],
                    config={"a": "{source}", "b": 2},
                )
            ),
            AgentNode(
                id="merge",
                type="agent",
                label="Merge: Add branches",
                data=AgentConfig(
                    agent_instructions="Use the add tool to complete this task",
                    tools=["add"],
                    config={"a": "{branch_1}", "b": "{branch_2}"},
                )
            )
        ],
        edges=[
            EdgeSpec(id="source_to_branch1", source="source", target="branch_1"),
            EdgeSpec(id="source_to_branch2", source="source", target="branch_2"),
            EdgeSpec(id="branch1_to_merge", source="branch_1", target="merge"),
            EdgeSpec(id="branch2_to_merge", source="branch_2", target="merge"),
        ],
        reasoning="Parallel branches workflow testing true parallel execution with DAG topology"
    )
    
    # Example 4: Weather Temperature Addition (Fixed)
    examples["weather_temps"] = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Weather Temperature Addition",
        description="Get weather for Paris and London, add their temperatures",
        nodes=[
            AgentNode(
                id="get_paris_temp",
                type="agent",
                label="Get Paris Temperature",
                data=AgentConfig(
                    agent_instructions="Use the get_weather tool to complete this task",
                    tools=["get_weather"],
                    config={"city": "Paris"},
                )
            ),
            AgentNode(
                id="get_london_temp",
                type="agent",
                label="Get London Temperature",
                data=AgentConfig(
                    agent_instructions="Use the get_weather tool to complete this task",
                    tools=["get_weather"],
                    config={"city": "London"},
                )
            ),
            AgentNode(
                id="add_temperatures",
                type="agent",
                label="Add Temperatures",
                data=AgentConfig(
                    agent_instructions="Use the add tool to complete this task",
                    tools=["add"],
                    config={"a": "{get_paris_temp.temp}", "b": "{get_london_temp.temp}"},
                )
            )
        ],
        edges=[
            EdgeSpec(id="paris_to_add", source="get_paris_temp", target="add_temperatures"),
            EdgeSpec(id="london_to_add", source="get_london_temp", target="add_temperatures"),
        ],
        reasoning="Weather workflow testing structured data access with direct .temp field resolution"
    )
    
    # Example 5: Agent-to-Agent Communication
    examples["joke_workflow"] = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Joke to Laughter Workflow",
        description="A workflow where a joke agent creates a joke, and an audience agent evaluates the severity of laughter",
        nodes=[
            AgentNode(
                id="joke_agent",
                type="agent",
                label="Create Joke",
                data=AgentConfig(
                    agent_instructions="Create a humorous joke that would make an audience laugh.",
                )
            ),
            AgentNode(
                id="audience_agent",
                type="agent", 
                label="Evaluate Joke",
                data=AgentConfig(
                    agent_instructions="Evaluate this joke: {joke_agent}. Rate the humor and expected laughter severity.",
                )
            )
        ],
        edges=[
            EdgeSpec(
                id="joke_to_eval",
                source="joke_agent",
                target="audience_agent",
                sourceHandle="joke_output",
                targetHandle="joke_input"
            )
        ],
        reasoning="Agent-to-agent workflow testing instruction template resolution with {node_id} references"
    )
    
    # Example 6: Mixed Tool and Agent Workflow
    examples["mixed_workflow"] = WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Mixed Tool and Agent Workflow",
        description="Get weather data with tools, then analyze it with an agent",
        nodes=[
            AgentNode(
                id="get_weather",
                type="agent",
                label="Get Weather",
                data=AgentConfig(
                    agent_instructions="Use the get_weather tool to complete this task",
                    tools=["get_weather"],
                    config={"city": "New York"},
                )
            ),
            AgentNode(
                id="analyze_weather",
                type="agent",
                label="Analyze Weather",
                data=AgentConfig(
                    agent_instructions="Analyze this weather data: {get_weather}. Provide recommendations for outdoor activities.",
                )
            )
        ],
        edges=[
            EdgeSpec(id="weather_to_analysis", source="get_weather", target="analyze_weather")
        ],
        reasoning="Mixed workflow combining tool execution with agent analysis"
    )
    
    return examples


def get_example_metadata() -> Dict[str, Dict[str, Any]]:
    """
    Get metadata for all workflow examples for UI display.
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping example_id to metadata
    """
    examples = create_workflow_examples()
    
    metadata = {}
    for key, workflow in examples.items():
        metadata[key] = {
            "title": workflow.title,
            "description": workflow.description,
            "node_count": len(workflow.nodes),
            "edge_count": len(workflow.edges),
            "types": list({node.type for node in workflow.nodes}),
            "complexity": _calculate_complexity(workflow)
        }
    
    return metadata


def _calculate_complexity(workflow: WorkflowSpec) -> str:
    """Calculate workflow complexity for UI display."""
    node_count = len(workflow.nodes)
    edge_count = len(workflow.edges)
    
    if node_count == 1:
        return "Simple"
    elif node_count <= 3 and edge_count <= 3:
        return "Basic"
    elif node_count <= 6 and edge_count <= 8:
        return "Intermediate"
    else:
        return "Advanced"


def get_example_by_id(example_id: str) -> WorkflowSpec:
    """
    Get a specific workflow example by ID.
    
    Args:
        example_id: The ID of the example to retrieve
        
    Returns:
        WorkflowSpec: The workflow specification
        
    Raises:
        KeyError: If the example_id doesn't exist
    """
    examples = create_workflow_examples()
    if example_id not in examples:
        available_ids = list(examples.keys())
        raise KeyError(f"Example '{example_id}' not found. Available: {available_ids}")
    
    return examples[example_id]


def get_examples_by_type(node_type: str) -> Dict[str, WorkflowSpec]:
    """
    Get all workflow examples that contain a specific node type.
    
    Args:
        node_type: The type of node to filter by ('tool', 'agent', 'decision', 'workflow_call')
        
    Returns:
        Dict[str, WorkflowSpec]: Dictionary of matching examples
    """
    examples = create_workflow_examples()
    
    filtered_examples = {}
    for example_id, workflow in examples.items():
        if any(node.type == node_type for node in workflow.nodes):
            filtered_examples[example_id] = workflow
    
    return filtered_examples


def get_examples_by_complexity(complexity: str) -> Dict[str, WorkflowSpec]:
    """
    Get all workflow examples of a specific complexity level.
    
    Args:
        complexity: The complexity level ('Simple', 'Basic', 'Intermediate', 'Advanced')
        
    Returns:
        Dict[str, WorkflowSpec]: Dictionary of matching examples
    """
    examples = create_workflow_examples()
    
    filtered_examples = {}
    for example_id, workflow in examples.items():
        if _calculate_complexity(workflow) == complexity:
            filtered_examples[example_id] = workflow
    
    return filtered_examples