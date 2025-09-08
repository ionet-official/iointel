#!/usr/bin/env python3
"""
Demonstration of WorkflowSpec conversion methods.
Shows how to convert WorkflowSpec to executable formats.
"""

import uuid
from pathlib import Path

from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec,
    NodeSpec,
    NodeData,
    EdgeSpec,
    EdgeData
)
from iointel.src.agent_methods.data_models.datamodels import AgentParams


def create_sample_workflow():
    """Create a sample workflow for demonstration."""
    return WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Weather Alert System",
        description="Monitor weather and send alerts based on conditions",
        nodes=[
            NodeSpec(
                id="get_weather",
                type="tool",
                label="Get Weather Data",
                data=NodeData(
                    tool_name="weather_api",
                    config={
                        "location": "San Francisco",
                        "units": "fahrenheit"
                    },
                    ins=[],
                    outs=["weather_data", "temperature", "conditions"]
                )
            ),
            NodeSpec(
                id="check_temp",
                type="decision",
                label="Check Temperature",
                data=NodeData(
                    tool_name="number_compare",
                    config={
                        "operator": ">",
                        "threshold": 85
                    },
                    ins=["temperature"],
                    outs=["is_hot", "details"]
                )
            ),
            NodeSpec(
                id="analyze_weather",
                type="agent",
                label="Analyze Conditions",
                data=NodeData(
                    agent_instructions="Analyze the weather data and determine if any alerts should be sent. Consider temperature, humidity, wind speed, and any severe weather warnings.",
                    config={
                        "model": "gpt-4",
                        "temperature": 0.3
                    },
                    ins=["weather_data", "is_hot"],
                    outs=["alert_message", "severity"]
                )
            ),
            NodeSpec(
                id="send_alert",
                type="tool",
                label="Send Alert",
                data=NodeData(
                    tool_name="send_notification",
                    config={
                        "channel": "email",
                        "recipients": ["alerts@example.com"]
                    },
                    ins=["alert_message", "severity"],
                    outs=["sent", "message_id"]
                )
            )
        ],
        edges=[
            EdgeSpec(
                id="weather_to_temp",
                source="get_weather",
                target="check_temp",
                sourceHandle="temperature",
                targetHandle="temperature"
            ),
            EdgeSpec(
                id="weather_to_analysis",
                source="get_weather",
                target="analyze_weather",
                sourceHandle="weather_data",
                targetHandle="weather_data"
            ),
            EdgeSpec(
                id="temp_to_analysis",
                source="check_temp",
                target="analyze_weather",
                sourceHandle="is_hot",
                targetHandle="is_hot"
            ),
            EdgeSpec(
                id="analysis_to_alert",
                source="analyze_weather",
                target="send_alert",
                sourceHandle="alert_message",
                targetHandle="alert_message",
                data=EdgeData(condition="severity != 'none'")
            )
        ]
    )


def main():
    """Demonstrate workflow conversion methods."""
    print("=== WorkflowSpec Conversion Demo ===\n")
    
    # Create a sample workflow
    workflow_spec = create_sample_workflow()
    print(f"Created workflow: {workflow_spec.title}")
    print(f"Description: {workflow_spec.description}")
    print(f"Nodes: {len(workflow_spec.nodes)}")
    print(f"Edges: {len(workflow_spec.edges)}")
    
    # Validate the workflow structure
    print("\n1. Validating workflow structure...")
    issues = workflow_spec.validate_structure()
    if issues:
        print(f"   ❌ Validation issues found: {issues}")
    else:
        print("   ✅ Workflow structure is valid")
    
    # Convert to WorkflowDefinition
    print("\n2. Converting to WorkflowDefinition...")
    workflow_def = workflow_spec.to_workflow_definition()
    print(f"   ✅ Created WorkflowDefinition: {workflow_def.name}")
    print(f"   - Tasks: {len(workflow_def.tasks)}")
    print(f"   - Client mode: {workflow_def.client_mode}")
    
    # Convert with custom agents
    print("\n3. Converting with custom agents...")
    custom_agents = [
        AgentParams(
            name="WeatherAnalyst",
            instructions="Expert weather analyst with focus on safety",
            model="gpt-4",
            tools=["weather_api", "send_notification"]
        )
    ]
    workflow_def_custom = workflow_spec.to_workflow_definition(
        agents=custom_agents,
        default_timeout=90,
        default_retries=2
    )
    print("   ✅ Converted with custom agents")
    print(f"   - Agent: {workflow_def_custom.agents[0].name}")
    print(f"   - Default timeout: {workflow_def_custom.tasks[0].execution_metadata['timeout']}s")
    
    # Convert to YAML
    print("\n4. Converting to YAML...")
    yaml_output = workflow_spec.to_yaml()
    print("   ✅ YAML output (first 500 chars):")
    print("   " + "-" * 50)
    print(yaml_output[:500] + "..." if len(yaml_output) > 500 else yaml_output)
    
    # Save YAML to file
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    yaml_file = output_dir / f"{workflow_spec.title.replace(' ', '_')}.yaml"
    yaml_file.write_text(yaml_output)
    print(f"\n5. Saved YAML to: {yaml_file}")
    
    # Demonstrate task details
    print("\n6. Task Details from WorkflowDefinition:")
    for i, task in enumerate(workflow_def.tasks, 1):
        print(f"\n   Task {i}: {task.name}")
        print(f"   - ID: {task.task_id}")
        print(f"   - Type: {task.type}")
        print(f"   - Objective: {task.objective}")
        
        if task.task_metadata:
            if "tool_name" in task.task_metadata:
                print(f"   - Tool: {task.task_metadata['tool_name']}")
            if "agent_instructions" in task.task_metadata:
                print(f"   - Instructions: {task.task_metadata['agent_instructions'][:80]}...")
            if "workflow_id" in task.task_metadata:
                print(f"   - Workflow ID: {task.task_metadata['workflow_id']}")
            
            if "ports" in task.task_metadata:
                ports = task.task_metadata["ports"]
                print(f"   - Inputs: {ports.get('inputs', [])}")
                print(f"   - Outputs: {ports.get('outputs', [])}")
    
    # Show edge conditions
    print("\n7. Edge Conditions:")
    for edge in workflow_spec.edges:
        if edge.data and edge.data.condition:
            print(f"   - {edge.source} → {edge.target}: {edge.data.condition}")
    
    print("\n✅ Demo completed successfully!")
    print(f"\nYAML file saved to: {yaml_file.absolute()}")


if __name__ == "__main__":
    main()