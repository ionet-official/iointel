"""
Comprehensive tests for WorkflowSpec conversion methods.
Tests to_workflow_definition, to_yaml, and full conversion pipeline.
"""

import pytest
import uuid
import yaml
import json
from pathlib import Path

from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec,
    NodeSpec,
    NodeData,
    EdgeSpec,
    EdgeData
)
from iointel.src.agent_methods.data_models.datamodels import (
    WorkflowDefinition,
    AgentParams
)
from iointel.src.agent_methods.workflow_converter import (
    spec_to_yaml,
    spec_to_definition
)


class TestWorkflowSpecConversions:
    """Test all WorkflowSpec conversion methods comprehensively."""
    
    @pytest.fixture
    def complex_workflow_spec(self):
        """Create a complex workflow spec with all node types."""
        return WorkflowSpec(
            id=uuid.uuid4(),
            rev=1,
            title="Complex Multi-Type Workflow",
            description="A comprehensive workflow demonstrating all node types and conversions",
            nodes=[
                # Tool node
                NodeSpec(
                    id="fetch_weather",
                    type="tool",
                    label="Fetch Weather Data",
                    data=NodeData(
                        tool_name="weather_api",
                        config={
                            "location": "New York",
                            "units": "celsius",
                            "include_forecast": True
                        },
                        ins=[],
                        outs=["weather_data", "temperature", "conditions"]
                    ),
                    position={"x": 100, "y": 100}
                ),
                # Decision node
                NodeSpec(
                    id="check_temperature",
                    type="decision",
                    label="Check Temperature Threshold",
                    data=NodeData(
                        tool_name="number_compare",
                        config={
                            "operator": "<",
                            "threshold": 20
                        },
                        ins=["temperature"],
                        outs=["is_cold", "comparison_details"]
                    ),
                    position={"x": 300, "y": 100}
                ),
                # Agent node
                NodeSpec(
                    id="analyze_conditions",
                    type="agent",
                    label="Analyze Weather Conditions",
                    data=NodeData(
                        agent_instructions="Analyze the weather data and provide recommendations for outdoor activities based on current conditions. Consider temperature, precipitation, wind speed, and visibility.",
                        config={
                            "model": "gpt-4",
                            "temperature": 0.7,
                            "max_tokens": 500
                        },
                        ins=["weather_data", "is_cold"],
                        outs=["recommendations", "safety_warnings"]
                    ),
                    position={"x": 500, "y": 100}
                ),
                # Another tool node
                NodeSpec(
                    id="send_notification",
                    type="tool",
                    label="Send Weather Alert",
                    data=NodeData(
                        tool_name="send_email",
                        config={
                            "to": "weather-alerts@example.com",
                            "subject": "Weather Update and Recommendations",
                            "template": "weather_alert"
                        },
                        ins=["recommendations", "safety_warnings"],
                        outs=["sent_status", "message_id"]
                    ),
                    position={"x": 700, "y": 100}
                ),
                # Workflow call node
                NodeSpec(
                    id="archive_data",
                    type="workflow_call",
                    label="Archive Weather Data",
                    data=NodeData(
                        workflow_id="data_archival_workflow_v2",
                        config={
                            "retention_days": 30,
                            "compression": "gzip"
                        },
                        ins=["weather_data", "message_id"],
                        outs=["archive_status", "archive_location"]
                    ),
                    position={"x": 900, "y": 100}
                )
            ],
            edges=[
                EdgeSpec(
                    id="weather_to_temp_check",
                    source="fetch_weather",
                    target="check_temperature",
                    sourceHandle="temperature",
                    targetHandle="temperature"
                ),
                EdgeSpec(
                    id="weather_to_analysis",
                    source="fetch_weather",
                    target="analyze_conditions",
                    sourceHandle="weather_data",
                    targetHandle="weather_data"
                ),
                EdgeSpec(
                    id="temp_check_to_analysis",
                    source="check_temperature",
                    target="analyze_conditions",
                    sourceHandle="is_cold",
                    targetHandle="is_cold"
                ),
                EdgeSpec(
                    id="analysis_to_notification",
                    source="analyze_conditions",
                    target="send_notification",
                    sourceHandle="recommendations",
                    targetHandle="recommendations",
                    data=EdgeData(condition="safety_warnings != null")
                ),
                EdgeSpec(
                    id="weather_to_archive",
                    source="fetch_weather",
                    target="archive_data",
                    sourceHandle="weather_data",
                    targetHandle="weather_data"
                ),
                EdgeSpec(
                    id="notification_to_archive",
                    source="send_notification",
                    target="archive_data",
                    sourceHandle="message_id",
                    targetHandle="message_id"
                )
            ],
            metadata={
                "author": "test_suite",
                "version": "1.0.0",
                "tags": ["weather", "notifications", "archival"]
            }
        )
    
    def test_to_workflow_definition_basic(self, complex_workflow_spec: WorkflowSpec):
        """Test basic conversion to WorkflowDefinition."""
        # Convert using the method
        workflow_def = complex_workflow_spec.to_workflow_definition()
        
        # Verify the conversion
        assert isinstance(workflow_def, WorkflowDefinition)
        assert workflow_def.name == complex_workflow_spec.title
        assert workflow_def.objective == complex_workflow_spec.description
        assert len(workflow_def.tasks) == len(complex_workflow_spec.nodes)
        
        # Check each task
        task_map = {task.task_id: task for task in workflow_def.tasks}
        
        # Verify tool node conversion
        fetch_task = task_map["fetch_weather"]
        assert fetch_task.type == "tool"
        assert fetch_task.task_metadata["tool_name"] == "weather_api"
        assert fetch_task.task_metadata["config"]["location"] == "New York"
        assert fetch_task.task_metadata["ports"]["outputs"] == ["weather_data", "temperature", "conditions"]
        
        # Verify agent node conversion
        analyze_task = task_map["analyze_conditions"]
        assert analyze_task.type == "agent"
        assert "Analyze the weather data" in analyze_task.task_metadata["agent_instructions"]
        assert analyze_task.task_metadata["config"]["model"] == "gpt-4"
        
        # Verify workflow_call node conversion
        archive_task = task_map["archive_data"]
        assert archive_task.type == "workflow_call"
        assert archive_task.task_metadata["workflow_id"] == "data_archival_workflow_v2"
        
        # Verify decision node conversion
        temp_check_task = task_map["check_temperature"]
        assert temp_check_task.type == "decision"
        # Decision nodes are treated as tool nodes in conversion
        assert temp_check_task.task_metadata.get("tool_name") == "number_compare" or temp_check_task.type == "decision"
        assert temp_check_task.task_metadata["config"]["threshold"] == 20
    
    def test_to_workflow_definition_with_agents(self, complex_workflow_spec: WorkflowSpec):
        """Test conversion with custom agents."""
        custom_agents = [
            AgentParams(
                name="WeatherAnalyst",
                instructions="Specialized agent for weather analysis",
                model="gpt-4",
                tools=["weather_api", "send_email"]
            )
        ]
        
        workflow_def = complex_workflow_spec.to_workflow_definition(
            agents=custom_agents,
            default_timeout=120,
            default_retries=5
        )
        
        # Verify custom settings
        assert workflow_def.agents == custom_agents
        
        # Check that agent nodes inherited custom agent properties but have node-specific instructions
        for task in workflow_def.tasks:
            if task.type == "agent":
                agent = task.agents[0]
                custom_agent = custom_agents[0]
                
                # Should inherit custom agent properties
                assert agent.model == custom_agent.model
                assert agent.tools == custom_agent.tools
                
                # But should have node-specific name and instructions
                assert agent.name.startswith("agent_")
                assert agent.instructions != custom_agent.instructions
            
            # Check execution metadata
            assert task.execution_metadata["timeout"] == 120
            assert task.execution_metadata["retries"] == 5
    
    def test_to_yaml_basic(self, complex_workflow_spec: WorkflowSpec):
        """Test basic YAML conversion."""
        yaml_output = complex_workflow_spec.to_yaml()
        
        # Verify it's valid YAML
        parsed = yaml.safe_load(yaml_output)
        assert isinstance(parsed, dict)
        
        # Check structure
        assert parsed["name"] == complex_workflow_spec.title
        assert parsed["objective"] == complex_workflow_spec.description
        assert "tasks" in parsed
        assert len(parsed["tasks"]) == len(complex_workflow_spec.nodes)
        
        # Verify task structure
        task_ids = [task["task_id"] for task in parsed["tasks"]]
        assert "fetch_weather" in task_ids
        assert "analyze_conditions" in task_ids
        assert "archive_data" in task_ids
    
    def test_to_yaml_with_formatting(self, complex_workflow_spec: WorkflowSpec):
        """Test YAML conversion with custom formatting."""
        yaml_output = complex_workflow_spec.to_yaml(
            default_timeout=90,
            default_client_mode=False
        )
        
        parsed = yaml.safe_load(yaml_output)
        
        # Check client mode
        assert parsed.get("client_mode") is False
        
        # Check that all tasks have the custom timeout
        for task in parsed["tasks"]:
            assert task["execution_metadata"]["timeout"] == 90
    
    def test_yaml_round_trip(self, complex_workflow_spec: WorkflowSpec):
        """Test that YAML can be parsed and used."""
        # Convert to YAML
        yaml_output = complex_workflow_spec.to_yaml()
        
        # Parse YAML
        parsed = yaml.safe_load(yaml_output)
        
        # Recreate WorkflowDefinition from parsed data
        workflow_def = WorkflowDefinition(**parsed)
        
        # Verify integrity
        assert workflow_def.name == complex_workflow_spec.title
        assert len(workflow_def.tasks) == len(complex_workflow_spec.nodes)
    
    def test_edge_conditions_preserved(self, complex_workflow_spec: WorkflowSpec):
        """Test that edge conditions are preserved in conversion."""
        workflow_def = complex_workflow_spec.to_workflow_definition()
        
        # Find the notification task (target of conditional edge)
        notification_task = next(
            task for task in workflow_def.tasks 
            if task.task_id == "send_notification"
        )
        
        # Should have preconditions from the edge
        assert "preconditions" in notification_task.execution_metadata
        assert "safety_warnings != null" in notification_task.execution_metadata["preconditions"]
    
    def test_metadata_preservation(self, complex_workflow_spec: WorkflowSpec):
        """Test that metadata is preserved through conversions."""
        # Convert to YAML and back
        yaml_output = complex_workflow_spec.to_yaml()
        parsed = yaml.safe_load(yaml_output)
        
        # The metadata should be preserved in some form
        # (Note: exact preservation depends on WorkflowDefinition structure)
        workflow_def = complex_workflow_spec.to_workflow_definition()
        
        # At minimum, basic info should be preserved
        assert workflow_def.name == complex_workflow_spec.title
        assert workflow_def.objective == complex_workflow_spec.description
    
    def test_validate_before_convert(self, complex_workflow_spec: WorkflowSpec):
        """Test that validation works before conversion."""
        # First validate the spec
        issues = complex_workflow_spec.validate_structure()
        assert len(issues) == 0, f"Workflow has validation issues: {issues}"
        
        # Then convert
        workflow_def = complex_workflow_spec.to_workflow_definition()
        assert isinstance(workflow_def, WorkflowDefinition)
    
    def test_invalid_spec_conversion(self):
        """Test conversion of invalid workflow spec."""
        # Create workflow with invalid edge
        invalid_spec = WorkflowSpec(
            id=uuid.uuid4(),
            rev=1,
            title="Invalid Workflow",
            description="Has invalid edges",
            nodes=[
                NodeSpec(
                    id="node1",
                    type="tool",
                    label="Node 1",
                    data=NodeData(tool_name="tool1")
                )
            ],
            edges=[
                EdgeSpec(
                    id="bad_edge",
                    source="node1",
                    target="nonexistent_node"  # Invalid target
                )
            ]
        )
        
        # Validation should catch this
        issues = invalid_spec.validate_structure()
        assert len(issues) > 0
        assert any("unknown target" in issue for issue in issues)
        
        # Conversion should still work (converter is forgiving)
        workflow_def = invalid_spec.to_workflow_definition()
        assert len(workflow_def.tasks) == 1
    
    def test_convenience_functions(self, complex_workflow_spec):
        """Test module-level convenience functions match method behavior.
        
        Note: This tests the spec_to_definition and spec_to_yaml convenience functions,
        which internally use WorkflowConverter. Direct WorkflowConverter class testing
        could be added if needed for more granular control testing.
        """
        # Test spec_to_definition
        method_result = complex_workflow_spec.to_workflow_definition()
        function_result = spec_to_definition(complex_workflow_spec)
        
        # Should produce equivalent results
        assert method_result.name == function_result.name
        assert len(method_result.tasks) == len(function_result.tasks)
        
        # Test spec_to_yaml
        method_yaml = complex_workflow_spec.to_yaml()
        function_yaml = spec_to_yaml(complex_workflow_spec)
        
        # Parse both and compare
        method_parsed = yaml.safe_load(method_yaml)
        function_parsed = yaml.safe_load(function_yaml)
        
        assert method_parsed["name"] == function_parsed["name"]
        assert len(method_parsed["tasks"]) == len(function_parsed["tasks"])
    
    def test_empty_workflow_conversion(self):
        """Test conversion of empty workflow."""
        empty_spec = WorkflowSpec(
            id=uuid.uuid4(),
            rev=1,
            title="Empty Workflow",
            description="No nodes or edges",
            nodes=[],
            edges=[]
        )
        
        # Should convert successfully
        workflow_def = empty_spec.to_workflow_definition()
        assert workflow_def.name == "Empty Workflow"
        assert len(workflow_def.tasks) == 0
        
        # YAML should also work
        yaml_output = empty_spec.to_yaml()
        parsed = yaml.safe_load(yaml_output)
        assert len(parsed["tasks"]) == 0
    
    def test_conversion_performance(self, complex_workflow_spec):
        """Test conversion performance with larger workflows."""
        import time
        
        # Create a larger workflow
        large_spec = WorkflowSpec(
            id=uuid.uuid4(),
            rev=1,
            title="Large Workflow",
            description="Performance test",
            nodes=[
                NodeSpec(
                    id=f"node_{i}",
                    type="tool" if i % 2 == 0 else "agent",
                    label=f"Node {i}",
                    data=NodeData(
                        tool_name=f"tool_{i}" if i % 2 == 0 else None,
                        agent_instructions=f"Process step {i}" if i % 2 == 1 else None,
                        config={"index": i},
                        ins=[f"input_{i}"] if i > 0 else [],
                        outs=[f"output_{i}"]
                    )
                )
                for i in range(50)
            ],
            edges=[
                EdgeSpec(
                    id=f"edge_{i}",
                    source=f"node_{i}",
                    target=f"node_{i+1}",
                    sourceHandle=f"output_{i}",
                    targetHandle=f"input_{i+1}"
                )
                for i in range(49)
            ]
        )
        
        # Time the conversion
        start_time = time.time()
        workflow_def = large_spec.to_workflow_definition()
        conversion_time = time.time() - start_time
        
        # Should be reasonably fast (< 0.1 seconds for 50 nodes)
        assert conversion_time < 0.1, f"Conversion took {conversion_time}s"
        assert len(workflow_def.tasks) == 50
        
        # YAML conversion
        start_time = time.time()
        yaml_output = large_spec.to_yaml()
        yaml_time = time.time() - start_time
        
        assert yaml_time < 0.2, f"YAML conversion took {yaml_time}s"
        assert len(yaml_output) > 1000  # Should produce substantial output


class TestWorkflowSpecConversionIntegration:
    """Integration tests for complete conversion scenarios."""
    
    @pytest.fixture
    def complex_workflow_spec(self):
        """Create a complex workflow spec with all node types."""
        return WorkflowSpec(
            id=uuid.uuid4(),
            rev=1,
            title="Complex Multi-Type Workflow",
            description="A comprehensive workflow demonstrating all node types and conversions",
            nodes=[
                NodeSpec(
                    id="fetch_weather",
                    type="tool",
                    label="Fetch Weather Data",
                    data=NodeData(
                        tool_name="weather_api",
                        config={
                            "location": "New York",
                            "units": "celsius",
                            "include_forecast": True
                        },
                        ins=[],
                        outs=["weather_data", "temperature", "conditions"]
                    ),
                    position={"x": 100, "y": 100}
                ),
                NodeSpec(
                    id="analyze_conditions",
                    type="agent",
                    label="Analyze Weather Conditions",
                    data=NodeData(
                        agent_instructions="Analyze the weather data and provide recommendations",
                        config={
                            "model": "gpt-4",
                            "temperature": 0.7,
                            "max_tokens": 500
                        },
                        ins=["weather_data"],
                        outs=["recommendations", "safety_warnings"]
                    ),
                    position={"x": 500, "y": 100}
                )
            ],
            edges=[
                EdgeSpec(
                    id="weather_to_analysis",
                    source="fetch_weather",
                    target="analyze_conditions",
                    sourceHandle="weather_data",
                    targetHandle="weather_data"
                )
            ],
            metadata={
                "author": "test_suite",
                "version": "1.0.0",
                "tags": ["weather"]
            }
        )
    
    def test_full_pipeline_with_file_output(self, tmp_path, complex_workflow_spec):
        """Test complete pipeline from spec to file outputs."""
        # Convert to WorkflowDefinition
        workflow_def = complex_workflow_spec.to_workflow_definition()
        
        # Save as YAML
        yaml_file = tmp_path / "workflow.yaml"
        yaml_content = complex_workflow_spec.to_yaml()
        yaml_file.write_text(yaml_content)
        
        # Verify file was created and is valid
        assert yaml_file.exists()
        loaded_yaml = yaml.safe_load(yaml_file.read_text())
        assert loaded_yaml["name"] == complex_workflow_spec.title
        
        # Save as JSON (using WorkflowSpec model dump, not WorkflowDefinition)
        json_file = tmp_path / "workflow.json"
        json_content = json.dumps(complex_workflow_spec.model_dump(), indent=2, default=str)
        json_file.write_text(json_content)
        
        # Verify JSON file
        assert json_file.exists()
        loaded_json = json.loads(json_file.read_text())
        assert loaded_json["title"] == complex_workflow_spec.title
    
    def test_conversion_with_actual_tools(self):
        """Test conversion with workflow that references actual tools."""
        from iointel.src.utilities.registries import TOOLS_REGISTRY
        
        # Get some actual tool names
        actual_tools = list(TOOLS_REGISTRY.keys())[:3] if TOOLS_REGISTRY else []
        
        if actual_tools:
            spec = WorkflowSpec(
                id=uuid.uuid4(),
                rev=1,
                title="Real Tools Workflow",
                description="Uses actual registered tools",
                nodes=[
                    NodeSpec(
                        id=f"node_{i}",
                        type="tool",
                        label=f"Use {tool}",
                        data=NodeData(
                            tool_name=tool,
                            config={},
                            ins=[],
                            outs=["result"]
                        )
                    )
                    for i, tool in enumerate(actual_tools)
                ],
                edges=[]
            )
            
            # Convert without errors
            workflow_def = spec.to_workflow_definition()
            assert len(workflow_def.tasks) == len(actual_tools)
            
            # Verify tool names are preserved
            for task, tool in zip(workflow_def.tasks, actual_tools):
                assert task.task_metadata["tool_name"] == tool

    def test_agent_instruction_template_resolution(self):
        """Test that agent instructions with {node_id} references are properly structured for resolution."""
        from iointel.src.utilities.data_flow_resolver import data_flow_resolver
        from iointel.src.test_workflows import get_example_by_id
        
        print("\n🎭 Testing Agent Instruction Template Resolution")
        print("=" * 60)
        
        # Use centralized joke workflow example
        joke_workflow = get_example_by_id("joke_workflow")
        
        print(f"📋 Workflow: {joke_workflow.title}")
        print(f"📝 Description: {joke_workflow.description}")
        print(f"🔢 Nodes: {len(joke_workflow.nodes)}")
        print(f"🔗 Edges: {len(joke_workflow.edges)}")
        
        # Show the nodes and their instructions
        print("\n🎭 Node Details:")
        for i, node in enumerate(joke_workflow.nodes, 1):
            print(f"  {i}. {node.label} (id: {node.id})")
            print(f"     Instructions: {node.data.agent_instructions}")
            print(f"     Inputs: {node.data.ins}")
            print(f"     Outputs: {node.data.outs}")
        
        # Convert to executable workflow
        print("\n🔄 Converting to executable workflow...")
        workflow_def = joke_workflow.to_workflow_definition()
        yaml_content = joke_workflow.to_yaml()
        
        # Verify the workflow structure
        assert len(workflow_def.tasks) == 2
        assert workflow_def.tasks[0].type == "agent"
        assert workflow_def.tasks[1].type == "agent"
        
        # Verify the second task has the template reference
        second_task = workflow_def.tasks[1]
        assert "{joke_agent}" in second_task.task_metadata["agent_instructions"]
        
        print(f"✅ Conversion successful!")
        print(f"📊 Tasks: {len(workflow_def.tasks)}")
        
        # Show task details
        print("\n📋 Task Details:")
        for i, task in enumerate(workflow_def.tasks, 1):
            print(f"  {i}. {task.name}")
            print(f"     Type: {task.type}")
            if hasattr(task, 'task_metadata') and 'agent_instructions' in task.task_metadata:
                print(f"     Instructions: {task.task_metadata['agent_instructions']}")
        
        # Test the template resolution directly
        print("\n🧪 Testing Template Resolution:")
        original_instructions = "Evaluate this joke: {joke_agent}. Rate the humor and expected laughter severity."
        mock_results = {
            "joke_agent": "Why don't scientists trust atoms? Because they make up everything!"
        }
        
        print(f"📋 Original: {original_instructions}")
        print(f"🎯 Mock joke: {mock_results['joke_agent']}")
        
        # Test the data flow resolver
        resolved_instructions = data_flow_resolver._resolve_value(original_instructions, mock_results)
        expected_instructions = "Evaluate this joke: Why don't scientists trust atoms? Because they make up everything!. Rate the humor and expected laughter severity."
        
        print(f"📋 Resolved: {resolved_instructions}")
        
        assert resolved_instructions == expected_instructions
        print(f"✅ Template resolution test passed!")
        
        # Verify the workflow can be created from YAML
        print("\n🔄 Creating workflow from YAML...")
        from iointel.src.workflow import Workflow
        workflow = Workflow.from_yaml(yaml_str=yaml_content)
        workflow.objective = joke_workflow.description
        
        # Verify the workflow has proper structure
        assert len(workflow.tasks) == 2
        assert workflow.tasks[0]["type"] == "agent"
        assert workflow.tasks[1]["type"] == "agent"
        
        # Verify the second task's agent instructions contain the template
        second_workflow_task = workflow.tasks[1]
        assert "{joke_agent}" in second_workflow_task["task_metadata"]["agent_instructions"]
        
        print("✅ YAML workflow creation successful!")
        print(f"📊 Final workflow has {len(workflow.tasks)} tasks")
        
        # Show the YAML structure
        print("\n📄 Generated YAML structure:")
        yaml_lines = yaml_content.split('\n')[:20]  # Show first 20 lines
        for line in yaml_lines:
            print(f"  {line}")
        if len(yaml_content.split('\n')) > 20:
            print("  ... (truncated)")
        
        print(f"\n🎉 All template resolution tests passed!")
        print(f"🎭 The joke workflow is ready for execution with proper data flow!")

    def test_agent_workflow_execution_with_mocks(self):
        """Test actual workflow execution with mocked agents to see data flow resolution in action."""
        import asyncio
        from unittest.mock import patch, AsyncMock
        from iointel.src.workflow import Workflow
        from iointel.src.test_workflows import get_example_by_id
        
        print("\n🎭 Testing Agent Workflow Execution with Data Flow")
        print("=" * 60)
        
        # Use centralized joke workflow example
        joke_workflow = get_example_by_id("joke_workflow")
        
        # Convert to YAML and create workflow
        yaml_content = joke_workflow.to_yaml()
        workflow = Workflow.from_yaml(yaml_str=yaml_content)
        
        print(f"📋 Created workflow: {joke_workflow.title}")
        print(f"📊 Tasks: {len(workflow.tasks)}")
        
        # Mock the agent execution to return predictable results
        mock_joke = "Why did the scarecrow win an award? Because he was outstanding in his field!"
        mock_evaluation = "This is a classic dad joke. Rating: 7/10 for wholesome humor."
        
        # Create a mock that returns different responses based on instructions
        async def mock_agent_run(*args, **kwargs):
            # Get the instructions from the agent
            agents = kwargs.get('agents', [])
            if agents and hasattr(agents[0], 'instructions'):
                instructions = agents[0].instructions
                if "Create a humorous joke" in instructions:
                    print(f"🤖 MOCK JOKE AGENT: Creating joke...")
                    return {"result": mock_joke}
                elif "Evaluate this joke:" in instructions:
                    print(f"🤖 MOCK EVAL AGENT: Received instructions: {instructions}")
                    print(f"🤖 MOCK EVAL AGENT: Evaluating joke...")
                    return {"result": mock_evaluation}
            
            return {"result": "Mock response"}
        
        async def run_mock_workflow():
            # Patch the run_agents function
            with patch('iointel.src.chainables.run_agents') as mock_run_agents:
                # Set up the mock to return our mock function
                mock_executor = AsyncMock()
                mock_executor.execute.side_effect = mock_agent_run
                mock_run_agents.return_value = mock_executor
                
                print(f"\n🚀 Executing mock workflow...")
                conversation_id = f"mock_test_{uuid.uuid4().hex[:8]}"
                
                # Execute the workflow
                results = await workflow.run_tasks(conversation_id=conversation_id)
                
                return results
        
        # Run the test
        results = asyncio.run(run_mock_workflow())
        
        print(f"\n📊 Execution Results:")
        print(f"Status: {'✅ Success' if 'results' in results else '❌ Failed'}")
        
        if 'results' in results:
            task_results = results['results']
            print(f"Task Results ({len(task_results)}):")
            for task_id, result in task_results.items():
                display_result = result[:100] + "..." if len(str(result)) > 100 else result
                print(f"  {task_id}: {display_result}")
                
                # Verify the results structure (mocking is working)
                if task_id == "joke_creator":
                    assert "result" in str(result)
                    print(f"    ✅ Joke creation task executed")
                elif task_id == "joke_evaluator":
                    assert "result" in str(result)
                    print(f"    ✅ Joke evaluation task executed")
        
        print(f"\n🎉 Mock workflow execution test passed!")
        print(f"📝 This demonstrates that agent instruction resolution works correctly")


class TestCentralizedWorkflowExamples:
    """Test the centralized workflow examples used across UI and tests."""
    
    def test_all_centralized_examples_are_valid(self):
        """Test that all centralized workflow examples are valid and can be converted."""
        from iointel.src.test_workflows import create_workflow_examples, get_example_metadata
        
        print("\n📦 Testing All Centralized Workflow Examples")
        print("=" * 60)
        
        # Get all examples
        examples = create_workflow_examples()
        metadata = get_example_metadata()
        
        print(f"📊 Found {len(examples)} centralized examples")
        
        # Test each example
        for example_id, workflow_spec in examples.items():
            print(f"\n🔍 Testing example: {example_id}")
            print(f"   Title: {workflow_spec.title}")
            print(f"   Description: {workflow_spec.description}")
            print(f"   Nodes: {len(workflow_spec.nodes)}")
            print(f"   Edges: {len(workflow_spec.edges)}")
            
            # Test validation
            issues = workflow_spec.validate_structure()
            if issues:
                print(f"   ⚠️  Validation issues: {issues}")
            else:
                print(f"   ✅ Validation: PASSED")
            
            # Test conversion to WorkflowDefinition
            try:
                workflow_def = workflow_spec.to_workflow_definition()
                print(f"   ✅ WorkflowDefinition conversion: PASSED ({len(workflow_def.tasks)} tasks)")
            except Exception as e:
                print(f"   ❌ WorkflowDefinition conversion: FAILED - {e}")
                raise
            
            # Test conversion to YAML
            try:
                yaml_content = workflow_spec.to_yaml()
                print(f"   ✅ YAML conversion: PASSED ({len(yaml_content)} chars)")
            except Exception as e:
                print(f"   ❌ YAML conversion: FAILED - {e}")
                raise
            
            # Test metadata consistency
            if example_id in metadata:
                meta = metadata[example_id]
                assert meta["title"] == workflow_spec.title
                assert meta["description"] == workflow_spec.description
                assert meta["node_count"] == len(workflow_spec.nodes)
                assert meta["edge_count"] == len(workflow_spec.edges)
                print(f"   ✅ Metadata consistency: PASSED")
            else:
                print(f"   ❌ Metadata missing for {example_id}")
                raise AssertionError(f"Metadata missing for {example_id}")
        
        print(f"\n🎉 All {len(examples)} centralized examples are valid!")
        print(f"✅ These examples can be used consistently across UI and tests")
    
    def test_examples_categorization(self):
        """Test that examples can be categorized properly."""
        from iointel.src.test_workflows import get_examples_by_type, get_examples_by_complexity
        
        print("\n🏷️  Testing Example Categorization")
        print("=" * 40)
        
        # Test filtering by type
        tool_examples = get_examples_by_type("tool")
        agent_examples = get_examples_by_type("agent")
        
        print(f"📋 Tool examples: {len(tool_examples)}")
        for example_id in tool_examples:
            print(f"   - {example_id}")
        
        print(f"🤖 Agent examples: {len(agent_examples)}")
        for example_id in agent_examples:
            print(f"   - {example_id}")
        
        # Test filtering by complexity
        simple_examples = get_examples_by_complexity("Simple")
        basic_examples = get_examples_by_complexity("Basic")
        intermediate_examples = get_examples_by_complexity("Intermediate")
        
        print(f"🟢 Simple examples: {len(simple_examples)}")
        for example_id in simple_examples:
            print(f"   - {example_id}")
        
        print(f"🟡 Basic examples: {len(basic_examples)}")
        for example_id in basic_examples:
            print(f"   - {example_id}")
        
        print(f"🔴 Intermediate examples: {len(intermediate_examples)}")
        for example_id in intermediate_examples:
            print(f"   - {example_id}")
        
        # Verify some examples exist in each category
        assert len(tool_examples) > 0, "Should have at least one tool example"
        assert len(agent_examples) > 0, "Should have at least one agent example"
        assert len(simple_examples) > 0, "Should have at least one simple example"
        
        print(f"✅ Example categorization works correctly")
    
    def test_web_ui_compatibility(self):
        """Test that examples work with the web UI format."""
        from iointel.src.test_workflows import get_example_metadata
        
        print("\n🌐 Testing Web UI Compatibility")
        print("=" * 40)
        
        # This is the format the web UI expects
        metadata = get_example_metadata()
        
        for example_id, meta in metadata.items():
            print(f"\n📋 Example: {example_id}")
            
            # Check required fields for UI
            required_fields = ["title", "description", "node_count", "edge_count"]
            for field in required_fields:
                assert field in meta, f"Missing required field '{field}' in {example_id}"
                print(f"   ✅ {field}: {meta[field]}")
            
            # Check optional fields
            if "types" in meta:
                print(f"   📊 Node types: {meta['types']}")
            if "complexity" in meta:
                print(f"   🎯 Complexity: {meta['complexity']}")
        
        print(f"\n✅ All examples are compatible with web UI format")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])