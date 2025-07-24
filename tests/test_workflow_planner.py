"""
Tests for WorkflowPlanner using actual agent system.
Uses real OpenAI API with credentials from creds.env
"""

import pytest
import json
import uuid
import os
from pathlib import Path
from dotenv import load_dotenv

from iointel.src.agent_methods.agents.workflow_planner import WorkflowPlanner, WORKFLOW_PLANNER_INSTRUCTIONS
from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec, 
    NodeSpec, 
    NodeData, 
    EdgeSpec, 
    EdgeData
)

# Load environment variables
env_path = Path(__file__).parent.parent / "creds.env"
load_dotenv(env_path)


# MIGRATED: Using centralized fixtures instead of local ones
# The tool_catalog fixture is now provided by tests/fixtures/workflow_test_fixtures.py
# This provides backward compatibility for existing tests

# Keep this for any tests that still reference it directly  
@pytest.fixture
def legacy_tool_catalog():
    """Legacy tool catalog - migrated tests should use mock_tool_catalog or real_tool_catalog."""
    return {
        "weather_api": {
            "name": "weather_api",
            "description": "Get weather information for a location", 
            "parameters": {"location": "string", "units": "string"},
            "returns": ["weather_data", "status"]
        },
        "send_email": {
            "name": "send_email",
            "description": "Send an email message",
            "parameters": {"to": "string", "subject": "string", "body": "string"},
            "returns": ["sent", "delivery_id"]
        },
        "data_processor": {
            "name": "data_processor",
            "description": "Process and transform data",
            "parameters": {"data": "any", "operation": "string"},
            "returns": ["processed_data", "metadata"]
        }
    }


@pytest.fixture
def real_tool_catalog():
    """Real tool catalog loaded from environment for integration tests."""
    from iointel.src.utilities.tool_registry_utils import create_tool_catalog
    from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env
    
    # Load tools from environment first (this registers them)
    load_tools_from_env('creds.env')
    
    # Create and return the real tool catalog
    return create_tool_catalog()


@pytest.fixture
def sample_workflow_spec():
    """Sample WorkflowSpec for testing."""
    return WorkflowSpec(
        id=uuid.uuid4(),
        rev=1,
        title="Weather Email Workflow",
        description="Get weather and send email",
        nodes=[
            NodeSpec(
                id="get_weather",
                type="tool",
                label="Get Weather",
                data=NodeData(
                    tool_name="weather_api",
                    config={"location": "London", "units": "celsius"},
                    ins=[],
                    outs=["weather_data"]
                )
            ),
            NodeSpec(
                id="send_notification",
                type="tool", 
                label="Send Email",
                data=NodeData(
                    tool_name="send_email",
                    config={"to": "user@example.com", "subject": "Weather Update"},
                    ins=["weather_data"],
                    outs=[]
                )
            )
        ],
        edges=[
            EdgeSpec(
                id="weather_to_email",
                source="get_weather",
                target="send_notification",
                sourceHandle="weather_data",
                targetHandle="weather_data"
            )
        ]
    )


class TestWorkflowPlanner:
    """Test cases for WorkflowPlanner class."""

    def test_workflow_planner_init(self):
        """Test WorkflowPlanner initialization."""
        planner = WorkflowPlanner(
            conversation_id="test_conv",
            debug=True
        )
        
        assert planner.conversation_id == "test_conv"
        assert planner.agent is not None
        assert planner.agent.name == "WorkflowPlanner"

    def test_workflow_planner_init_with_defaults(self):
        """Test WorkflowPlanner with default parameters."""
        planner = WorkflowPlanner()
        
        assert planner.conversation_id is not None
        assert planner.agent is not None

    def test_workflow_planner_instructions(self):
        """Test that workflow planner instructions are comprehensive."""
        assert WORKFLOW_PLANNER_INSTRUCTIONS is not None
        assert "WorkflowPlanner-GPT" in WORKFLOW_PLANNER_INSTRUCTIONS
        
        # Check for critical node type distinctions
        assert "data_source" in WORKFLOW_PLANNER_INSTRUCTIONS
        assert "agent" in WORKFLOW_PLANNER_INSTRUCTIONS
        assert "workflow_call" in WORKFLOW_PLANNER_INSTRUCTIONS
        
        # Check for key concepts
        assert "edge" in WORKFLOW_PLANNER_INSTRUCTIONS.lower()
        assert "node" in WORKFLOW_PLANNER_INSTRUCTIONS.lower()
        assert "user_input" in WORKFLOW_PLANNER_INSTRUCTIONS.lower()
        assert "condition" in WORKFLOW_PLANNER_INSTRUCTIONS.lower()

    @pytest.mark.asyncio  
    async def test_generate_workflow_success(self, mock_tool_catalog):
        """MIGRATED: Test successful workflow generation using centralized fixtures."""
        # Skip if no API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No OPENAI_API_KEY available")
            
        print("\n=== WORKFLOW GENERATION TEST (MIGRATED) ===")
        print(f"Tool Catalog: {json.dumps(mock_tool_catalog, indent=2)}")
        
        planner = WorkflowPlanner(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",  # Use cheaper model for tests
            debug=True
        )
        
        query = "Get weather for London and email it to the team"
        print(f"\nQuery: {query}")
        print("\nGenerating workflow...")
        
        result = await planner.generate_workflow(
            query=query,
            tool_catalog=mock_tool_catalog
        )
        
        print("\n=== GENERATED WORKFLOW ===")
        print(f"Reasoning: {result.reasoning}")
        print(f"Title: {result.title}")
        print(f"Description: {result.description}")
        print(f"ID: {result.id}")
        print(f"Revision: {result.rev}")
        
        print(f"\nNodes ({len(result.nodes)}):")
        for node in result.nodes:
            print(f"  - {node.id} ({node.type}): {node.label}")
            if node.type == "data_source" and hasattr(node.data, 'source_name') and node.data.source_name:
                print(f"    Source: {node.data.source_name}")
            elif node.type == "agent" and hasattr(node.data, 'tools') and node.data.tools:
                print(f"    Tools: {node.data.tools}")
            if node.data.config:
                print(f"    Config: {json.dumps(node.data.config, indent=6)}")
            print(f"    Inputs: {node.data.ins}")
            print(f"    Outputs: {node.data.outs}")
        
        print(f"\nEdges ({len(result.edges)}):")
        for edge in result.edges:
            condition = f" [if {edge.data.condition}]" if edge.data and edge.data.condition else ""
            print(f"  - {edge.source} -> {edge.target}{condition}")
            print(f"    Handles: {edge.sourceHandle} -> {edge.targetHandle}")
        
        assert isinstance(result, WorkflowSpec)
        assert result.title is not None
        assert len(result.nodes) >= 2  # Should have at least weather and email nodes
        assert len(result.edges) >= 1  # Should connect the nodes
        
        # Validate structure
        issues = result.validate_structure()
        print(f"\nValidation: {'✅ PASSED' if not issues else '❌ FAILED'}")
        assert len(issues) == 0, f"Generated workflow has issues: {issues}"

    @pytest.mark.asyncio
    async def test_generate_workflow_complex_query(self, tool_catalog):
        """Test workflow generation with complex requirements."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No OPENAI_API_KEY available")
            
        planner = WorkflowPlanner(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            debug=False
        )
        
        complex_query = """
        Create a workflow that:
        1. Gets weather data for multiple cities
        2. Processes the data to find the warmest city
        3. Sends an email with the results
        Include error handling for API failures.
        """
        
        result = await planner.generate_workflow(
            query=complex_query,
            tool_catalog=tool_catalog
        )
        
        assert isinstance(result, WorkflowSpec)
        # Should have multiple nodes for complex workflow
        assert len(result.nodes) >= 2
        
        # Check for data processing logic
        has_processing = any(
            node.type == "agent" or 
            (node.data.tool_name == "data_processor" if node.data.tool_name else False)
            for node in result.nodes
        )
        assert has_processing, "Complex workflow should include data processing"

    @pytest.mark.asyncio
    async def test_refine_workflow_success(self, sample_workflow_spec):
        """Test successful workflow refinement."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No OPENAI_API_KEY available")
            
        planner = WorkflowPlanner(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            debug=True
        )
        
        result = await planner.refine_workflow(
            workflow_spec=sample_workflow_spec,
            feedback="Add error handling and retry logic"
        )
        
        assert isinstance(result, WorkflowSpec)
        assert result.id == sample_workflow_spec.id  # Same workflow ID
        assert result.rev > sample_workflow_spec.rev  # Version incremented
        
        # Should still be valid
        issues = result.validate_structure()
        assert len(issues) == 0, f"Refined workflow has issues: {issues}"

    def test_create_example_workflow(self):
        """Test example workflow creation."""
        planner = WorkflowPlanner()
        
        workflow = planner.create_example_workflow("Test Example")
        
        assert isinstance(workflow, WorkflowSpec)
        assert workflow.title == "Test Example"
        assert len(workflow.nodes) >= 2  # Should have at least a few nodes
        assert len(workflow.edges) >= 1  # Should have connections
        
        # Test structure validation
        issues = workflow.validate_structure()
        assert len(issues) == 0, f"Example workflow has issues: {issues}"

    def test_create_example_workflow_variations(self):
        """Test creating different example workflows."""
        planner = WorkflowPlanner()
        
        titles = [
            "Data Processing Pipeline",
            "Email Notification Workflow",
            "Complex Multi-Step Process"
        ]
        
        for title in titles:
            workflow = planner.create_example_workflow(title)
            assert workflow.title == title
            assert isinstance(workflow.id, uuid.UUID)
            assert workflow.rev == 1
            
            # Validate each one
            issues = workflow.validate_structure()
            assert len(issues) == 0, f"Workflow '{title}' has issues: {issues}"


class TestWorkflowPlannerIntegration:
    """Integration tests for WorkflowPlanner."""

    @pytest.mark.asyncio
    async def test_end_to_end_workflow_generation(self, tool_catalog):
        """Test complete workflow generation pipeline."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No OPENAI_API_KEY available")
            
        planner = WorkflowPlanner(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            debug=False
        )
        
        # Add more tools for complex workflow
        extended_catalog = tool_catalog.copy()
        extended_catalog.update({
            "api_fetcher": {
                "name": "api_fetcher",
                "description": "Fetch data from any API",
                "parameters": {"url": "string", "headers": "dict"},
                "returns": ["data", "status_code"]
            },
            "data_validator": {
                "name": "data_validator",
                "description": "Validate data against rules",
                "parameters": {"data": "any", "rules": "dict"},
                "returns": ["valid", "errors"]
            },
            "logger": {
                "name": "logger",
                "description": "Log messages and data",
                "parameters": {"message": "string", "level": "string"},
                "returns": ["logged"]
            }
        })
        
        # Test generation
        query = "Create a data processing pipeline that fetches data, validates it, processes it, and sends results via email"
        print("\n=== END-TO-END WORKFLOW TEST ===")
        print(f"Query: {query}")
        print(f"\nExtended Tool Catalog includes: {list(extended_catalog.keys())}")
        
        result = await planner.generate_workflow(
            query=query,
            tool_catalog=extended_catalog
        )
        
        print("\n=== GENERATED PIPELINE ===")
        print
        print(f"Title: {result.title}")
        print(f"Description: {result.description}")
        
        print(f"\nNodes ({len(result.nodes)}):")
        for i, node in enumerate(result.nodes, 1):
            print(f"\n{i}. {node.id} ({node.type}): {node.label}")
            if node.data.tool_name:
                print(f"   Tool: {node.data.tool_name}")
            if node.data.agent_instructions:
                print(f"   Instructions: {node.data.agent_instructions[:100]}...")
            if node.data.config:
                print(f"   Config: {json.dumps(node.data.config, indent=3)}")
            print(f"   Inputs: {node.data.ins}")
            print(f"   Outputs: {node.data.outs}")
        
        print("\nWorkflow Flow:")
        for edge in result.edges:
            condition = f" [if {edge.data.condition}]" if edge.data and edge.data.condition else ""
            print(f"  {edge.source} -> {edge.target}{condition}")
        
        # Validate result
        assert isinstance(result, WorkflowSpec)
        assert len(result.nodes) >= 3  # Should have multiple steps
        assert len(result.edges) >= 2  # Should have connections
        
        # Test workflow structure
        issues = result.validate_structure()
        print(f"\nValidation: {'✅ PASSED' if not issues else '❌ FAILED'}")
        assert len(issues) == 0, f"Generated workflow has issues: {issues}"
        
        # Test conversion to YAML
        yaml_output = result.to_yaml()
        print("\nYAML Output Preview (first 300 chars):")
        print(yaml_output[:300] + "..." if len(yaml_output) > 300 else yaml_output)
        assert result.title.lower() in yaml_output.lower()
        assert "tasks:" in yaml_output

    @pytest.mark.asyncio
    async def test_workflow_memory_integration(self, tool_catalog):
        """Test that WorkflowPlanner properly uses memory."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No OPENAI_API_KEY available")
            
        # Skip memory test if no database URL
        pytest.skip("Memory integration requires database setup")
        
        conversation_id = str(uuid.uuid4())
        planner = WorkflowPlanner(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            memory=memory,
            conversation_id=conversation_id
        )
        
        # Generate workflow
        await planner.generate_workflow(
            query="Simple test workflow",
            tool_catalog=tool_catalog
        )
        
        # Check memory was used
        history = await memory.get_message_history(conversation_id)
        assert len(history) > 0, "Conversation should be stored in memory"

    @pytest.mark.asyncio
    async def test_workflow_with_conditional_logic(self, tool_catalog):
        """Test generating workflows with conditional branching."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No OPENAI_API_KEY available")
            
        planner = WorkflowPlanner(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini"
        )
        
        result = await planner.generate_workflow(
            query="Create a workflow that checks weather and only sends email if it's raining",
            tool_catalog=tool_catalog
        )
        
        assert isinstance(result, WorkflowSpec)
        
        # Should have conditional logic
        any(
            edge.data and edge.data.condition 
            for edge in result.edges
        )
        
        # Print workflow for debugging
        print(f"\nConditional Workflow: {result.title}")
        for edge in result.edges:
            if edge.data and edge.data.condition:
                print(f"  Edge {edge.source} -> {edge.target}: {edge.data.condition}")

    @pytest.mark.asyncio
    async def test_agent_decision_tools_workflow(self):
        """Test agent generating workflow that properly uses decision tools for complex conditionals."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No OPENAI_API_KEY available")
            
        print("\n=== Testing Agent-Generated Decision Tools Workflow ===")
        
        # Create tool catalog including decision tools
        decision_tool_catalog = {
            "weather_api": {
                "name": "weather_api",
                "description": "Get weather information for a location",
                "parameters": {"location": "string", "units": "string"},
                "returns": ["weather_data", "status"]
            },
            "json_evaluator": {
                "name": "json_evaluator",
                "description": "Evaluate JSON data against expressions for decision making",
                "parameters": {"data": "dict", "expression": "string"},
                "returns": ["result", "details", "confidence"]
            },
            "number_compare": {
                "name": "number_compare", 
                "description": "Compare numbers using operators (>, <, ==, etc.)",
                "parameters": {"value": "number", "operator": "string", "threshold": "number"},
                "returns": ["result", "details", "confidence"]
            },
            "string_contains": {
                "name": "string_contains",
                "description": "Check if string contains substring or regex pattern",
                "parameters": {"text": "string", "substring": "string", "case_sensitive": "boolean"},
                "returns": ["result", "details", "confidence"]
            },
            "conditional_router": {
                "name": "conditional_router",
                "description": "Route to different paths based on structured decisions",
                "parameters": {"decision": "dict", "routes": "dict", "decision_path": "string"},
                "returns": ["routed_to", "route_data", "matched_condition"]
            },
            "send_alert": {
                "name": "send_alert",
                "description": "Send emergency weather alert",
                "parameters": {"message": "string", "severity": "string", "recipients": "list"},
                "returns": ["sent", "alert_id"]
            },
            "send_notification": {
                "name": "send_notification", 
                "description": "Send normal weather notification",
                "parameters": {"message": "string", "recipients": "list"},
                "returns": ["sent", "notification_id"]
            }
        }
        
        print(f"Decision tool catalog: {list(decision_tool_catalog.keys())}")
        
        # Create workflow planner
        planner = WorkflowPlanner(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            debug=True
        )
        
        # Request a complex conditional workflow
        conditional_query = """
        Create a smart weather alert system that:
        1. Gets weather data for New York
        2. Checks if temperature is below 0°C (freezing)
        3. Checks if the weather condition contains 'storm' or 'severe'
        4. If either condition is true, send an emergency alert
        5. Otherwise, send a normal notification
        
        IMPORTANT: Use the decision tools (json_evaluator, number_compare, string_contains, conditional_router) 
        to implement the conditional logic. Do NOT use string conditions in edges - use explicit decision nodes instead.
        """
        
        print(f"\nQuery: {conditional_query}")
        print("\nGenerating workflow with decision tools...")
        
        # Generate workflow
        workflow = await planner.generate_workflow(
            query=conditional_query,
            tool_catalog=decision_tool_catalog
        )
        
        print("\n=== GENERATED DECISION TOOLS WORKFLOW ===")
        print(f"Title: {workflow.title}")
        print(f"Description: {workflow.description}")
        
        print(f"\nNodes ({len(workflow.nodes)}):")
        decision_nodes = []
        for i, node in enumerate(workflow.nodes, 1):
            print(f"\n{i}. {node.id} ({node.type}): {node.label}")
            if node.data.tool_name:
                print(f"   Tool: {node.data.tool_name}")
            if node.data.config:
                print(f"   Config: {node.data.config}")
            if node.data.agent_instructions:
                print(f"   Instructions: {node.data.agent_instructions[:100]}...")
            print(f"   Inputs: {node.data.ins}")
            print(f"   Outputs: {node.data.outs}")
            
            # Detect decision nodes
            if (node.type == "decision" or 
                (node.data.tool_name and node.data.tool_name in ["json_evaluator", "number_compare", "string_contains", "conditional_router"])):
                decision_nodes.append(node)
                print("   ✓ DECISION NODE DETECTED")
        
        print(f"\nEdges ({len(workflow.edges)}):")
        for edge in workflow.edges:
            condition_str = f" [condition: {edge.data.condition}]" if edge.data and edge.data.condition else ""
            print(f"  {edge.source} -> {edge.target} ({edge.sourceHandle} -> {edge.targetHandle}){condition_str}")
        
        # Comprehensive validation
        print("\n=== DECISION TOOLS VALIDATION ===")
        
        # 1. Check for decision node usage
        assert len(decision_nodes) > 0, f"Workflow should include decision nodes, found: {len(decision_nodes)}"
        print(f"✓ Found {len(decision_nodes)} decision nodes")
        
        # 2. Check for specific decision tool usage
        decision_tool_names = ["json_evaluator", "number_compare", "string_contains", "conditional_router"]
        used_decision_tools = [node.data.tool_name for node in workflow.nodes if node.data.tool_name in decision_tool_names]
        assert len(used_decision_tools) > 0, f"Workflow should use decision tools, found: {used_decision_tools}"
        print(f"✓ Uses decision tools: {used_decision_tools}")
        
        # 3. Validate workflow structure
        issues = workflow.validate_structure()
        assert len(issues) == 0, f"Generated workflow has structural issues: {issues}"
        print("✓ Workflow structure is valid")
        
        # 4. Check that the workflow includes weather API
        weather_nodes = [node for node in workflow.nodes if node.data.tool_name == "weather_api"]
        assert len(weather_nodes) > 0, "Workflow should include weather API call"
        print("✓ Includes weather API")
        
        # 5. Check for alert/notification routing
        node_tools = [node.data.tool_name for node in workflow.nodes if node.data.tool_name]
        has_alert_tools = any(tool in ["send_alert", "send_notification"] for tool in node_tools)
        assert has_alert_tools, "Workflow should include alert or notification tools"
        print("✓ Includes alert/notification routing")
        
        # 6. Check that decision logic is properly connected
        decision_node_ids = [node.id for node in decision_nodes]
        connected_decisions = []
        for edge in workflow.edges:
            if edge.source in decision_node_ids or edge.target in decision_node_ids:
                connected_decisions.append(edge)
        
        assert len(connected_decisions) > 0, "Decision nodes should be connected to workflow"
        print(f"✓ Decision nodes are connected ({len(connected_decisions)} connections)")
        
        print("\n✅ AGENT SUCCESSFULLY GENERATED CONDITIONAL WORKFLOW USING DECISION TOOLS")
        print(f"   - {len(workflow.nodes)} nodes including {len(decision_nodes)} decision nodes")
        print(f"   - {len(workflow.edges)} edges with {len(connected_decisions)} decision connections")
        print(f"   - Decision tools used: {', '.join(used_decision_tools)}")
        
        return workflow

    @pytest.mark.asyncio
    async def test_capri_travel_decision_workflow(self):
        """Test the specific Capri travel example to verify proper conditional logic."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No OPENAI_API_KEY available")
            
        print("\n=== Testing Capri Travel Decision Workflow ===")
        
        # Create tool catalog with travel-specific tools
        travel_tool_catalog = {
            "weather_api": {
                "name": "weather_api",
                "description": "Get weather information for a location",
                "parameters": {"location": "string", "units": "string"},
                "returns": ["weather_data", "temperature", "status"]
            },
            "number_compare": {
                "name": "number_compare", 
                "description": "Compare numbers using operators (>, <, ==, etc.)",
                "parameters": {"value": "number", "operator": "string", "threshold": "number"},
                "returns": ["result", "details", "confidence"]
            },
            "boolean_mux": {
                "name": "boolean_mux",
                "description": "Route based on boolean condition",
                "parameters": {"condition": "boolean", "true_value": "any", "false_value": "any"},
                "returns": ["routed_to", "route_data", "matched_condition"]
            },
            "send_email": {
                "name": "send_email",
                "description": "Send an email message",
                "parameters": {"to": "string", "subject": "string", "body": "string"},
                "returns": ["sent", "delivery_id"]
            },
            "book_tickets": {
                "name": "book_tickets",
                "description": "Book travel tickets to destination",
                "parameters": {"destination": "string", "passengers": "number", "date": "string"},
                "returns": ["booking_id", "confirmation"]
            }
        }
        
        print(f"Travel tool catalog: {list(travel_tool_catalog.keys())}")
        
        # Create workflow planner
        planner = WorkflowPlanner(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            debug=True
        )
        
        # Request the exact Capri travel workflow
        capri_query = """
        Check weather in Capri, and if it is less than 65 degrees, send alex@travel.com an email 
        that we should go to Ibiza instead. If it is above 65 degrees, book tickets to Capri.
        
        Use decision tools (number_compare, boolean_mux) for the conditional logic.
        Do NOT use string conditions in edges.
        """
        
        print(f"\nCapri Query: {capri_query}")
        print("\nGenerating Capri travel workflow...")
        
        # Generate workflow
        workflow = await planner.generate_workflow(
            query=capri_query,
            tool_catalog=travel_tool_catalog
        )
        
        print("\n=== GENERATED CAPRI TRAVEL WORKFLOW ===")
        print(f"Title: {workflow.title}")
        print(f"Description: {workflow.description}")
        
        print(f"\nNodes ({len(workflow.nodes)}):")
        weather_nodes = []
        decision_nodes = []
        email_nodes = []
        booking_nodes = []
        
        for i, node in enumerate(workflow.nodes, 1):
            print(f"\n{i}. {node.id} ({node.type}): {node.label}")
            if node.data.tool_name:
                print(f"   Tool: {node.data.tool_name}")
            if node.data.config:
                print(f"   Config: {node.data.config}")
            print(f"   Inputs: {node.data.ins}")
            print(f"   Outputs: {node.data.outs}")
            
            # Categorize nodes
            if node.data.tool_name == "weather_api":
                weather_nodes.append(node)
                print("   ✓ WEATHER NODE")
            elif node.type == "decision" or (node.data.tool_name and node.data.tool_name in ["number_compare", "boolean_mux"]):
                decision_nodes.append(node)
                print("   ✓ DECISION NODE")
            elif node.data.tool_name == "send_email":
                email_nodes.append(node)
                print("   ✓ EMAIL NODE")
            elif node.data.tool_name == "book_tickets":
                booking_nodes.append(node)
                print("   ✓ BOOKING NODE")
        
        print(f"\nEdges ({len(workflow.edges)}):")
        conditional_edges = []
        for edge in workflow.edges:
            condition_str = f" [condition: {edge.data.condition}]" if edge.data and edge.data.condition else ""
            print(f"  {edge.source} -> {edge.target} ({edge.sourceHandle} -> {edge.targetHandle}){condition_str}")
            
            if edge.data and edge.data.condition:
                conditional_edges.append(edge)
        
        # Comprehensive validation for Capri travel scenario
        print("\n=== CAPRI TRAVEL VALIDATION ===")
        
        # 1. Should have weather check
        assert len(weather_nodes) > 0, "Workflow should check weather"
        weather_node = weather_nodes[0]
        assert "capri" in str(weather_node.data.config).lower(), "Should check Capri weather"
        print(f"✓ Checks Capri weather: {weather_node.data.config}")
        
        # 2. Should have temperature decision logic
        assert len(decision_nodes) > 0, "Workflow should have decision nodes"
        temp_decision_nodes = [n for n in decision_nodes if n.data.tool_name == "number_compare"]
        assert len(temp_decision_nodes) > 0, "Should have temperature comparison"
        temp_node = temp_decision_nodes[0]
        assert temp_node.data.config.get("threshold") == 65, "Should compare against 65 degrees"
        print(f"✓ Temperature decision: {temp_node.data.config}")
        
        # 3. Should have routing logic
        routing_nodes = [n for n in decision_nodes if n.data.tool_name == "boolean_mux"]
        if len(routing_nodes) > 0:
            print(f"✓ Has routing logic: {routing_nodes[0].data.config}")
        
        # 4. Should have email to alex@travel.com
        assert len(email_nodes) > 0, "Workflow should send email"
        email_node = email_nodes[0]
        assert "alex@travel.com" in str(email_node.data.config), "Should email alex@travel.com"
        assert "ibiza" in str(email_node.data.config).lower(), "Email should mention Ibiza"
        print(f"✓ Email to Alex about Ibiza: {email_node.data.config.get('to')}")
        
        # 5. Should have Capri booking
        assert len(booking_nodes) > 0, "Workflow should book tickets"
        booking_node = booking_nodes[0]
        assert "capri" in str(booking_node.data.config).lower(), "Should book Capri tickets"
        print(f"✓ Books Capri tickets: {booking_node.data.config}")
        
        # 6. CRITICAL: Should NOT have string conditions in edges
        assert len(conditional_edges) == 0, f"Should have NO conditional edges, found: {[e.data.condition for e in conditional_edges]}"
        print("✓ NO string conditions in edges")
        
        # 7. Validate workflow structure
        issues = workflow.validate_structure()
        assert len(issues) == 0, f"Generated workflow has structural issues: {issues}"
        print("✓ Workflow structure is valid")
        
        print("\n✅ CAPRI TRAVEL WORKFLOW SUCCESSFULLY GENERATED WITH PROPER DECISION LOGIC")
        print(f"   - Weather check: {len(weather_nodes)} nodes")
        print(f"   - Decision logic: {len(decision_nodes)} nodes") 
        print(f"   - Email action: {len(email_nodes)} nodes")
        print(f"   - Booking action: {len(booking_nodes)} nodes")
        print(f"   - No conditional edges: {len(conditional_edges) == 0}")
        
        return workflow


class TestWorkflowPlannerEdgeCases:
    """Test edge cases and error scenarios."""

    def test_create_example_workflow_structure(self):
        """Test that example workflows have valid structure."""
        planner = WorkflowPlanner()
        
        for title in ["Test 1", "Complex Workflow", "Simple Process"]:
            workflow = planner.create_example_workflow(title)
            
            # Basic validation
            assert workflow.title == title
            assert len(workflow.nodes) > 0
            
            # Structure validation
            issues = workflow.validate_structure()
            assert len(issues) == 0, f"Example '{title}' has issues: {issues}"
            
            # Each node should have valid IDs
            node_ids = [node.id for node in workflow.nodes]
            assert len(node_ids) == len(set(node_ids)), "Duplicate node IDs found"

    def test_workflow_spec_edge_conditions(self):
        """Test workflows with decision nodes instead of edge conditions."""
        # Create a workflow with decision nodes for routing
        nodes = [
            NodeSpec(
                id="validator",
                type="tool",
                label="Validate Input",
                data=NodeData(
                    tool_name="validator",
                    config={"rules": "strict"},
                    ins=["input"],
                    outs=["valid_data", "validation_status"]
                )
            ),
            NodeSpec(
                id="check_validity",
                type="decision",
                label="Check if Valid",
                data=NodeData(
                    tool_name="json_evaluator",
                    config={"expression": "status == 'success'"},
                    ins=["validation_status"],
                    outs=["result", "details"]
                )
            ),
            NodeSpec(
                id="success_handler",
                type="agent",
                label="Process Valid Data",
                data=NodeData(
                    agent_instructions="Process the validated data",
                    ins=["valid_data"],
                    outs=["result"]
                )
            ),
            NodeSpec(
                id="error_handler",
                type="tool",
                label="Handle Error",
                data=NodeData(
                    tool_name="error_reporter",
                    config={"notify": "admin@example.com"},
                    ins=["validation_status"],
                    outs=["error_report"]
                )
            )
        ]
        
        edges = [
            EdgeSpec(
                id="validator_to_decision",
                source="validator",
                target="check_validity",
                sourceHandle="validation_status",
                targetHandle="validation_status"
            ),
            EdgeSpec(
                id="success_path",
                source="check_validity",
                target="success_handler",
                sourceHandle="result",
                targetHandle="valid_data",
                data=EdgeData(condition="result == true")
            ),
            EdgeSpec(
                id="error_path",
                source="validator",
                target="error_handler",
                sourceHandle="validation_status",
                targetHandle="validation_status",
                data=EdgeData(condition="validation_status == 'error'")
            )
        ]
        
        workflow = WorkflowSpec(
            id=uuid.uuid4(),
            rev=1,
            title="Conditional Workflow",
            description="Workflow with conditional branching",
            nodes=nodes,
            edges=edges
        )
        
        # Validate the workflow
        issues = workflow.validate_structure()
        assert len(issues) == 0, f"Conditional workflow has issues: {issues}"

    @pytest.mark.parametrize("node_count", [10, 20, 30])
    def test_workflow_performance_scaling(self, node_count):
        """Test workflow creation and validation with different sizes."""
        nodes = []
        edges = []
        
        # Create a linear workflow with node_count nodes
        for i in range(node_count):
            node = NodeSpec(
                id=f"node_{i}",
                type="tool" if i % 2 == 0 else "agent",
                label=f"Step {i}",
                data=NodeData(
                    tool_name=f"tool_{i}" if i % 2 == 0 else None,
                    agent_instructions=f"Process step {i}" if i % 2 == 1 else None,
                    config={"index": i},
                    ins=[f"input_{i}"] if i > 0 else [],
                    outs=[f"output_{i}"]
                )
            )
            nodes.append(node)
            
            # Connect to previous node
            if i > 0:
                edge = EdgeSpec(
                    id=f"edge_{i-1}_to_{i}",
                    source=f"node_{i-1}",
                    target=f"node_{i}",
                    sourceHandle=f"output_{i-1}",
                    targetHandle=f"input_{i}"
                )
                edges.append(edge)
        
        workflow = WorkflowSpec(
            id=uuid.uuid4(),
            rev=1,
            title=f"Performance Test - {node_count} nodes",
            description=f"Testing with {node_count} nodes",
            nodes=nodes,
            edges=edges
        )
        
        # Time the validation
        import time
        start_time = time.time()
        issues = workflow.validate_structure()
        validation_time = time.time() - start_time
        
        # Should validate successfully
        assert len(issues) == 0, f"Workflow with {node_count} nodes has issues: {issues}"
        
        # Validation should be reasonably fast (< 0.1s even for 30 nodes)
        assert validation_time < 0.1, f"Validation took {validation_time}s for {node_count} nodes"


@pytest.mark.asyncio
async def test_stock_agent_data_source_vs_agent_nodes(real_tool_catalog):
    """
    Test that 'stock agent' query correctly creates:
    - data_source node for user input (NOT for API tools)
    - agent node with stock analysis tools
    
    This test verifies the fix for the data_source/agent node confusion issue.
    """
    planner = WorkflowPlanner()
    tool_catalog = real_tool_catalog
    
    # Test with simple 'stock agent' query
    workflow = await planner.generate_workflow(
        query='stock agent',
        tool_catalog=tool_catalog
    )
    
    # Verify workflow was created
    assert workflow is not None
    assert workflow.title is not None
    assert len(workflow.nodes) >= 2  # Should have at least user_input and agent
    
    # Find the nodes
    data_source_nodes = [n for n in workflow.nodes if n.type == 'data_source']
    agent_nodes = [n for n in workflow.nodes if n.type == 'agent']
    
    # Verify we have the right node types
    assert len(data_source_nodes) >= 1, "Should have at least one data_source node for user input"
    assert len(agent_nodes) >= 1, "Should have at least one agent node for stock analysis"
    
    # Verify data_source is only used for user_input or prompt_tool
    for ds_node in data_source_nodes:
        assert ds_node.data.source_name in ['user_input', 'prompt_tool'], \
            f"data_source node should only be user_input or prompt_tool, got: {ds_node.data.source_name}"
    
    # Verify agent node has stock analysis tools
    stock_agent = None
    for agent_node in agent_nodes:
        if agent_node.data.tools and any('stock' in tool or 'price' in tool for tool in agent_node.data.tools):
            stock_agent = agent_node
            break
    
    assert stock_agent is not None, "Should have an agent with stock analysis tools"
    assert len(stock_agent.data.tools) >= 1, "Stock agent should have tools"
    
    # Verify no data_source nodes have API tools like get_current_stock_price
    for ds_node in data_source_nodes:
        if hasattr(ds_node.data, 'source_name'):
            assert 'get_current_stock_price' not in str(ds_node.data.source_name), \
                "data_source nodes should NOT have API tools like get_current_stock_price"
    
    print(f"✅ Test passed!")
    print(f"Generated workflow: {workflow.title}")
    print(f"Nodes created:")
    for node in workflow.nodes:
        print(f"  - {node.id}: {node.type} ({node.label})")
        if node.type == 'data_source':
            print(f"    source_name: {node.data.source_name}")
        elif node.type == 'agent' and node.data.tools:
            print(f"    tools: {node.data.tools}")


# ==========================================
# NEW LAYERED TESTS (using centralized system)  
# ==========================================

class TestWorkflowPlannerAgenticLayer:
    """MIGRATED: Agentic layer tests using centralized test repository."""
    
    @pytest.mark.asyncio
    async def test_stock_agent_generation_centralized(self, stock_analysis_prompts, real_tool_catalog):
        """
        MIGRATED: Test stock agent generation using centralized test cases.
        This replaces the old hardcoded test with repository-driven test data.
        """
        planner = WorkflowPlanner()
        
        for prompt in stock_analysis_prompts:
            if not prompt:
                continue
                
            print(f"\n=== Testing prompt: '{prompt}' ===")
            workflow = await planner.generate_workflow(
                query=prompt,
                tool_catalog=real_tool_catalog
            )
            
            # Verify the fix for data_source vs agent node confusion
            assert workflow is not None
            assert len(workflow.nodes) >= 2  # Should have user input + agent
            
            # Check node types
            data_source_nodes = [n for n in workflow.nodes if n.type == 'data_source']
            agent_nodes = [n for n in workflow.nodes if n.type == 'agent']
            
            assert len(data_source_nodes) >= 1, "Should have data_source node for user input"
            assert len(agent_nodes) >= 1, "Should have agent node for stock analysis"
            
            # Critical: Verify data_source nodes are ONLY for user_input/prompt_tool
            for node in data_source_nodes:
                assert hasattr(node.data, 'source_name'), f"data_source node {node.id} missing source_name"
                assert node.data.source_name in ['user_input', 'prompt_tool'], \
                    f"data_source node {node.id} has invalid source_name: {node.data.source_name}"
            
            # Critical: Verify agent nodes have stock-related tools
            found_stock_tools = False
            for node in agent_nodes:
                if hasattr(node.data, 'tools') and node.data.tools:
                    stock_tools = [t for t in node.data.tools 
                                  if any(keyword in t.lower() for keyword in ['stock', 'finance', 'yfinance', 'price'])]
                    if stock_tools:
                        found_stock_tools = True
                        print(f"✅ Agent {node.id} has stock tools: {stock_tools}")
            
            assert found_stock_tools, f"No agent nodes found with stock-related tools. Agent tools: {[getattr(n.data, 'tools', []) for n in agent_nodes]}"
            
            print(f"✅ Test passed for '{prompt}': {len(data_source_nodes)} data_source nodes, {len(agent_nodes)} agent nodes")

    @pytest.mark.layer("agentic")
    @pytest.mark.category("stock_analysis")
    @pytest.mark.asyncio
    async def test_smart_stock_generation(self, smart_test_data, real_tool_catalog):
        """Example of using smart fixture dispatcher for agentic tests."""
        generation_cases = smart_test_data.get('generation_cases', [])
        
        planner = WorkflowPlanner()
        
        for test_case in generation_cases:
            if test_case.user_prompt and 'stock' in test_case.user_prompt.lower():
                workflow = await planner.generate_workflow(
                    query=test_case.user_prompt,
                    tool_catalog=test_case.tool_catalog or real_tool_catalog
                )
                
                # Verify expected results if specified
                if test_case.expected_result:
                    expected = test_case.expected_result
                    if expected.get('has_data_source_nodes'):
                        data_source_nodes = [n for n in workflow.nodes if n.type == 'data_source']
                        assert len(data_source_nodes) > 0
                    
                    if expected.get('has_agent_nodes'):
                        agent_nodes = [n for n in workflow.nodes if n.type == 'agent']
                        assert len(agent_nodes) > 0


class TestWorkflowPlannerLogicalLayer:
    """MIGRATED: Logical layer tests for pure workflow validation."""
    
    def test_workflow_validation_centralized(self, validation_test_cases, mock_tool_catalog):
        """Test workflow validation using centralized test cases."""
        from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec
        
        for test_case in validation_test_cases:
            if not test_case.workflow_spec:
                continue
                
            try:
                workflow = WorkflowSpec(**test_case.workflow_spec)
                issues = workflow.validate_structure(mock_tool_catalog)
                
                if test_case.should_pass:
                    assert len(issues) == 0, f"Expected test '{test_case.name}' to pass but got issues: {issues}"
                else:
                    assert len(issues) > 0, f"Expected test '{test_case.name}' to fail but got no issues"
                    
                    # Check expected errors if specified
                    if test_case.expected_errors:
                        for expected_error in test_case.expected_errors:
                            assert any(expected_error in issue for issue in issues), \
                                f"Expected error '{expected_error}' not found in {issues}"
            
            except Exception as e:
                if test_case.should_pass:
                    pytest.fail(f"Expected test '{test_case.name}' to pass but got exception: {e}")

    def test_conditional_routing_centralized(self, conditional_routing_cases):
        """Test conditional routing using centralized test cases."""
        for test_case in conditional_routing_cases:
            workflow_spec = test_case.workflow_spec
            assert workflow_spec is not None
            assert "nodes" in workflow_spec
            assert "edges" in workflow_spec
            
            # Test routing logic
            edges = workflow_spec["edges"]
            for edge in edges:
                if "data" in edge and "route_index" in edge["data"]:
                    # New route index system should be integers
                    assert isinstance(edge["data"]["route_index"], int)
                    print(f"✅ Edge {edge.get('id', 'unknown')} uses route_index: {edge['data']['route_index']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])