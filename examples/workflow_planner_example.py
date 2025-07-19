"""
Example usage of the WorkflowPlanner agent.

This script demonstrates how to use the WorkflowPlanner to generate
React Flow compatible workflow specifications from natural language queries.
"""

import asyncio
import json
from iointel.src.agent_methods.agents.workflow_planner import WorkflowPlanner
from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpec


async def basic_example():
    """Basic example of generating a workflow from a query."""
    print("=== Basic WorkflowPlanner Example ===")
    
    # Create the WorkflowPlanner
    planner = WorkflowPlanner(debug=True)
    
    # Example tool catalog (simplified)
    tool_catalog = {
        "fetch_sales_data": {
            "description": "Fetch sales data from database",
            "parameters": {"date_range": "string", "format": "string"},
            "returns": "sales_data"
        },
        "calculate_sum": {
            "description": "Calculate sum of numeric values",
            "parameters": {"data": "array", "column": "string"},
            "returns": "total"
        },
        "send_slack_message": {
            "description": "Send message to Slack channel",
            "parameters": {"channel": "string", "message": "string", "auth": "slack_token"},
            "returns": "success"
        }
    }
    
    # User query
    query = "Every morning at 9 AM fetch yesterday's sales CSV, sum revenue, and Slack it to finance."
    
    try:
        # Generate workflow
        workflow_spec = await planner.generate_workflow(
            query=query,
            tool_catalog=tool_catalog
        )
        
        print("Generated Workflow:")
        print(json.dumps(workflow_spec.model_dump(), indent=2, default=str))
        
        return workflow_spec
        
    except Exception as e:
        print(f"Error generating workflow: {e}")
        return None


async def refinement_example():
    """Example of refining a workflow based on feedback."""
    print("\n=== Workflow Refinement Example ===")
    
    planner = WorkflowPlanner()
    
    # Start with a simple example workflow
    workflow_spec = planner.create_example_workflow("Sales Report Workflow")
    
    print("Original Workflow:")
    print(f"Title: {workflow_spec.title}")
    print(f"Nodes: {len(workflow_spec.nodes)}")
    print(f"Edges: {len(workflow_spec.edges)}")
    
    # Refine based on feedback
    feedback = "Add error handling and retry logic for failed API calls"
    
    try:
        refined_spec = await planner.refine_workflow(
            workflow_spec=workflow_spec,
            feedback=feedback
        )
        
        print("\nRefined Workflow:")
        print(f"Title: {refined_spec.title}")
        print(f"Nodes: {len(refined_spec.nodes)}")
        print(f"Edges: {len(refined_spec.edges)}")
        
        return refined_spec
        
    except Exception as e:
        print(f"Error refining workflow: {e}")
        return None


async def complex_example():
    """Example with a more complex workflow requirement."""
    print("\n=== Complex Workflow Example ===")
    
    planner = WorkflowPlanner()
    
    # More comprehensive tool catalog
    tool_catalog = {
        "web_scraper": {
            "description": "Scrape data from web pages",
            "parameters": {"url": "string", "selectors": "object"},
            "returns": "scraped_data"
        },
        "data_validator": {
            "description": "Validate data against schema",
            "parameters": {"data": "object", "schema": "object"},
            "returns": "validation_result"
        },
        "ml_classifier": {
            "description": "Classify data using ML model",
            "parameters": {"data": "array", "model": "string"},
            "returns": "classifications"
        },
        "database_insert": {
            "description": "Insert data into database",
            "parameters": {"table": "string", "data": "object", "connection": "string"},
            "returns": "insert_result"
        },
        "email_notification": {
            "description": "Send email notification",
            "parameters": {"to": "string", "subject": "string", "body": "string"},
            "returns": "email_sent"
        }
    }
    
    # Complex query
    query = """
    Create a workflow that:
    1. Scrapes product data from multiple e-commerce websites
    2. Validates the scraped data for completeness
    3. Classifies products into categories using ML
    4. Stores valid data in the database
    5. Sends email notifications for any failures
    6. Runs this process every hour during business hours
    """
    
    try:
        workflow_spec = await planner.generate_workflow(
            query=query,
            tool_catalog=tool_catalog,
            context={"schedule": "hourly", "business_hours": "9-17"}
        )
        
        print("Complex Workflow Generated:")
        print(f"Title: {workflow_spec.title}")
        print(f"Description: {workflow_spec.description}")
        print(f"Nodes: {len(workflow_spec.nodes)}")
        print(f"Edges: {len(workflow_spec.edges)}")
        
        # Print node details
        print("\nNodes:")
        for node in workflow_spec.nodes:
            print(f"  - {node.id}: {node.label} ({node.type})")
        
        # Print edge details
        print("\nEdges:")
        for edge in workflow_spec.edges:
            condition = f" [when: {edge.data.condition}]" if edge.data.condition else ""
            print(f"  - {edge.source} → {edge.target}{condition}")
        
        return workflow_spec
        
    except Exception as e:
        print(f"Error generating complex workflow: {e}")
        return None


def validate_workflow_spec(workflow_spec: WorkflowSpec) -> bool:
    """Validate that a workflow specification is well-formed."""
    print(f"\n=== Validating Workflow: {workflow_spec.title} ===")
    
    issues = []
    
    # Check that all edge sources and targets exist as nodes
    node_ids = {node.id for node in workflow_spec.nodes}
    
    for edge in workflow_spec.edges:
        if edge.source not in node_ids:
            issues.append(f"Edge {edge.id} references unknown source node: {edge.source}")
        if edge.target not in node_ids:
            issues.append(f"Edge {edge.id} references unknown target node: {edge.target}")
    
    # Check for required fields
    for node in workflow_spec.nodes:
        if not node.label:
            issues.append(f"Node {node.id} has no label")
        if not node.data.config and node.type == "tool":
            issues.append(f"Tool node {node.id} has no configuration")
    
    # Check for cycles (simple check)
    if has_cycle(workflow_spec):
        issues.append("Workflow contains cycles")
    
    if issues:
        print("❌ Validation Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Workflow specification is valid!")
        return True


def has_cycle(workflow_spec: WorkflowSpec) -> bool:
    """Simple cycle detection using DFS."""
    from collections import defaultdict
    
    # Build adjacency list
    graph = defaultdict(list)
    for edge in workflow_spec.edges:
        graph[edge.source].append(edge.target)
    
    visited = set()
    rec_stack = set()
    
    def dfs(node):
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        
        rec_stack.remove(node)
        return False
    
    # Check all nodes
    for node in workflow_spec.nodes:
        if node.id not in visited:
            if dfs(node.id):
                return True
    
    return False


async def main():
    """Run all examples."""
    print("🚀 WorkflowPlanner Examples\n")
    
    # Basic example
    basic_result = await basic_example()
    if basic_result:
        validate_workflow_spec(basic_result)
    
    # Refinement example
    refined_result = await refinement_example()
    if refined_result:
        validate_workflow_spec(refined_result)
    
    # Complex example
    complex_result = await complex_example()
    if complex_result:
        validate_workflow_spec(complex_result)
    
    print("\n🎉 All examples completed!")


if __name__ == "__main__":
    asyncio.run(main())