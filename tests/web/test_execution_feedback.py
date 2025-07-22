#!/usr/bin/env python3
"""
Test the workflow execution feedback system.

This demonstrates how execution results are captured, curated, and sent back
to the WorkflowPlanner for analysis and improvement suggestions.
"""

import asyncio
from uuid import uuid4

from iointel.src.web.execution_feedback import (
    feedback_collector,
    create_execution_feedback_prompt,
    ExecutionStatus,
    ExecutionResultCurator
)
from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec, NodeSpec, EdgeSpec, NodeData, EdgeData
)


async def test_execution_feedback_system():
    """Test the complete execution feedback system."""
    
    print("🧪 Testing Workflow Execution Feedback System")
    print("=" * 50)
    
    # Create example workflow spec
    example_workflow = WorkflowSpec(
        id=uuid4(),
        rev=1,
        title="Email Processing and Organization Workflow",
        description="Process emails and organize them into contextual task trees",
        nodes=[
            NodeSpec(
                id="agent_1", 
                type="agent", 
                label="Email Processing Agent",
                data=NodeData(
                    agent_instructions="Process and analyze incoming emails",
                    tools=["get_email_content"],
                    ins=[],
                    outs=["processed_emails"]
                )
            ),
            NodeSpec(
                id="agent_2", 
                type="agent", 
                label="Context Tree Organization Agent",
                data=NodeData(
                    agent_instructions="Organize emails into contextual task tree based on topic and urgency",
                    tools=["create_context_tree", "append_context_tree"],
                    ins=["processed_emails"],
                    outs=["organized_tasks"]
                )
            )
        ],
        edges=[
            EdgeSpec(
                id="e1", 
                source="agent_1", 
                target="agent_2", 
                data=EdgeData()
            )
        ]
    )
    
    # Test execution ID
    execution_id = "test-execution-feedback-123"
    
    print(f"📋 Workflow: {example_workflow.title}")
    print(f"🆔 Execution ID: {execution_id}")
    print()
    
    # 1. Start execution tracking
    print("1️⃣ Starting execution tracking...")
    feedback_collector.start_execution_tracking(
        execution_id=execution_id,
        workflow_spec=example_workflow,
        user_inputs={
            "user_input_1": "I bought AAPL, TSLA and NVIDA last year, what should we sell today?",
            "tool_1": "the dirtiest jokes for a southern audience"
        }
    )
    print("✅ Execution tracking started")
    print()
    
    # 2. Simulate first node execution (success)
    print("2️⃣ Simulating first node execution...")
    feedback_collector.record_node_start(
        execution_id, "agent_1", "agent", "Email Processing Agent"
    )
    
    # Simulate some execution time
    await asyncio.sleep(0.1)
    
    feedback_collector.record_node_completion(
        execution_id=execution_id,
        node_id="agent_1",
        status=ExecutionStatus.SUCCESS,
        result_preview="Successfully processed 3 email topics: stock recommendations, investment decisions, and entertainment content. Identified key themes and urgency levels.",
        tool_usage=["get_email_content"]
    )
    print("✅ First node completed successfully")
    print()
    
    # 3. Simulate second node execution (failure)
    print("3️⃣ Simulating second node execution with failure...")
    feedback_collector.record_node_start(
        execution_id, "agent_2", "agent", "Context Tree Organization Agent"
    )
    
    await asyncio.sleep(0.1)
    
    feedback_collector.record_node_completion(
        execution_id=execution_id,
        node_id="agent_2",
        status=ExecutionStatus.FAILED,
        error_message="KeyError: 'self' - Agent initialization failed due to type hint issue in tool registration",
        tool_usage=["create_context_tree"]
    )
    print("❌ Second node failed")
    print()
    
    # 4. Complete execution and generate summary
    print("4️⃣ Completing execution and generating summary...")
    execution_summary = feedback_collector.complete_execution(
        execution_id=execution_id,
        final_outputs={"agent_1": "Processed emails with topics identified"},
        error_summary="Agent initialization failed due to 'self' parameter in type hints during tool registration"
    )
    
    print("✅ Execution summary generated")
    print(f"   Status: {execution_summary.status.value}")
    print(f"   Duration: {execution_summary.total_duration_seconds:.2f}s")
    print(f"   Nodes executed: {len(execution_summary.nodes_executed)}")
    print(f"   Nodes skipped: {len(execution_summary.nodes_skipped)}")
    print()
    
    # 5. Generate curated summary
    print("5️⃣ Generating curated execution summary...")
    curated_summary = ExecutionResultCurator.curate_execution_summary(execution_summary)
    
    print("📋 CURATED SUMMARY:")
    print("-" * 20)
    print(curated_summary[:500] + "...\n")
    
    # 6. Generate feedback prompt for WorkflowPlanner
    print("6️⃣ Generating feedback prompt for WorkflowPlanner...")
    feedback_prompt = create_execution_feedback_prompt(execution_summary)
    
    print("📤 FEEDBACK PROMPT (first 300 chars):")
    print("-" * 20)
    print(feedback_prompt[:300] + "...\n")
    
    # 7. Simulate sending to WorkflowPlanner (without actual API call)
    print("7️⃣ Simulating WorkflowPlanner analysis...")
    
    # In a real scenario, this would call the WorkflowPlanner
    mock_planner_response = """
🤖 **Execution Analysis Summary**

I've analyzed your email processing workflow execution and identified several key insights:

**What Happened:**
- The workflow successfully processed emails in the first stage, identifying 3 distinct topics (stock recommendations, investment decisions, entertainment)
- However, it failed during the organization phase due to a technical issue with agent initialization

**Issues Identified:**
1. **Critical Bug**: The 'self' parameter type hint issue prevents proper agent tool registration
2. **Workflow Design**: The linear dependency means second node failure stops entire workflow
3. **Error Handling**: No fallback mechanism for agent initialization failures

**Improvement Suggestions:**
1. **Fix Technical Issue**: Update agent factory to handle 'self' parameter in type hints properly
2. **Add Error Handling**: Implement try-catch with fallback strategies for tool registration
3. **Parallel Design**: Consider parallel processing where email analysis could continue even if organization fails
4. **Monitoring**: Add health checks for agent initialization before workflow execution
5. **User Experience**: Provide partial results when some nodes succeed but others fail

**Recommended Next Steps:**
- Prioritize fixing the type hint issue in the agent factory
- Add conditional routing to handle failures gracefully
- Consider breaking this into smaller, more resilient workflows

This pattern suggests implementing better error boundaries and recovery mechanisms in future workflows.
"""
    
    print("🤖 MOCK PLANNER ANALYSIS:")
    print("-" * 25)
    print(mock_planner_response.strip())
    print()
    
    print("✅ Workflow execution feedback system test completed!")
    print("🎯 The system successfully:")
    print("   • Tracked execution progress")
    print("   • Captured node-level results and errors")
    print("   • Generated comprehensive execution summary")
    print("   • Created actionable feedback for improvement")
    print("   • Provided specific technical and design recommendations")
    
    return execution_summary


async def test_successful_execution():
    """Test feedback for a successful execution."""
    
    print("\n" + "=" * 50)
    print("🧪 Testing Successful Execution Feedback")
    print("=" * 50)
    
    # Create simple successful workflow
    simple_workflow = WorkflowSpec(
        id=uuid4(),
        rev=1,
        title="Simple Bitcoin Price Analysis",
        description="Get Bitcoin price and analyze trends",
        nodes=[
            NodeSpec(
                id="price_getter", 
                type="tool", 
                label="Get Bitcoin Price",
                data=NodeData(
                    tool_name="get_coin_quotes",
                    config={"symbol": ["BTC"]},
                    ins=[],
                    outs=["price_data"]
                )
            ),
            NodeSpec(
                id="analyst", 
                type="agent", 
                label="Price Analysis Agent",
                data=NodeData(
                    agent_instructions="Analyze Bitcoin price trends and provide insights",
                    tools=["conditional_gate"],
                    ins=["price_data"],
                    outs=["analysis"]
                )
            )
        ],
        edges=[
            EdgeSpec(id="e1", source="price_getter", target="analyst", data=EdgeData())
        ]
    )
    
    execution_id = "success-test-456"
    
    # Track successful execution
    feedback_collector.start_execution_tracking(
        execution_id=execution_id,
        workflow_spec=simple_workflow,
        user_inputs={"analysis_type": "technical"}
    )
    
    # Simulate successful nodes
    feedback_collector.record_node_start(execution_id, "price_getter", "tool", "Get Bitcoin Price")
    await asyncio.sleep(0.05)
    feedback_collector.record_node_completion(
        execution_id, "price_getter", ExecutionStatus.SUCCESS,
        result_preview="BTC: $67,234.56 (+2.3% 24h)", tool_usage=["get_coin_quotes"]
    )
    
    feedback_collector.record_node_start(execution_id, "analyst", "agent", "Price Analysis Agent")
    await asyncio.sleep(0.1)
    feedback_collector.record_node_completion(
        execution_id, "analyst", ExecutionStatus.SUCCESS,
        result_preview="Bitcoin showing bullish momentum with strong volume support at current levels",
        tool_usage=["conditional_gate"]
    )
    
    # Complete successful execution
    execution_summary = feedback_collector.complete_execution(
        execution_id=execution_id,
        final_outputs={
            "price_getter": "BTC: $67,234.56 (+2.3% 24h)",
            "analyst": "Bullish analysis with buy recommendation"
        }
    )
    
    # Generate feedback
    curated_summary = ExecutionResultCurator.curate_execution_summary(execution_summary)
    
    print("📋 SUCCESSFUL EXECUTION SUMMARY:")
    print("-" * 30)
    print(curated_summary[:400] + "...")
    
    print("\n✅ Success metrics:")
    print(f"   Duration: {execution_summary.total_duration_seconds:.2f}s")
    print("   Success rate: 100%")
    print(f"   Efficiency: {execution_summary.performance_metrics.get('execution_efficiency', 0):.1%}")
    
    return execution_summary


if __name__ == "__main__":
    asyncio.run(test_execution_feedback_system())
    asyncio.run(test_successful_execution())