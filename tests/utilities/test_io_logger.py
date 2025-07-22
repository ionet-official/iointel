"""
Test the IO.net logging system with real workflow execution scenarios.

This test verifies that our logging system properly captures execution reports,
workflow analysis, and provides the telemetry needed for debugging and monitoring.
"""

import pytest
import sys
from io import StringIO
from contextlib import contextmanager
from uuid import uuid4

from iointel.src.utilities.io_logger import (
    IOLogger, 
    execution_logger, 
    workflow_logger, 
    agent_logger,
    get_component_logger
)
from iointel.src.agent_methods.data_models.workflow_spec import (
    WorkflowSpec, 
    NodeSpec, 
    NodeData, 
    EdgeSpec, 
    EdgeData,
    SLARequirements
)


@contextmanager
def capture_output():
    """Capture stdout for testing log output."""
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    try:
        yield mystdout
    finally:
        sys.stdout = old_stdout


class TestIOLogger:
    """Test the IO.net logger functionality."""
    
    def test_basic_logging_levels(self):
        """Test that all logging levels work and produce expected output."""
        logger = IOLogger("TEST")
        
        with capture_output() as output:
            logger.debug("Debug message", {"key": "value"})
            logger.info("Info message")
            logger.success("Success message", {"count": 5})
            logger.warning("Warning message")
            logger.error("Error message", {"error_code": 500})
            logger.critical("Critical message")
        
        log_output = output.getvalue()
        
        # Check that all log levels appear
        assert "🔍" in log_output  # DEBUG
        assert "📋" in log_output  # INFO
        assert "✅" in log_output  # SUCCESS
        assert "⚠️" in log_output  # WARNING
        assert "❌" in log_output  # ERROR
        assert "💥" in log_output  # CRITICAL
        
        # Check component appears
        assert "[TEST]" in log_output
        
        # Check structured data appears
        assert "key: value" in log_output
        assert "count: 5" in log_output
        assert "error_code: 500" in log_output

    def test_execution_context_tracking(self):
        """Test that execution IDs are properly tracked in logs."""
        logger = IOLogger("EXECUTION")
        execution_id = "test-exec-123"
        
        with capture_output() as output:
            logger.info("Starting execution", execution_id=execution_id)
            logger.success("Node completed", {"node": "agent_1"}, execution_id)
            logger.error("Node failed", {"node": "agent_2"}, execution_id)
        
        log_output = output.getvalue()
        
        # Check execution ID appears in context
        assert "exec:test-exe" in log_output  # Truncated to 8 chars
        assert "[EXECUTION|exec:test-exe]" in log_output

    def test_structured_logging_mode(self):
        """Test JSON structured logging for machine parsing."""
        logger = IOLogger("STRUCTURED", structured=True)
        
        with capture_output() as output:
            logger.info("Test message", {"key": "value"}, "exec-456")
        
        log_output = output.getvalue().strip()
        
        # Should be valid JSON
        import json
        log_data = json.loads(log_output)
        
        assert log_data["level"] == "INFO"
        assert log_data["component"] == "STRUCTURED"
        assert log_data["message"] == "Test message"
        assert log_data["execution_id"] == "exec-456"
        assert log_data["data"]["key"] == "value"
        assert "timestamp" in log_data

    def test_execution_report_formatting(self):
        """Test the special execution report formatting."""
        logger = IOLogger("EXECUTION")
        
        # Create a test workflow spec
        spec = WorkflowSpec(
            id=uuid4(),
            rev=1,
            title="Test Workflow",
            description="A test workflow for logging",
            nodes=[
                NodeSpec(
                    id="test_node",
                    type="decision",
                    label="Test Agent",
                    data=NodeData(
                        agent_instructions="Test agent instructions",
                        tools=["conditional_gate"],
                        sla=SLARequirements(enforce_usage=True, final_tool_must_be="conditional_gate")
                    )
                )
            ],
            edges=[]
        )
        
        report_data = {
            "execution_id": "test-execution-789",
            "status": "completed", 
            "duration": "45.2s",
            "nodes_executed": 3,
            "nodes_skipped": 1,
            "workflow_spec": spec,
            "feedback_prompt": "This is a test feedback prompt with WORKFLOW SPECIFICATION and EXECUTION RESULTS and EXPECTED EXECUTION PATTERNS"
        }
        
        with capture_output() as output:
            logger.execution_report("Test Execution Analysis", report_data, "exec-789")
        
        log_output = output.getvalue()
        
        # Check report formatting
        assert "📈 EXECUTION REPORT: Test Execution Analysis" in log_output
        assert "=" * 48 in log_output  # Title separator
        assert "🔹 EXECUTION_ID: test-execution-789" in log_output
        assert "🔹 STATUS: completed" in log_output
        assert "📋 WORKFLOW SPECIFICATION:" in log_output
        assert "🤖 GENERATED FEEDBACK PROMPT:" in log_output
        assert "✓ Contains workflow specification" in log_output
        assert "✓ Contains execution results" in log_output
        assert "✓ Contains expected patterns analysis" in log_output


class TestWorkflowExecutionLogging:
    """Test logging integration with real workflow execution scenarios."""
    
    def create_test_workflow_spec(self) -> WorkflowSpec:
        """Create a test workflow spec for logging tests."""
        return WorkflowSpec(
            id=uuid4(),
            rev=1,
            title="Stock Trading Decision Workflow",
            description="Test conditional routing workflow",
            nodes=[
                NodeSpec(
                    id="decision_agent",
                    type="decision",
                    label="Market Decision Agent", 
                    data=NodeData(
                        agent_instructions="Analyze market and decide to buy or sell",
                        tools=["conditional_gate"],
                        sla=SLARequirements(
                            enforce_usage=True,
                            final_tool_must_be="conditional_gate"
                        )
                    )
                ),
                NodeSpec(
                    id="buy_agent",
                    type="agent",
                    label="Buy Agent",
                    data=NodeData(agent_instructions="Execute buy orders")
                ),
                NodeSpec(
                    id="sell_agent", 
                    type="agent",
                    label="Sell Agent",
                    data=NodeData(agent_instructions="Execute sell orders")
                )
            ],
            edges=[
                EdgeSpec(
                    id="buy_edge",
                    source="decision_agent",
                    target="buy_agent",
                    data=EdgeData(condition="routed_to == 'buy'")
                ),
                EdgeSpec(
                    id="sell_edge",
                    source="decision_agent", 
                    target="sell_agent",
                    data=EdgeData(condition="routed_to == 'sell'")
                )
            ]
        )

    def test_workflow_execution_completion_logging(self):
        """Test logging of workflow execution completion."""
        spec = self.create_test_workflow_spec()
        execution_id = "test-exec-456"
        
        with capture_output() as output:
            execution_logger.success(
                "Workflow execution completed",
                data={
                    "execution_id": execution_id,
                    "workflow_title": spec.title,
                    "total_results": 2,
                    "execution_time": 12.5,
                    "nodes_executed": 2,
                    "nodes_skipped": 1,
                    "status": "completed"
                },
                execution_id=execution_id
            )
        
        log_output = output.getvalue()
        
        # Verify execution completion logging
        assert "✅" in log_output  # Success emoji
        assert "Workflow execution completed" in log_output
        assert spec.title in log_output
        assert "total_results: 2" in log_output
        assert "execution_time: 12.5" in log_output
        assert "nodes_executed: 2" in log_output
        assert "nodes_skipped: 1" in log_output
        assert f"exec:{execution_id[:8]}" in log_output

    def test_workflow_planner_analysis_logging(self):
        """Test logging of WorkflowPlanner analysis results."""
        spec = self.create_test_workflow_spec()
        execution_id = "test-exec-789"
        
        # Simulate WorkflowPlanner analysis response
        analysis_text = ("The conditional routing worked correctly. The decision agent "
                        "properly routed to the sell agent based on market analysis. "
                        "Only one branch executed as expected for conditional workflows.")
        
        with capture_output() as output:
            execution_logger.execution_report(
                f"WorkflowPlanner Analysis for {spec.title}",
                report_data={
                    "execution_id": execution_id,
                    "status": "completed",
                    "duration": "8.3s", 
                    "nodes_executed": 2,
                    "nodes_skipped": 1,
                    "workflow_spec": spec,
                    "feedback_prompt": f"Generated feedback prompt for {spec.title} with WORKFLOW SPECIFICATION and EXECUTION RESULTS"
                },
                execution_id=execution_id
            )
            
            execution_logger.success(
                "WorkflowPlanner analysis completed",
                data={
                    "analysis_length": len(analysis_text),
                    "analysis_preview": analysis_text[:100] + "...",
                    "has_workflow_spec": True,
                    "feedback_conversation_id": "feedback_123"
                },
                execution_id=execution_id
            )
        
        log_output = output.getvalue()
        
        # Verify analysis report logging
        assert "📈 EXECUTION REPORT: WorkflowPlanner Analysis" in log_output
        assert spec.title in log_output
        assert "📋 WORKFLOW SPECIFICATION:" in log_output
        assert "Market Decision Agent" in log_output  # From spec
        assert "🎯 decision_agent" in log_output  # Decision node indicator
        assert "[SLA]" in log_output  # SLA enforcement indicator
        assert "CONDITIONAL ROUTING EXPECTED:" in log_output
        assert "🤖 GENERATED FEEDBACK PROMPT:" in log_output
        
        # Verify analysis completion logging
        assert "WorkflowPlanner analysis completed" in log_output
        assert f"analysis_length: {len(analysis_text)}" in log_output
        assert "The conditional routing worked correctly" in log_output  # Check preview content

    def test_workflow_error_logging(self):
        """Test logging of workflow execution errors."""
        execution_id = "test-exec-error"
        
        with capture_output() as output:
            execution_logger.error(
                "Workflow execution failed",
                data={
                    "execution_id": execution_id,
                    "error_type": "NodeExecutionError",
                    "failed_node": "decision_agent",
                    "error_message": "Tool conditional_gate not found",
                    "execution_time": 5.2,
                    "completed_nodes": 0
                },
                execution_id=execution_id
            )
        
        log_output = output.getvalue()
        
        # Verify error logging
        assert "❌" in log_output  # Error emoji
        assert "Workflow execution failed" in log_output
        assert "error_type: NodeExecutionError" in log_output
        assert "failed_node: decision_agent" in log_output
        assert "Tool conditional_gate not found" in log_output
        assert f"exec:{execution_id[:8]}" in log_output

    def test_component_specific_loggers(self):
        """Test that different component loggers work correctly."""
        execution_id = "test-comp-123"
        
        with capture_output() as output:
            execution_logger.info("Execution started", execution_id=execution_id)
            workflow_logger.info("Workflow validated", execution_id=execution_id)
            agent_logger.info("Agent initialized", execution_id=execution_id)
            
            custom_logger = get_component_logger("CUSTOM")
            custom_logger.info("Custom component active", execution_id=execution_id)
        
        log_output = output.getvalue()
        
        # Verify different component loggers
        assert "[EXECUTION|exec:test-com]" in log_output
        assert "[WORKFLOW|exec:test-com]" in log_output
        assert "[AGENT|exec:test-com]" in log_output
        assert "[CUSTOM|exec:test-com]" in log_output


if __name__ == "__main__":
    # Run the tests directly
    pytest.main([__file__, "-v"])