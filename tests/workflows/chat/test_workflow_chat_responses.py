"""
Test chat-only responses in workflow planner using null nodes/edges.

This test suite verifies that:
1. WorkflowSpecLLM allows null nodes/edges for chat-only responses
2. Chat responses use the reasoning field for conversational messages
3. Normal workflows still work as expected
4. Different chat response scenarios work correctly
"""

import pytest
from iointel.src.agent_methods.data_models.workflow_spec import WorkflowSpecLLM


class TestWorkflowChatResponses:
    """Test chat response functionality using null nodes/edges convention."""
    
    def test_workflow_spec_llm_allows_null_nodes_edges(self):
        """Test that WorkflowSpecLLM model allows null nodes/edges for chat responses."""
        # Chat-only response with null nodes/edges
        chat_spec = WorkflowSpecLLM(
            title=None,
            description="Chat response",
            reasoning="I need more information about what specific aspect of machine learning you're interested in. Are you looking for recent papers, foundational work, or papers on a specific topic?",
            nodes=None,
            edges=None
        )
        
        assert chat_spec.nodes is None
        assert chat_spec.edges is None
        assert chat_spec.reasoning == "I need more information about what specific aspect of machine learning you're interested in. Are you looking for recent papers, foundational work, or papers on a specific topic?"
        assert chat_spec.description == "Chat response"
        
    def test_normal_workflow_spec_llm(self):
        """Test normal workflow with nodes and edges."""
        normal_spec = WorkflowSpecLLM(
            title="Data Processing Workflow",
            description="Process and analyze CSV data",
            reasoning="Created a simple workflow to process your CSV data using the calculator tool.",
            nodes=[
                {
                    "type": "tool",
                    "label": "Calculate Sum",
                    "data": {
                        "tool_name": "calculator",
                        "config": {"expression": "sum([1,2,3,4,5])"},
                        "ins": [],
                        "outs": ["result"]
                    }
                }
            ],
            edges=[]
        )
        
        assert normal_spec.nodes is not None
        assert len(normal_spec.nodes) == 1
        assert normal_spec.edges == []
        assert normal_spec.title == "Data Processing Workflow"
        assert normal_spec.reasoning == "Created a simple workflow to process your CSV data using the calculator tool."
    
    def test_chat_response_for_execution_feedback(self):
        """Test chat response providing feedback on execution results."""
        feedback_spec = WorkflowSpecLLM(
            title=None,
            description="Execution feedback",
            reasoning="Great! The workflow executed successfully and found 15 relevant papers. The analysis shows that transformer architectures are dominating recent research. Would you like me to create a follow-up workflow to dive deeper into transformer papers specifically?",
            nodes=None,
            edges=None
        )
        
        assert feedback_spec.nodes is None
        assert feedback_spec.edges is None
        assert "successfully" in feedback_spec.reasoning
        assert "transformer" in feedback_spec.reasoning
    
    def test_chat_response_asking_clarification(self):
        """Test chat response asking for clarification."""
        clarification_spec = WorkflowSpecLLM(
            title=None,
            description="Need clarification",
            reasoning="I understand you want to modify the search query. What specific terms should I search for instead? Please provide the exact keywords or phrases you'd like me to use.",
            nodes=None,
            edges=None
        )
        
        assert clarification_spec.nodes is None
        assert clarification_spec.edges is None
        assert "specific terms" in clarification_spec.reasoning
        assert "What" in clarification_spec.reasoning


if __name__ == "__main__":
    pytest.main([__file__, "-v"])