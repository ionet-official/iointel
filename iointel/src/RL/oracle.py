from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from iointel import Agent
from .oracle import EvaluationResult

class OracleAgent:
    """An agent that acts as an oracle for evaluating responses"""
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.0
    ):
        self.agent = Agent(
            name="Oracle",
            instructions="""
            You are an oracle agent responsible for evaluating responses against ground truth.
            Your task is to:
            1. Compare the agent's response with the ground truth
            2. Evaluate the correctness and completeness of the response
            3. Provide detailed feedback on any discrepancies
            4. Assign a score between 0.0 and 1.0, ranking how well the response correctly answers the question
            
            You must return your evaluation in the following format:
            {
                "correct": boolean,
                "score": float (0.0 to 1.0),
                "feedback": string,
                "details": {
                    "matching_fields": list of matching fields,
                    "missing_fields": list of missing fields,
                    "incorrect_values": dict of incorrect values,
                    "additional_insights": string
                }
            }
            """,
            model=model,
            temperature=temperature,
            output_type=EvaluationResult, 
            api_key=api_key
        )
    
    async def evaluate(
        self,
        agent_response: Any,
        ground_truth: Any,
        task_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Evaluate agent response against ground truth using the oracle agent
        
        Args:
            agent_response: The response from the agent
            ground_truth: The correct answer/expected response
            task_description: Description of the task
            context: Additional context for evaluation
            
        Returns:
            EvaluationResult with correctness, score, and feedback
        """
        # Prepare the evaluation prompt
        prompt = f"""
        Task Description: {task_description}
        
        Agent Response:
        {agent_response}
        
        Ground Truth:
        {ground_truth}
        
        Context:
        {context if context else 'No additional context provided'}
        
        Please evaluate the agent's response against the ground truth.
        """
        
        # Get evaluation from the oracle agent
        response = await self.agent.run(prompt)
        return response
        
    
    async def evaluate_tool_usage(
        self,
        tool_results: List[Dict[str, Any]],
        expected_tools: List[str],
        task_description: str
    ) -> EvaluationResult:
        """
        Evaluate if the agent used the correct tools in the right way
        
        Args:
            tool_results: List of tool execution results
            expected_tools: List of expected tool names
            task_description: Description of the task
            
        Returns:
            EvaluationResult with tool usage evaluation
        """
        prompt = f"""
        Task Description: {task_description}
        
        Tool Results:
        {tool_results}
        
        Expected Tools:
        {expected_tools}
        
        Please evaluate if the agent used the correct tools effectively.
        """
        
        response = await self.agent.run(prompt)
        return response