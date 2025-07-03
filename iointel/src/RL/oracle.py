from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from iointel import Agent

from iointel.src.agents import ToolUsageResult
from iointel.src.RL.utils import tool_usage_results_to_string
from iointel.src.RL.prompts import ORACLE_INSTRUCTIONS, create_oracle_prompt


class EvaluationResult(BaseModel):
    """Result of oracle evaluation"""

    correct: bool
    score: float  # 0.0 to 1.0
    feedback: str
    details: Dict[str, Any] = {}


class OracleAgent:
    """An agent that acts as an oracle for evaluating responses"""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        verbose: bool = True,
    ):
        self.agent = Agent(
            name="Oracle",
            instructions=ORACLE_INSTRUCTIONS,
            model=model,
            output_type=EvaluationResult,
            api_key=api_key,
            base_url=base_url,
            model_settings={"temperature": temperature}
            if temperature is not None
            else {},
        )
        self.verbose = verbose

    async def evaluate(
        self,
        agent_response: Any,
        ground_truth: Any,
        task_description: str,
        agent_actions: List[ToolUsageResult],
        required_tools: List[str],
        context: Optional[Dict[str, Any]] = None,
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
        agent_actions_string = tool_usage_results_to_string(agent_actions)
        # Prepare the evaluation prompt using the unified prompt creator
        prompt = create_oracle_prompt(
            task_description=task_description,
            required_tools=required_tools,
            agent_response=str(agent_response),
            agent_actions_string=agent_actions_string,
            ground_truth=str(ground_truth),
            context=str(context) if context else None
        )
        if self.verbose:
            print(f"Oracle prompt: {prompt}")
        # Get evaluation from the oracle agent
        response = (await self.agent.run(prompt))["result"]
        return response