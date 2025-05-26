from typing import Dict, List, Optional
from pydantic import BaseModel
from iointel import Agent
from iointel.src.agents import ToolUsageResult
import os
import asyncio
from dotenv import load_dotenv


load_dotenv(os.path.join(os.path.dirname(__file__), '../../..', 'creds.env'))

class CriticFeedback(BaseModel):
    """Feedback from the critic on agent performance"""
    score: float  # 0.0 to 1.0
    better_query: str
    metrics: Dict[str, float]
    new_instructions: Optional[str] = None  # For meta-learning: a better agent instruction

class CriticAgent:
    """Agentic LLM-based Critic for evaluating agent performance"""
    def __init__(self, model="gpt-4o", api_key=None, base_url=None, verbose=True):
        self.agent = Agent(
            name="CriticAgent",
            instructions="""
            You are a critic agent. Given a task, agent actions, tool results, the agent's final response, and the ground truth, evaluate the agent's performance.
            Estimate the following as best as possible:
            - score: a float from 0.0 to 1.0 (overall performance)
            - better_query: a better query to solve the task (more specific, more detailed, more accurate, a hint, etc.). If current query is good, just return the same query. This new query SHOULD NOT lose information, but rather add more information.
             eg: Divide the current temperature in London by the square root of 99, then multiply by 5.5. SHOULD NOT BE "What is the square root of 99?" or "What is the weather in London?", capish?
            - metrics: a dict of named metrics (e.g., tool_usage_efficiency, response_accuracy, action_relevance, response_completeness), each a float from 0.0 to 1.0
            - new_instructions: (optional) a better instruction prompt for the agent to solve this task next time. Should be GENERAL instructions you would give to an LLM agent as instructions, not specific to this task.
            
            Output ONLY valid JSON that can be parsed as the CriticFeedback Pydantic model.
            """,
            model=model,
            api_key=api_key,
            base_url=base_url,
            output_type=CriticFeedback
        )
        self.verbose = verbose

    async def evaluate_performance(
        self,
        task: str,
        agent_actions: List[ToolUsageResult],
        final_response: str,
        goal_seek: Optional[str] = None, # if the task has a goal seek outcome, include it here
    ) -> CriticFeedback:
        """Evaluate agent performance on a task using the LLM agent"""
        prompt = f"""
        Task:
        {task}

        Agent Actions:
        {agent_actions}

        Final Response:
        {final_response}
        """
        if goal_seek:
            prompt += f"""
            Goal Seek Outcome:
            {goal_seek}
            """
            
        if self.verbose:
            print(f"Critic prompt: {prompt}")
        return (await self.agent.run(prompt))['result']
    

if __name__ == "__main__":
    async def main():
        critic = CriticAgent(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))
        
        # Perfect execution
        task1 = "What is the weather in Tokyo?"
        agent_actions1 = [
            ToolUsageResult(
                tool_name="get_weather",
                tool_args={"city": "Tokyo"},
                tool_result={"result": "Sunny", "metadata": {}}
            )
        ]
        final_response1 = "The weather in Tokyo is sunny."
        res1 = await critic.evaluate_performance(task1, agent_actions1, final_response1)
        print("\nPerfect execution:")
        print(res1['result'])

        # Partially correct execution
        task2 = "What is 5 + 3 multiplied by 2?"
        agent_actions2 = [
            ToolUsageResult(
                tool_name="add",
                tool_args={"a": 5, "b": 3},
                tool_result={"result": 8, "metadata": {}}
            )
            # Missing multiplication step
        ]
        final_response2 = "The result is 8" # Should be 16
        res2 = await critic.evaluate_performance(task2, agent_actions2, final_response2)
        print("\nPartially correct execution:")
        print(res2['result'])

        # Wrong execution
        task3 = "What is the weather in Paris?"
        agent_actions3 = [
            ToolUsageResult(
                tool_name="multiply",
                tool_args={"a": 10, "b": 5},
                tool_result={"result": 50, "metadata": {}}
            ) # Wrong tool
        ]
        final_response3 = "The result is 50" # Completely wrong response
        res3 = await critic.evaluate_performance(task3, agent_actions3, final_response3)
        print("\nWrong execution:")
        print(res3['result'])

    asyncio.run(main())