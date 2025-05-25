from typing import Any, Dict, List, Optional, Callable
from pydantic import BaseModel
from iointel.src.RL.task_manager import Task, TaskManager
from iointel.src.RL.critic import CriticAgent, CriticFeedback
from iointel.src.RL.oracle import OracleAgent, EvaluationResult
from iointel import Agent, PersonaConfig
from iointel.src.agents  import ToolUsageResult
import asyncio
import os
from dotenv import load_dotenv
from iointel.src.RL.example_tools import add, subtract, multiply, divide, get_weather

load_dotenv(os.path.join(os.path.dirname(__file__), '../../..', 'creds.env'))


class RLState(BaseModel):
    task: Task
    step_count: int = 0
    agent_result: Optional[Dict[str, str  | list[ToolUsageResult]] | None] = None
    best_query: str = ''
    best_instructions: str = ''
    critic_feedback: Optional[CriticFeedback] = None
    oracle_result: Optional[EvaluationResult] = None # TODO: add oracle result
    done: bool = False

class RLEnvironment:
    """Agentic RL environment: orchestrates agent, critic, oracle, and task manager."""
    def __init__(self, name: str, agent_instructions: str, task_manager: TaskManager, critic: CriticAgent, oracle: OracleAgent, tools: List[Callable], max_steps=10, meta_learn_instructions=False, persona=None, agent_class=None, task_file_path="tasks.json", model=None, api_key=None, base_url=None, threshold: float = 0.95):
        self.name: str = name
        self.agent_instructions: str = agent_instructions
        self.persona: PersonaConfig = persona
        self.agent_class: Agent = agent_class
        self.task_manager: TaskManager = task_manager
        self.critic: CriticAgent = critic
        self.oracle: OracleAgent = oracle
        self.tools: List[Callable] = tools
        self.max_steps = max_steps
        self.meta_learn_instructions = meta_learn_instructions
        self.state: RLState = None
        self.task_file_path: str = task_file_path
        self.model: str = model
        self.api_key: str = api_key
        self.base_url: str = base_url
        self.threshold: float = threshold

    def generate_tasks(self, num_tasks: int = 10, verbose: bool = False):
        tasks = self.task_manager.generate_tasks(self.tools, num_tasks)
        self.task_manager.save_tasks(self.task_file_path)
        if verbose:
            print(f"Generated {len(tasks)} tasks and saved to {self.task_file_path}")
            for task in tasks:
                print(f"Task {task.id}: {task.description}")
        return tasks
    
    def load_tasks(self, verbose: bool = False):
        self.task_manager.load_tasks(self.task_file_path)
        if verbose:
            print(f"Loaded {len(self.task_manager.tasks)} tasks from {self.task_file_path}")
            for task in self.task_manager.tasks:
                print(f"Task {task.id}: {task.description}")
        return self.task_manager.tasks

    def reset(self, task: Task = None, difficulty: float = 0.5):
        if task is None:
            task = self.task_manager.get_task(difficulty)
        print(f"Resetting environment with task: {task.description}")
        self.state = RLState(task=task)
        return self.state

    async def run_episode(self, verbose=True):
        state = self.reset()
        task = state.task
        instructions = self.agent_instructions
        best_instructions = None
        critic_feedback = None
        context = '' # we can also vary this, but for later....
        for step in range(self.max_steps):
            ##########################The Agent Learns###############################
            # 1. Instantiate agent (with updated instructions if meta-learning)
            agent = self.agent_class(
                name=self.name,
                instructions=instructions,
                tools=self.tools,
                context=context,
                persona=self.persona,
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url
            )
            # Compose input: task.description (+ critic feedback if not meta-learn-instructions)
            if step == 0 or self.meta_learn_instructions:
                query = task.description
            else:
                query = f"{task.description}\n\n{critic_feedback.better_query if critic_feedback else ''}"
            result = await agent.run(query, conversation_id=task.id)
            ##########################The Agent Learns###############################
            state.agent_result = result
            state.step_count = step + 1

            # 2. Critic feedback
            critic_feedback = await self.critic.evaluate_performance(
                task=task,
                agent_actions=result.get("tool_usage_results", []),
                final_response=result.get("result"),
            )

            # 3. Oracle evaluation
            oracle_result = await self.oracle.evaluate(
                agent_response=result.get("result"),
                ground_truth=task.ground_truth,
                task_description=task.description
            )

            # 4. Print/log everything
            if verbose:
                print("="*60)
                print(f"Query: {query}")
                print(f"Step {step+1}:")
                print("Agent output:", result.get("result"))
                print("Tool usage:", result.get("tool_usage_results"))
                print("Critic:", getattr(critic_feedback, 'model_dump_json', lambda **_: critic_feedback)())
                print("Oracle:", getattr(oracle_result, 'model_dump_json', lambda **_: oracle_result)())
                print("-"*60)

            # 5. Meta-learn: update instructions if critic suggests new ones
            if self.meta_learn_instructions and getattr(critic_feedback, 'new_instructions', None):
                instructions = critic_feedback.new_instructions

            # 6. Check for success
            if hasattr(oracle_result, 'score') and oracle_result.score > self.threshold:
                print("Task solved!")
                state.done = True
                best_instructions = instructions
                break
        
        state.best_instructions = best_instructions
        state.best_query = query
        state.critic_feedback = critic_feedback
        state.oracle_result = oracle_result
        return state

if __name__ == "__main__":
    async def main():
        tools = [add, subtract, multiply, divide, get_weather]
        model = "gpt-4o"
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE")

        critic = CriticAgent(model=model, api_key=api_key, base_url=base_url)
        task_manager = TaskManager(model=model, api_key=api_key, base_url=base_url)
        oracle = OracleAgent(model=model, api_key=api_key, base_url=base_url)

        environment = RLEnvironment(
            name="padwan",
            agent_instructions='',
            task_manager=task_manager,
            critic=critic,
            oracle=oracle,
            tools=tools,
            max_steps=3,
            agent_class=Agent,
            model=model,
            api_key=api_key,
            base_url=base_url
        )
        generate_new = False
        if generate_new:
            environment.generate_tasks(num_tasks=10, verbose=True)
        else:
            environment.load_tasks(verbose=True)
        best_state = await environment.run_episode(verbose=True)
        print("="*80)
        print(f"\n\nBest state: {best_state}")
    asyncio.run(main())