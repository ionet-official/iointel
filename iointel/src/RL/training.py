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
import random
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
    def __init__(self, name: str, agent_instructions: str, task_manager: TaskManager, critic: CriticAgent, oracle: OracleAgent, tools: List[Callable], max_steps=10, meta_learn_instructions=False, persona=None, agent_class=None, task_file_path="tasks.json", model=None, api_key=None, base_url=None, threshold: float = 0.90):
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

    def generate_tasks(self, num_tasks: int = 10, verbose: bool = False) -> List[Task]:
        tasks = self.task_manager.generate_tasks(self.tools, num_tasks)
        self.task_manager.save_tasks(self.task_file_path)
        if verbose:
            print(f"Generated {len(tasks)} tasks and saved to {self.task_file_path}")
            for task in tasks:
                print(f"Task {task.id}: {task.description}")
                print('='*60)
        return tasks
    
    def load_tasks(self, verbose: bool = False) -> List[Task]:
        self.task_manager.load_tasks(self.task_file_path)
        if verbose:
            print(f"Loaded {len(self.task_manager.tasks)} tasks from {self.task_file_path}")
            for task in self.task_manager.tasks:
                print(f"Task {task.id}: {task.description}")
                print('='*60)
        return self.task_manager.tasks

    def reset(self, task: Task = None, difficulty: Optional[float] = None) -> RLState:
        random.seed(random.randint(0, 1000000))
        if task is None:
            task = self.task_manager.get_task(difficulty)
        print('-'*30)
        print(f"==== Resetting environment with task: {task.description}")
        self.state = RLState(task=task)
        return self.state

    async def run_episode(self, task: Task = None, difficulty: Optional[float] = None, verbose=True) -> Optional[RLState]:
        state = self.reset(task=task, difficulty=difficulty)
        if not state.task:
            print("==== No task found ====")
            return None
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
            #########################################################
            conversation_id = f"{task.id}-{task.description}-training-episode"
            result = await agent.run(query, conversation_id=conversation_id)
            ##########################The Agent Learns###############################
            state.agent_result = result['full_result']
            state.step_count = step + 1

            # 2. Critic feedback
            critic_feedback = await self.critic.evaluate_performance(
                task=task.description,
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
                print("\n" + "-"*60)
                print(f"\nStep {step+1}:")
                print(f"\nTask/Query:\n{query}")
                print("\nAgent output:\n" + str(result.get("result")))
                print("\nTool usage:\n" + str(result.get("tool_usage_results")))
                print("\nCritic:\n" + str(getattr(critic_feedback, 'model_dump_json', lambda **_: critic_feedback)()))
                print("\nOracle:\n" + str(getattr(oracle_result, 'model_dump_json', lambda **_: oracle_result)()))
                print("\n" + "-"*60 + "\n")

            # 5. Meta-learn: update instructions if critic suggests new ones
            if self.meta_learn_instructions and getattr(critic_feedback, 'new_instructions', None):
                instructions = critic_feedback.new_instructions

            # 6. Check for success
            if hasattr(oracle_result, 'score') and oracle_result.score >= self.threshold:
                print("Task solved!")
                state.done = True
                best_instructions = instructions
                break
        
        # pretty print the state
        print("\n" + "-"*60)
        print('='*60)
        print("Final Results:")
        print("\n * Task:")
        print(f"    {task.description}")
        print("\n * Best Instructions:")
        print(f"    {best_instructions}")
        print("\n * Best Query:")
        print(f"    {query}")
        print("\n * Critic Feedback:")
        print(f"    {critic_feedback}")
        print("\nOracle Result:") 
        print(f"    {oracle_result}")
        print("\n" + '='*60)
        state.best_instructions = best_instructions
        state.best_query = query
        state.critic_feedback = critic_feedback
        state.oracle_result = oracle_result
        return state
    
    async def run_all_tasks(self, verbose=True):
        tasks = self.load_tasks(verbose=verbose)
        best_states = []
        for task in tasks:
            state = await self.run_episode(task=task, verbose=verbose)
            best_states.append(state)
        self.best_states = best_states
        return self.best_states
    
if __name__ == "__main__":
    PADWAN_INSTRUCTIONS = """
You are a tool-using assistant.

MANDATORY CONTRACT
1.  **Never** invent external facts or numbers.  
    • If a value is unknown, you must call an appropriate tool  
      or clearly state “UNKNOWN”.
2.  For every action, emit exactly this pair of lines:

    STEP {n}: <brief reasoning sentence>
    TOOL {n}: <tool_name>(<json_args>) -> <result OR PENDING>

   Keep numbering consecutive.
3.  After your final TOOL line, give a one-paragraph answer
   that references your computed values.
4.  If a required tool is unavailable or fails, output only:

    CANNOT COMPLETE – REASON: <short explanation>

Begin.
"""


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
            agent_instructions=PADWAN_INSTRUCTIONS,
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

        run_all = True
        if run_all:
            best_states = await environment.run_all_tasks(verbose=True)
            print("="*80)
            for state in best_states:
                print(f"\n\nBest state:")
                print(f"Task: {state.task.description}")
                print(f"Best Instructions: {state.best_instructions}")
                print(f"Best Query: {state.best_query}")
                print(f"Critic Feedback: {state.critic_feedback}")
                print(f"Oracle Result: {state.oracle_result}")
                print("="*80)
        else:
            best_state = await environment.run_episode(verbose=True, difficulty=0.8)
            print("="*80)
            print(f"\n\nBest state:")
            print(f"Task: {best_state.task.description}")
            print(f"Best Instructions: {best_state.best_instructions}")
            print(f"Best Query: {best_state.best_query}")
            print(f"Critic Feedback: {best_state.critic_feedback}")
            print(f"Oracle Result: {best_state.oracle_result}")
            print("="*80)
    asyncio.run(main())