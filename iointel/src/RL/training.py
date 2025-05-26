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
from iointel.src.RL.example_tools import add, subtract, multiply, divide, get_weather, square_root
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

def pprint_state(task: Task, query: str, critic_feedback: CriticFeedback, oracle_result: EvaluationResult, best_instructions: str, state: RLState) -> None:
        print("\n" + "-"*60)
        print('='*60)
        print("Final Results:")
        
        print("\n * Initial Task:")
        print(f"    {task.description if task else 'No task'}")
        
        print("\n * Best Query (after critic and oracle feedback):")
        print(f"    {query if query else 'No query'}")
        
        if critic_feedback:
            print("\n * Critic Feedback Metrics:")
            print(f"    {critic_feedback.metrics}")
        
        if oracle_result:   
            print("\n * Oracle Result:")

            print(f"    correct:   {oracle_result.correct}")
            print(f"    score:     {oracle_result.score}")
            print(f"    feedback:  {oracle_result.feedback}")
            print(f"    details:   {oracle_result.details}")
   
        if best_instructions:
            print("\n * Best Instructions:")
            print(f"    {best_instructions}")
        
        print("\n * Done:")
        print(f"    {state.done if state else False}")
        
        print("\n" + '='*60)


class RLEnvironment:
    """Agentic RL environment: orchestrates agent, critic, oracle, and task manager."""
    def __init__(self, name: str, agent_instructions: str, task_manager: TaskManager, critic: CriticAgent, oracle: OracleAgent, tools: List[Callable], max_steps=10, meta_learn_instructions=True, persona=None, agent_class=None, task_file_path="tasks.json", model=None, api_key=None, base_url=None, threshold: float = 0.90):
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
        if task is None:
            task = self.task_manager.get_task(difficulty)
        print('-'*30)
        print(f"==== Resetting environment with task: {task.description}")
        self.state = RLState(task=task)
        return self.state

    async def run_episode(self, task: Task = None, difficulty: Optional[float] = None, verbose=True, use_chat_history=False) -> Optional[RLState]:
        state = self.reset(task=task, difficulty=difficulty)
        if not state.task:
            print("==== No task found ====")
            return None
        task = state.task
        OG_INSTRUCTIONS = self.agent_instructions
        instructions = self.agent_instructions
        critic_feedback = None
        context = '' # we can also vary this, but for later....
        query = task.description
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

            #########################################################
            if use_chat_history:
                conversation_id = f"{task.id}-{task.description}-training-episode"
                print(f"=== Using chat history for task {task.id}")
                result = await agent.run(query, conversation_id=conversation_id)
            else:
                print(f"=== Not using chat history for task {task.id}")
                result = await agent.run(query)
            ##########################The Agent Learns###############################
            state.agent_result = result['full_result']
            state.step_count = step + 1

            # 2. Oracle evaluation
            oracle_result: EvaluationResult = await self.oracle.evaluate(
                agent_response=result.get("result"),
                ground_truth=task.ground_truth,
                task_description=task.description,
                agent_actions=result.get("tool_usage_results", []),
                required_tools=task.required_tools
            )
            if oracle_result.correct: # skip the rest of the steps if the task is solved, no more learning is needed
                print("********** Task solved! **********")
                state.done = True
                break
            # get feedback from the oracle
            feedback = oracle_result.feedback + "\n\n" + oracle_result.details.get("additional_insights", "")

            # 3. Critic steering, using the oracle feedback
            critic_feedback: CriticFeedback = await self.critic.generate_critical_feedback(
                task=task.description,
                agent_actions=result.get("tool_usage_results", []),
                final_response=result.get("result"),
                feedback=feedback
            )

            # # 4. Meta-learn: update instructions if critic suggests new ones
            if self.meta_learn_instructions and (critic_feedback and critic_feedback.agent_prompt_instructions):
                # Refine the instruction string for the *next* agent instance
                instructions = f"{OG_INSTRUCTIONS}\n\n{critic_feedback.agent_prompt_instructions}"
                # Keep the original task text so the agent always sees context
                query = f"{critic_feedback.better_query}"
            elif critic_feedback and critic_feedback.better_query:
                # No meta-instruction learning; just pass the improved query
                query = f"{critic_feedback.better_query}"

            # 5. Print/log everything
            if verbose:
                print("\n" + "-"*60)
                print(f"\n ****** Step {step+1}: ******")
                pprint_state(task, query, critic_feedback, oracle_result, instructions, state)
        
        pprint_state(task, query, critic_feedback, oracle_result, instructions, state)
        state.best_instructions = instructions
        state.best_query = query
        state.critic_feedback = critic_feedback
        state.oracle_result = oracle_result
        return state
    
    async def run_all_tasks(self, verbose=True, sample_size: Optional[int]=None):
        tasks = self.load_tasks(verbose=verbose)
        best_states = []
        if sample_size:
            tasks = random.sample(tasks, sample_size)
        for task in tasks:
            state = await self.run_episode(task=task, verbose=verbose)
            best_states.append(state)
        self.best_states = best_states
        return self.best_states
    
if __name__ == "__main__":
    PADWAN_INSTRUCTIONS = """
You are a tool-using assistant."""

    async def main():
        tools = [add, subtract, multiply, divide, get_weather, square_root]
        model = "gpt-4o"
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE")

        critic = CriticAgent(model=model, api_key=api_key, base_url=base_url)
        task_manager = TaskManager(model=model, api_key=api_key, base_url=base_url)
        oracle = OracleAgent(model=model, api_key=api_key, base_url=base_url)

        environment = RLEnvironment(
            name="Padawan learner",
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
            await environment.run_all_tasks(verbose=True, sample_size=3)
        else:
            await environment.run_episode(verbose=True, difficulty=0.8)
    asyncio.run(main())