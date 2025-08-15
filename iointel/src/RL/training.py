from typing import Dict, List, Optional, Callable
from pydantic import BaseModel
from iointel.src.RL.task_manager import Task, TaskManager
from iointel.src.RL.critic import CriticAgent, CriticFeedback
from iointel.src.RL.oracle import OracleAgent, EvaluationResult
from iointel import Agent, PersonaConfig
from iointel.src.agents import ToolUsageResult
from pydantic_ai.settings import ModelSettings
import asyncio
import os
from dotenv import load_dotenv
# Import our curated working tools filter
from iointel.src.utilities.working_tools_filter import get_curated_tool_list
from iointel.src.utilities.working_tools_filter import filter_available_tools
from iointel.src.utilities.constants import get_model_config
from iointel.src.agent_methods.tools.tool_loader import load_tools_from_env
from iointel.src.utilities.registries import TOOLS_REGISTRY

# Keep example tools for backward compatibility
from iointel.src.RL.example_tools import (
    add,
    subtract,
    multiply,
    divide,
    get_weather,
    square_root,
)
import random

# Models that require special model settings handling
MODELS_THAT_REQUIRE_MODEL_SETTINGS = [
    "deepseek-ai/DeepSeek-R1-0528",
    "meta-llama/Llama-3.3-70B-Instruct",
]


class RLState(BaseModel):
    task: Task
    step_count: int = 0
    agent_result: Optional[Dict[str, str | list[ToolUsageResult]] | None] = None
    best_query: str = ""
    best_instructions: str = ""
    critic_feedback: Optional[CriticFeedback] = None
    oracle_result: Optional[EvaluationResult] = None  # TODO: add oracle result
    done: bool = False


def pprint_state(
    task: Task,
    query: str,
    critic_feedback: CriticFeedback,
    oracle_result: EvaluationResult,
    best_instructions: str,
    state: RLState,
) -> None:
    print(
        f"\n{'=' * 60}\n"
        f"Final Results:\n\n"
        f" * Initial Task:\n"
        f"    {task.description if task else 'No task'}\n\n"
        f" * Best Query (after critic and oracle feedback):\n"
        f"    {query if query else 'No query'}\n"
        f"{f'\n * Critic Feedback Metrics:\n    {critic_feedback.metrics}' if critic_feedback else ''}\n"
        f"{f'\n * Oracle Result:\n    correct:   {oracle_result.correct}\n    score:     {oracle_result.score}\n    feedback:  {oracle_result.feedback}\n    details:   {oracle_result.details}' if oracle_result else ''}\n"
        f"{f'\n * Best Instructions:\n    {best_instructions}' if best_instructions else ''}\n\n"
        f" * Done:\n"
        f"    {state.done if state else False}\n\n"
        f"{'=' * 60}"
    )


class RLEnvironment:
    """Agentic RL environment: orchestrates agent, critic, oracle, and task manager."""

    def __init__(
        self,
        name: str,
        agent_instructions: str,
        task_manager: TaskManager,
        critic: CriticAgent,
        oracle: OracleAgent,
        tools: List[Callable],
        max_steps=10,
        meta_learn_instructions=True,
        persona=None,
        agent_class=None,
        task_file_path="iointel/src/RL/tests/tasks.json",
        model=None,
        api_key=None,
        base_url=None,
        threshold: float = 0.90,
        needs_model_settings: list[str] = [],
    ):
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
        self.needs_model_settings: list[str] = needs_model_settings

    def generate_tasks(self, num_tasks: int = 10, verbose: bool = False) -> List[Task]:
        tasks = self.task_manager.generate_tasks(self.tools, num_tasks)
        self.task_manager.save_tasks(self.task_file_path)
        if verbose:
            print(f"Generated {len(tasks)} tasks and saved to {self.task_file_path}")
            for task in tasks:
                print(f"Task {task.id}: {task.description}")
                print("=" * 60)
        return tasks

    def load_tasks(self, verbose: bool = False) -> List[Task]:
        self.task_manager.load_tasks(self.task_file_path)
        if verbose:
            print(
                f"Loaded {len(self.task_manager.tasks)} tasks from {self.task_file_path}"
            )
            for task in self.task_manager.tasks:
                print(f"Task {task.id}: {task.description}")
                print("=" * 60)
        return self.task_manager.tasks

    def reset(self, task: Task = None, difficulty: Optional[float] = None) -> RLState:
        if task is None:
            task = self.task_manager.get_task(difficulty)
        print("-" * 30)
        print(f"==== Resetting environment with task: {task.description}")
        self.state = RLState(task=task)
        return self.state

    async def run_episode(
        self,
        task: Task = None,
        difficulty: Optional[float] = None,
        verbose=True,
        use_chat_history=False,
    ) -> Optional[RLState]:
        state = self.reset(task=task, difficulty=difficulty)
        if not state.task:
            print("==== No task found ====")
            return None
        task = state.task
        OG_INSTRUCTIONS = self.agent_instructions
        instructions = self.agent_instructions
        critic_feedback = None
        context = ""  # we can also vary this, but for later....
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
                base_url=self.base_url,
                model_settings=ModelSettings(
                    dict(supports_tool_choice_required=True),
                    extra_body={"tool_choice": "auto"},
                )
                if self.model in self.needs_model_settings
                else None,
                show_tool_calls=True,
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
            state.agent_result = result["full_result"]
            state.step_count = step + 1

            # 2. Oracle evaluation
            oracle_result: EvaluationResult = await self.oracle.evaluate(
                agent_response=result.get("result"),
                ground_truth=task.ground_truth,
                task_description=task.description,
                agent_actions=result.get("tool_usage_results", []),
                required_tools=task.required_tools,
            )
            if oracle_result.correct:  # skip the rest of the steps if the task is solved, no more learning is needed
                print("********** Task solved! **********")
                state.done = True
                break
            else:
                print("********** Task not solved! Whomp Whomp **********")
            # get feedback from the oracle
            feedback = (
                oracle_result.feedback
                + "\n\n"
                + oracle_result.details.get("additional_insights", "")
            )

            # 3. Critic steering, using the oracle feedback
            critic_feedback: CriticFeedback = (
                await self.critic.generate_critical_feedback(
                    task=task.description,
                    agent_actions=result.get("tool_usage_results", []),
                    final_response=result.get("result"),
                    feedback=feedback,
                )
            )

            # # 4. Meta-learn: update instructions if critic suggests new ones
            if self.meta_learn_instructions and (
                critic_feedback and critic_feedback.agent_prompt_instructions
            ):
                # Refine the instruction string for the *next* agent instance
                instructions = (
                    f"{OG_INSTRUCTIONS}\n\n{critic_feedback.agent_prompt_instructions}"
                )
                # Keep the original task text so the agent always sees context
                query = f"{critic_feedback.better_query}"
            elif critic_feedback and critic_feedback.better_query:
                # No meta-instruction learning; just pass the improved query
                query = f"{critic_feedback.better_query}"

            # 5. Print/log everything
            if verbose:
                print("\n" + "-" * 60)
                print(f"\n ****** Step {step + 1}: ******")
                pprint_state(
                    task, query, critic_feedback, oracle_result, instructions, state
                )

        pprint_state(task, query, critic_feedback, oracle_result, instructions, state)
        state.best_instructions = instructions
        state.best_query = query
        state.critic_feedback = critic_feedback
        state.oracle_result = oracle_result
        return state

    async def run_all_tasks(self, verbose=True, sample_size: Optional[int] = None):
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
    # Load environment variables from creds.env
    load_dotenv("creds.env")

    PADWAN_INSTRUCTIONS = """
You are a tool-using assistant."""

    async def main():
        # Register our example tools first (they're better than agno calculator tools)
        example_tools = [add, subtract, multiply, divide, get_weather, square_root]
        example_tool_names = ["add", "subtract", "multiply", "divide", "get_weather", "square_root"]
        
        print("📝 Registering example tools...")
        for tool in example_tools:
            # The @register_tool decorator should have already registered them
            # but let's make sure they're in the registry
            if hasattr(tool, '__name__') and tool.__name__ in TOOLS_REGISTRY:
                print(f"  ✓ {tool.__name__} already registered")
            else:
                print(f"  ⚠️ {tool.__name__} not found in registry")
        
        # Load additional working tools from environment
        print("\n🔧 Loading additional working tools...")
        tool_names = load_tools_from_env("creds.env")
        working_tool_names = filter_available_tools(tool_names)
        
        # Remove agno calculator tools if they exist (we prefer our simple ones)
        agno_calculator_tools = [
            "calculator_add", "calculator_subtract", "calculator_multiply", 
            "calculator_divide", "calculator_square_root", "calculator_factorial",
            "calculator_exponentiate", "calculator_is_prime"
        ]
        filtered_working_tools = [t for t in working_tool_names if t not in agno_calculator_tools]
        
        # Combine example tools with filtered working tools
        all_tool_names = example_tool_names + filtered_working_tools
        
        print(f"\n✅ Total tools for RL training: {len(all_tool_names)}")
        print(f"  - Example tools: {example_tool_names}")
        print(f"  - Additional tools: {len(filtered_working_tools)} (filtered from {len(working_tool_names)})")
        print(f"  - First 10 additional tools: {filtered_working_tools[:10]}")
        
        # Use combined tool list
        tools = all_tool_names
        
        # Use centralized model config function
        others = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"  # works
        agent_model = os.getenv("RL_MODEL", "gpt-4o")  # Model for main agent execution
        evaluation_model = "gpt-4o"  # Always use GPT for Oracle/Critic (needs structured output)
        
        agent_config = get_model_config(agent_model)
        eval_config = get_model_config(evaluation_model)
        
        print(f"🤖 Agent model: {agent_config['model']}")
        print(f"🧠 Evaluation model (Oracle/Critic): {eval_config['model']}")
        print(f"🌐 Agent API endpoint: {agent_config['base_url']}")
        
        # Detect if we're using a Llama model for the agent and adjust instructions accordingly
        is_llama_model = "llama" in agent_config["model"].lower()
        
        if is_llama_model:
            print("🦙 Detected Llama agent model - using optimized function calling instructions")
            PADWAN_INSTRUCTIONS = """You are a helpful assistant with tool calling capabilities. 

When you need to use a tool, respond with a JSON function call in this exact format:
{"name": "tool_name", "parameters": {"param1": "value1", "param2": "value2"}}

Available functions will be automatically provided to you. Always use the exact function names and parameter names as specified.

You can perform calculations, check weather information, scrape websites, search the web, query financial data, and more depending on what tools are available.
Always explain your reasoning before using a tool and explain the result after using it."""
        else:
            PADWAN_INSTRUCTIONS = """You are a tool-using assistant with access to various tools.
You can perform calculations, check weather information, scrape websites, search the web, query financial data, and more depending on what tools are available.
Always explain your reasoning after using a tool and provide detailed explanations."""

        # Oracle/Critic always use GPT models (they need structured output with tool_choice='required')
        critic = CriticAgent(
            model=eval_config["model"], 
            api_key=eval_config["api_key"], 
            base_url=eval_config["base_url"]
        )
        task_manager = TaskManager(
            model=eval_config["model"], 
            api_key=eval_config["api_key"], 
            base_url=eval_config["base_url"]
        )
        oracle = OracleAgent(
            model=eval_config["model"], 
            api_key=eval_config["api_key"], 
            base_url=eval_config["base_url"]
        )

        environment = RLEnvironment(
            name="Padawan learner",
            agent_instructions=PADWAN_INSTRUCTIONS,
            task_manager=task_manager,
            critic=critic,
            oracle=oracle,
            tools=tools,
            max_steps=3,
            agent_class=Agent,
            model=agent_config["model"],
            api_key=agent_config["api_key"],
            base_url=agent_config["base_url"],
            needs_model_settings=MODELS_THAT_REQUIRE_MODEL_SETTINGS,
        )
        generate_new = False
        if generate_new:
            environment.generate_tasks(num_tasks=2, verbose=True)
        else:
            environment.load_tasks(verbose=True)

        run_all = True
        if run_all:
            await environment.run_all_tasks(verbose=True, sample_size=10)
        else:
            await environment.run_episode(verbose=True, difficulty=0.8)

    asyncio.run(main())
