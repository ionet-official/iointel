from typing import Any, Dict, List, Optional, Callable
from pydantic import BaseModel
import random
import inspect
from iointel import Agent
import asyncio
import os
from iointel.src.RL.example_tools import add, subtract, multiply, divide, get_weather, square_root
from dotenv import load_dotenv
import json
load_dotenv(os.path.join(os.path.dirname(__file__), '../../..', 'creds.env'))

print("API KEY:", os.getenv("OPENAI_API_KEY")[:20], "..." if os.getenv("OPENAI_API_KEY") else "NOT FOUND")
print("BASE URL:", os.getenv("OPENAI_API_BASE")[:20], "..." if os.getenv("OPENAI_API_BASE") else "NOT FOUND")

class Task(BaseModel):
    """Represents a task for the agent to solve"""
    id: int
    description: str
    ground_truth: Dict[str, Any]
    required_tools: List[str]
    difficulty: float  # 0.0 to 1.0
    context: Optional[Dict[str, Any]] = None
    goal_seek: Optional[str] = None

class TaskGeneratorAgent:
    def __init__(self, model:str, api_key:str=None, base_url:str=None, verbose:bool=True):
        self.Task = Task
        self.agent = Agent(
            name="TaskGenerator",
            instructions="""
            You are a task generation agent. Given a set of tool definitions (name, docstring, parameters), 
            generate a list of diverse and interesting tasks as Pydantic Task objects. 
            Each task should specify a ordered numeric id, description, ground_truth, required_tools, and difficulty (between 0.0 and 1.0).
            if during the task generation, you need to specify a goal seek outcome, include it here, otherwise leave it blank.

            If it is difficult to generate a known ground truth (imagine the roots of a quintic polynomial), you can estimate it and its likelihood, such as ranges, or expected values, or probabilities.

            Output only valid JSON that can be parsed into the Task Pydantic model.
            """,
            model=model,
            api_key=api_key,
            base_url=base_url,
            output_type=List[Task]
        )
        self.verbose = verbose
        
    async def generate_tasks(self, tools: List[Callable], num_tasks: int = 5, context: str = None, goal_seek: bool = False) -> List[Task]:
        tool_descriptions = []
        for tool in tools:
            sig = str(inspect.signature(tool))
            doc = tool.__doc__ or ""
            tool_descriptions.append(f"{tool.__name__}{sig}: {doc}")
        prompt = f"""Tools available:\n{chr(10).join(tool_descriptions)}\n\nContext: {context or 'None'}\n\nPlease generate {num_tasks} diverse Task objects as JSON.
        """
        if goal_seek:
            prompt += f"""  
            Generate a goal seek outcome for each task.
            """
        if self.verbose:
            print(f"TaskGenerator prompt:\n{prompt}")
        res = await self.agent.run(prompt)
        return res['result']#.output

class TaskManager:
    """Manages task generation and evaluation"""
    
    def __init__(self, model="gpt-4o", api_key=None, base_url=None):
        self.tasks: List[Task] = []
        self.generator = TaskGeneratorAgent(model=model, api_key=api_key, base_url=base_url)
        
    def get_task(self, difficulty: Optional[float] = None) -> Task:
        """Get a random task, optionally filtered by difficulty"""
        ## add random seed
        random.seed(random.randint(0, 1000000))

        if difficulty is not None:
            filtered_tasks = [t for t in self.tasks if abs(t.difficulty - difficulty) < 0.2]
            if filtered_tasks:
                return random.choice(filtered_tasks)
        return random.choice(self.tasks)
    
    def get_task_by_id(self, task_id: int) -> Optional[Task]:
        """Get a specific task by ID"""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def add_task(self, task: Task):
        """Add a new task to the manager"""
        self.tasks.append(task)
    
    def remove_task(self, task_id: int):
        """Remove a task by ID"""
        self.tasks = [t for t in self.tasks if t.id != task_id]
    
    def get_all_tasks(self) -> List[Task]:
        """Get all available tasks"""
        return self.tasks.copy()
    
    def save_tasks(self, file_path: str):
        """Save tasks to a file"""
        with open(file_path, 'w') as f:
            json.dump([t.model_dump() for t in self.tasks], f)
    
    def load_tasks(self, file_path: str):
        """Load tasks from a file"""
        with open(file_path, 'r') as f:
            self.tasks = [Task(**t) for t in json.load(f)]
    
    def get_tasks_by_difficulty(self, min_difficulty: float, max_difficulty: float) -> List[Task]:
        """Get tasks within a difficulty range"""
        return [
            t for t in self.tasks 
            if min_difficulty <= t.difficulty <= max_difficulty
        ] 
    
    async def generate_tasks(self, tools: List[Callable], num_tasks: int = 5, context: str = None, goal_seek: bool = False) -> List[Task]:
        tasks = await self.generator.generate_tasks(tools, num_tasks, context, goal_seek)
        self.tasks.extend(tasks)
        return tasks
    

# add a if n
if __name__ == "__main__":
    async def main():
        task_manager = TaskManager(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))
        tools = [add, subtract, multiply, divide, get_weather, square_root]
        
        generate_new = False
        if generate_new:
            tasks = await task_manager.generate_tasks(tools, num_tasks=10, context="more difficult tasks can mix weather and arithmetic, and inputs should be floats etc. Must generate a few with difficult ground truths and difficulty>0.8.", goal_seek=False)
            task_manager.save_tasks("tasks.json")
            print("Generated new tasks:")
        else:
            print("Loading saved tasks:")
            task_manager.load_tasks("tasks.json")
            tasks = task_manager.get_all_tasks()
        
        print("Tasks:")
        print("-" * 50)
        for task in tasks:
            print('Task(id={},\n    description={},\n    ground_truth={},\n    required_tools={},\n    difficulty={},\n    context={})'.format(task.id, task.description, task.ground_truth, task.required_tools, task.difficulty, task.context))
            print("-" * 50)
        # Test task manager methods with clear output labels
        print("\n=== Testing TaskManager Methods ===")
        
        print("\n[1] Getting random task with difficulty ~0.5:")
        print("-" * 50)
        print(task_manager.get_task(difficulty=0.5))
        
        print("\n[2] Getting task with ID '1':")
        print("-" * 50) 
        print(task_manager.get_task_by_id(1))
        
        print("\n[3] Getting all available tasks:")
        print("-" * 50)
        print(task_manager.get_all_tasks())

        print("\nEasy tasks (0.0-0.2):")
        print(task_manager.get_tasks_by_difficulty(0.0, 0.2))

        print("\nMedium difficulty (0.5-0.7):")
        print(task_manager.get_tasks_by_difficulty(0.5, 0.7))
        
        print("\nHard tasks (0.8-1.0):")
        print(task_manager.get_tasks_by_difficulty(0.8, 1.0))
            
    asyncio.run(main())