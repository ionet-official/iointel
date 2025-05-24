from typing import Any, Dict, List, Optional
from pydantic import BaseModel
import random

class Task(BaseModel):
    """Represents a task for the agent to solve"""
    id: str
    description: str
    ground_truth: Dict[str, Any]
    required_tools: List[str]
    difficulty: float  # 0.0 to 1.0
    context: Optional[Dict[str, Any]] = None

class TaskManager:
    """Manages task generation and evaluation"""
    
    def __init__(self):
        self.tasks: List[Task] = []
        self._initialize_tasks()
    
    def _initialize_tasks(self):
        """Initialize a set of sample tasks"""
        self.tasks = [
            Task(
                id="task_001",
                description="Investigate suspicious login activity from IP 192.168.1.100",
                ground_truth={
                    "source_ip": "192.168.1.100",
                    "event_type": "login",
                    "user": "admin",
                    "timestamp": "2024-03-20T10:00:00Z"
                },
                required_tools=["splunk_query"],
                difficulty=0.3
            ),
            Task(
                id="task_002",
                description="Find all failed login attempts in the last hour",
                ground_truth={
                    "event_type": "failed_login",
                    "count": 5,
                    "time_range": "-1h to now"
                },
                required_tools=["splunk_query"],
                difficulty=0.5
            ),
            Task(
                id="task_003",
                description="Investigate potential data exfiltration by user 'admin'",
                ground_truth={
                    "user": "admin",
                    "event_type": "data_transfer",
                    "destination": "external_server",
                    "data_size": "1.2GB"
                },
                required_tools=["splunk_query"],
                difficulty=0.8
            )
        ]
    
    def get_task(self, difficulty: Optional[float] = None) -> Task:
        """Get a random task, optionally filtered by difficulty"""
        if difficulty is not None:
            filtered_tasks = [t for t in self.tasks if abs(t.difficulty - difficulty) < 0.2]
            if filtered_tasks:
                return random.choice(filtered_tasks)
        return random.choice(self.tasks)
    
    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """Get a specific task by ID"""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def add_task(self, task: Task):
        """Add a new task to the manager"""
        self.tasks.append(task)
    
    def remove_task(self, task_id: str):
        """Remove a task by ID"""
        self.tasks = [t for t in self.tasks if t.id != task_id]
    
    def get_all_tasks(self) -> List[Task]:
        """Get all available tasks"""
        return self.tasks.copy()
    
    def get_tasks_by_difficulty(self, min_difficulty: float, max_difficulty: float) -> List[Task]:
        """Get tasks within a difficulty range"""
        return [
            t for t in self.tasks 
            if min_difficulty <= t.difficulty <= max_difficulty
        ] 