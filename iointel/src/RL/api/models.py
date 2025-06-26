from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    models: List[str] = Field(
        ..., 
        description="List of model names to evaluate",
        min_length=1
    )
    num_tasks: Optional[int] = Field(
        default=3,
        description="Number of tasks to evaluate per model",
        ge=1
    )
    timeout: Optional[int] = Field(
        default=120,
        description="Timeout in seconds for each task evaluation",
        ge=10,
        le=600
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for model access (uses IO_API_KEY env var if not provided)"
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Base URL for model API (uses IO_BASE_URL env var if not provided)"
    )


class CriticFeedback(BaseModel):
    score: float = Field(..., description="Critic score for the evaluation")
    better_query: str = Field(..., description="Improved query suggested by critic")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Evaluation metrics")
    agent_prompt_instructions: str = Field(..., description="Instructions for agent prompt")


class OracleDetails(BaseModel):
    matching_fields: List[str] = Field(default_factory=list, description="Fields that match ground truth")
    missing_fields: List[str] = Field(default_factory=list, description="Fields missing from result")
    incorrect_values: Dict[str, Any] = Field(default_factory=dict, description="Fields with incorrect values")
    additional_insights: str = Field(default="", description="Additional insights from oracle")


class OracleResult(BaseModel):
    correct: bool = Field(..., description="Whether the result is correct")
    score: float = Field(..., description="Oracle score")
    feedback: str = Field(..., description="Oracle feedback")
    details: Optional[OracleDetails] = None


class TaskResult(BaseModel):
    model: str = Field(..., description="Model name evaluated")
    task_id: str = Field(..., description="Task identifier")
    task_description: str = Field(..., description="Task description")
    task_difficulty: TaskDifficulty = Field(..., description="Task difficulty level")
    task_required_tools: List[str] = Field(default_factory=list, description="Required tools for task")
    task_ground_truth: Dict[str, Any] = Field(default_factory=dict, description="Expected result")
    task_context: Dict[str, Any] = Field(default_factory=dict, description="Task context")
    task_goal_seek: str = Field(default="", description="Goal seek information")
    
    step_count: int = Field(default=0, description="Number of steps taken")
    agent_result: Optional[Dict[str, Any]] = Field(default=None, description="Agent execution result")
    tool_usage_results: str = Field(default="", description="Linearized tool usage results")
    best_query: str = Field(default="", description="Best query found")
    best_instructions: str = Field(default="", description="Best instructions found")
    
    critic_feedback: Optional[CriticFeedback] = None
    oracle_result: Optional[OracleResult] = None
    
    error: Optional[str] = Field(default=None, description="Error message if evaluation failed")
    execution_time: Optional[float] = Field(default=None, description="Execution time in seconds")


class EvaluationResponse(BaseModel):
    status: str = Field(..., description="Evaluation status")
    total_models: int = Field(..., description="Total number of models evaluated")
    total_tasks: int = Field(..., description="Total number of tasks evaluated")
    results: List[TaskResult] = Field(..., description="Detailed results for each task")
    summary: Dict[str, Any] = Field(default_factory=dict, description="Summary statistics")


class EvaluationStatus(BaseModel):
    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Current status of the evaluation")
    models_completed: int = Field(default=0, description="Number of models completed")
    total_models: int = Field(..., description="Total number of models to evaluate")
    current_model: Optional[str] = Field(default=None, description="Currently evaluating model")
    estimated_time_remaining: Optional[float] = Field(default=None, description="Estimated time remaining in seconds")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")