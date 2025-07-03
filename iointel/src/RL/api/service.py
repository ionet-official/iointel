import os
import asyncio
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables from creds.env
load_dotenv("creds.env")

from iointel.src.RL.task_manager import TaskManager
from iointel.src.RL.critic import CriticAgent
from iointel.src.RL.oracle import OracleAgent
from iointel.src.RL.training import RLEnvironment
from iointel import Agent
from iointel.src.RL.example_tools import (
    add,
    subtract,
    multiply,
    divide,
    get_weather,
    square_root,
)

from iointel.src.agent_methods.tools.firecrawl import Crawler
from iointel.src.RL.utils import tool_usage_results_to_string
from iointel.src.RL.prompts import get_agent_instructions
from .models import (
    TaskResult,
    CriticFeedback,
    OracleResult,
    OracleDetails,
    TaskDifficulty,
    EvaluationResponse,
)
from .config import settings

logger = logging.getLogger(__name__)

# OpenAI model prefixes
OPENAI_MODELS = ["gpt-4", "gpt-3.5", "o1-preview", "o1-mini", "chatgpt"]


def is_openai_model(model_name):
    """Check if a model is an OpenAI model based on its name"""
    return any(model_name.startswith(prefix) for prefix in OPENAI_MODELS)

firecrawl = Crawler()

class EvaluationService:
    def __init__(self):
        self.tools = [add, subtract, multiply, divide, get_weather, square_root, firecrawl.scrape_url]
        self.active_evaluations = {}
        
    def _linearize_tool_usage(self, tool_usage_results) -> str:
        if not tool_usage_results:
            return ""
        try:
            return tool_usage_results_to_string(tool_usage_results, prefix="")
        except Exception:
            return json.dumps(
                [
                    tur.model_dump() if hasattr(tur, "model_dump") else dict(tur)
                    for tur in tool_usage_results
                ]
            )

    def _linearize_dict(self, d) -> str:
        if d is None:
            return ""
        try:
            return json.dumps(d, ensure_ascii=False)
        except Exception:
            return str(d)

    def _linearize_list(self, lst) -> List[str]:
        if lst is None:
            return []
        return [str(x) for x in lst]

    def _linearize_agent_result(self, agent_result) -> Dict[str, Any]:
        if agent_result is None:
            return {}
        try:
            if isinstance(agent_result, dict):
                return agent_result
            return json.loads(json.dumps(agent_result, ensure_ascii=False))
        except Exception:
            return {"raw": str(agent_result)}

    async def evaluate_single_model(
        self,
        model_name: str,
        num_tasks: int = 3,
        timeout: int = 120,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> List[TaskResult]:
        # Auto-detect API type if not provided
        if not api_key or not base_url:
            if is_openai_model(model_name):
                api_key = api_key or os.getenv("OPENAI_API_KEY")
                base_url = base_url or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
                if not api_key:
                    raise ValueError("API key must be provided or set in OPENAI_API_KEY environment variable")
            else:
                api_key = api_key or os.getenv("IO_API_KEY")
                base_url = base_url or os.getenv("IO_BASE_URL")
                if not api_key:
                    raise ValueError("API key must be provided or set in IO_API_KEY environment variable")
                if not base_url:
                    raise ValueError("Base URL must be provided or set in IO_BASE_URL environment variable")

        logger.info(f"Starting evaluation for model: {model_name}")
        
        critic = CriticAgent(model=model_name, api_key=api_key, base_url=base_url)
        task_manager = TaskManager(model=model_name, api_key=api_key, base_url=base_url)
        oracle = OracleAgent(model=model_name, api_key=api_key, base_url=base_url)
        
        environment = RLEnvironment(
            name=f"RL-{model_name}",
            agent_instructions=get_agent_instructions("default"),
            task_manager=task_manager,
            critic=critic,
            oracle=oracle,
            tools=self.tools,
            max_steps=3,
            agent_class=Agent,
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            needs_model_settings=settings.models_requiring_settings,
        )
        
        environment.load_tasks(verbose=False)
        tasks = environment.task_manager.get_all_tasks()
        
        if num_tasks:
            import random
            tasks = random.sample(tasks, min(num_tasks, len(tasks)))
        
        results = []
        for task in tasks:
            start_time = time.time()
            
            # Convert float difficulty to enum
            difficulty_mapping = {
                lambda x: x < 0.4: TaskDifficulty.EASY,
                lambda x: x < 0.7: TaskDifficulty.MEDIUM,
                lambda x: x >= 0.7: TaskDifficulty.HARD
            }
            task_difficulty = TaskDifficulty.MEDIUM  # default
            for condition, difficulty in difficulty_mapping.items():
                if condition(task.difficulty):
                    task_difficulty = difficulty
                    break
            
            result = TaskResult(
                model=model_name,
                task_id=str(task.id),
                task_description=task.description,
                task_difficulty=task_difficulty,
                task_required_tools=self._linearize_list(task.required_tools),
                task_ground_truth=task.ground_truth or {},
                task_context=task.context or {},
                task_goal_seek=task.goal_seek or "",
            )
            
            try:
                logger.info(f"Running episode for task {task.id} with model {model_name}")
                state = await asyncio.wait_for(
                    environment.run_episode(task=task, verbose=False), 
                    timeout=timeout
                )
                logger.info(f"Episode completed for task {task.id}, state: {type(state)}")
                
                result.step_count = getattr(state, "step_count", 0)
                result.agent_result = self._linearize_agent_result(
                    getattr(state, "agent_result", None)
                )
                
                # Extract tool usage results
                agent_result = getattr(state, "agent_result", None)
                tool_usage_results = None
                if agent_result and isinstance(agent_result, dict):
                    tool_usage_results = agent_result.get("tool_usage_results")
                result.tool_usage_results = self._linearize_tool_usage(tool_usage_results)
                
                result.best_query = getattr(state, "best_query", "")
                result.best_instructions = getattr(state, "best_instructions", "")
                
                # Process critic feedback
                critic_feedback = getattr(state, "critic_feedback", None)
                if critic_feedback:
                    result.critic_feedback = CriticFeedback(
                        score=getattr(critic_feedback, "score", 0.0),
                        better_query=getattr(critic_feedback, "better_query", ""),
                        metrics=getattr(critic_feedback, "metrics", {}) or {},
                        agent_prompt_instructions=getattr(
                            critic_feedback, "agent_prompt_instructions", ""
                        ),
                    )
                
                # Process oracle result
                oracle_result = getattr(state, "oracle_result", None)
                if oracle_result:
                    details = getattr(oracle_result, "details", None)
                    oracle_details = None
                    if details:
                        oracle_details = OracleDetails(
                            matching_fields=details.get("matching_fields", []),
                            missing_fields=details.get("missing_fields", []),
                            incorrect_values=details.get("incorrect_values", {}),
                            additional_insights=details.get("additional_insights", ""),
                        )
                    
                    result.oracle_result = OracleResult(
                        correct=getattr(oracle_result, "correct", False),
                        score=getattr(oracle_result, "score", 0.0),
                        feedback=getattr(oracle_result, "feedback", ""),
                        details=oracle_details,
                    )
                
            except asyncio.TimeoutError:
                result.error = f"Timeout after {timeout}s"
                logger.warning(f"Task {task.id} timed out for model {model_name}")
            except Exception as e:
                result.error = f"Error: {str(e)}"
                logger.error(f"Error evaluating task {task.id} for model {model_name}: {e}", exc_info=True)
            
            result.execution_time = time.time() - start_time
            results.append(result)
            
        logger.info(f"Completed evaluation for model: {model_name}")
        return results

    async def evaluate_models(
        self,
        models: List[str],
        num_tasks: int = 3,
        timeout: int = 120,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> EvaluationResponse:
        all_results = []
        start_time = time.time()
        
        for model in models:
            try:
                model_results = await self.evaluate_single_model(
                    model_name=model,
                    num_tasks=num_tasks,
                    timeout=timeout,
                    api_key=api_key,
                    base_url=base_url
                )
                all_results.extend(model_results)
            except Exception as e:
                logger.error(f"Failed to evaluate model {model}: {e}")
                error_result = TaskResult(
                    model=model,
                    task_id="error",
                    task_description="Evaluation failed",
                    task_difficulty=TaskDifficulty.MEDIUM,
                    error=str(e)
                )
                all_results.append(error_result)
        
        # Calculate summary statistics
        summary = self._calculate_summary(all_results)
        
        return EvaluationResponse(
            status="completed",
            total_models=len(models),
            total_tasks=len(all_results),
            results=all_results,
            summary=summary
        )

    def _calculate_summary(self, results: List[TaskResult]) -> Dict[str, Any]:
        summary = {
            "total_evaluations": len(results),
            "successful_evaluations": sum(1 for r in results if not r.error),
            "failed_evaluations": sum(1 for r in results if r.error),
            "models": {},
            "average_execution_time": 0.0,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Group by model
        model_stats = {}
        for result in results:
            if result.model not in model_stats:
                model_stats[result.model] = {
                    "total_tasks": 0,
                    "successful": 0,
                    "failed": 0,
                    "average_oracle_score": 0.0,
                    "average_critic_score": 0.0,
                    "oracle_correct_count": 0,
                    "execution_times": []
                }
            
            stats = model_stats[result.model]
            stats["total_tasks"] += 1
            
            if result.error:
                stats["failed"] += 1
            else:
                stats["successful"] += 1
                if result.oracle_result:
                    stats["average_oracle_score"] += result.oracle_result.score
                    if result.oracle_result.correct:
                        stats["oracle_correct_count"] += 1
                if result.critic_feedback:
                    stats["average_critic_score"] += result.critic_feedback.score
            
            if result.execution_time:
                stats["execution_times"].append(result.execution_time)
        
        # Calculate averages
        for model, stats in model_stats.items():
            if stats["successful"] > 0:
                stats["average_oracle_score"] /= stats["successful"]
                stats["average_critic_score"] /= stats["successful"]
            stats["average_execution_time"] = (
                sum(stats["execution_times"]) / len(stats["execution_times"])
                if stats["execution_times"] else 0.0
            )
            stats["oracle_accuracy"] = (
                stats["oracle_correct_count"] / stats["successful"]
                if stats["successful"] > 0 else 0.0
            )
            del stats["execution_times"]
        
        summary["models"] = model_stats
        all_times = [r.execution_time for r in results if r.execution_time]
        summary["average_execution_time"] = (
            sum(all_times) / len(all_times) if all_times else 0.0
        )
        
        return summary