import concurrent.futures
from .runners import run_agents
from typing import Optional, Any

class BaseStage:
    def run(self, agents, task_metadata, default_text) -> any:
        raise NotImplementedError("Subclasses must implement this method.")

class SimpleStage(BaseStage):
    def __init__(self, objective: str, context: Optional[dict] = None, result_type: Optional[Any] = None):
        self.objective = objective
        self.context = context or {}
        self.result_type = result_type

    def run(self, agents, task_metadata, default_text) -> Any:
        # Merge the stage context with a default input.
        merged_context = dict(self.context)
        if "input" not in merged_context:
            merged_context["input"] = default_text
        # Pass result_type if provided.
        return run_agents(
            objective=self.objective,
            agents=agents,
            context=merged_context,
            result_type=self.result_type
        ).execute()
class SequentialStage(BaseStage):
    def __init__(self, stages: list):
        self.stages = stages

    def run(self, agents, task_metadata, default_text) -> list:
        results = []
        for stage in self.stages:
            result = stage.run(agents, task_metadata, default_text)
            results.append(result)
        return results

class ParallelStage(BaseStage):
    def __init__(self, stages: list):
        self.stages = stages

    def run(self, agents, task_metadata, default_text) -> list:
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(stage.run, agents, task_metadata, default_text)
                       for stage in self.stages]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        return results

class WhileStage(BaseStage):
    def __init__(self, condition: callable, stage: BaseStage, max_iterations: int = 100):
        """
        :param condition: A callable that returns True if the loop should continue.
        :param stage: The stage to execute repeatedly.
        :param max_iterations: Safety limit to prevent infinite loops.
        """
        self.condition = condition
        self.stage = stage
        self.max_iterations = max_iterations

    def run(self, agents, task_metadata, default_text) -> list:
        results = []
        iterations = 0
        while self.condition() and iterations < self.max_iterations:
            result = self.stage.run(agents, task_metadata, default_text)
            results.append(result)
            iterations += 1
        return results

class FallbackStage(BaseStage):
    def __init__(self, primary: BaseStage, fallback: BaseStage):
        """
        :param primary: The primary stage to attempt.
        :param fallback: The fallback stage to execute if the primary fails.
        """
        self.primary = primary
        self.fallback = fallback

    def run(self, agents, task_metadata, default_text) -> any:
        try:
            return self.primary.run(agents, task_metadata, default_text)
        except Exception as e:
            print(f"Primary stage failed with error: {e}. Running fallback stage.")
            return self.fallback.run(agents, task_metadata, default_text)