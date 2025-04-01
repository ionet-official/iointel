import concurrent.futures
from .runners import run_agents
from typing import Optional, Any, get_type_hints


from ..agent_methods.data_models.datamodels import BaseStage, SimpleStage, SequentialStage, \
    ParallelStage, WhileStage, FallbackStage

_STAGE_RUNNERS = {}

def execute_stage(stage: BaseStage, agents, task_metadata, default_text):
    try:
        runner = _STAGE_RUNNERS[type(stage)]
    except KeyError:
        raise NotImplementedError(f"Stage {stage.__class__} not supported yet")
    return runner(stage, agents, task_metadata, default_text)

def register_stage_runner(func):
    stage = get_type_hints(func).get('stage', None)
    assert stage and issubclass(stage, BaseStage) and stage not in _STAGE_RUNNERS
    _STAGE_RUNNERS[stage] = func
    return func

@register_stage_runner
def _run_simple(stage: SimpleStage, agents, task_metadata, default_text) -> Any:
    # Merge the stage context with a default input.
    merged_context = dict(stage.context)
    if "input" not in merged_context:
        merged_context["input"] = default_text
    # Pass result_stage if provided.
    return run_agents(
        objective=stage.objective,
        agents=stage.agents,
        context=merged_context,
        result_type=stage.result_type
    ).execute()

@register_stage_runner
def _run_sequential(stage: SequentialStage, agents, task_metadata, default_text) -> list:
    results = []
    for substage in stage.stages:
        results.append(execute_stage(substage, agents, task_metadata, default_text))
    return results

@register_stage_runner
def _run_parallel(stage: ParallelStage, agents, task_metadata, default_text) -> list:
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(execute_stage, substage, agents, task_metadata, default_text)
                    for substage in stage.stages]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results

@register_stage_runner
def _run_while(stage: WhileStage, agents, task_metadata, default_text) -> list:
    results = []
    iterations = 0
    while stage.condition() and iterations < stage.max_iterations:
        result = execute_stage(stage.stage, agents, task_metadata, default_text)
        results.append(result)
        iterations += 1
    return results

@register_stage_runner
def _run_fallback(stage: FallbackStage, agents, task_metadata, default_text) -> any:
    try:
        return execute_stage(stage.primary, agents, task_metadata, default_text)
    except Exception as e:
        print(f"Primary stage failed with error: {e}. Running fallback stage.")
        return execute_stage(stage.fallback, agents, task_metadata, default_text)
