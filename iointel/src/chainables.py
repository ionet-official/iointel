from typing import List, Optional, Dict
import logging

# from .task import CUSTOM_WORKFLOW_REGISTRY
from .utilities.runners import run_agents
from .utilities.decorators import register_custom_task
from .utilities.registries import CHAINABLE_METHODS, CUSTOM_WORKFLOW_REGISTRY
# Tool usage enforcement now handled at DAG level via node_execution_wrapper
from .agents import Agent
from .agent_methods.agents.agents_factory import create_agent
from .agent_methods.data_models.datamodels import AgentParams, AgentResultFormat
from .agent_methods.data_models.execution_models import AgentExecutionResult, AgentRunResponse, ExecutionStatus
import time

logger = logging.getLogger(__name__)

##############################################
# Example Executor Functions
##############################################
"""
The executor functions below are examples of how to implement custom tasks.
These functions are registered with the @register_custom_task decorator.
The decorator takes a string argument that is the name of the custom task.
The function should take the following arguments:
    - task_metadata: A dictionary of metadata for the task. This can include any additional information needed for the task.
    - objective: The text to process. This is the input to the task.
    - agents: A list of agents to use for the task. These agents can be used to run sub-tasks.
    - execution_metadata: A dictionary of metadata for the execution. This can include any additional information needed for the execution like client mode, etc.

"""


@register_custom_task("schedule_reminder")
def execute_schedule_reminder(
    task_metadata: dict, objective: str, agents: List[Agent], execution_metadata: dict
):
    from ..client.client import schedule_task

    client_mode = execution_metadata.get("client_mode", False)
    if client_mode:
        return schedule_task(command=objective)
    else:
        response = run_agents(
            objective="Schedule a reminder",
            instructions="Schedule a reminder and track the time.",
            agents=agents,
            context={"command": objective},
            output_type=str,
        )
        return response.execute()


@register_custom_task("solve_with_reasoning")
async def execute_solve_with_reasoning(
    task_metadata: dict, objective: str, agents: List[Agent], execution_metadata: dict
):
    from .agent_methods.prompts.instructions import REASONING_INSTRUCTIONS
    from .agent_methods.data_models.datamodels import ReasoningStep

    client_mode = execution_metadata.get("client_mode", False)
    if client_mode:
        from ..client.client import run_reasoning_task

        return run_reasoning_task(objective)
    else:
        # For example, loop until a validated solution is found.
        while True:
            response: ReasoningStep = await run_agents(
                objective=REASONING_INSTRUCTIONS,
                output_type=ReasoningStep,
                agents=agents,
                context={"goal": objective},
            ).execute()
            if response.found_validated_solution:
                # Optionally, double-check the solution.
                if run_agents(
                    objective="""
                            Check your solution to be absolutely sure that it is correct and meets all requirements of the goal. Return True if it does.
                            """,
                    output_type=bool,
                    context={"goal": objective},
                    agents=agents,
                ).execute():
                    return response.proposed_solution


@register_custom_task("summarize_text")
def execute_summarize_text(
    task_metadata: dict, objective: str, agents: List[Agent], execution_metadata: dict
):
    from ..client.client import summarize_task
    from .agent_methods.data_models.datamodels import SummaryResult

    max_words = task_metadata.get("max_words")
    client_mode = execution_metadata.get("client_mode", False)
    if client_mode:
        return summarize_task(text=objective, max_words=max_words)
    else:
        summary = run_agents(
            objective=f"Summarize the given text: {objective}\n into no more than {max_words} words and list key points",
            output_type=SummaryResult,
            # context={"text": text},
            agents=agents,
        )
        return summary.execute()


@register_custom_task("sentiment")
async def execute_sentiment(
    task_metadata: dict, objective: str, agents: List[Agent], execution_metadata: dict
):
    from ..client.client import sentiment_analysis

    client_mode = execution_metadata.get("client_mode", False)

    if client_mode:
        return sentiment_analysis(text=objective)
    else:
        result = await run_agents(
            objective=f"Classify the sentiment of the text as a value between 0 and 1.\nText: {objective}",
            agents=agents,
            output_type=float,
            # result_validator=between(0, 1),
            # context={"text": text},
        ).execute()
        # Extract the actual result value from the response dict
        sentiment_val = result.get("result", result) if isinstance(result, dict) else result
        if not isinstance(sentiment_val, float):
            try:
                return float(sentiment_val)
            except ValueError:
                pass
        return sentiment_val


@register_custom_task("extract_categorized_entities")
def execute_extract_entities(
    task_metadata: dict, objective: str, agents: List[Agent], execution_metadata: dict
):
    from ..client.client import extract_entities

    client_mode = execution_metadata.get("client_mode", False)
    if client_mode:
        return extract_entities(text=objective)
    else:
        extracted = run_agents(
            objective=f"""from this text: {objective}

                        Extract named entities from the text above and categorize them,
                            Return a JSON dictionary with the following keys:
                            - 'persons': List of person names
                            - 'organizations': List of organization names
                            - 'locations': List of location names
                            - 'dates': List of date references
                            - 'events': List of event names
                            Only include keys if entities of that type are found in the text.
                            """,
            agents=agents,
            output_type=Dict[str, List[str]],
            # context={"text": text},
        )
        return extracted.execute()


@register_custom_task("translate_text")
def execute_translate_text(
    task_metadata: dict, objective: str, agents: List[Agent], execution_metadata: dict
):
    target_lang = task_metadata["target_language"]
    from ..client.client import translate_text_task

    client_mode = execution_metadata.get("client_mode", False)
    if client_mode:
        return translate_text_task(text=objective, target_language=target_lang)
    else:
        translated = run_agents(
            objective=f"Translate the given text:{objective} into {target_lang}",
            # output_type=TranslationResult,
            # context={"text": text, "target_language": target_lang},
            agents=agents,
        )
        result = translated.execute()
        # Assuming the model has an attribute 'translated'
        return result


@register_custom_task("classify")
def execute_classify(
    task_metadata: dict, objective: str, agents: List[Agent], execution_metadata: dict
):
    from ..client.client import classify_text

    client_mode = execution_metadata.get("client_mode", False)
    classify_by = task_metadata.get("classify_by")

    if client_mode:
        return classify_text(text=objective, classify_by=classify_by)
    else:
        classification = run_agents(
            objective=f"""Take this text: {objective}

            Classify it into the appropriate category.
            Category must be one of: {", ".join(classify_by)}.
            Return only the determined category, omit the thoughts.""",
            agents=agents,
            output_type=str,
            # context={"text": text},
        )
        return classification.execute()


@register_custom_task("moderation")
async def execute_moderation(
    task_metadata: dict, objective: str, agents: List[Agent], execution_metadata: dict
):
    from .agent_methods.data_models.datamodels import (
        ViolationActivation,
        ModerationException,
    )
    from ..client.client import moderation_task

    client_mode = execution_metadata.get("client_mode", False)
    threshold = task_metadata["threshold"]

    if client_mode:
        result = moderation_task(text=objective, threshold=threshold)
        # Raise exceptions based on result thresholds if necessary.
        return result
    else:
        result: ViolationActivation = await run_agents(
            objective=f" from the text: {objective}:\n Check the text for violations and return activation levels",
            agents=agents,
            output_type=ViolationActivation,
            # context={"text": text},
        ).execute()

        if result["extreme_profanity"] > threshold:
            raise ModerationException("Extreme profanity detected", violations=result)
        elif result["sexually_explicit"] > threshold:
            raise ModerationException(
                "Sexually explicit content detected", violations=result
            )
        elif result["hate_speech"] > threshold:
            raise ModerationException("Hate speech detected", violations=result)
        elif result["harassment"] > threshold:
            raise ModerationException("Harassment detected", violations=result)
        elif result["self_harm"] > threshold:
            raise ModerationException("Self harm detected", violations=result)
        elif result["dangerous_content"] > threshold:
            raise ModerationException("Dangeme profanity detected", violations=result)

        return result


@register_custom_task("custom")
def execute_custom(
    task_metadata: dict, objective: str, agents: List[Agent], execution_metadata: dict
):
    client_mode = execution_metadata.get("client_mode", False)
    name = task_metadata["name"]

    if name in CUSTOM_WORKFLOW_REGISTRY:
        custom_fn = CUSTOM_WORKFLOW_REGISTRY[name]
        result = custom_fn(task_metadata, run_agents, objective)
        if hasattr(result, "execute") and callable(result.execute):
            result = result.execute()
        return result
    else:
        if client_mode:
            from ..client.client import custom_workflow

            return custom_workflow(
                name=name,
                objective=objective,
                agents=agents,
                context=task_metadata.get("kwargs", {}),
            )
        else:
            response = run_agents(
                name=name,
                objective=objective,
                agents=agents,
                context=task_metadata.get("kwargs", {}),
                output_type=str,
            )
            return response.execute()


@register_custom_task("agent")  # Core agent executor
async def execute_agent_task(
    task_metadata: dict, objective: str, agents: List[Agent], execution_metadata: dict
) -> "AgentExecutionResult":
    """
    Generic agent task executor that handles type='agent' tasks from WorkflowSpec.
    This executor:
    1. Converts AgentParams to Agent instances using agents_factory
    2. Uses agent_instructions from task_metadata as the primary objective
    3. Passes available results from previous tasks as context to the agent
    4. Combines agent instructions with available data in the objective
    5. Executes using the standard run_agents function
    6. Follows the same pattern as other chainable tasks
    """
    execution_metadata.get("client_mode", False)
    agent_instructions = task_metadata.get("agent_instructions", "")
    
    # Determine result format for workflow/agent chaining
    agent_result_format_str = execution_metadata.get("agent_result_format", "full")
    
    # CRITICAL FIX: Decision agents need "workflow" format to include tool_usage_results
    # The DAG executor requires tool_usage_results to find conditional_gate routing info
    node_type = task_metadata.get("node_type", task_metadata.get("type", ""))
    if node_type == "decision" and agent_result_format_str == "full":
        agent_result_format_str = "workflow"
        print(f"🔧 execute_agent_task: Override decision agent to workflow format for routing")
    
    # print(f"🔧 execute_agent_task: agent_result_format = {agent_result_format_str}")
    
    # Convert string format to AgentResultFormat instance
    if agent_result_format_str == "chat":
        result_format = AgentResultFormat.chat()
    elif agent_result_format_str == "chat_w_tools":
        result_format = AgentResultFormat.chat_w_tools()
    elif agent_result_format_str == "workflow":
        result_format = AgentResultFormat.workflow()
    elif agent_result_format_str == "minimal":
        # Legacy support - map to workflow format
        result_format = AgentResultFormat.workflow()
    else:
        # Default to full format
        result_format = AgentResultFormat.full()
    
    # print(f"🔧 execute_agent_task: using format with fields = {result_format.get_included_fields()}")
    
    # Convert AgentParams to Agent instances if needed
    if not agents:
        # This should never happen for properly constructed agent nodes
        # The workflow_converter ALWAYS provides AgentParams for agent nodes
        raise ValueError(
            f"No agent configuration provided for agent node. "
            f"Task metadata: {list(task_metadata.keys())}. "
            f"This indicates a workflow construction error."
        )
    
    if isinstance(agents[0], AgentParams):
        # Convert AgentParams to Agent instances using the factory
        # This properly handles model, tools, instructions, and all other fields
        print(f"🔧 execute_agent_task: Converting {len(agents)} AgentParams to Agent instances")
        agents_to_use = [create_agent(ap) for ap in agents]
        # Log what was created
        for i, agent in enumerate(agents_to_use):
            print(f"========== Agent {i}: model={getattr(agent.model, 'model_name', 'unknown')}, tools={len(agent.tools)}, name={agent.name}")
    else:
        # Already Agent instances
        agents_to_use = agents
    context = task_metadata.get("kwargs", {}).copy()
    available_results = task_metadata.get("available_results", {})
    
    # Extract actual result values from the result structure
    processed_results = {}
    for key, value in available_results.items():
        if isinstance(value, dict) and 'result' in value:
            # Extract the actual result value
            processed_results[key] = value['result']
        else:
            processed_results[key] = value
    
    # Add available results to context so agent can access them
    if processed_results:
        context["available_results"] = processed_results
        # Also add individual results for easy access
        context.update(processed_results)
    
    # Build task objective that includes both instructions and data context
    # Check which inputs this node expects (from edges/ports)
    expected_inputs = task_metadata.get("ports", {}).get("inputs", [])
    
    # CRITICAL: For agents that expect input, check if we have direct input data
    # This respects the edge topology - only use inputs we're actually connected to
    primary_input_value = None
    if expected_inputs:
        # Look for the first expected input in our results
        for input_name in expected_inputs:
            # Check if any result key matches this input
            for key, value in processed_results.items():
                # Direct match or the key is the expected input source
                if key == input_name or (input_name == "user_message" and key.startswith("user_input")):
                    primary_input_value = value
                    logger.info(f"🎯 Found expected input '{input_name}' from '{key}': {primary_input_value}")
                    break
            if primary_input_value:
                break
    
    # Determine task objective
    if primary_input_value:
        # Use the primary input as the objective
        task_objective = primary_input_value
    elif agent_instructions:
        if processed_results:
            # Include information about available data in the objective
            available_data_info = ", ".join([f"{k}: {v}" for k, v in processed_results.items()])
            task_objective = f"{agent_instructions}\n\nAvailable data from previous tasks:\n{available_data_info}"
        else:
            task_objective = agent_instructions
    else:
        task_objective = objective
    
    # Agent execution - SLA enforcement handled at DAG level
    agent_name = agents_to_use[0].name if agents_to_use else "unknown"
    
    # Extract conversation_id from task_metadata for proper conversation continuity
    conversation_id = task_metadata.get("conversation_id")
    print(f"🔧 execute_agent_task: Using conversation_id = {conversation_id}")
    
    # Fallback to execution_metadata if not found in task_metadata
    if not conversation_id:
        conversation_id = execution_metadata.get("conversation_id")
        if conversation_id:
            print(f"🔧 execute_agent_task: Using conversation_id from execution_metadata = {conversation_id}")
    
    # Define the agent execution function for enforcement wrapping
    async def execute_agent():
        return await run_agents(
            objective=task_objective,
            agents=agents_to_use,
            context=context,
            conversation_id=conversation_id,
            output_type=str,
            result_format=result_format,
        ).execute()
    
    # Execute the agent normally - SLA enforcement now handled at DAG level
    print(f"ℹ️ Executing agent '{agent_name}' (SLA enforcement handled at DAG level)")
    start_time = time.time()
    
    try:
        response = await execute_agent()
        
        # Convert dict response to typed AgentRunResponse
        typed_response = AgentRunResponse.from_dict(response)
        
        # Return typed execution result
        return AgentExecutionResult(
            status=ExecutionStatus.COMPLETED,
            agent_response=typed_response,
            execution_time=time.time() - start_time,
            node_id=task_metadata.get("node_id") or task_metadata.get("task_id")
        )
        
    except Exception as e:
        # Return typed error result
        return AgentExecutionResult(
            status=ExecutionStatus.FAILED,
            error=str(e),
            execution_time=time.time() - start_time,
            node_id=task_metadata.get("node_id") or task_metadata.get("task_id")
        )


@register_custom_task("data_source")  # Core data_source executor
async def execute_data_source_task(task_metadata, objective, agents, execution_metadata) -> "DataSourceResult":
    """Clean data source executor using standardized Pydantic models."""
    from .agent_methods.data_sources import get_data_source
    from .agent_methods.data_sources.models import DataSourceRequest, DataSourceResponse
    from .agent_methods.data_models.execution_models import DataSourceResult, ExecutionStatus
    import time
    import logging
    
    logger = logging.getLogger("iointel.chainables.data_source_executor")
    start_time = time.time()
    
    # Get source_name (new) or tool_name (legacy compatibility)
    source_name = task_metadata.get("source_name") or task_metadata.get("tool_name")
    config = task_metadata.get("config", {})
    
    logger.info(f"[CHAINABLES] Executing data source: {source_name}")
    logger.debug(f"    Config: {config}")
    
    try:
        # Get data source implementation
        data_source_func = get_data_source(source_name)
        
        # Create standardized request
        # Only include fields that exist in DataSourceRequest model
        request_kwargs = {
            "message": config.get("message", "")
        }
        
        # Only add optional fields if they have values
        if "default_value" in config and config["default_value"] is not None:
            request_kwargs["default_value"] = config["default_value"]
            
        request = DataSourceRequest(**request_kwargs)
        
        # Execute data source
        response: DataSourceResponse = data_source_func(request, execution_metadata=execution_metadata)
        
        logger.info(f"[CHAINABLES] Data source '{source_name}' completed: {response.status}")
        
        # Convert to legacy DataSourceResult format for compatibility
        return DataSourceResult(
            tool_type=source_name,
            status=ExecutionStatus.COMPLETED if response.status == "completed" else ExecutionStatus.PENDING,
            result=response.message if response.status == "completed" else response.model_dump(),
            message=f"Data source '{source_name}' executed successfully",
            metadata={
                "execution_time": time.time() - start_time,
                "response": response.model_dump()
            }
        )
        
    except ValueError as e:
        error_msg = f"Data source '{source_name}' not found: {str(e)}"
        logger.error(error_msg)
        return DataSourceResult(
            tool_type=source_name or "unknown",
            status=ExecutionStatus.FAILED,
            error=error_msg,
            metadata={"execution_time": time.time() - start_time}
        )
    except Exception as e:
        error_msg = f"Data source '{source_name}' failed: {str(e)}"
        logger.error(error_msg)
        return DataSourceResult(
            tool_type=source_name or "unknown",
            status=ExecutionStatus.FAILED,
            error=error_msg,
            metadata={"execution_time": time.time() - start_time}
        )


@register_custom_task("tool")  # Backward compatibility for actual tools
async def execute_tool_task(task_metadata, objective, agents, execution_metadata) -> "DataSourceResult":
    """Portable, backend-agnostic tool executor for 'tool' nodes."""
    from .utilities.registries import TOOLS_REGISTRY
    from .agent_methods.data_models.execution_models import DataSourceResult, ExecutionStatus
    import inspect
    import logging
    import time
    
    logger = logging.getLogger("iointel.chainables.tool_executor")
    start_time = time.time()
    
    tool_name = task_metadata.get("tool_name")
    config = task_metadata.get("config", {})
    logger.info(f"[CHAINABLES] Executing tool: {tool_name}")
    logger.debug(f"    Config: {config}")
    
    tool = TOOLS_REGISTRY.get(tool_name)
    if not tool:
        error_msg = f"Tool '{tool_name}' not found in TOOLS_REGISTRY"
        logger.error(error_msg)
        return DataSourceResult(
            tool_type=tool_name or "unknown",
            status=ExecutionStatus.FAILED,
            error=error_msg,
            metadata={"execution_time": time.time() - start_time}
        )
    
    try:
        # Pass execution_metadata inside the arguments dict - Tool.run() expects it there
        # The Tool class will extract it and pass it as additional_args to the actual function
        config_with_metadata = config.copy()
        config_with_metadata['execution_metadata'] = execution_metadata
        result = tool.run(config_with_metadata)
        if inspect.isawaitable(result):
            result = await result
        
        logger.info(f"[CHAINABLES] Tool '{tool_name}' completed: {result}")
        
        return DataSourceResult(
            tool_type=tool_name,
            status=ExecutionStatus.COMPLETED,
            result=result,
            message=f"Tool '{tool_name}' executed successfully",
            metadata={"execution_time": time.time() - start_time}
        )
        
    except Exception as e:
        error_msg = f"Tool '{tool_name}' failed: {str(e)}"
        logger.error(error_msg)
        return DataSourceResult(
            tool_type=tool_name,
            status=ExecutionStatus.FAILED,
            error=error_msg,
            metadata={"execution_time": time.time() - start_time}
        )


@register_custom_task("decision")  # Core decision executor
async def execute_decision_task(task_metadata, objective, agents, execution_metadata) -> "AgentExecutionResult":
    """Core decision executor - delegates to agent executor for decision agents."""
    # Decision nodes are just agents with routing tools, so delegate to agent executor
    return await execute_agent_task(task_metadata, objective, agents, execution_metadata)


##############################################
# CHAINABLES
##############################################
"""
The chainable methods below are used to chain tasks together in a workflow.
Each method takes a 'self' argument, which is the task object being chained.
The method should return the 'self' object with the task appended to the 'tasks' list.
The 'tasks' list is used to store the tasks in the workflow.

"""


def schedule_reminder(self, delay: int = 0, agents: Optional[List[Agent]] = None):
    # WIP
    self.tasks.append(
        {
            "type": "schedule_reminder",
            "command": self.objective,
            "task_metadata": {"delay": delay},
            "agents": self.agents if agents is None else agents,
        }
    )
    return self


def solve_with_reasoning(self, agents: Optional[List[Agent]] = None):
    self.tasks.append(
        {
            "type": "solve_with_reasoning",
            "objective": self.objective,
            "agents": self.agents if agents is None else agents,
        }
    )
    return self


def summarize_text(self, max_words: int = 100, agents: Optional[List[Agent]] = None):
    self.tasks.append(
        {
            "type": "summarize_text",
            "objective": self.objective,
            "agents": self.agents if agents is None else agents,
            "task_metadata": {"max_words": max_words},
        }
    )
    return self


def sentiment(self, agents: Optional[List[Agent]] = None):
    self.tasks.append(
        {
            "type": "sentiment",
            "objective": self.objective,
            "agents": self.agents if agents is None else agents,
        }
    )
    return self


def extract_categorized_entities(self, agents: Optional[List[Agent]] = None):
    self.tasks.append(
        {
            "type": "extract_categorized_entities",
            "objective": self.objective,
            "agents": self.agents if agents is None else agents,
        }
    )
    return self


def translate_text(self, target_language: str, agents: Optional[List[Agent]] = None):
    self.tasks.append(
        {
            "type": "translate_text",
            "objective": self.objective,
            "task_metadata": {"target_language": target_language},
            "agents": self.agents if agents is None else agents,
        }
    )
    return self


def classify(self, classify_by: list, agents: Optional[List[Agent]] = None):
    self.tasks.append(
        {
            "type": "classify",
            "task_metadata": {"classify_by": classify_by},
            "objective": self.objective,
            "agents": self.agents if agents is None else agents,
        }
    )
    return self


def moderation(self, threshold: float, agents: Optional[List[Agent]] = None):
    self.tasks.append(
        {
            "type": "moderation",
            "objective": self.objective,
            "task_metadata": {"threshold": threshold},
            "agents": self.agents if agents is None else agents,
        }
    )
    return self


# def custom(self, name: str, objective: str, agents: Optional[List[Agent]] = None, instructions: str = "", **kwargs):
def custom(
    self, name: str, objective: str, agents: Optional[List[Agent]] = None, **kwargs
):
    """
    Allows users to define a custom workflow (or step) that can be chained
    like the built-in tasks. 'name' can help identify the custom workflow
    in run_tasks().

    :param name: Unique identifier for this custom workflow step.
    :param objective: The main objective or prompt for run_agents.
    :param agents: List of agents used (if None, a default can be used).
    :param kwargs: Additional data needed for this custom workflow.
    """
    self.tasks.append(
        {
            "type": "custom",
            "objective": objective,
            "task_metadata": {"name": name, "kwargs": kwargs},
            "agents": self.agents if agents is None else agents,
        }
    )
    return self


# Dictionary mapping method names to functions
CHAINABLE_METHODS.update(
    {
        "schedule_reminder": schedule_reminder,
        "solve_with_reasoning": solve_with_reasoning,
        "summarize_text": summarize_text,
        "sentiment": sentiment,
        "extract_categorized_entities": extract_categorized_entities,
        "translate_text": translate_text,
        "classify": classify,
        "moderation": moderation,
        "custom": custom,
    }
)
