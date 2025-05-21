from typing import List, Optional, Any
import uuid
import inspect
import yaml
from pathlib import Path
from collections import defaultdict

from .agent_methods.agents.agents_factory import (
    agent_or_swarm,
    create_agent,
    create_swarm,
)
from .agent_methods.data_models.datamodels import (
    WorkflowDefinition,
    TaskDefinition,
    AgentParams,
)
from .agents import Agent

from .utilities.runners import run_agents_stream
from .utilities.registries import TASK_EXECUTOR_REGISTRY
from .utilities.stages import execute_stage
from .utilities.helpers import make_logger
from .utilities.stages import (
    SimpleStage,
    SequentialStage,
    ParallelStage,
    WhileStage,
    FallbackStage,
)

from .utilities.graph_nodes import WorkflowState, TaskNode, make_task_node
from pydantic_graph import Graph, End, GraphRunContext

from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)


logger = make_logger(__name__)


def _get_task_key(task: dict) -> str:
    return (
        task.get("task_id")
        or task.get("task_metadata", {}).get("name")
        or task.get("type")
        or "task"
    )


class Workflow:
    """
    Manages a chain of tasks and runs them sequentially.

    Example usage:
        workflow = Workflow(text="Some input text", client_mode=False, agents=[swarm])
        workflow.summarize_text(max_words=50).custom(name="do-fancy-thing", objective="Fancy step", agents=[my_agent])
        results = await workflow.run_tasks()
    """

    def __init__(
        self,
        objective: str = "",
        client_mode: bool = True,
        agents: Optional[List[Any]] = None,
    ):
        self.tasks: List[dict] = []
        self.objective = objective
        self.client_mode = client_mode
        self.agents = agents

    def _instantiate_node(self, node_cls: type[TaskNode]) -> TaskNode:
        return node_cls(
            task=node_cls.task,
            default_text=node_cls.default_text,
            default_agents=node_cls.default_agents,
            conversation_id=node_cls.conversation_id,
            next_task=node_cls.next_task,
        )

    def __call__(
        self,
        objective: str,
        client_mode: bool = True,
        agents: Optional[List[Any]] = None,
    ):
        self.objective = objective
        self.client_mode = client_mode
        self.agents = agents
        return self

    def add_task(self, task: dict):
        if not task.get("agents"):
            task = dict(task, agents=self.agents)
        self.tasks.append(task)
        return self

    async def run_task(
        self,
        task: dict,
        default_text: str,
        default_agents: list,
        conversation_id: Optional[str] = None,
    ) -> Any:
        """
        Execute a single task.
        - If the task's execution_metadata contains declarative stage definitions,
        create the corresponding stage objects, run them, and then convert the result
        list into a dictionary keyed by the parent task's name plus stage order.
        - Otherwise, look up the custom executor from TASK_EXECUTOR_REGISTRY and call it.
        """

        if default_agents is None:
            default_agents = [Agent.make_default()]

        text_for_task = task.get("objective", default_text)
        agents_for_task = task.get("agents") or default_agents
        execution_metadata = task.get("execution_metadata", {})
        # Ensure conversation_id is in metadata
        if task.get("conversation_id"):
            execution_metadata["conversation_id"] = task["conversation_id"]
        if conversation_id:
            execution_metadata["conversation_id"] = conversation_id
        # client_mode = execution_metadata.get("client_mode", self.client_mode)

        if stage_defs := execution_metadata.get("stages"):
            stage_objects = []
            for stage_def in stage_defs:
                stage_type = stage_def.get("stage_type", "simple")
                rtype = stage_def.get("output_type", None)
                context = stage_def.get("context", {})
                if stage_type == "simple":
                    stage_objects.append(
                        SimpleStage(
                            objective=stage_def["objective"],
                            context=context,
                            output_type=rtype,
                        )
                    )
                elif stage_type == "while":
                    condition = stage_def["condition"]
                    nested_stage_def = stage_def["stage"]
                    nested_context = nested_stage_def.get("context", {})
                    nested_rtype = nested_stage_def.get("output_type", None)
                    nested_stage = SimpleStage(
                        objective=nested_stage_def["objective"],
                        context=nested_context,
                        output_type=nested_rtype,
                    )
                    stage_objects.append(
                        WhileStage(
                            condition=condition,
                            stage=nested_stage,
                            max_iterations=stage_def.get("max_iterations", 100),
                        )
                    )
                elif stage_type == "parallel":
                    nested_defs = stage_def.get("stages", [])
                    nested_objs = [
                        SimpleStage(
                            objective=nd["objective"],
                            context=nd.get("context", {}),
                            output_type=nd.get("output_type", None),
                        )
                        for nd in nested_defs
                    ]
                    stage_objects.append(ParallelStage(stages=nested_objs))
                elif stage_type == "fallback":
                    primary_obj = SimpleStage(
                        objective=stage_def["primary"]["objective"],
                        context=stage_def["primary"].get("context", {}),
                        output_type=stage_def["primary"].get("output_type", None),
                    )
                    fallback_obj = SimpleStage(
                        objective=stage_def["fallback"]["objective"],
                        context=stage_def["fallback"].get("context", {}),
                        output_type=stage_def["fallback"].get("output_type", None),
                    )
                    stage_objects.append(
                        FallbackStage(primary=primary_obj, fallback=fallback_obj)
                    )
                else:
                    stage_objects.append(
                        SimpleStage(
                            objective=stage_def["objective"],
                            context=context,
                            output_type=rtype,
                        )
                    )
            container_mode = execution_metadata.get("execution_mode", "sequential")
            if container_mode == "parallel":
                container = ParallelStage(stages=stage_objects)
            else:
                container = SequentialStage(stages=stage_objects)

            result = await execute_stage(
                container,
                agents_for_task,
                task.get("task_metadata", {}),
                text_for_task,
            )
            if isinstance(result, list):
                base = _get_task_key(task)
                result = {f"{base}_stage_{i + 1}": val for i, val in enumerate(result)}
            return result
        else:
            task_type = task.get("type") or task.get("name")
            executor = TASK_EXECUTOR_REGISTRY.get(task_type)
            if executor is None:
                raise ValueError(f"No executor registered for task type: {task_type}")
            result = executor(
                task_metadata=task.get("task_metadata", {}),
                objective=text_for_task,
                agents=agents_for_task,
                execution_metadata=execution_metadata,
            )
            if inspect.isawaitable(result):
                result = await result
            if hasattr(result, "execute") and callable(result.execute):
                result = await result.execute()
            return result

    async def execute_graph_streaming(self, graph, initial_state):
        nodes = list(graph.node_defs.values())
        total_tasks = len(nodes)

        state = initial_state

        completed_tasks = 0

        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            task_progress = progress.add_task(
                "[cyan]Executing Tasks...", total=total_tasks
            )

            current_node_cls = nodes[0].node  # class object
            current_node = self._instantiate_node(current_node_cls)

            while current_node:
                task_type = current_node.task.get("type") or current_node.task.get(
                    "name"
                )
                conversation_id = (
                    current_node.task.get("conversation_id")
                    or state.conversation_id
                    or str(uuid.uuid4())
                )

                agents_to_use = (
                    current_node.task.get("agents") or current_node.default_agents
                )

                if task_type in TASK_EXECUTOR_REGISTRY:
                    executor = TASK_EXECUTOR_REGISTRY[task_type]
                    result = executor(
                        task_metadata=current_node.task.get("task_metadata", {}),
                        objective=current_node.task["objective"],
                        agents=agents_to_use,
                        execution_metadata={"conversation_id": conversation_id},
                    )
                    if inspect.isawaitable(result):
                        result = await result
                    if callable(getattr(result, "execute", None)):
                        result = await result.execute()
                    final_result = result
                else:
                    final_result = await run_agents_stream(
                        objective=current_node.task["objective"],
                        agents=agents_to_use,
                        conversation_id=conversation_id,
                    ).execute()

                task_key = _get_task_key(current_node.task)
                state.results[task_key] = final_result.get("result", final_result)

                progress.advance(task_progress, 1)
                completed_tasks += 1
                next_node = current_node.next_task
                if isinstance(next_node, type) and issubclass(next_node, TaskNode):
                    current_node = self._instantiate_node(next_node)
                else:
                    current_node = next_node

        return state

    async def execute_graph(self, graph: Graph, initial_state: WorkflowState):
        nodes = list(graph.node_defs.values())
        total_tasks = len(nodes)

        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            task_progress = progress.add_task(
                "[cyan]Executing Tasks...", total=total_tasks
            )

            current_node_cls = nodes[0].node  # class object
            current_node = self._instantiate_node(current_node_cls)
            state = initial_state
            completed_tasks = 0

            while current_node:
                result = await current_node.run(GraphRunContext(state=state, deps={}))

                progress.advance(task_progress, 1)
                completed_tasks += 1

                if isinstance(result, End):
                    break
                elif isinstance(result, type) and issubclass(result, TaskNode):
                    current_node = self._instantiate_node(result)
                else:
                    raise ValueError(f"Unexpected node result type: {type(result)}")

            if completed_tasks < total_tasks:
                progress.update(task_progress, completed=total_tasks)

        return state

    def build_workflow_graph(self, conversation_id: Optional[str] = None) -> Graph:
        conversation_id = conversation_id or str(uuid.uuid4())
        node_classes = []
        prev_node_cls = None

        for task in self.tasks:
            NodeCls = make_task_node(task, self.objective, self.agents, conversation_id)
            node_classes.append(NodeCls)

            # chain the classes, not instances
            if prev_node_cls:
                prev_node_cls.next_task = NodeCls
            prev_node_cls = NodeCls

        # pass the class objects to Graph
        return Graph(
            nodes=tuple(node_classes),
            state_type=WorkflowState,
            run_end_type=WorkflowState,
        )

    async def run_tasks(self, conversation_id: Optional[str] = None, **kwargs) -> dict:
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        initial_state = WorkflowState(
            conversation_id=conversation_id, initial_text=self.objective, results={}
        )

        self.graph = self.build_workflow_graph(conversation_id)

        final_state = await self.execute_graph(self.graph, initial_state)

        return dict(conversation_id=conversation_id, results=final_state.results)

    async def run_tasks_streaming(
        self, conversation_id: Optional[str] = None, **kwargs
    ) -> dict:
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        initial_state = WorkflowState(
            conversation_id=conversation_id, initial_text=self.objective, results={}
        )

        self.graph = self.build_workflow_graph(conversation_id)

        final_state = await self.execute_graph_streaming(self.graph, initial_state)

        return dict(conversation_id=conversation_id, results=final_state.results)

    def to_yaml(
        self,
        workflow_name: str = "My YAML Workflow",
        file_path: Optional[str] = None,
        store_creds: bool = False,
    ) -> str:
        agent_params_list = []
        if self.agents:
            for agent_obj in self.agents:
                agent_params_list.extend(agent_or_swarm(agent_obj, store_creds))

        task_models = []
        for t in self.tasks:
            task_metadata = t.get("task_metadata") or {}
            if "client_mode" not in task_metadata:
                task_metadata["client_mode"] = t.get("client_mode", self.client_mode)
            task_model = TaskDefinition(
                task_id=t.get("task_id", t.get("type", str(uuid.uuid4()))),
                name=t.get("name", t.get("type", "Unnamed Task")),
                objective=t.get("objective"),
                task_metadata=task_metadata,
                execution_metadata=t.get("execution_metadata") or {},
            )
            # Process task-level agents similarly.
            step_agents_params = []
            if t.get("agents"):
                for agent in t["agents"]:
                    step_agents_params.extend(agent_or_swarm(agent, store_creds))
                task_model.agents = step_agents_params
            task_models.append(task_model)

        # Build the WorkflowDefinition.
        wf_def = WorkflowDefinition(
            name=workflow_name,
            objective=self.objective,
            client_mode=self.client_mode,
            agents=agent_params_list,
            tasks=task_models,
        )
        wf_dict = wf_def.model_dump(
            mode="json",
            exclude={
                "memory": True,
                "memories": True,
                "agents": {"__all__": {"memory": True}},
                "tasks": {"__all__": {"agents": {"__all__": {"memory": True}}}},
            },
        )
        yaml_str = yaml.safe_dump(wf_dict, sort_keys=False)
        if file_path:
            Path(file_path).write_text(yaml_str, encoding="utf-8")
        return yaml_str

    def from_yaml(self, yaml_str: str = None, file_path: str = None) -> "Workflow":
        if not yaml_str and not file_path:
            raise ValueError("Either yaml_str or file_path must be provided.")
        if yaml_str:
            data = yaml.safe_load(yaml_str)
        else:
            data = yaml.safe_load(Path(file_path).read_text(encoding="utf-8"))

        wf_def = WorkflowDefinition(**data)
        self.objective = wf_def.objective or ""

        # --- Rehydrate Top-Level Agents ---
        swarm_lookup = {}  # key: swarm_name, value: list of AgentParams objects
        individual_agents = []  # list of AgentParams without a swarm_name
        if wf_def.agents:
            for agent_data in wf_def.agents:
                if (
                    hasattr(agent_data, "swarm_name")
                    and agent_data.swarm_name is not None
                ):
                    swarm_name = agent_data.swarm_name
                    logger.debug(
                        f"Top-level agent '{agent_data.name}' is part of swarm '{swarm_name}'"
                    )
                    swarm_lookup.setdefault(swarm_name, []).append(agent_data)
                else:
                    individual_agents.append(agent_data)

        real_agents = []

        for swarm_name, members_list in swarm_lookup.items():
            logger.debug(
                f" Group for swarm '{swarm_name}': {len(members_list)} member(s)"
            )
            members = [create_agent(member) for member in members_list]
            swarm_obj = create_swarm(members)
            # Explicitly set the swarm's name.
            swarm_obj.name = swarm_name
            real_agents.append(swarm_obj)
        # Rehydrate individual agents.
        for agent_data in individual_agents:
            real_agents.append(create_agent(agent_data))
        self.agents = real_agents

        top_level_swarm_lookup = {
            swarm_obj.name: swarm_obj
            for swarm_obj in real_agents
            if hasattr(swarm_obj, "members")
        }

        # --- Rehydrate Tasks ---
        self.tasks.clear()
        for task in wf_def.tasks:
            new_task = {
                "task_id": task.task_id,
                "name": task.name,
                "objective": task.objective,
                "task_metadata": dict(task.task_metadata or {}, name=task.name),
                "execution_metadata": task.execution_metadata or {},
            }
            if task.agents:
                step_agents = []
                # Group task-level agents by swarm_name.
                swarm_groups = defaultdict(list)
                individual = []
                for agent in task.agents:
                    swarm_name = None
                    if isinstance(agent, dict):
                        swarm_name = agent.get("swarm_name")
                    else:
                        swarm_name = getattr(agent, "swarm_name", None)

                    if swarm_name:
                        swarm_groups[swarm_name].append(agent)
                    else:
                        individual.append(agent)

                logger.debug(
                    f"Task '{task.name}': Found {len(swarm_groups)} swarm group(s) and {len(individual)} individual agent(s)"
                )

                for swarm_name, members_list in swarm_groups.items():
                    logger.debug(
                        f"  Group for swarm '{swarm_name}': {len(members_list)} member(s)"
                    )

                    if swarm_name in top_level_swarm_lookup:
                        logger.debug(
                            f"Task '{task.name}'  Using top-level swarm '{swarm_name}'"
                        )
                        step_agents.append(top_level_swarm_lookup[swarm_name])
                    else:
                        members = [
                            create_agent(AgentParams.model_validate(m))
                            for m in members_list
                        ]
                        swarm_obj = create_swarm(members)
                        swarm_obj.name = swarm_name  # set the swarm name explicitly
                        logger.debug(
                            f" Task '{task.name}' Created new swarm '{swarm_obj.name}' with {len(swarm_obj.members)} members"
                        )
                        step_agents.append(swarm_obj)

                for agent in individual:
                    rehydrated = create_agent(AgentParams.model_validate(agent))

                    logger.debug(f"  Rehydrated individual agent: {rehydrated.name}")
                    step_agents.append(rehydrated)
                new_task["agents"] = step_agents
            self.tasks.append(new_task)
        return self


# has to be down here else circular import
from .chainables import CHAINABLE_METHODS  # noqa: E402

for method_name, func in CHAINABLE_METHODS.items():
    setattr(Workflow, method_name, func)
