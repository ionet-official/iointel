from typing import Callable, List, Optional, Any, Union
import uuid
import inspect
import warnings
from pydantic import BaseModel
import yaml
from pathlib import Path
from collections import defaultdict

from iointel.src.agent_methods.agents.agents_factory import (
    agent_or_swarm,
    create_agent,
    create_swarm,
)
from iointel.src.agent_methods.data_models.datamodels import (
    Tool,
    WorkflowDefinition,
    TaskDefinition,
    AgentParams,
)
from iointel.src.agents import Agent

from iointel.src.utilities.runners import run_agents_stream
from iointel.src.utilities.registries import TASK_EXECUTOR_REGISTRY
from iointel.src.utilities.stages import execute_stage
from iointel.src.utilities.helpers import make_logger
from iointel.src.utilities.io_logger import get_component_logger
from iointel.src.utilities.registries import TOOLS_REGISTRY
from iointel.src.utilities.stages import (
    SimpleStage,
    SequentialStage,
    ParallelStage,
    WhileStage,
    FallbackStage,
)
from iointel.src.utilities.rich import pretty_output

from iointel.src.utilities.graph_nodes import WorkflowState, TaskNode, make_task_node
from iointel.src.utilities.dag_executor import DAGExecutor
from pydantic_graph import Graph, End, GraphRunContext

from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)


logger = make_logger(__name__)


def dict_to_task_definition(task_dict: dict) -> TaskDefinition:
    """
    Convert a runtime dict to a validated TaskDefinition.
    
    This bridges the ontological gap between the three workflow formats:
    1. Runtime dicts (Workflow.add_task)
    2. TaskDefinition (WorkflowDefinition YAML)
    3. NodeSpec (WorkflowSpec DAG)
    """
    task_copy = task_dict.copy()
    
    # Ensure required fields with sensible defaults
    if not task_copy.get("task_id"):
        task_copy["task_id"] = task_copy.get("name") or task_copy.get("type", "task") + "_" + str(uuid.uuid4())[:8]
    if not task_copy.get("name"):
        task_copy["name"] = task_copy.get("task_id", "Unnamed Task")
    
    return TaskDefinition.model_validate(task_copy)


def task_definition_to_dict(task_def: TaskDefinition) -> dict:
    """
    Convert a TaskDefinition back to a dict for backward compatibility.
    """
    return task_def.model_dump()


def _get_task_key(task: dict) -> str:
    return (
        task.get("task_id")
        or task.get("name")  # Check top-level name first (for runtime dict format)
        or task.get("task_metadata", {}).get("name")
        or task.get("type")
        or "task"
    )


class Workflow:
    """
    Manages a chain of tasks and runs them sequentially.

    Example usage:
        workflow = Workflow(objective="Some input text", client_mode=False, agents=[swarm])
        workflow.summarize_text(max_words=50).custom(name="do-fancy-thing", objective="Fancy step", agents=[my_agent])
        results = await workflow.run_tasks()
    """

    def __init__(
        self,
        objective: str = "",
        text: str | None = None,
        client_mode: bool = True,
        agents: Optional[List[Any]] = None,
    ):
        if text is not None:
            if objective:
                raise ValueError("Both `text` and `objective` parameters set")
            objective = text
            warnings.warn(
                "`text` parameter is deprecated, please use `objective` instead"
            )
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

    def add_task(self, task: Union[dict, TaskDefinition]):
        """
        Add a task to the workflow.
        
        Args:
            task: Either a dict (legacy) or TaskDefinition instance.
                  Dicts will be validated and converted to ensure consistent schema.
        """
        # Convert to TaskDefinition for validation and consistency
        if isinstance(task, dict):
            try:
                # Validate using conversion helper
                validated_task = dict_to_task_definition(task)
                logger.debug(f"✅ Validated task: {validated_task.name} (ID: {validated_task.task_id})")
                # Convert back to dict for backward compatibility with existing workflow execution
                task = task_definition_to_dict(validated_task)
            except Exception as e:
                logger.warning(f"⚠️  Task validation failed, using original dict: {e}")
                # Fall back to original dict for backward compatibility
                pass
        elif isinstance(task, TaskDefinition):
            # Already validated, convert to dict for workflow execution
            task = task_definition_to_dict(task)
            logger.debug(f"✅ Using TaskDefinition: {task.get('name')} (ID: {task.get('task_id')})")
        
        # Apply default agents if not specified
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
        **kwargs,
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

        if text_for_task := task.get("objective"):
            context_for_task = {"main input": default_text}
        else:
            text_for_task, context_for_task = default_text, {}
        agents_for_task = task.get("agents") or default_agents
        
        # Create workflow logger for debug output  
        workflow_logger = get_component_logger("WORKFLOW")
        
        # Debug: Check task structure for agent tasks
        if task.get("type") == "agent":
            workflow_logger.info("🔍 Agent task analysis", data={
                "task_name": task.get('name', 'unnamed'),
                "agent_count": len(agents_for_task),
                "has_agents_key": 'agents' in task,
                "task_keys": list(task.keys()),
                "task_type": "agent"
            })
            if agents_for_task and hasattr(agents_for_task[0], 'tools'):
                workflow_logger.debug("Agent tools configuration", data={
                    "tool_count": len(agents_for_task[0].tools),
                    "tools": [str(tool) for tool in agents_for_task[0].tools]
                })
        
        execution_metadata = task.get("execution_metadata") or {}
        
        # Debug: Show what we're starting with
        workflow_logger.debug("🔍 Task execution starting", data={
            "task_keys": list(task.keys()),
            "initial_execution_metadata": execution_metadata,
            "task_type": task.get("type"),
            "task_name": task.get("name")
        })
        
        # Add current task/node ID to execution metadata for tool context
        task_id = task.get("task_id") or task.get("id") or task.get("name")
        if task_id:
            execution_metadata["task_id"] = task_id
            execution_metadata["node_id"] = task_id  # Alias for compatibility
            workflow_logger.debug("🔍 Task ID resolved", data={
                "task_id": task_id,
                "metadata_updated": True
            })
        else:
            workflow_logger.warning("🔍 No task_id found in task", data={
                "available_keys": list(task.keys()),
                "impact": "Task context may be limited"
            })
        
        workflow_logger.debug("🔍 Final execution metadata prepared", data={
            "execution_metadata": execution_metadata
        })
        
        # Ensure conversation_id is in metadata
        if task.get("conversation_id"):
            execution_metadata["conversation_id"] = task["conversation_id"]
        if conversation_id:
            execution_metadata["conversation_id"] = conversation_id
        # Add agent result format preference (passed via kwargs in workflow execution)
        agent_result_format = kwargs.get('agent_result_format', 'full')
        execution_metadata["agent_result_format"] = agent_result_format
        print(f"🔧 Workflow: adding agent_result_format = {agent_result_format} to execution_metadata")
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
            task_metadata = task.get("task_metadata", {})
            task_metadata["kwargs"] = context_for_task | task_metadata.get("kwargs", {})
            result = executor(
                task_metadata=task_metadata,
                objective=text_for_task,
                agents=agents_for_task,
                execution_metadata=execution_metadata,
            )
            if inspect.isawaitable(result):
                result = await result
            if hasattr(result, "execute") and callable(result.execute):
                result = await result.execute()
            # --- PATCH: Always wrap result in a dict with a single 'result' key unless already so ---
            if not (isinstance(result, dict) and list(result.keys()) == ["result"]):
                result = {"result": result}
            return result

    async def execute_graph_streaming(
        self, graph, initial_state, pretty: Optional[bool] = None
    ):
        if pretty is None:
            pretty = pretty_output.is_enabled
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
            disable=not pretty,
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

    async def execute_graph(
        self, graph: Graph, initial_state: WorkflowState, pretty: Optional[bool] = None, agent_result_format: str = "full"
    ):
        if pretty is None:
            pretty = pretty_output.is_enabled
        nodes = list(graph.node_defs.values())
        total_tasks = len(nodes)

        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            transient=True,
            disable=not pretty,
        ) as progress:
            task_progress = progress.add_task(
                "[cyan]Executing Tasks...", total=total_tasks
            )

            current_node_cls = nodes[0].node  # class object
            current_node = self._instantiate_node(current_node_cls)
            state = initial_state
            completed_tasks = 0

            while current_node:
                # Pass agent_result_format through to node execution
                current_node._agent_result_format = agent_result_format
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
    
    def _has_dag_structure(self) -> bool:
        """Check if this workflow has DAG structure (WorkflowSpec converted) vs simple task list."""
        # If tasks have 'edges' metadata or complex dependencies, use DAG execution
        for task in self.tasks:
            task_metadata = task.get("task_metadata", {})
            if "edges" in task_metadata or "dag_structure" in task_metadata:
                return True
        return False

    def _ensure_tools_loaded(self):
        """Ensure tools are loaded before workflow execution."""
        if not TOOLS_REGISTRY:
            try:
                from .agent_methods.tools.tool_loader import load_tools_from_env
                logger.info("Loading tools for workflow execution...")
                available_tools = load_tools_from_env("creds.env")
                logger.info(f"Loaded {len(available_tools)} tools for workflow execution")
            except Exception as e:
                logger.warning(f"Could not load tools: {e}")

    async def run_tasks(self, conversation_id: Optional[str] = None, agent_result_format: str = "full", **kwargs) -> dict:
        # Ensure tools are loaded before execution
        self._ensure_tools_loaded()
        
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        initial_state = WorkflowState(
            conversation_id=conversation_id, initial_text=self.objective, results={}
        )
        
        # print(f"🔧 Workflow: running with agent_result_format = {agent_result_format}")

        # Always use DAG execution - convert sequential tasks to DAG format if needed
        if self._has_dag_structure():
            print("🔄 Using DAG execution for workflow with edge topology")
            logger.info("Using DAG execution for workflow with edge topology")
            final_state = await self._run_tasks_dag(initial_state, conversation_id, agent_result_format=agent_result_format)
        else:
            print("🔄 Converting sequential tasks to DAG format and using DAG execution")
            logger.info("Converting sequential tasks to DAG format and using DAG execution")
            final_state = await self._run_tasks_sequential_as_dag(initial_state, conversation_id, agent_result_format=agent_result_format)

        return dict(conversation_id=conversation_id, results=final_state.results)
    
    async def _run_tasks_dag(self, initial_state: WorkflowState, conversation_id: str, agent_result_format: str = "full") -> WorkflowState:
        """Run tasks using DAG executor for proper topology and parallel execution."""
        # Extract DAG structure from task metadata
        dag_structure = None
        for task in self.tasks:
            task_metadata = task.get("task_metadata", {})
            if "dag_structure" in task_metadata:
                dag_structure = task_metadata["dag_structure"]
                break
        
        if not dag_structure:
            raise ValueError("DAG structure not found in task metadata")
        
        # Reconstruct WorkflowSpec from metadata
        from .agent_methods.data_models.workflow_spec import NodeSpec, EdgeSpec
        
        nodes = [NodeSpec.model_validate(node_data) for node_data in dag_structure["nodes"]]
        edges = [EdgeSpec.model_validate(edge_data) for edge_data in dag_structure["edges"]]
        
        # Extract execution_metadata and agents from original workflow tasks
        execution_metadata_by_node = {}
        agents_by_node = {}
        for task in self.tasks:
            node_id = task.get("task_id") or task.get("id")
            if node_id:
                # Extract execution metadata
                if "execution_metadata" in task:
                    execution_metadata_by_node[node_id] = task["execution_metadata"]
                    print(f"🔍 Extracted execution_metadata for node {node_id}")
                    if "user_inputs" in task["execution_metadata"]:
                        print(f"🔍 User inputs found for node {node_id}: {task['execution_metadata']['user_inputs']}")
                else:
                    print(f"🔍 Task missing execution_metadata: {node_id}")
                
                # Extract task-specific agents
                if "agents" in task and task["agents"]:
                    agents_by_node[node_id] = task["agents"]
                    print(f"🔍 Extracted {len(task['agents'])} agents for node {node_id}")
            else:
                print(f"🔍 Task missing task_id/id: {task}")
        
        # Create DAG executor
        executor = DAGExecutor()
        executor.build_execution_graph(
            nodes=nodes,
            edges=edges,
            objective=self.objective,
            agents=self.agents,
            conversation_id=conversation_id,
            execution_metadata_by_node=execution_metadata_by_node,
            agents_by_node=agents_by_node
        )
        
        # Execute DAG
        logger.info(f"Executing DAG with {len(nodes)} nodes and {len(edges)} edges")
        summary = executor.get_execution_summary()
        logger.info(f"DAG execution plan: {summary['total_batches']} batches, max parallelism: {summary['max_parallelism']}")
        
        return await executor.execute_dag(initial_state)
    
    async def _run_tasks_sequential_as_dag(self, initial_state: WorkflowState, conversation_id: str, agent_result_format: str = "full") -> WorkflowState:
        """Convert sequential tasks to DAG format and run using DAG executor."""
        from .agent_methods.data_models.workflow_spec import NodeSpec, EdgeSpec, NodeData, EdgeData
        
        # Convert tasks to DAG nodes
        nodes = []
        edges = []
        
        for i, task in enumerate(self.tasks):
            task_id = task.get("task_id") or task.get("name") or f"task_{i}"
            task_type = task.get("type", "tool")
            task_name = task.get("name", f"Task {i+1}")
            
            # Extract task metadata
            task_metadata = task.get("task_metadata", {})
            
            # Create node data
            node_data = NodeData(
                tool_name=task_metadata.get("tool_name"),
                agent_instructions=task_metadata.get("agent_instructions"),
                config=task_metadata.get("config", {}),
                tools=task_metadata.get("tools", []),
                model=task_metadata.get("model"),
                ins=[] if i == 0 else [f"task_{i-1}_out"],  # Sequential: depend on previous task
                outs=[f"{task_id}_out"]
            )
            
            # Create node
            node = NodeSpec(
                id=task_id,
                type=task_type,
                label=task_name,
                data=node_data
            )
            nodes.append(node)
            
            # Create edge to next task (except for last task)
            if i > 0:
                edge = EdgeSpec(
                    id=f"edge_{i-1}_to_{i}",
                    source=self.tasks[i-1].get("task_id") or self.tasks[i-1].get("name") or f"task_{i-1}",
                    target=task_id,
                    data=EdgeData()
                )
                edges.append(edge)
        
        # Extract execution metadata and agents from tasks
        execution_metadata_by_node = {}
        agents_by_node = {}
        
        for i, task in enumerate(self.tasks):
            task_id = task.get("task_id") or task.get("name") or f"task_{i}"
            
            # Extract execution metadata
            if "execution_metadata" in task and task["execution_metadata"]:
                execution_metadata_by_node[task_id] = task["execution_metadata"]
            
            # Extract agents
            if "agents" in task and task["agents"]:
                agents_by_node[task_id] = task["agents"]
        
        # Create DAG executor
        executor = DAGExecutor()
        executor.build_execution_graph(
            nodes=nodes,
            edges=edges,
            objective=self.objective,
            agents=self.agents,
            conversation_id=conversation_id,
            execution_metadata_by_node=execution_metadata_by_node,
            agents_by_node=agents_by_node
        )
        
        # Execute DAG
        logger.info(f"Executing converted sequential workflow as DAG with {len(nodes)} nodes and {len(edges)} edges")
        summary = executor.get_execution_summary()
        logger.info(f"Sequential-to-DAG execution plan: {summary['total_batches']} batches, max parallelism: {summary['max_parallelism']}")
        
        return await executor.execute_dag(initial_state)

    async def run_tasks_streaming(
        self, conversation_id: Optional[str] = None, agent_result_format: str = "full", **kwargs
    ) -> dict:
        # Ensure tools are loaded before execution
        self._ensure_tools_loaded()
        
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        initial_state = WorkflowState(
            conversation_id=conversation_id, initial_text=self.objective, results={}
        )
        
        # Store agent result format preference for task execution
        self.agent_result_format = agent_result_format

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
                type=t.get("type", "custom"),
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

    @classmethod
    def from_yaml(
        cls,
        yaml_str: Optional[str] = None,
        file_path: Optional[str] = None,
        instantiate_agent: Callable[[AgentParams], Agent] | None = None,
        instantiate_tool: Callable[[Tool, dict | None], BaseModel | None] | None = None,
    ) -> "Workflow":
        if not yaml_str and not file_path:
            raise ValueError("Either yaml_str or file_path must be provided.")
        if yaml_str:
            data = yaml.safe_load(yaml_str)
        else:
            data = yaml.safe_load(Path(file_path).read_text(encoding="utf-8"))

        wf_def = WorkflowDefinition(**data)

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
            members = [
                create_agent(member, instantiate_agent, instantiate_tool)
                for member in members_list
            ]
            swarm_obj = create_swarm(members)
            # Explicitly set the swarm's name.
            swarm_obj.name = swarm_name
            real_agents.append(swarm_obj)
        # Rehydrate individual agents.
        for agent_data in individual_agents:
            real_agents.append(
                create_agent(agent_data, instantiate_agent, instantiate_tool)
            )

        top_level_swarm_lookup = {
            swarm_obj.name: swarm_obj
            for swarm_obj in real_agents
            if hasattr(swarm_obj, "members")
        }

        # --- Rehydrate Tasks ---
        tasks: list[dict] = []
        for task in wf_def.tasks:
            new_task = {
                "task_id": task.task_id,
                "type": task.type,
                "name": task.name,
                "objective": task.objective,
                "task_metadata": dict(task.task_metadata or {}, name=task.name),
                "execution_metadata": task.execution_metadata or {},
            }
            if task.agents:
                logger.debug(f"YAML Loading - Task '{task.name}' has {len(task.agents)} agents")
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
                            create_agent(
                                AgentParams.model_validate(m),
                                instantiate_agent,
                                instantiate_tool,
                            )
                            for m in members_list
                        ]
                        swarm_obj = create_swarm(members)
                        swarm_obj.name = swarm_name  # set the swarm name explicitly
                        logger.debug(
                            f" Task '{task.name}' Created new swarm '{swarm_obj.name}' with {len(swarm_obj.members)} members"
                        )
                        step_agents.append(swarm_obj)

                for agent in individual:
                    # Keep as AgentParams for task executors, don't convert to Agent yet
                    agent_params = AgentParams.model_validate(agent)
                    logger.debug(f"  Prepared AgentParams for task: {agent_params.name}")
                    step_agents.append(agent_params)
                new_task["agents"] = step_agents
            tasks.append(new_task)

        self = Workflow(
            objective=wf_def.objective or "",
            client_mode=wf_def.client_mode,
            agents=real_agents,
        )
        self.tasks = tasks
        return self


# has to be down here else circular import
from iointel.src.chainables import CHAINABLE_METHODS  # noqa: E402

for method_name, func in CHAINABLE_METHODS.items():
    setattr(Workflow, method_name, func)
