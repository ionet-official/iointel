"""Direct node execution - bypassing the dict conversion madness."""

from typing import Any, Dict, List, Optional
from iointel.src.agent_methods.data_models.workflow_spec import (
    NodeSpec, 
    EdgeSpec, 
    WorkflowSpec,
    AgentNode,
    DataSourceNode,
    DecisionNode
)
from iointel.src.agent_methods.data_models.datamodels import AgentParams
from iointel.src.agent_methods.data_sources import get_data_source
from iointel.src.agent_methods.data_sources.models import DataSourceRequest
from iointel.src.agents import Agent
from iointel.src.utilities.runners import run_agents
from iointel.src.agent_methods.agents.agents_factory import create_agent
from iointel.src.utilities.data_flow_resolver import data_flow_resolver
from iointel.src.agent_methods.data_models.datamodels import AgentResultFormat
import logging

logger = logging.getLogger(__name__)


class DirectNodeExecutor:
    """Execute nodes directly from NodeSpec and EdgeSpec without dict conversion."""
    
    def __init__(self, conversation_id: str, agents: Optional[List[Agent]] = None):
        self.conversation_id = conversation_id
        self.available_agents = agents or []
        self._agent_cache = {}
    
    async def execute_node(
        self, 
        node: NodeSpec,
        edges: List[EdgeSpec],
        state_results: Dict[str, Any],
        objective: Optional[str] = None,
        workflow_spec: Optional[WorkflowSpec] = None
    ) -> Any:
        """Execute a node based on its type."""
        logger.info(f"🎯 Direct execution of node '{node.id}' (type: {node.type})")
        
        # Store workflow_spec for reference
        self.workflow_spec = workflow_spec
        
        if node.type == "agent":
            return await self._execute_agent_node(node, edges, state_results, objective)
        elif node.type == "data_source":
            return await self._execute_data_source_node(node, state_results)
        elif node.type == "decision":
            return await self._execute_decision_node(node, edges, state_results, objective)
        elif node.type == "tool":
            return await self._execute_tool_node(node, edges, state_results)
        else:
            raise ValueError(f"Unknown node type: {node.type}")
    
    async def _execute_agent_node(
        self, 
        node: NodeSpec,
        edges: List[EdgeSpec],
        state_results: Dict[str, Any],
        objective: Optional[str] = None
    ) -> Any:
        """Execute an agent node directly."""
        # Get or create agent
        agent = self._get_or_create_agent(node)
        
        # Resolve variables in agent instructions
        agent_instructions = node.data.agent_instructions or "SHOULD NEVER HAPPEN"
        # this should not be needed:
        # if agent_instructions and state_results:
        #     agent_instructions = data_flow_resolver._resolve_value(agent_instructions, state_results)
        
        # Build context from inputs using edges
        context = {}
        # AgentNode and DecisionNode types don't have ins/outs - they use edge-based flow
        # Only the old NodeData had ins/outs arrays
        # Find edges that target this node
        incoming_edges = [e for e in edges if e.target == node.id]
        
        for edge in incoming_edges:
            # Get the result from the source node
            source_result_key = f"{edge.source}_result"
            if source_result_key in state_results:
                result_value = state_results[source_result_key]
                
                # Extract actual value if it's a DataSourceResult
                from iointel.src.agent_methods.data_models.execution_models import DataSourceResult
                if isinstance(result_value, DataSourceResult):
                    result_value = result_value.result
                
                # Add to context using source node id as key
                context[edge.source] = result_value
                    
                # Special case: if this is from a user_input node, use it as objective
                source_node_key = edge.source
                if not objective and "user_input" in source_node_key:
                    objective = result_value
                    logger.info(f"📝 Using user input as objective: {objective}")
        
        # If still no objective, use agent instructions
        if not objective:
            objective = agent_instructions or "Process the provided context"
        
        # Determine result format
        result_format = AgentResultFormat.full()
        if node.type == "decision":
            result_format = AgentResultFormat.workflow()
        
        logger.info(f"🤖 Executing agent with objective: {objective[:100]}...")
        logger.info(f"📊 Context keys: {list(context.keys())}")
        
        # Execute agent
        response = await run_agents(
            objective=objective,
            agents=[agent] if agent else [],
            context=context,
            conversation_id=self.conversation_id,
            output_type=str,
            result_format=result_format
        ).execute()
        
        return response
    
    async def _execute_data_source_node(
        self, 
        node: NodeSpec, 
        state_results: Dict[str, Any]
    ) -> Any:
        """Execute a data source node directly."""
        # Type checking - we know this is a DataSourceNode
        if not isinstance(node, DataSourceNode):
            raise ValueError(f"Expected DataSourceNode, got {type(node)}")
            
        source_name = node.data.source_name
        config = node.data.config
        
        # config is a DataSourceConfig Pydantic model
        config_dict = {
            "message": config.message,
            "default_value": config.default_value
        }
        
        # Resolve variables in config
        if state_results:
            config_dict = data_flow_resolver.resolve_config(config_dict, state_results)
        
        # Get data source function
        data_source_func = get_data_source(source_name)
        
        # Create request based on source type
        if source_name == "user_input":
            request = DataSourceRequest(
                message=config_dict.get("message", ""),
                default_value=config_dict.get("default_value")
            )
        else:
            # For other data sources, pass config directly
            request = config_dict
        
        logger.info(f"📊 Executing data source '{source_name}'")
        
        # Execute data source
        response = data_source_func(request, execution_metadata={"node_id": node.id})
        
        # Extract result value based on response type
        from iointel.src.agent_methods.data_sources.models import DataSourceResponse
        
        if isinstance(response, DataSourceResponse):
            # DataSourceResponse has 'message' field for the actual content
            return response.message
        else:
            # Fallback for other response types
            return response
    
    async def _execute_decision_node(
        self, 
        node: NodeSpec,
        edges: List[EdgeSpec],
        state_results: Dict[str, Any],
        objective: Optional[str] = None
    ) -> Any:
        """Execute a decision node (special agent with conditional_gate tool)."""
        # Decision nodes are just agents with special tools
        return await self._execute_agent_node(node, edges, state_results, objective)
    
    async def _execute_tool_node(
        self, 
        node: NodeSpec,
        edges: List[EdgeSpec],
        state_results: Dict[str, Any]
    ) -> Any:
        """Execute a tool node directly."""
        # This would integrate with the tool registry
        # For now, raise NotImplementedError
        raise NotImplementedError("Tool node execution not yet implemented in direct executor")
    
    def _get_or_create_agent(self, node: NodeSpec) -> Optional[Agent]:
        """Get or create an agent for the node."""
        # Check cache
        if node.id in self._agent_cache:
            return self._agent_cache[node.id]

        # Type checking - we know this is an AgentNode or DecisionNode
        if not isinstance(node, (AgentNode, DecisionNode)):
            return None
            
        # Now we can safely access typed fields
        if node.data.agent_instructions:
            agent_params = AgentParams(
                name=f"agent_{node.id}",
                instructions=node.data.agent_instructions,
                model=node.data.model or "gpt-4o",
                tools=node.data.tools or [],
                api_key=None,  # Use defaults
            )
            agent = create_agent(agent_params)
            self._agent_cache[node.id] = agent
            return agent
        
        return None