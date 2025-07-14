"""
Secure wrapper for pydantic-ai Agent that integrates with our existing security features
and provides backward compatibility with the current Agent class.
"""

from typing import Callable, Dict, Any, Optional, Union, List
from functools import wraps
import inspect

from pydantic_ai import Agent as PydanticAIAgent, Tool as PydanticTool
from pydantic_ai.tools import RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from pydantic import BaseModel, SecretStr

from .data_models.datamodels import Tool, AgentResult, ToolUsageResult
from ..utilities.hybrid_tool_registry import secure_registry
from ..utilities.registries import TOOLS_REGISTRY
from ..utilities.constants import get_api_url, get_base_model, get_api_key
from ..utilities.helpers import make_logger, supports_tool_choice_required, flatten_union_types
from ..memory import AsyncMemory

logger = make_logger(__name__)


class SecurePydanticAgent:
    """
    A secure wrapper around pydantic-ai Agent that provides:
    1. Security validation for all tool functions
    2. Backward compatibility with existing Agent interface
    3. Integration with our existing tool registry
    4. Enhanced security features on top of pydantic-ai's validation
    """
    
    def __init__(
        self,
        name: str,
        instructions: str,
        model: Optional[Union[OpenAIModel, str]] = None,
        tools: Optional[List[Union[str, Callable, Tool, PydanticTool]]] = None,
        api_key: Optional[SecretStr | str] = None,
        base_url: Optional[str] = None,
        memory: Optional[AsyncMemory] = None,
        use_registry: bool = True,
        validate_security: bool = True,
        deps_type: Optional[type] = None,
        output_type: Optional[type] = str,
        model_settings: Optional[ModelSettings | Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize a secure pydantic-ai agent.
        
        Args:
            name: Agent name
            instructions: System instructions/prompt
            model: Model to use (OpenAIModel instance or string)
            tools: List of tools to register
            api_key: API key for the model
            base_url: Base URL for the model API
            memory: Memory instance for conversation history
            use_registry: Whether to use the tool registry
            validate_security: Whether to validate tool security
            deps_type: Type for dependencies (pydantic-ai feature)
            output_type: Expected output type
            model_settings: Settings for the model
            **kwargs: Additional arguments
        """
        self.name = name
        self.instructions = instructions
        self.memory = memory
        self.use_registry = use_registry
        self.validate_security = validate_security
        
        # Resolve API credentials
        resolved_api_key = (
            api_key
            if isinstance(api_key, SecretStr)
            else SecretStr(api_key or get_api_key())
        )
        resolved_base_url = base_url or get_api_url()
        
        # Prepare model
        if isinstance(model, OpenAIModel):
            resolved_model = model
        else:
            provider = OpenAIProvider(
                base_url=resolved_base_url,
                api_key=resolved_api_key.get_secret_value(),
            )
            resolved_model = OpenAIModel(
                model_name=model if isinstance(model, str) else get_base_model(),
                provider=provider,
                **kwargs
            )
        
        # Process tools for security validation
        secure_tools = self._process_tools(tools) if tools else []
        
        # Create the underlying pydantic-ai agent
        self._pydantic_agent = PydanticAIAgent(
            model=resolved_model,
            tools=secure_tools,
            system_prompt=instructions,
            deps_type=deps_type,
            output_type=output_type,
            model_settings=model_settings or {},
            **kwargs
        )
        
        # Store tools for later reference
        self._tools = secure_tools
        
        logger.debug(f"Created secure pydantic-ai agent '{name}' with {len(secure_tools)} tools")
    
    def _process_tools(
        self, 
        tools: List[Union[str, Callable, Tool, PydanticTool]]
    ) -> List[PydanticTool]:
        """
        Process and validate tools for security.
        
        Args:
            tools: List of tools to process
            
        Returns:
            List of validated PydanticTool objects
        """
        secure_tools = []
        
        for tool in tools:
            if isinstance(tool, str):
                # String reference - look up in registry
                if self.use_registry and tool in TOOLS_REGISTRY:
                    registry_tool = TOOLS_REGISTRY[tool]
                    func = registry_tool.fn
                    
                    # Create secure pydantic tool
                    pydantic_tool = secure_registry.create_pydantic_tool(
                        func=func,
                        name=tool,
                        validate_security=self.validate_security
                    )
                    secure_tools.append(pydantic_tool)
                    
                elif not self.use_registry:
                    raise ValueError(
                        f"String tool reference '{tool}' not supported when use_registry=False"
                    )
                else:
                    raise ValueError(f"Tool '{tool}' not found in registry")
            
            elif callable(tool):
                # Callable function - create tool directly
                pydantic_tool = secure_registry.create_pydantic_tool(
                    func=tool,
                    validate_security=self.validate_security
                )
                secure_tools.append(pydantic_tool)
            
            elif isinstance(tool, Tool):
                # Our Tool object - extract function and create pydantic tool
                func = tool.fn
                pydantic_tool = secure_registry.create_pydantic_tool(
                    func=func,
                    name=tool.name,
                    validate_security=self.validate_security
                )
                secure_tools.append(pydantic_tool)
            
            elif isinstance(tool, PydanticTool):
                # Already a PydanticTool - validate and use
                if self.validate_security:
                    # Extract function from PydanticTool for validation
                    func = tool.function
                    secure_registry._validate_function_security(tool.name, func)
                
                secure_tools.append(tool)
            
            else:
                raise ValueError(f"Invalid tool type: {type(tool)}")
        
        return secure_tools
    
    def add_tool(self, tool: Union[str, Callable, Tool, PydanticTool]) -> None:
        """
        Add a tool to the agent.
        
        Args:
            tool: Tool to add
        """
        secure_tool = self._process_tools([tool])[0]
        self._tools.append(secure_tool)
        
        # We need to recreate the pydantic agent with the new tool
        # This is a limitation of the current pydantic-ai API
        self._pydantic_agent = PydanticAIAgent(
            model=self._pydantic_agent.model,
            tools=self._tools,
            system_prompt=self.instructions,
            deps_type=self._pydantic_agent.deps_type,
            output_type=self._pydantic_agent.output_type,
            model_settings=self._pydantic_agent.model_settings,
        )
        
        logger.debug(f"Added tool to agent '{self.name}'")
    
    def tool(self, func: Optional[Callable] = None, *, name: Optional[str] = None) -> Callable:
        """
        Decorator to register a tool that takes RunContext.
        
        Args:
            func: Function to register
            name: Optional custom name
            
        Returns:
            Decorated function
        """
        def decorator(f: Callable) -> Callable:
            # Register with security validation
            secure_registry.register_tool_with_agent(
                agent=self._pydantic_agent,
                func=f,
                name=name,
                takes_ctx=True,
                validate_security=self.validate_security
            )
            return f
        
        if func is None:
            return decorator
        else:
            return decorator(func)
    
    def tool_plain(self, func: Optional[Callable] = None, *, name: Optional[str] = None) -> Callable:
        """
        Decorator to register a tool that doesn't take RunContext.
        
        Args:
            func: Function to register
            name: Optional custom name
            
        Returns:
            Decorated function
        """
        def decorator(f: Callable) -> Callable:
            # Register with security validation
            secure_registry.register_tool_with_agent(
                agent=self._pydantic_agent,
                func=f,
                name=name,
                takes_ctx=False,
                validate_security=self.validate_security
            )
            return f
        
        if func is None:
            return decorator
        else:
            return decorator(func)
    
    async def run(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        deps: Optional[Any] = None,
        **kwargs
    ) -> AgentResult:
        """
        Run the agent asynchronously.
        
        Args:
            query: The query to run
            conversation_id: Optional conversation ID
            deps: Optional dependencies for pydantic-ai
            **kwargs: Additional arguments
            
        Returns:
            AgentResult with the response
        """
        # Load message history if memory is available
        message_history = None
        if self.memory and conversation_id:
            try:
                message_history = await self.memory.get_message_history(conversation_id, 100)
            except Exception as e:
                logger.warning(f"Error loading message history: {e}")
        
        # Add message history to kwargs if available
        if message_history:
            kwargs['message_history'] = message_history
        
        # Run the pydantic-ai agent
        if deps is not None:
            result = await self._pydantic_agent.run(query, deps=deps, **kwargs)
        else:
            result = await self._pydantic_agent.run(query, **kwargs)
        
        # Store in memory if available
        if self.memory and conversation_id:
            try:
                await self.memory.store_run_history(conversation_id, result)
            except Exception as e:
                logger.warning(f"Error storing run history: {e}")
        
        # Convert to our AgentResult format
        return self._convert_to_agent_result(result, query, conversation_id)
    
    def run_sync(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        deps: Optional[Any] = None,
        **kwargs
    ) -> AgentResult:
        """
        Run the agent synchronously.
        
        Args:
            query: The query to run
            conversation_id: Optional conversation ID
            deps: Optional dependencies for pydantic-ai
            **kwargs: Additional arguments
            
        Returns:
            AgentResult with the response
        """
        if deps is not None:
            result = self._pydantic_agent.run_sync(query, deps=deps, **kwargs)
        else:
            result = self._pydantic_agent.run_sync(query, **kwargs)
        
        # Convert to our AgentResult format
        return self._convert_to_agent_result(result, query, conversation_id)
    
    def _convert_to_agent_result(
        self,
        result: AgentRunResult,
        query: str,
        conversation_id: Optional[str]
    ) -> AgentResult:
        """
        Convert pydantic-ai result to our AgentResult format.
        
        Args:
            result: The pydantic-ai result
            query: The original query
            conversation_id: Optional conversation ID
            
        Returns:
            AgentResult instance
        """
        # Extract tool usage results (simplified for now)
        tool_usage_results = []
        
        # TODO: Extract tool usage from pydantic-ai result
        # This would require parsing the result.all_messages() if available
        
        return AgentResult(
            result=result.output,
            conversation_id=conversation_id or "default",
            full_result=result,
            tool_usage_results=tool_usage_results,
        )
    
    @property
    def tools(self) -> List[PydanticTool]:
        """Get the list of registered tools."""
        return self._tools.copy()
    
    @property
    def model(self) -> OpenAIModel:
        """Get the underlying model."""
        return self._pydantic_agent.model
    
    def __repr__(self) -> str:
        return f"SecurePydanticAgent(name='{self.name}', tools={len(self._tools)})"