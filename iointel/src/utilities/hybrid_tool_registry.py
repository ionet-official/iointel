"""
Hybrid tool registry that combines pydantic-ai's native tool system with our security features.
This provides the best of both worlds: pydantic-ai's advanced tool validation and schema generation
with our security validation and anti-spoofing measures.
"""

import inspect
import hashlib
from typing import Callable, Dict, Any, Optional, Union, List
from functools import wraps

from pydantic_ai import Agent as PydanticAIAgent, Tool as PydanticTool
from pydantic_ai.tools import RunContext, ToolDefinition

from .registries import TOOLS_REGISTRY
from .helpers import make_logger
from ..agent_methods.data_models.datamodels import Tool

logger = make_logger(__name__)


class SecureToolRegistry:
    """
    A secure tool registry that wraps pydantic-ai's tool system with security validation.
    Maintains function signature validation and anti-spoofing measures.
    """
    
    def __init__(self):
        self._function_signatures: Dict[str, bytes] = {}
        self._registered_tools: Dict[str, Callable] = {}
        self._pydantic_agents: Dict[str, PydanticAIAgent] = {}
    
    def _compute_function_signature(self, func: Callable) -> bytes:
        """Compute a signature hash for a function to detect spoofing."""
        try:
            # Get function bytecode for comparison
            if hasattr(func, '__code__'):
                signature_data = func.__code__.co_code
            else:
                # For built-in functions or other cases
                signature_data = str(func).encode()
            
            # Include function name and module for additional verification
            func_info = f"{func.__module__}.{func.__qualname__}".encode()
            
            return hashlib.sha256(signature_data + func_info).digest()
        except Exception as e:
            logger.warning(f"Could not compute signature for {func}: {e}")
            return b""
    
    def _validate_function_security(self, tool_name: str, func: Callable) -> bool:
        """Validate that a function hasn't been spoofed."""
        current_signature = self._compute_function_signature(func)
        
        if tool_name in self._function_signatures:
            stored_signature = self._function_signatures[tool_name]
            if current_signature != stored_signature:
                raise ValueError(
                    f"Tool '{tool_name}' signature mismatch. Potential spoofing detected."
                )
        else:
            self._function_signatures[tool_name] = current_signature
        
        return True
    
    def register_tool_with_agent(
        self, 
        agent: PydanticAIAgent, 
        func: Callable, 
        name: Optional[str] = None,
        takes_ctx: bool = False,
        validate_security: bool = True
    ) -> Callable:
        """
        Register a tool with a pydantic-ai agent while maintaining security validation.
        
        Args:
            agent: The pydantic-ai agent to register the tool with
            func: The function to register as a tool
            name: Optional custom name for the tool
            takes_ctx: Whether the tool takes RunContext as first parameter
            validate_security: Whether to perform security validation
        
        Returns:
            The original function (for decorator usage)
        """
        tool_name = name or func.__name__
        
        if validate_security:
            self._validate_function_security(tool_name, func)
        
        # Store reference to the function for later validation
        self._registered_tools[tool_name] = func
        
        # Create a wrapper that validates the function on each call
        @wraps(func)
        def secure_wrapper(*args, **kwargs):
            if validate_security:
                # Re-validate function signature on each call
                self._validate_function_security(tool_name, func)
            return func(*args, **kwargs)
        
        # Register with pydantic-ai agent
        if takes_ctx:
            agent.tool(secure_wrapper, name=tool_name)
        else:
            agent.tool_plain(secure_wrapper, name=tool_name)
        
        # Also register in our legacy registry for backward compatibility
        if tool_name not in TOOLS_REGISTRY:
            TOOLS_REGISTRY[tool_name] = Tool.from_function(func, name=tool_name)
        
        logger.debug(f"Registered secure tool '{tool_name}' with agent")
        return func
    
    def register_tool_function(
        self, 
        func: Callable, 
        name: Optional[str] = None,
        validate_security: bool = True
    ) -> Callable:
        """
        Register a tool function for later use with agents.
        This is similar to the old @register_tool decorator but with security validation.
        
        Args:
            func: The function to register as a tool
            name: Optional custom name for the tool
            validate_security: Whether to perform security validation
        
        Returns:
            The original function (for decorator usage)
        """
        tool_name = name or func.__name__
        
        if validate_security:
            self._validate_function_security(tool_name, func)
        
        # Store in legacy registry for backward compatibility
        TOOLS_REGISTRY[tool_name] = Tool.from_function(func, name=tool_name)
        self._registered_tools[tool_name] = func
        
        logger.debug(f"Registered tool function '{tool_name}' in registry")
        return func
    
    def get_tool_function(self, tool_name: str) -> Optional[Callable]:
        """Get a registered tool function by name."""
        return self._registered_tools.get(tool_name)
    
    def create_pydantic_tool(
        self, 
        func: Callable, 
        name: Optional[str] = None,
        takes_ctx: bool = False,
        validate_security: bool = True
    ) -> PydanticTool:
        """
        Create a pydantic-ai Tool object with security validation.
        
        Args:
            func: The function to create a tool from
            name: Optional custom name for the tool
            takes_ctx: Whether the tool takes RunContext as first parameter
            validate_security: Whether to perform security validation
        
        Returns:
            A pydantic-ai Tool object
        """
        tool_name = name or func.__name__
        
        if validate_security:
            self._validate_function_security(tool_name, func)
        
        # Create wrapper with security validation
        @wraps(func)
        def secure_wrapper(*args, **kwargs):
            if validate_security:
                self._validate_function_security(tool_name, func)
            return func(*args, **kwargs)
        
        # Create pydantic-ai Tool
        pydantic_tool = PydanticTool(secure_wrapper, name=tool_name, takes_ctx=takes_ctx)
        
        # Also register in legacy registry
        if tool_name not in TOOLS_REGISTRY:
            TOOLS_REGISTRY[tool_name] = Tool.from_function(func, name=tool_name)
        
        self._registered_tools[tool_name] = func
        
        logger.debug(f"Created secure pydantic-ai tool '{tool_name}'")
        return pydantic_tool
    
    def validate_tool_execution(self, tool_name: str, func: Callable) -> bool:
        """
        Validate that a tool function is safe to execute.
        Called before tool execution to ensure no spoofing.
        """
        if tool_name not in self._registered_tools:
            logger.warning(f"Tool '{tool_name}' not found in secure registry")
            return False
        
        registered_func = self._registered_tools[tool_name]
        
        # Validate function signatures match
        current_sig = self._compute_function_signature(func)
        registered_sig = self._compute_function_signature(registered_func)
        
        if current_sig != registered_sig:
            logger.error(f"Tool '{tool_name}' signature mismatch during execution")
            return False
        
        return True
    
    def list_registered_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._registered_tools.keys())
    
    def clear_registry(self):
        """Clear all registered tools (useful for testing)."""
        self._function_signatures.clear()
        self._registered_tools.clear()
        self._pydantic_agents.clear()


# Global secure tool registry instance
secure_registry = SecureToolRegistry()


def register_secure_tool(
    _fn: Optional[Callable] = None,
    name: Optional[str] = None,
    takes_ctx: bool = False,
    validate_security: bool = True
) -> Callable:
    """
    Decorator to register a tool with security validation.
    This is a drop-in replacement for the old @register_tool decorator
    that adds security validation and pydantic-ai compatibility.
    
    Args:
        _fn: The function to register (when used as @register_secure_tool)
        name: Optional custom name for the tool
        takes_ctx: Whether the tool takes RunContext as first parameter
        validate_security: Whether to perform security validation
    
    Usage:
        @register_secure_tool
        def my_tool(param: str) -> str:
            return f"Result: {param}"
        
        @register_secure_tool(name="custom_name", takes_ctx=True)
        def my_context_tool(ctx: RunContext, param: str) -> str:
            return f"Result: {param}"
    """
    def decorator(func: Callable) -> Callable:
        return secure_registry.register_tool_function(
            func=func,
            name=name,
            validate_security=validate_security
        )
    
    # Handle usage as @register_secure_tool or @register_secure_tool(...)
    if _fn is None:
        return decorator
    else:
        return decorator(_fn)


def create_agent_with_secure_tools(
    model: str,
    tools: Optional[List[Union[str, Callable, PydanticTool]]] = None,
    validate_security: bool = True,
    **kwargs
) -> PydanticAIAgent:
    """
    Create a pydantic-ai Agent with secure tool registration.
    
    Args:
        model: The model name for the agent
        tools: List of tools (function names, callables, or PydanticTool objects)
        validate_security: Whether to perform security validation
        **kwargs: Additional arguments passed to PydanticAgent
    
    Returns:
        A pydantic-ai Agent with securely registered tools
    """
    # Create the pydantic-ai agent
    agent = PydanticAIAgent(model=model, **kwargs)
    
    if tools:
        pydantic_tools = []
        
        for tool in tools:
            if isinstance(tool, str):
                # String reference - look up in registry
                if tool in TOOLS_REGISTRY:
                    func = TOOLS_REGISTRY[tool].fn
                    pydantic_tool = secure_registry.create_pydantic_tool(
                        func=func,
                        name=tool,
                        validate_security=validate_security
                    )
                    pydantic_tools.append(pydantic_tool)
                else:
                    raise ValueError(f"Tool '{tool}' not found in registry")
            
            elif callable(tool):
                # Callable function - create tool directly
                pydantic_tool = secure_registry.create_pydantic_tool(
                    func=tool,
                    validate_security=validate_security
                )
                pydantic_tools.append(pydantic_tool)
            
            elif isinstance(tool, PydanticTool):
                # Already a PydanticTool - use as is
                pydantic_tools.append(tool)
            
            else:
                raise ValueError(f"Invalid tool type: {type(tool)}")
        
        # Add tools to agent
        agent = PydanticAIAgent(model=model, tools=pydantic_tools, **kwargs)
    
    return agent