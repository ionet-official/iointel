from typing import Dict, Callable, TYPE_CHECKING
import logging


if TYPE_CHECKING:
    from pydantic import BaseModel
    from ..agent_methods.data_models.datamodels import Tool

logger = logging.getLogger(__name__)

# A global registry mapping task types to executor functions.
TASK_EXECUTOR_REGISTRY: Dict[str, Callable] = {}

# A global registry mapping chainable method names to functions.
CHAINABLE_METHODS: Dict[str, Callable] = {}

# A global or module-level registry of custom workflows
CUSTOM_WORKFLOW_REGISTRY: Dict[str, Callable] = {}

# Enhanced tool registry with conflict resolution
class SmartToolRegistry(dict):
    """
    Smart tool registry that handles conflicts intelligently:
    - Prevents duplicate registrations
    - Prioritizes bound methods over unbound methods
    - Provides clear logging of registration decisions
    """
    
    def __setitem__(self, name: str, tool: "Tool"):
        # Normalize bound methods to plain functions BEFORE storing
        normalized_tool = self._normalize_tool(tool)
        
        if name in self:
            existing_tool = self[name]
            
            # Check if this is a bound method vs unbound method (on original tools)
            existing_is_bound = self._is_bound_method(existing_tool.fn)
            new_is_bound = self._is_bound_method(tool.fn)  # Check original tool, not normalized
            
            # Priority: bound methods > unbound methods
            if new_is_bound and not existing_is_bound:
                logger.info(f"🔄 Upgrading tool '{name}' from unbound to bound method")
                super().__setitem__(name, normalized_tool)
            elif not new_is_bound and existing_is_bound:
                logger.debug(f"⏭️  Skipping unbound registration of '{name}' (bound version exists)")
                return
            elif new_is_bound and existing_is_bound:
                # Both bound - check if they're the same instance
                if self._same_instance_source(existing_tool.fn, tool.fn):
                    logger.debug(f"⏭️  Skipping duplicate bound registration of '{name}'")
                    return
                else:
                    logger.warning(f"⚠️  Multiple bound methods for '{name}' - using latest")
                    super().__setitem__(name, normalized_tool)
            else:
                # Both unbound - check if they're the same function
                if existing_tool.fn == normalized_tool.fn:
                    logger.debug(f"⏭️  Skipping duplicate unbound registration of '{name}'")
                    return
                else:
                    # For agno tools, check if they have unique IDs
                    existing_id = getattr(existing_tool.fn, '_agno_tool_id', None)
                    new_id = getattr(normalized_tool.fn, '_agno_tool_id', None)
                    
                    if existing_id and new_id and existing_id != new_id:
                        logger.warning(f"⚠️  Different agno tools with same name '{name}' - this is wrong!")
                        logger.warning(f"   Existing ID: {existing_id}")
                        logger.warning(f"   New ID: {new_id}")
                        # Don't overwrite - keep the existing one
                        return
                    else:
                        logger.warning(f"⚠️  Multiple unbound functions for '{name}' - using latest")
                        super().__setitem__(name, normalized_tool)
        else:
            # First registration
            method_type = "bound" if self._is_bound_method(tool.fn) else "unbound"
            logger.debug(f"✅ Registered {method_type} tool '{name}' (normalized to function)")
            super().__setitem__(name, normalized_tool)
    
    def _is_bound_method(self, func) -> bool:
        """Check if a function is a bound method"""
        return hasattr(func, '__self__') or (hasattr(func, '_bound_method') and func._bound_method)
    
    def _same_instance_source(self, func1, func2) -> bool:
        """Check if two bound methods come from the same instance"""
        if hasattr(func1, '__self__') and hasattr(func2, '__self__'):
            return func1.__self__ is func2.__self__
        return False
    
    def _normalize_tool(self, tool: "Tool") -> "Tool":
        """
        Normalize a tool to ensure it uses a plain function without 'self' parameter.
        
        For bound methods, creates a wrapper function that calls the bound method
        without exposing the 'self' parameter to pydantic-ai.
        
        Args:
            tool: Original tool (may have bound method)
            
        Returns:
            Tool with normalized function (no 'self' parameter)
        """
        if not self._is_bound_method(tool.fn):
            # Already a plain function, return as-is
            return tool
        
        # Create a wrapper function that calls the bound method
        bound_method = tool.fn
        
        # Get the original function signature (without 'self')
        import inspect
        sig = inspect.signature(bound_method)
        
        # Remove 'self' parameter from signature
        params = list(sig.parameters.values())[1:]  # Skip 'self'
        new_sig = sig.replace(parameters=params)
        
        # Create wrapper function
        def wrapper(*args, **kwargs):
            return bound_method(*args, **kwargs)
        
        # Copy metadata from original bound method
        wrapper.__name__ = bound_method.__name__
        wrapper.__doc__ = bound_method.__doc__
        wrapper.__signature__ = new_sig
        wrapper.__qualname__ = getattr(bound_method, '__qualname__', bound_method.__name__)
        
        # Copy any special attributes
        if hasattr(bound_method, '_agno_tool_id'):
            wrapper._agno_tool_id = bound_method._agno_tool_id
        
        # Create new tool with wrapper function
        if hasattr(tool, 'model_copy'):
            # Pydantic v2 style
            return tool.model_copy(update={"fn": wrapper})
        else:
            # Fallback - create new tool manually
            from ..agent_methods.data_models.datamodels import Tool
            return Tool(
                name=tool.name,
                description=tool.description,
                fn=wrapper,
                fn_metadata=getattr(tool, 'fn_metadata', None)
            )
    
    def get_tool_info(self, name: str) -> dict:
        """Get diagnostic information about a tool"""
        if name not in self:
            return {"exists": False}
        
        tool = self[name]
        return {
            "exists": True,
            "name": tool.name,
            "is_bound": self._is_bound_method(tool.fn),
            "function_name": tool.fn.__name__,
            "qualified_name": getattr(tool.fn, '__qualname__', 'unknown'),
            "source": str(type(tool.fn.__self__)) if hasattr(tool.fn, '__self__') else 'function'
        }

# Replace the simple dict with our smart registry
TOOLS_REGISTRY = SmartToolRegistry()

# A global registry of classes which instance methods are registered as tools
TOOL_SELF_REGISTRY: "Dict[str, type[BaseModel]]" = {}
