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
        
        if name in self:
            existing_tool = self[name]
            
            # Check if this is a bound method vs unbound method
            existing_is_bound = self._is_bound_method(existing_tool.fn)
            new_is_bound = self._is_bound_method(tool.fn)
            
            # Priority: bound methods > unbound methods
            if new_is_bound and not existing_is_bound:
                logger.info(f"🔄 Upgrading tool '{name}' from unbound to bound method")
                super().__setitem__(name, tool)
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
                    super().__setitem__(name, tool)
            else:
                # Both unbound - check if they're the same function
                if existing_tool.fn == tool.fn:
                    logger.debug(f"⏭️  Skipping duplicate unbound registration of '{name}'")
                    return
                else:
                    # For agno tools, check if they have unique IDs
                    existing_id = getattr(existing_tool.fn, '_agno_tool_id', None)
                    new_id = getattr(tool.fn, '_agno_tool_id', None)
                    
                    if existing_id and new_id and existing_id != new_id:
                        logger.warning(f"⚠️  Different agno tools with same name '{name}' - this is wrong!")
                        logger.warning(f"   Existing ID: {existing_id}")
                        logger.warning(f"   New ID: {new_id}")
                        # Don't overwrite - keep the existing one
                        return
                    else:
                        logger.warning(f"⚠️  Multiple unbound functions for '{name}' - using latest")
                        super().__setitem__(name, tool)
        else:
            # First registration
            method_type = "bound" if self._is_bound_method(tool.fn) else "unbound"
            logger.debug(f"✅ Registered {method_type} tool '{name}'")
            super().__setitem__(name, tool)
    
    def _is_bound_method(self, func) -> bool:
        """Check if a function is a bound method"""
        return hasattr(func, '__self__') or (hasattr(func, '_bound_method') and func._bound_method)
    
    def _same_instance_source(self, func1, func2) -> bool:
        """Check if two bound methods come from the same instance"""
        if hasattr(func1, '__self__') and hasattr(func2, '__self__'):
            return func1.__self__ is func2.__self__
        return False
    
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
