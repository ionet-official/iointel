from pydantic import BaseModel
from agno.tools import Toolkit

from iointel.src.agent_methods.tools.utils import register_tool


def _create_bound_wrapper(bound_method, original_method, tool_name):
    """Create a wrapper function for bound methods with clean qualified names."""
    import inspect
    
    # Get the original method signature and remove 'self' if present
    try:
        sig = inspect.signature(original_method)
    except (ValueError, TypeError):
        # If we can't get signature from original method, try bound method
        sig = inspect.signature(bound_method)
    
    params = list(sig.parameters.values())
    
    # Remove 'self' parameter if it exists
    if params and params[0].name == 'self':
        params = params[1:]
    
    # Create new signature without 'self'
    new_sig = sig.replace(parameters=params)
    
    # Create wrapper function that calls the bound method
    # Use exec to create a unique function body for each tool
    wrapper_code = f"""
def {tool_name}(*args, **kwargs):
    '''Tool wrapper for {tool_name}'''
    return bound_method(*args, **kwargs)
"""
    
    # Create a unique namespace for this wrapper
    namespace = {'bound_method': bound_method}
    exec(wrapper_code, namespace)
    agno_tool_wrapper = namespace[tool_name]
    
    # Set the correct signature and metadata
    agno_tool_wrapper.__signature__ = new_sig
    agno_tool_wrapper.__name__ = tool_name
    agno_tool_wrapper.__qualname__ = tool_name  # Clean qualified name
    agno_tool_wrapper.__doc__ = getattr(original_method, '__doc__', f"Wrapped agno tool: {tool_name}")
    
    # Copy annotations from original method, excluding 'self'
    original_annotations = getattr(original_method, '__annotations__', {})
    new_annotations = {k: v for k, v in original_annotations.items() if k != 'self'}
    agno_tool_wrapper.__annotations__ = new_annotations
    
    # Set a unique marker to distinguish different agno tools
    agno_tool_wrapper._agno_tool_id = f"{tool_name}_{id(bound_method)}"
    
    return agno_tool_wrapper


class DisableAgnoRegistryMixin:
    """
    Put this as first parent class when inheriting
    from Agno tool to disable Agno registry,
    because we only care about our own registry."""

    def _register_tools(self):
        """Disabled in favour of iointel registry."""

    def register(self, function, name=None):
        """Disabled in favour of iointel registry."""


def make_base(agno_tool_cls: type[Toolkit]):
    class BaseAgnoTool(BaseModel):
        class Inner(DisableAgnoRegistryMixin, agno_tool_cls):
            pass

        def _get_tool(self) -> Inner:
            raise NotImplementedError()

        _tool: Inner

        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            self._tool = self._get_tool()
            
            # Register bound methods for tools marked with @wrap_tool
            self._register_bound_methods()
        
        def _register_bound_methods(self):
            """Register bound methods for tools marked with @wrap_tool."""
            import os
            
            # Import io_logger for better logging
            try:
                from ....utilities.io_logger import IOLogger
                logger = IOLogger()
            except ImportError:
                logger = None
            
            verbose_mode = os.getenv("IOINTEL_VERBOSE_TOOL_REGISTRATION", "false").lower() in ("true", "1", "yes")
            registered_tools = []
            
            if verbose_mode and logger:
                logger.info(f"Starting bound method registration for {self.__class__.__name__}")
            
            # Look for methods that were marked by @wrap_tool
            # Use the class's dict to avoid accessing class-only attributes
            for attr_name in dir(self.__class__):
                if attr_name.startswith('_') and not attr_name.startswith('__'):
                    continue  # Skip private attributes
                try:
                    attr = getattr(self, attr_name)
                    if hasattr(attr, '_should_register') and attr._should_register:
                        tool_name = attr._tool_name
                        agno_method = attr._agno_method
                        
                        if verbose_mode and logger:
                            logger.debug(f"Registering bound tool '{tool_name}'...")
                        
                        # Create the bound wrapper and register it
                        bound_wrapper = _create_bound_wrapper(attr, agno_method, tool_name)
                        
                        # Set a unique __doc__ to ensure Tool has unique body when getsource fails
                        # This prevents all agno tools from having body=None and being treated as identical
                        bound_wrapper.__doc__ = f"{bound_wrapper.__doc__ or ''}\n[AgnoTool:{self.__class__.__name__}.{tool_name}]"
                        
                        register_tool(name=tool_name)(bound_wrapper)
                        registered_tools.append(tool_name)
                        
                        if verbose_mode and logger:
                            logger.success(f"Registered bound tool '{tool_name}' from {self.__class__.__name__}")
                        
                except AttributeError:
                    # Skip attributes that can't be accessed (like __signature__)
                    continue
                except Exception as e:
                    # Special handling for Pydantic CallableSchema errors - these are known issues
                    if "CallableSchema" in str(e):
                        if logger:
                            logger.error(f"Failed to register {attr_name}: Cannot generate a JsonSchema for core_schema.CallableSchema")
                        else:
                            print(f"❌ Failed to register {attr_name}: Cannot generate a JsonSchema for core_schema.CallableSchema")
                        # Don't print full traceback for known CallableSchema issues
                    else:
                        if logger:
                            logger.error(f"Failed to register {attr_name}: {e}")
                        else:
                            print(f"❌ Failed to register {attr_name}: {e}")
                        import traceback
                        traceback.print_exc()
            
            # Log summary if not in verbose mode and tools were registered
            if not verbose_mode and registered_tools and logger:
                tool_count = len(registered_tools)
                tool_preview = ", ".join(registered_tools[:3])
                if tool_count > 3:
                    tool_preview += f", +{tool_count - 3} more"
                
                logger.info(
                    f"Registered {tool_count} tools from {self.__class__.__name__}: {tool_preview}",
                    data={"tool_count": tool_count, "active_agno_tools": registered_tools, "class": self.__class__.__name__}
                )

    return BaseAgnoTool


def wrap_tool(name, agno_method):
    def wrapper(func):
        # Don't register the tool immediately - just mark it for registration
        # The registration will happen when the class is instantiated with bound methods
        func._tool_name = name
        func._agno_method = agno_method
        func._should_register = True
        
        # Ensure no immediate registration by completely bypassing any decorators
        # Return the original function without any registration
        return func
    return wrapper
