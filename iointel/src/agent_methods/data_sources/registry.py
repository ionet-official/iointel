"""
Data Source Registry
===================

Registry for data source implementations, separate from the tools registry.
"""

from typing import Dict, Callable, List
from iointel.src.agent_methods.data_sources.models import DataSourceRequest, DataSourceResponse

# Global registry for data sources
_DATA_SOURCE_REGISTRY: Dict[str, Callable[[DataSourceRequest], DataSourceResponse]] = {}


def register_data_source(name: str):
    """Decorator to register a data source implementation."""
    def decorator(func: Callable[[DataSourceRequest], DataSourceResponse]):
        _DATA_SOURCE_REGISTRY[name] = func
        return func
    return decorator


def get_data_source(name: str) -> Callable[[DataSourceRequest], DataSourceResponse]:
    """Get a data source implementation by name."""
    # Auto-load data sources if not already loaded
    if not _DATA_SOURCE_REGISTRY:
        _load_builtin_data_sources()
    
    if name not in _DATA_SOURCE_REGISTRY:
        raise ValueError(f"Data source '{name}' not found. Available: {list(_DATA_SOURCE_REGISTRY.keys())}")
    return _DATA_SOURCE_REGISTRY[name]


def list_data_sources() -> List[str]:
    """List all registered data source names."""
    # Auto-load data sources if not already loaded
    if not _DATA_SOURCE_REGISTRY:
        _load_builtin_data_sources()
    return list(_DATA_SOURCE_REGISTRY.keys())


def _load_builtin_data_sources():
    """Load built-in data sources."""
    try:
        # Import to trigger registration
        from . import input_sources  # noqa: F401
    except ImportError:
        pass  # Skip if not available