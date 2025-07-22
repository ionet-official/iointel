"""
Data Flow Resolver - Resolves variable references in workflow task configurations.

This module provides the missing piece for workflow data flow: resolving {node_id.field}
references in task configurations using results from previous task executions.
"""

import re
from typing import Any, Dict
from ..utilities.helpers import make_logger

logger = make_logger(__name__)


class DataFlowResolver:
    """Resolves variable references in workflow task configurations."""
    
    def __init__(self):
        # Pattern to match {node_id} or {node_id.field} or {node_id.field.subfield}
        self.reference_pattern = re.compile(r'\{([^}]+)\}')
    
    def resolve_config(self, config: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve variable references in a configuration dictionary.
        
        Args:
            config: Configuration dictionary that may contain {node_id.field} references
            results: Results from previous task executions (WorkflowState.results)
            
        Returns:
            Configuration with all variable references resolved
            
        Example:
            config = {"a": "{add_numbers.result}", "b": 3}
            results = {"add_numbers": 15.0}
            returns {"a": 15.0, "b": 3}
        """
        logger.debug(f"Resolving config: {config}")
        logger.debug(f"Available results: {list(results.keys())}")
        
        resolved = {}
        for key, value in config.items():
            resolved[key] = self._resolve_value(value, results)
        
        logger.debug(f"Resolved config: {resolved}")
        return resolved
    
    def _resolve_value(self, value: Any, results: Dict[str, Any]) -> Any:
        """Resolve a single value that may contain variable references."""
        if not isinstance(value, str):
            return value
        
        # Check if the entire value is a single reference like "{node_id.field}"
        if value.startswith('{') and value.endswith('}') and value.count('{') == 1:
            reference = value[1:-1]  # Remove { and }
            return self._resolve_reference(reference, results)
        
        # Handle multiple references in a string (template substitution)
        def replace_reference(match):
            reference = match.group(1)
            resolved_value = self._resolve_reference(reference, results)
            return str(resolved_value)
        
        return self.reference_pattern.sub(replace_reference, value)
    
    def _resolve_reference(self, reference: str, results: Dict[str, Any]) -> Any:
        """
        Resolve a single reference like 'node_id.field' or 'node_id'.
        
        Args:
            reference: Reference string like "add_numbers.result" or "add_numbers"
            results: Available results dictionary
            
        Returns:
            The resolved value
            
        Raises:
            ValueError: If reference cannot be resolved
        """
        parts = reference.split('.')
        node_id = parts[0]
        
        if node_id not in results:
            available_nodes = list(results.keys())
            raise ValueError(
                f"Cannot resolve reference '{reference}': node '{node_id}' not found. "
                f"Available nodes: {available_nodes}"
            )
        
        node_result = results[node_id]
        
        # If no field specified, return the entire result
        if len(parts) == 1:
            return node_result
        
        # Navigate through nested fields
        current_value = node_result
        for field in parts[1:]:
            if isinstance(current_value, dict):
                if field not in current_value:
                    available_fields = list(current_value.keys()) if isinstance(current_value, dict) else "not a dict"
                    raise ValueError(
                        f"Cannot resolve reference '{reference}': field '{field}' not found in {node_id}. "
                        f"Available fields: {available_fields}"
                    )
                current_value = current_value[field]
            else:
                # Try to access as attribute for objects
                if hasattr(current_value, field):
                    current_value = getattr(current_value, field)
                else:
                    raise ValueError(
                        f"Cannot resolve reference '{reference}': field '{field}' not accessible in {node_id}. "
                        f"Value type: {type(current_value)}"
                    )
        
        return current_value
    
    def validate_references(self, config: Dict[str, Any], available_nodes: set = None) -> list[str]:
        """
        Validate that all references in config can potentially be resolved.
        
        Args:
            config: Configuration to validate
            available_nodes: Set of node IDs that will be available (optional)
            
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        def check_value(value, path=""):
            if isinstance(value, str):
                matches = self.reference_pattern.findall(value)
                for reference in matches:
                    parts = reference.split('.')
                    node_id = parts[0]
                    
                    if available_nodes and node_id not in available_nodes:
                        issues.append(
                            f"Reference '{reference}' at {path} refers to unknown node '{node_id}'"
                        )
            elif isinstance(value, dict):
                for k, v in value.items():
                    check_value(v, f"{path}.{k}" if path else k)
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    check_value(v, f"{path}[{i}]" if path else f"[{i}]")
        
        for key, value in config.items():
            check_value(value, key)
        
        return issues


# Global instance for use throughout the workflow system
data_flow_resolver = DataFlowResolver()