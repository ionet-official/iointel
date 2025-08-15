"""
Data Flow Resolver - Resolves variable references in workflow task configurations.

This module provides the missing piece for workflow data flow: resolving {node_id.field}
references in task configurations using results from previous task executions.
"""

import re
from typing import Optional, Any, Dict
from .io_logger import get_component_logger

logger = get_component_logger("DATA_FLOW", grouped=True)


class DataFlowResolver:
    """Resolves variable references in workflow task configurations."""
    
    def __init__(self) -> None:
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
        with logger.group("Config Resolution"):
            logger.info("Starting config resolution", {
                "config_keys": list(config.keys()),
                "available_nodes": list(results.keys()),
                "config_size": len(config)
            })
            
            resolved = {}
            for key, value in config.items():
                with logger.group(f"Resolving '{key}'", suppress_header=True):
                    resolved_value = self._resolve_value(value, results)
                    resolved[key] = resolved_value
                    
                    # Log resolution result
                    if value != resolved_value:
                        logger.success(f"Resolved '{key}'", {
                            "original": str(value)[:100] + "..." if len(str(value)) > 100 else value,
                            "resolved": str(resolved_value)[:100] + "..." if len(str(resolved_value)) > 100 else resolved_value,
                            "type_changed": type(value).__name__ != type(resolved_value).__name__
                        })
                    else:
                        logger.debug(f"No resolution needed for '{key}' (literal value)")
            
            logger.success("Config resolution complete", {
                "resolved_count": len([k for k in config if config[k] != resolved[k]]),
                "unchanged_count": len([k for k in config if config[k] == resolved[k]])
            })
            
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
        
        with logger.group(f"Reference: {reference}"):
            logger.debug("Parsing reference", {
                "full_reference": reference,
                "node_id": node_id,
                "field_path": parts[1:] if len(parts) > 1 else None,
                "available_nodes": list(results.keys())
            })
            
            if node_id not in results:
                available_nodes = list(results.keys())
                error_msg = f"Cannot resolve reference '{reference}': node '{node_id}' not found"
                logger.error(error_msg, {
                    "requested_node": node_id,
                    "available_nodes": available_nodes
                })
                raise ValueError(f"{error_msg}. Available nodes: {available_nodes}")
            
            node_result = results[node_id]
            result_type = type(node_result).__name__
            
            # If no field specified, extract human-readable content if possible
            if len(parts) == 1:
                # Try to extract clean text from typed result objects
                from ..agent_methods.data_models.execution_models import AgentExecutionResult, DataSourceResult
                
                if isinstance(node_result, AgentExecutionResult) and node_result.agent_response:
                    logger.info("Extracting from AgentExecutionResult", {
                        "has_response": bool(node_result.agent_response),
                        "has_result": bool(node_result.agent_response and node_result.agent_response.result)
                    })
                    return node_result.agent_response.result
                elif isinstance(node_result, DataSourceResult):
                    logger.info("Extracting from DataSourceResult", {
                        "result_preview": str(node_result.result)[:50] + "..." if len(str(node_result.result)) > 50 else node_result.result
                    })
                    return node_result.result
                else:
                    logger.info(f"Returning raw {result_type} (no field specified)")
                    return node_result
            
            # Navigate through nested fields
            with logger.group("Field Navigation"):
                current_value = node_result
                
                for i, field in enumerate(parts[1:], 1):
                    value_type = type(current_value).__name__
                    
                    if isinstance(current_value, dict):
                        if field not in current_value:
                            available_fields = list(current_value.keys())
                            logger.error(f"Field '{field}' not found in dict", {
                                "field": field,
                                "available_fields": available_fields,
                                "path_so_far": ".".join(parts[:i])
                            })
                            raise ValueError(
                                f"Cannot resolve reference '{reference}': field '{field}' not found in {node_id}. "
                                f"Available fields: {available_fields}"
                            )
                        current_value = current_value[field]
                        logger.debug(f"Accessed dict['{field}']", {
                            "new_type": type(current_value).__name__,
                            "is_final": i == len(parts) - 1
                        })
                    else:
                        # Try to access as attribute for objects
                        if hasattr(current_value, field):
                            current_value = getattr(current_value, field)
                            logger.debug(f"Accessed .{field} attribute", {
                                "object_type": value_type,
                                "new_type": type(current_value).__name__,
                                "is_final": i == len(parts) - 1
                            })
                        else:
                            # List available attributes for debugging
                            available_attrs = [attr for attr in dir(current_value) if not attr.startswith('_')][:10]
                            logger.error(f"Field '{field}' not accessible", {
                                "field": field,
                                "object_type": value_type,
                                "sample_attributes": available_attrs,
                                "path_so_far": ".".join(parts[:i])
                            })
                            raise ValueError(
                                f"Cannot resolve reference '{reference}': field '{field}' not accessible in {node_id}. "
                                f"Value type: {type(current_value)}"
                            )
                
                logger.success("Field navigation complete", {
                    "final_type": type(current_value).__name__,
                    "final_value_preview": str(current_value)[:50] + "..." if len(str(current_value)) > 50 else current_value
                })
                
                return current_value
    
    def validate_references(self, config: Dict[str, Any], available_nodes: Optional[set] = None) -> list[str]:
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