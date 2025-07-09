"""
Decision-making tools for workflow conditionals.
These tools provide explicit, testable alternatives to string conditions in edges.
"""

import json
import re
from typing import Any, Dict, Union, Optional
from pydantic import BaseModel, Field

from ...utilities.decorators import register_tool


class DecisionResult(BaseModel):
    """Structured result for decision tools."""
    result: bool
    details: str
    confidence: Optional[float] = None


class RouterResult(BaseModel):
    """Result from routing tools."""
    routed_to: str
    route_data: Dict[str, Any] = Field(default_factory=dict)
    matched_condition: Optional[str] = None


@register_tool
def json_evaluator(
    data: Union[Dict, str],
    expression: str,
    context: Optional[Dict] = None
) -> DecisionResult:
    """
    Evaluate JSON data against a simple expression.
    
    Args:
        data: JSON data to evaluate (dict or JSON string)
        expression: Simple expression like "weather.condition == 'rain'" or "temperature > 20"
        context: Additional context variables
        
    Returns:
        DecisionResult with boolean result and details
    """
    try:
        # Parse JSON string if needed
        if isinstance(data, str):
            data = json.loads(data)
        
        # Simple expression evaluation for safety
        # Only allow basic operations and property access
        if not _is_safe_expression(expression):
            return DecisionResult(
                result=False,
                details=f"Unsafe expression: {expression}",
                confidence=0.0
            )
        
        # Create evaluation context
        eval_context = {"data": data}
        if context:
            eval_context.update(context)
        
        # Evaluate the expression
        try:
            result = eval(expression, {"__builtins__": {}}, eval_context)
            return DecisionResult(
                result=bool(result),
                details=f"Expression '{expression}' evaluated to {result}",
                confidence=1.0
            )
        except Exception as e:
            return DecisionResult(
                result=False,
                details=f"Evaluation error: {str(e)}",
                confidence=0.0
            )
            
    except Exception as e:
        return DecisionResult(
            result=False,
            details=f"JSON parsing error: {str(e)}",
            confidence=0.0
        )


@register_tool
def number_compare(
    value: Union[int, float, str],
    operator: str,
    threshold: Union[int, float],
    tolerance: Optional[float] = None
) -> DecisionResult:
    """
    Compare a number against a threshold using specified operator.
    
    Args:
        value: Number to compare (can be string that converts to number)
        operator: Comparison operator ('>', '<', '>=', '<=', '==', '!=')
        threshold: Value to compare against
        tolerance: For float comparisons, tolerance for equality (default: 1e-9)
        
    Returns:
        DecisionResult with comparison result
    """
    try:
        # Convert value to number if it's a string
        if isinstance(value, str):
            try:
                value = float(value) if '.' in value else int(value)
            except ValueError:
                return DecisionResult(
                    result=False,
                    details=f"Cannot convert '{value}' to number",
                    confidence=0.0
                )
        
        # Set default tolerance for float comparisons
        if tolerance is None:
            tolerance = 1e-9
        
        # Perform comparison
        if operator == '>':
            result = value > threshold
        elif operator == '<':
            result = value < threshold
        elif operator == '>=':
            result = value >= threshold
        elif operator == '<=':
            result = value <= threshold
        elif operator == '==':
            if isinstance(value, float) or isinstance(threshold, float):
                result = abs(value - threshold) <= tolerance
            else:
                result = value == threshold
        elif operator == '!=':
            if isinstance(value, float) or isinstance(threshold, float):
                result = abs(value - threshold) > tolerance
            else:
                result = value != threshold
        else:
            return DecisionResult(
                result=False,
                details=f"Unknown operator: {operator}",
                confidence=0.0
            )
        
        return DecisionResult(
            result=result,
            details=f"{value} {operator} {threshold} = {result}",
            confidence=1.0
        )
        
    except Exception as e:
        return DecisionResult(
            result=False,
            details=f"Comparison error: {str(e)}",
            confidence=0.0
        )


@register_tool
def string_contains(
    text: str,
    substring: str,
    case_sensitive: bool = True,
    regex: bool = False
) -> DecisionResult:
    """
    Check if a string contains a substring or matches a pattern.
    
    Args:
        text: Text to search in
        substring: Substring to find (or regex pattern if regex=True)
        case_sensitive: Whether comparison is case sensitive
        regex: Whether to treat substring as regex pattern
        
    Returns:
        DecisionResult with search result
    """
    try:
        if not case_sensitive:
            text = text.lower()
            if not regex:
                substring = substring.lower()
        
        if regex:
            # Use regex matching
            flags = 0 if case_sensitive else re.IGNORECASE
            pattern = re.compile(substring, flags)
            match = pattern.search(text)
            result = match is not None
            details = f"Regex pattern '{substring}' {'found' if result else 'not found'} in text"
            if match:
                details += f" (matched: '{match.group()}')"
        else:
            # Simple substring check
            result = substring in text
            details = f"Substring '{substring}' {'found' if result else 'not found'} in text"
        
        return DecisionResult(
            result=result,
            details=details,
            confidence=1.0
        )
        
    except Exception as e:
        return DecisionResult(
            result=False,
            details=f"String search error: {str(e)}",
            confidence=0.0
        )


@register_tool
def boolean_mux(
    condition: bool,
    true_value: Any = "true_path",
    false_value: Any = "false_path"
) -> RouterResult:
    """
    Simple multiplexer that routes based on boolean condition.
    
    Args:
        condition: Boolean condition to check
        true_value: Value/path to return if condition is True
        false_value: Value/path to return if condition is False
        
    Returns:
        RouterResult with selected path
    """
    selected = true_value if condition else false_value
    
    return RouterResult(
        routed_to=str(selected),
        route_data={"condition": condition, "selected": selected},
        matched_condition=f"condition == {condition}"
    )


@register_tool
def conditional_router(
    decision: Union[Dict, str],
    routes: Dict[str, str],
    default_route: Optional[str] = None,
    decision_path: str = "action"
) -> RouterResult:
    """
    Route to different paths based on structured decision data.
    
    Args:
        decision: Decision data (dict or JSON string)
        routes: Mapping of decision values to route names
        default_route: Default route if no match found
        decision_path: Path in decision data to use for routing (e.g., "action", "result.type")
        
    Returns:
        RouterResult with routing information
    """
    try:
        # Parse decision if it's a string
        if isinstance(decision, str):
            decision = json.loads(decision)
        
        # Extract decision value using path
        decision_value = _get_nested_value(decision, decision_path)
        
        # Find matching route
        if str(decision_value) in routes:
            route = routes[str(decision_value)]
            matched = str(decision_value)
        elif decision_value in routes:
            route = routes[decision_value]
            matched = decision_value
        elif default_route:
            route = default_route
            matched = "default"
        else:
            route = "no_route"
            matched = None
        
        return RouterResult(
            routed_to=route,
            route_data={
                "decision_value": decision_value,
                "available_routes": list(routes.keys()),
                "decision_path": decision_path
            },
            matched_condition=f"{decision_path} == {matched}" if matched else None
        )
        
    except Exception as e:
        return RouterResult(
            routed_to=default_route or "error",
            route_data={"error": str(e)},
            matched_condition=None
        )


@register_tool
def threshold_check(
    value: Union[int, float],
    thresholds: Dict[str, Union[int, float]],
    default_result: str = "normal"
) -> RouterResult:
    """
    Check value against multiple thresholds and return appropriate result.
    
    Args:
        value: Numeric value to check
        thresholds: Dict of threshold_name -> threshold_value (checked in order)
        default_result: Result if no thresholds are exceeded
        
    Returns:
        RouterResult with threshold result
    """
    try:
        # Sort thresholds by value (descending) to check highest first
        sorted_thresholds = sorted(thresholds.items(), key=lambda x: x[1], reverse=True)
        
        for threshold_name, threshold_value in sorted_thresholds:
            if value >= threshold_value:
                return RouterResult(
                    routed_to=threshold_name,
                    route_data={
                        "value": value,
                        "threshold": threshold_value,
                        "exceeded": True
                    },
                    matched_condition=f"value >= {threshold_value}"
                )
        
        # No thresholds exceeded
        return RouterResult(
            routed_to=default_result,
            route_data={
                "value": value,
                "thresholds": thresholds,
                "exceeded": False
            },
            matched_condition="value < all thresholds"
        )
        
    except Exception as e:
        return RouterResult(
            routed_to=default_result,
            route_data={"error": str(e)},
            matched_condition=None
        )


# Helper functions

def _is_safe_expression(expression: str) -> bool:
    """
    Check if expression is safe for evaluation.
    Only allows basic comparisons and property access.
    """
    # List of dangerous keywords/functions
    dangerous = [
        'import', 'exec', 'eval', 'open', 'file', '__',
        'compile', 'globals', 'locals', 'vars', 'dir',
        'getattr', 'setattr', 'delattr', 'hasattr'
    ]
    
    expression_lower = expression.lower()
    for danger in dangerous:
        if danger in expression_lower:
            return False
    
    # Only allow basic comparison operators and property access
    allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._[](){}\'\" <>!=+-*/% ')
    if not all(c in allowed_chars for c in expression):
        return False
    
    return True


def _get_nested_value(data: Dict, path: str) -> Any:
    """
    Get nested value from dict using dot notation path.
    Example: "result.status" gets data["result"]["status"]
    """
    keys = path.split('.')
    value = data
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    
    return value


# Export the tools for easy importing
__all__ = [
    'json_evaluator',
    'number_compare', 
    'string_contains',
    'boolean_mux',
    'conditional_router',
    'threshold_check',
    'DecisionResult',
    'RouterResult'
]