"""
Production-grade conditional gating system for workflow control.

This module provides a generic, composable conditional routing system that enables
true data flow termination in workflows. Built for first-order logic operations
in production environments (trading, automation, etc.).

CRITICAL: This system is designed for production use where incorrect routing
could have financial or operational consequences. All conditions are validated
and logged for audit trails.
"""

import json
import operator
from typing import Any, Dict, List, Optional, Union, Literal
from enum import Enum
from pydantic import BaseModel, Field, field_validator
import logging

from ...utilities.decorators import register_tool


# Configure production logging
logger = logging.getLogger(__name__)


# Define the allowed action literals
RouteActionLiteral = Literal["continue", "terminate", "branch"]

# Keep the enum for backward compatibility and constant references
class RouteAction(str, Enum):
    """Standardized routing actions for workflows."""
    CONTINUE = "continue"
    TERMINATE = "terminate"
    BRANCH = "branch"

class ComparisonOperator(str, Enum):
    """Supported comparison operators with explicit semantics."""
    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="
    EQ = "=="
    NEQ = "!="
    IN = "in"
    NOT_IN = "not_in"
    BETWEEN = "between"
    OUTSIDE = "outside"


class SimpleCondition(BaseModel):
    """
    A simple, LLM-friendly condition for routing decisions.
    """
    field: str = Field(..., description="Field name to check in the data (e.g., 'price_change', 'sentiment')")
    operator: Literal[">", "<", ">=", "<=", "==", "!=", "in", "between"] = Field(
        ..., 
        description="Comparison operator to use"
    )
    value: Union[str, int, float, List[Union[str, int, float]]] = Field(
        ..., 
        description="Value to compare against. Use list for 'in' and 'between' operators"
    )
    route: str = Field(..., description="Route name to go to if this condition matches")
    action: RouteActionLiteral = Field(
        "branch", 
        description="Action to take if condition matches: 'branch', 'continue', or 'terminate'"
    )
    
    @field_validator('value')
    def validate_value_for_operator(cls, v, info):
        """Validate value based on operator."""
        if hasattr(info, 'data') and 'operator' in info.data:
            op = info.data['operator']
            if op in ['in'] and not isinstance(v, list):
                raise ValueError(f"Operator '{op}' requires a list value")
            elif op == 'between':
                if not isinstance(v, list) or len(v) != 2:
                    raise ValueError("Operator 'between' requires a list of exactly 2 values")
                if not all(isinstance(x, (int, float)) for x in v):
                    raise ValueError("Operator 'between' requires numeric values")
        return v


class RouterConfig(BaseModel):
    """
    Configuration for the conditional router - the 'dial' that routes to K outgoing edges.
    """
    conditions: List[SimpleCondition] = Field(
        ..., 
        description="List of conditions to evaluate. Any condition that matches routes data to its output."
    )
    default_route: str = Field(
        "terminate", 
        description="Route name to use if no conditions match"
    )
    default_action: RouteActionLiteral = Field(
        "terminate", 
        description="Action to take if no conditions match"
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "conditions": [
                        {
                            "field": "price_change",
                            "operator": ">",
                            "value": 10,
                            "route": "sell_signal",
                            "action": "branch"
                        },
                        {
                            "field": "price_change", 
                            "operator": "<",
                            "value": -10,
                            "route": "buy_signal",
                            "action": "branch"
                        },
                        {
                            "field": "sentiment",
                            "operator": "in",
                            "value": ["bullish", "positive"],
                            "route": "positive_sentiment",
                            "action": "branch"
                        }
                    ],
                    "default_route": "hold",
                    "default_action": "terminate"
                }
            ]
        }


class ConditionRule(BaseModel):
    """A single condition rule for evaluation."""
    field_path: str = Field(..., description="Path to field in data, e.g., 'price' or 'result.value'")
    operator: ComparisonOperator
    threshold: Union[float, int, str, List[Any]] = Field(..., description="Value(s) to compare against")
    tolerance: Optional[float] = Field(None, description="Tolerance for float comparisons")
    
    @field_validator('threshold')
    def validate_threshold(cls, v, info):
        """Validate threshold based on operator."""
        # Get the operator from the data being validated
        if hasattr(info, 'data') and 'operator' in info.data:
            op = info.data['operator']
            if op in [ComparisonOperator.BETWEEN, ComparisonOperator.OUTSIDE]:
                if not isinstance(v, list) or len(v) != 2:
                    raise ValueError(f"Operator {op} requires list of 2 values")
                if not all(isinstance(x, (int, float)) for x in v):
                    raise ValueError(f"Operator {op} requires numeric values")
        return v


class RouteConfig(BaseModel):
    """Configuration for a routing decision."""
    route_name: str = Field(..., description="Name of this route (e.g., 'buy', 'sell')")
    action: RouteActionLiteral = Field(
        "branch", 
        description="What action to take: 'continue' (execute next node), 'terminate' (stop execution), or 'branch' (conditional path)"
    )
    conditions: List[ConditionRule] = Field(..., description="All conditions must be true")
    condition_logic: str = Field("AND", description="AND or OR logic for multiple conditions")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional routing metadata")


class ConditionalGateConfig(BaseModel):
    """Configuration for the entire conditional gate."""
    routes: List[RouteConfig] = Field(..., description="Routes evaluated in order")
    default_route: str = Field("terminate", description="Default route if no conditions match")
    default_action: RouteActionLiteral = Field(
        "terminate", 
        description="Default action if no conditions match: 'continue', 'terminate', or 'branch'"
    )
    require_all_fields: bool = Field(False, description="Fail if any field_path is missing")
    audit_log: bool = Field(True, description="Log all routing decisions for audit")


class GateResult(BaseModel):
    """Result from conditional gate evaluation. Will match the route name if the condition is met."""
    routed_to: str = Field(..., description="Route name for DAG executor")
    route_index: int = Field(..., description="Route index (0, 1, 2...) for DAG executor matching that maps to edge where information flows")
    action: RouteActionLiteral = Field(..., description="Action taken: 'continue', 'terminate', or 'branch'")
    matched_route: Optional[str] = None
    evaluated_conditions: List[Dict[str, Any]] = Field(default_factory=list)
    decision_reason: str
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    audit_trail: Dict[str, Any] = Field(default_factory=dict)


class MultiGateResult(BaseModel):
    """Result from multi-conditional gate evaluation with multiple routes."""
    routed_to: List[str] = Field(..., description="List of route names for DAG executor")
    matched_routes: List[str] = Field(..., description="All routes that matched conditions")
    actions: List[RouteActionLiteral] = Field(default_factory=list, description="Actions for each matched route")
    evaluated_conditions: List[Dict[str, Any]] = Field(default_factory=list)
    decision_reason: str
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    audit_trail: Dict[str, Any] = Field(default_factory=dict)


# Operator mapping for safe evaluation
OPERATOR_MAP = {
    ComparisonOperator.GT: operator.gt,
    ComparisonOperator.LT: operator.lt,
    ComparisonOperator.GTE: operator.ge,
    ComparisonOperator.LTE: operator.le,
    ComparisonOperator.EQ: operator.eq,
    ComparisonOperator.NEQ: operator.ne,
}


def extract_field_value(data: Dict[str, Any], field_path: str) -> Any:
    """
    Safely extract nested field value from data.
    
    Args:
        data: Data dictionary
        field_path: Dot-separated path like "result.price" or "metrics.change_percent"
        
    Returns:
        Field value or None if not found
    """
    try:
        parts = field_path.split('.')
        current = data
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
                
        return current
    except Exception as e:
        logger.warning(f"Failed to extract field {field_path}: {e}")
        return None


class ConditionalGateBase:
    """
    Base class for conditional gate tools with shared evaluation logic.
    """
    
    @staticmethod
    def _evaluate_simple_condition(data: Dict[str, Any], condition: SimpleCondition) -> tuple[bool, str]:
        """
        Evaluate a single SimpleCondition against data.
        
        Returns:
            Tuple of (result, explanation)
        """
        # Get field value from data
        field_value = data.get(condition.field)
        if field_value is None:
            return False, f"Field '{condition.field}' not found in data"
        
        # Evaluate condition
        try:
            result = False
            reason = ""
            
            if condition.operator == ">":
                result = float(field_value) > float(condition.value)
                reason = f"{field_value} > {condition.value} = {result}"
            elif condition.operator == "<":
                result = float(field_value) < float(condition.value)
                reason = f"{field_value} < {condition.value} = {result}"
            elif condition.operator == ">=":
                result = float(field_value) >= float(condition.value)
                reason = f"{field_value} >= {condition.value} = {result}"
            elif condition.operator == "<=":
                result = float(field_value) <= float(condition.value)
                reason = f"{field_value} <= {condition.value} = {result}"
            elif condition.operator == "==":
                result = field_value == condition.value
                reason = f"{field_value} == {condition.value} = {result}"
            elif condition.operator == "!=":
                result = field_value != condition.value
                reason = f"{field_value} != {condition.value} = {result}"
            elif condition.operator == "in":
                if isinstance(condition.value, list):
                    # Support both exact matching and substring/contains matching
                    if isinstance(field_value, str):
                        # For strings, check if any keyword is contained in the field value
                        result = any(str(keyword).lower() in field_value.lower() for keyword in condition.value)
                        matched_keywords = [str(k) for k in condition.value if str(k).lower() in field_value.lower()]
                        reason = f"'{field_value}' contains {matched_keywords} from {condition.value} = {result}"
                    else:
                        # For non-strings, use exact matching
                        result = field_value in condition.value
                        reason = f"{field_value} in {condition.value} = {result}"
                else:
                    result = False
                    reason = "'in' operator requires list value"
            elif condition.operator == "between":
                if isinstance(condition.value, list) and len(condition.value) == 2:
                    result = float(condition.value[0]) <= float(field_value) <= float(condition.value[1])
                    reason = f"{condition.value[0]} <= {field_value} <= {condition.value[1]} = {result}"
                else:
                    result = False
                    reason = "'between' operator requires list of 2 values"
            else:
                result = False
                reason = f"Unknown operator: {condition.operator}"
            
            return result, reason
                
        except (ValueError, TypeError) as e:
            return False, f"Evaluation error: {e}"
    
    @staticmethod
    def _parse_and_validate_inputs(data, router_config):
        """
        Parse and validate input data and router configuration.
        
        Returns:
            Tuple of (parsed_data, parsed_config, error_result)
            If error_result is not None, return it immediately.
        """
        # Parse input data
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError as e:
                return None, None, {
                    "routed_to": "error",
                    "route_index": -999,
                    "action": "terminate", 
                    "matched_route": None,
                    "decision_reason": f"Invalid JSON data: {e}",
                    "confidence": 0.0,
                    "audit_trail": {}
                }
        
        # Parse and validate router config
        if isinstance(router_config, dict):
            try:
                router_config = RouterConfig.model_validate(router_config)
            except Exception as e:
                return None, None, {
                    "routed_to": "error",
                    "route_index": -999,
                    "action": "terminate",
                    "matched_route": None,
                    "decision_reason": f"Invalid router configuration: {e}",
                    "confidence": 0.0,
                    "audit_trail": {}
                }
        
        return data, router_config, None
    
    @staticmethod
    def _evaluate_all_conditions(data: Dict[str, Any], router_config: RouterConfig) -> tuple[List[str], List[str], List[int], Dict[str, Any]]:
        """
        Evaluate all conditions and return matched routes, actions, indexes, and audit trail.
        
        Returns:
            Tuple of (matched_routes, matched_actions, matched_indexes, audit_trail)
        """
        matched_routes = []
        matched_actions = []
        matched_indexes = []
        audit_trail = {
            "input_data": {"keys": list(data.keys())},
            "evaluated_conditions": []
        }
        
        # Evaluate ALL conditions and collect matches
        for index, condition in enumerate(router_config.conditions):
            result, reason = ConditionalGateBase._evaluate_simple_condition(data, condition)
            
            audit_trail["evaluated_conditions"].append({
                "field": condition.field,
                "operator": condition.operator,
                "value": condition.value,
                "result": result,
                "reason": reason,
                "route_index": index
            })
            
            # If condition matches, add to matched routes
            if result:
                matched_routes.append(condition.route)
                matched_actions.append(condition.action)
                matched_indexes.append(index)
        
        return matched_routes, matched_actions, matched_indexes, audit_trail


def evaluate_condition(data: Dict[str, Any], rule: ConditionRule) -> tuple[bool, str]:
    """
    Evaluate a single condition rule against data.
    
    Returns:
        Tuple of (result, explanation)
    """
    value = extract_field_value(data, rule.field_path)
    
    if value is None:
        return False, f"Field '{rule.field_path}' not found"
    
    try:
        # Type conversion for numeric comparisons
        if rule.operator in [ComparisonOperator.GT, ComparisonOperator.LT, 
                           ComparisonOperator.GTE, ComparisonOperator.LTE,
                           ComparisonOperator.BETWEEN, ComparisonOperator.OUTSIDE]:
            if isinstance(value, str):
                value = float(value)
                
        # Apply operator
        if rule.operator in OPERATOR_MAP:
            op_func = OPERATOR_MAP[rule.operator]
            threshold = rule.threshold
            
            # Handle float tolerance
            if (rule.operator in [ComparisonOperator.EQ, ComparisonOperator.NEQ] and 
                isinstance(value, float) and isinstance(threshold, (int, float))):
                tolerance = rule.tolerance or 1e-9
                if rule.operator == ComparisonOperator.EQ:
                    result = abs(value - threshold) <= tolerance
                else:
                    result = abs(value - threshold) > tolerance
            else:
                result = op_func(value, threshold)
                
        elif rule.operator == ComparisonOperator.IN:
            result = value in rule.threshold
            
        elif rule.operator == ComparisonOperator.NOT_IN:
            result = value not in rule.threshold
            
        elif rule.operator == ComparisonOperator.BETWEEN:
            low, high = rule.threshold
            result = low <= value <= high
            
        elif rule.operator == ComparisonOperator.OUTSIDE:
            low, high = rule.threshold
            result = value < low or value > high
            
        else:
            return False, f"Unknown operator: {rule.operator}"
            
        explanation = f"{rule.field_path} ({value}) {rule.operator} {rule.threshold} = {result}"
        return result, explanation
        
    except Exception as e:
        return False, f"Evaluation error: {str(e)}"


@register_tool
def routing_gate(
    data: Any,
    route_index: int,
    route_name: Optional[str] = None,
    action: str = "branch"
) -> GateResult:
    """
    Simple routing gate that directly routes to a specified index.
    
    This is a streamlined version without complex JSON conditions.
    The agent analyzes the data and directly specifies which route to take.
    
    Args:
        data: Input data (kept for compatibility, can be anything)
        route_index: The route index to select (0-based) matching edge route_index
        route_name: Optional human-readable route name
        action: Action to take ('branch', 'continue', or 'terminate')
    
    Returns:
        GateResult with routing decision for the DAG executor
        
    Example:
        >>> routing_gate(data="pwd", route_index=4, route_name="System & Shell")
        GateResult(routed_to="System & Shell", route_index=4, action="branch", ...)
    """
    # Use route_name if provided, otherwise generate from index
    routed_to = route_name if route_name else f"route_{route_index}"
    
    return GateResult(
        routed_to=routed_to,
        route_index=route_index,
        action=action,
        matched_route=routed_to,
        decision_reason=f"Direct routing to {routed_to} (index {route_index})",
        confidence=1.0,
        audit_trail={"input_data": str(data)[:100], "direct_route": True}
    )


@register_tool
def conditional_gate(
    data: Union[Dict, str],
    router_config: Union[RouterConfig, Dict[str, Any]]
) -> GateResult:
    """
    Beautiful conditional router - the 'dial' that routes to K outgoing edges.
    
    This is a streamlined router that evaluates typed conditions and returns
    routing decisions for workflow control. The router acts as a composable
    'dial' that can route data to multiple downstream paths based on conditions.
    
    IMPORTANT: This router returns the FIRST matching condition (single route).
    For multiple simultaneous routes, use conditional_multi_gate.
    
    Args:
        data: Input data to evaluate (dict or JSON string)
        router_config: Router configuration with typed conditions (RouterConfig or dict)
        
    Returns:
        GateResult with routing decision for the DAG executor
        
    Examples:
        >>> # Trading decision router
        >>> conditional_gate(
        ...     data={"price_change": 15.5},
        ...     router_config={
        ...         "conditions": [
        ...             {"field": "price_change", "operator": ">", "value": 10, "route": "sell"},
        ...             {"field": "price_change", "operator": "<", "value": -10, "route": "buy"}
        ...         ],
        ...         "default_route": "hold"
        ...     }
        ... )
        GateResult(routed_to="sell", action="branch", ...)
    """
    # Parse and validate inputs
    data, router_config, error_result = ConditionalGateBase._parse_and_validate_inputs(data, router_config)
    if error_result:
        return GateResult(**error_result)
    
    # Get evaluation results
    matched_routes, matched_actions, matched_indexes, audit_trail = ConditionalGateBase._evaluate_all_conditions(data, router_config)
    
    # Return first match (single route behavior)
    if matched_routes:
        return GateResult(
            routed_to=matched_routes[0],
            route_index=matched_indexes[0],
            action=matched_actions[0],
            matched_route=matched_routes[0],
            decision_reason=f"Matched condition: {matched_routes[0]} (index {matched_indexes[0]})",
            confidence=1.0,
            audit_trail=audit_trail
        )
    
    # No conditions matched - use default (use -1 to indicate default route)
    return GateResult(
        routed_to=router_config.default_route,
        route_index=-1,
        action=router_config.default_action,
        matched_route=None,
        decision_reason=f"No conditions matched, using default '{router_config.default_route}' (index -1)",
        confidence=1.0,
        audit_trail=audit_trail
    )


@register_tool
def conditional_multi_gate(
    data: Union[Dict, str],
    router_config: Union[RouterConfig, Dict[str, Any]]
) -> MultiGateResult:
    """
    Multi-output conditional router - routes data to ALL matching outputs simultaneously.
    
    This router evaluates ALL conditions and returns ALL matched routes for proper
    data flow routing where multiple conditions can trigger simultaneously.
    
    Args:
        data: Input data to evaluate (dict or JSON string)
        router_config: Router configuration with typed conditions (RouterConfig or dict)
        
    Returns:
        MultiGateResult with all matching routes for the DAG executor
        
    Examples:
        >>> # Multiple conditions match simultaneously
        >>> conditional_multi_gate(
        ...     data={"sentiment": "bullish", "confidence": 0.85, "volume": 5000},
        ...     router_config={
        ...         "conditions": [
        ...             {"field": "sentiment", "operator": "in", "value": ["bullish", "positive"], "route": "positive"},
        ...             {"field": "confidence", "operator": ">", "value": 0.7, "route": "high_confidence"},
        ...             {"field": "volume", "operator": ">", "value": 1000, "route": "high_volume"}
        ...         ],
        ...         "default_route": "neutral"
        ...     }
        ... )
        MultiGateResult(routed_to=["positive", "high_confidence", "high_volume"], ...)
        
        >>> # Fan-out pattern - alert multiple systems
        >>> conditional_multi_gate(
        ...     data={"price_change": 15.5, "alert_level": "critical"},
        ...     router_config={
        ...         "conditions": [
        ...             {"field": "price_change", "operator": ">", "value": 10, "route": "trading_alert"},
        ...             {"field": "alert_level", "operator": "==", "value": "critical", "route": "notification_system"},
        ...             {"field": "price_change", "operator": ">", "value": 5, "route": "logging_system"}
        ...         ]
        ...     }
        ... )
        # Routes to: trading_alert, notification_system, logging_system
    """
    # Parse and validate inputs
    data, router_config, error_result = ConditionalGateBase._parse_and_validate_inputs(data, router_config)
    if error_result:
        return MultiGateResult(**error_result)
    
    # Get evaluation results
    matched_routes, matched_actions, audit_trail = ConditionalGateBase._evaluate_all_conditions(data, router_config)
    
    # Return all matches (multi-route behavior)
    if matched_routes:
        return MultiGateResult(
            routed_to=matched_routes,
            matched_routes=matched_routes,
            actions=matched_actions,
            decision_reason=f"Matched {len(matched_routes)} conditions: {', '.join(matched_routes)}",
            confidence=1.0,
            audit_trail=audit_trail
        )
    else:
        # No conditions matched - use default route
        return MultiGateResult(
            routed_to=[router_config.default_route],
            matched_routes=[],
            actions=[router_config.default_action],
            decision_reason=f"No conditions matched, using default '{router_config.default_route}'",
            confidence=1.0,
            audit_trail=audit_trail
        )


# threshold_gate removed - use conditional_gate with threshold conditions instead:
# Example: conditional_gate(condition="price >= 70", routes={"true": "high_price", "false": "low_price"})




# Export all tools
__all__ = [
    'conditional_gate',
    'conditional_multi_gate',
    'RouteAction',
    'ComparisonOperator',
    'ConditionRule',
    'RouteConfig',
    'ConditionalGateConfig',
    'GateResult',
    'MultiGateResult',
    'ConditionalGateBase'
]