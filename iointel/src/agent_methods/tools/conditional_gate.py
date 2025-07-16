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
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
from pydantic import BaseModel, Field, field_validator
import logging

from ...utilities.decorators import register_tool


# Configure production logging
logger = logging.getLogger(__name__)


class RouteAction(str, Enum):
    """Standardized routing actions for workflows."""
    CONTINUE = Field(
        "continue",
        description="Continue execution to the next node in the workflow sequence"
    )
    TERMINATE = Field(
        "terminate", 
        description="Stop all downstream execution and end the workflow branch"
    )
    BRANCH = Field(
        "branch",
        description="Branch execution to a specific conditional path in the workflow"
    )

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
    action: RouteAction = Field(RouteAction.BRANCH, description="What action to take")
    conditions: List[ConditionRule] = Field(..., description="All conditions must be true")
    condition_logic: str = Field("AND", description="AND or OR logic for multiple conditions")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional routing metadata")


class ConditionalGateConfig(BaseModel):
    """Configuration for the entire conditional gate."""
    routes: List[RouteConfig] = Field(..., description="Routes evaluated in order")
    default_route: str = Field("terminate", description="Default route if no conditions match")
    default_action: RouteAction = Field(RouteAction.TERMINATE, description="Default action")
    require_all_fields: bool = Field(False, description="Fail if any field_path is missing")
    audit_log: bool = Field(True, description="Log all routing decisions for audit")


class GateResult(BaseModel):
    """Result from conditional gate evaluation."""
    routed_to: str = Field(..., description="Route name for DAG executor")
    action: RouteAction
    matched_route: Optional[str] = None
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
def conditional_gate(
    data: Union[Dict, str],
    gate_config: Union[Dict, ConditionalGateConfig],
    trace: bool = False
) -> GateResult:
    """
    Generic conditional gate for workflow routing decisions.
    
    This is the production-grade router that evaluates complex conditions
    and returns routing decisions that the DAG executor uses to control
    data flow.
    
    Args:
        data: Input data to evaluate (dict or JSON string)
        gate_config: Gate configuration (dict or ConditionalGateConfig)
        trace: Enable detailed trace logging for debugging
        
    Returns:
        GateResult with routing decision
        
    Example:
        gate_config = {
            "routes": [
                {
                    "route_name": "execute_trade",
                    "action": "branch",
                    "conditions": [
                        {"field_path": "signal.strength", "operator": ">", "threshold": 0.8},
                        {"field_path": "risk.value", "operator": "<", "threshold": 0.2}
                    ]
                },
                {
                    "route_name": "monitor_only", 
                    "action": "branch",
                    "conditions": [
                        {"field_path": "signal.strength", "operator": "between", "threshold": [0.3, 0.8]}
                    ]
                }
            ],
            "default_route": "terminate",
            "default_action": "terminate"
        }
    """
    # Parse input data
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            return GateResult(
                routed_to="error",
                action=RouteAction.TERMINATE,
                decision_reason=f"Invalid JSON data: {e}",
                confidence=0.0
            )
    
    # Parse gate config
    if isinstance(gate_config, dict):
        try:
            gate_config = ConditionalGateConfig.model_validate(gate_config)
        except Exception as e:
            return GateResult(
                routed_to="error",
                action=RouteAction.TERMINATE,
                decision_reason=f"Invalid gate configuration: {e}",
                confidence=0.0
            )
    
    evaluated_conditions = []
    audit_trail = {
        "input_data": data if trace else {"keys": list(data.keys())},
        "evaluated_routes": []
    }
    
    # Evaluate each route in order
    for route in gate_config.routes:
        route_eval = {
            "route_name": route.route_name,
            "conditions": [],
            "matched": False
        }
        
        # Evaluate all conditions for this route
        condition_results = []
        for condition in route.conditions:
            result, explanation = evaluate_condition(data, condition)
            condition_eval = {
                "rule": condition.model_dump(),
                "result": result,
                "explanation": explanation
            }
            route_eval["conditions"].append(condition_eval)
            condition_results.append(result)
            
            if trace:
                logger.info(f"Condition: {explanation}")
        
        # Apply condition logic (AND/OR)
        if route.condition_logic == "OR":
            route_matched = any(condition_results)
        else:  # Default AND
            route_matched = all(condition_results)
            
        route_eval["matched"] = route_matched
        audit_trail["evaluated_routes"].append(route_eval)
        
        if route_matched:
            # Route matched - return routing decision
            result = GateResult(
                routed_to=route.route_name,
                action=route.action,
                matched_route=route.route_name,
                evaluated_conditions=evaluated_conditions,
                decision_reason=f"Matched route '{route.route_name}' with {len(condition_results)} conditions",
                confidence=1.0,
                audit_trail=audit_trail
            )
            
            if gate_config.audit_log:
                logger.info(f"Conditional gate routed to '{route.route_name}': {result.decision_reason}")
                
            return result
    
    # No routes matched - use default
    result = GateResult(
        routed_to=gate_config.default_route,
        action=gate_config.default_action,
        matched_route=None,
        evaluated_conditions=evaluated_conditions,
        decision_reason=f"No routes matched, using default '{gate_config.default_route}'",
        confidence=1.0,
        audit_trail=audit_trail
    )
    
    if gate_config.audit_log:
        logger.info(f"Conditional gate defaulted to '{gate_config.default_route}'")
        
    return result


@register_tool
def threshold_gate(
    value: Union[float, int, str],
    thresholds: Dict[str, Union[float, int]],
    value_name: str = "value",
    default_route: str = "terminate"
) -> GateResult:
    """
    Simplified threshold-based gate for common numeric routing.
    
    Routes based on which threshold is exceeded (checked in descending order).
    
    Args:
        value: Numeric value to check
        thresholds: Dict of route_name -> threshold_value
        value_name: Name of the value for logging
        default_route: Route if no thresholds exceeded
        
    Example:
        thresholds = {
            "critical": 90,
            "warning": 70,
            "normal": 50
        }
        # If value=85, routes to "warning"
    """
    # Convert value if string
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return GateResult(
                routed_to="error",
                action=RouteAction.TERMINATE,
                decision_reason=f"Cannot convert '{value}' to number",
                confidence=0.0
            )
    
    # Sort thresholds by value (descending)
    sorted_thresholds = sorted(thresholds.items(), key=lambda x: x[1], reverse=True)
    
    for route_name, threshold in sorted_thresholds:
        if value >= threshold:
            return GateResult(
                routed_to=route_name,
                action=RouteAction.BRANCH,
                matched_route=route_name,
                decision_reason=f"{value_name} ({value}) >= {threshold}",
                confidence=1.0,
                audit_trail={
                    "value": value,
                    "threshold": threshold,
                    "all_thresholds": thresholds
                }
            )
    
    # No threshold exceeded
    return GateResult(
        routed_to=default_route,
        action=RouteAction.TERMINATE,
        decision_reason=f"{value_name} ({value}) below all thresholds",
        confidence=1.0,
        audit_trail={
            "value": value,
            "all_thresholds": thresholds
        }
    )


@register_tool  
def percentage_change_gate(
    current_value: Union[float, int],
    reference_value: Union[float, int],
    buy_threshold: float = -5.0,
    sell_threshold: float = 5.0,
    action_routes: Optional[Dict[str, str]] = None
) -> GateResult:
    """
    Specialized gate for percentage-based routing (common in trading).
    
    Args:
        current_value: Current value (e.g., price)
        reference_value: Reference value to compare against
        buy_threshold: Percentage below which to trigger buy (negative)
        sell_threshold: Percentage above which to trigger sell (positive)
        action_routes: Custom route names (default: buy_path, sell_path, hold_path)
        
    Returns:
        GateResult with routing decision
    """
    if action_routes is None:
        action_routes = {
            "buy": "buy_path",
            "sell": "sell_path", 
            "hold": "terminate"  # No action = terminate data flow
        }
    
    # Calculate percentage change
    percent_change = ((current_value - reference_value) / reference_value) * 100
    
    # Determine action
    if percent_change <= buy_threshold:
        action = "buy"
    elif percent_change >= sell_threshold:
        action = "sell"
    else:
        action = "hold"
    
    route = action_routes.get(action, "terminate")
    
    return GateResult(
        routed_to=route,
        action=RouteAction.BRANCH if action != "hold" else RouteAction.TERMINATE,
        matched_route=route if action != "hold" else None,
        decision_reason=(
            f"Price change {percent_change:.2f}% - Action: {action} "
            f"(current: {current_value}, reference: {reference_value})"
        ),
        confidence=1.0,
        audit_trail={
            "current_value": current_value,
            "reference_value": reference_value, 
            "percent_change": percent_change,
            "action": action,
            "thresholds": {
                "buy": buy_threshold,
                "sell": sell_threshold
            }
        }
    )


# Export all tools
__all__ = [
    'conditional_gate',
    'threshold_gate', 
    'percentage_change_gate',
    'RouteAction',
    'ComparisonOperator',
    'ConditionRule',
    'RouteConfig',
    'ConditionalGateConfig',
    'GateResult'
]