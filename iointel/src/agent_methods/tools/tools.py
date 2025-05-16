from datetime import datetime
from typing import Any, Callable, Optional
from ...utilities.decorators import register_tool

@register_tool(name="between")
def between(
    min_value: Optional[Any] = None,
    max_value: Optional[Any] = None,
) -> Callable[[Any], Any]:
    """
    Create a validator function for values with optional minimum and maximum.

    Args:
        min_value: The minimum allowed value (inclusive). If None, no minimum is enforced.
        max_value: The maximum allowed value (inclusive). If None, no maximum is enforced.

    Returns:
        A function that validates a value based on the specified constraints.

    Raises:
        ValueError: If the value is less than min_value or greater than max_value.

    Example:
        >>> validator = between(min_value=0, max_value=100)
        >>> validator(50)  # Returns 50
        >>> validator(-1)  # Raises ValueError
        >>> validator(101)  # Raises ValueError
    """

    def validate(value: Any) -> Any:
        if min_value is not None and value < min_value:
            raise ValueError(f"Value must be greater than or equal to {min_value}")
        if max_value is not None and value > max_value:
            raise ValueError(f"Value must be less than or equal to {max_value}")
        return value

    return validate


@register_tool(name="get_current_datetime")
def get_current_datetime() -> str:
    """
    Return the current datetime as a string in YYYY-MM-DD HH:MM:SS format.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
