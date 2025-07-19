from agno.tools.calculator import CalculatorTools as AgnoCalculatorTools
from ..utils import register_tool

# Create a global calculator instance for function calls
_calculator = AgnoCalculatorTools()

# Register tools as static functions to avoid 'self' parameter issues
@register_tool
def calculator_add(a: float, b: float) -> str:
    """Add two numbers together."""
    return _calculator.add(a, b)

@register_tool  
def calculator_subtract(a: float, b: float) -> str:
    """Subtract the second number from the first."""
    return _calculator.subtract(a, b)

@register_tool
def calculator_multiply(a: float, b: float) -> str:
    """Multiply two numbers together.""" 
    return _calculator.multiply(a, b)

@register_tool
def calculator_divide(a: float, b: float) -> str:
    """Divide the first number by the second."""
    return _calculator.divide(a, b)

@register_tool
def calculator_exponentiate(a: float, b: float) -> str:
    """Raise the first number to the power of the second."""
    return _calculator.exponentiate(a, b)

@register_tool
def calculator_square_root(n: float) -> str:
    """Calculate the square root of a number."""
    return _calculator.square_root(n)

@register_tool
def calculator_factorial(n: int) -> str:
    """Calculate the factorial of a number."""
    return _calculator.factorial(n)

@register_tool
def calculator_is_prime(n: int) -> str:
    """Check if a number is prime."""
    return _calculator.is_prime(n)
