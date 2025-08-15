from agno.tools.calculator import CalculatorTools as AgnoCalculatorTools
from ..utils import register_tool

# Create a global calculator instance for function calls
_calculator = AgnoCalculatorTools()

# Register tools as static functions to avoid 'self' parameter issues
@register_tool
def calculator_add(a: float, b: float) -> str:
    """Add two numbers together.
    
    Args:
        a: The first number to add
        b: The second number to add
    
    Returns:
        The sum of the two numbers as a string
    """
    return _calculator.add(a, b)

@register_tool  
def calculator_subtract(a: float, b: float) -> str:
    """Subtract the second number from the first.
    
    Args:
        a: The number to subtract from (minuend)
        b: The number to subtract (subtrahend)
    
    Returns:
        The difference (a - b) as a string
    """
    return _calculator.subtract(a, b)

@register_tool
def calculator_multiply(a: float, b: float) -> str:
    """Multiply two numbers together.
    
    Args:
        a: The first number to multiply
        b: The second number to multiply
    
    Returns:
        The product of the two numbers as a string
    """ 
    return _calculator.multiply(a, b)

@register_tool
def calculator_divide(a: float, b: float) -> str:
    """Divide the first number by the second.
    
    Args:
        a: The dividend (number to be divided)
        b: The divisor (number to divide by)
    
    Returns:
        The quotient (a / b) as a string
    """
    return _calculator.divide(a, b)

@register_tool
def calculator_exponentiate(a: float, b: float) -> str:
    """Raise the first number to the power of the second.
    
    Args:
        a: The base number
        b: The exponent (power to raise to)
    
    Returns:
        The result of a^b as a string
    """
    return _calculator.exponentiate(a, b)

@register_tool
def calculator_square_root(n: float) -> str:
    """Calculate the square root of a number.
    
    Args:
        n: The number to find the square root of (must be non-negative)
    
    Returns:
        The square root of n as a string
    """
    return _calculator.square_root(n)

@register_tool
def calculator_factorial(n: int) -> str:
    """Calculate the factorial of a number.
    
    Args:
        n: The number to calculate factorial for (must be non-negative integer)
    
    Returns:
        The factorial of n (n!) as a string
    """
    return _calculator.factorial(n)

@register_tool
def calculator_is_prime(n: int) -> str:
    """Check if a number is prime.
    
    Args:
        n: The number to check for primality (must be a positive integer)
    
    Returns:
        'True' if the number is prime, 'False' otherwise
    """
    return _calculator.is_prime(n)
