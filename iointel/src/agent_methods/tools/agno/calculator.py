from functools import wraps
from typing import Union
from pydantic import BaseModel
from agno.tools.calculator import CalculatorTools as AgnoCalculatorTools

from ..utils import register_tool
from .common import DisableAgnoRegistryMixin


class Calculator(BaseModel, DisableAgnoRegistryMixin, AgnoCalculatorTools):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @register_tool(name="calculator_add")
    @wraps(AgnoCalculatorTools.add)
    def add(self, a:float, b: float) -> str:
        """
        Add two numbers.
        """
        return super().add(a, b)

    @register_tool(name="calculator_subtract")
    @wraps(AgnoCalculatorTools.subtract)
    def subtract(self, a: float, b: float) -> str:
        """
        Subtract b from a.
        """
        return super().subtract(a, b)

    @register_tool(name="calculator_multiply")
    @wraps(AgnoCalculatorTools.multiply)
    def multiply(self, a: float, b: float) -> str:
        """
        Multiply two numbers.
        """
        return super().multiply(a, b)

    @register_tool(name="calculator_divide")
    @wraps(AgnoCalculatorTools.divide)
    def divide(self, a: float, b: float) -> str:
        """
        Divide a by b.
        """
        return super().divide(a, b)

    @register_tool(name="calculator_exponentiate")
    @wraps(AgnoCalculatorTools.exponentiate)
    def exponentiate(self, a: float, b: float) -> str:
        """
        Raise a number to a power.
        """
        return super().exponentiate(a, b)

    @register_tool(name="calculator_square_root")
    @wraps(AgnoCalculatorTools.square_root)
    def square_root(self, n: float) -> str:
        """
        Calculate the square root of a number.
        """
        return super().square_root(n)

    @register_tool(name="calculator_factorial")
    @wraps(AgnoCalculatorTools.factorial)
    def factorial(self, n: int) -> str:
        """
        Calculate the factorial of a number.
        Accepts either 'number' or 'n' as argument for compatibility.
        """
        return super().factorial(n)

    @register_tool(name="calculator_is_prime")
    @wraps(AgnoCalculatorTools.is_prime)
    def is_prime(self, n: int) -> str:
        """
        Check if a number is prime.
        """
        return super().is_prime(n)
