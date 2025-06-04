from agno.tools.calculator import CalculatorTools as AgnoCalculatorTools

from .common import make_base, wrap_tool


class Calculator(make_base(AgnoCalculatorTools)):
    def _get_tool(self):
        return self.Inner()

    @wrap_tool("calculator_add", AgnoCalculatorTools.add)
    def add(self, a: float, b: float) -> str:
        return self._tool.add(a, b)

    @wrap_tool("calculator_subtract", AgnoCalculatorTools.subtract)
    def subtract(self, a: float, b: float) -> str:
        return self._tool.subtract(a, b)

    @wrap_tool("calculator_multiply", AgnoCalculatorTools.multiply)
    def multiply(self, a: float, b: float) -> str:
        return self._tool.multiply(a, b)

    @wrap_tool("calculator_divide", AgnoCalculatorTools.divide)
    def divide(self, a: float, b: float) -> str:
        return self._tool.divide(a, b)

    @wrap_tool("calculator_exponentiate", AgnoCalculatorTools.exponentiate)
    def exponentiate(self, a: float, b: float) -> str:
        return self._tool.exponentiate(a, b)

    @wrap_tool("calculator_square_root", AgnoCalculatorTools.square_root)
    def square_root(self, n: float) -> str:
        return self._tool.square_root(n)

    @wrap_tool("calculator_factorial", AgnoCalculatorTools.factorial)
    def factorial(self, n: int) -> str:
        return self._tool.factorial(n)

    @wrap_tool("calculator_is_prime", AgnoCalculatorTools.is_prime)
    def is_prime(self, n: int) -> str:
        return self._tool.is_prime(n)
