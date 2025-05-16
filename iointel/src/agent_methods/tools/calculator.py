import json
import math

import logging

from ...utilities.decorators import register_tool


logger = logging.getLogger(__name__)


class CalculatorTools:
    def __init__(self):
        pass

    @register_tool(name="calculator_add")
    def add(self, a: float, b: float) -> str:
        """Add two numbers and return the result.

        Args:
            a (float): First number.
            b (float): Second number.

        Returns:
            str: JSON string of the result.
        """
        result = a + b
        logger.info(f"Adding {a} and {b} to get {result}")
        return json.dumps({"operation": "addition", "result": result})

    @register_tool(name="calculator_subtract")
    def subtract(self, a: float, b: float) -> str:
        """Subtract second number from first and return the result.

        Args:
            a (float): First number.
            b (float): Second number.

        Returns:
            str: JSON string of the result.
        """
        result = a - b
        logger.info(f"Subtracting {b} from {a} to get {result}")
        return json.dumps({"operation": "subtraction", "result": result})

    @register_tool(name="calculator_multiply")
    def multiply(self, a: float, b: float) -> str:
        """Multiply two numbers and return the result.

        Args:
            a (float): First number.
            b (float): Second number.

        Returns:
            str: JSON string of the result.
        """
        result = a * b
        logger.info(f"Multiplying {a} and {b} to get {result}")
        return json.dumps({"operation": "multiplication", "result": result})

    @register_tool(name="calculator_divide")
    def divide(self, a: float, b: float) -> str:
        """Divide first number by second and return the result.

        Args:
            a (float): Numerator.
            b (float): Denominator.

        Returns:
            str: JSON string of the result.
        """
        if b == 0:
            logger.error("Attempt to divide by zero")
            return json.dumps(
                {"operation": "division", "error": "Division by zero is undefined"}
            )
        try:
            result = a / b
        except Exception as e:
            return json.dumps({"operation": "division", "error": e, "result": "Error"})
        logger.info(f"Dividing {a} by {b} to get {result}")
        return json.dumps({"operation": "division", "result": result})

    @register_tool(name="calculator_exponentiate")
    def exponentiate(self, a: float, b: float) -> str:
        """Raise first number to the power of the second number and return the result.

        Args:
            a (float): Base.
            b (float): Exponent.

        Returns:
            str: JSON string of the result.
        """
        result = math.pow(a, b)
        logger.info(f"Raising {a} to the power of {b} to get {result}")
        return json.dumps({"operation": "exponentiation", "result": result})

    @register_tool(name="calculator_factorial")
    def factorial(self, n: int) -> str:
        """Calculate the factorial of a number and return the result.

        Args:
            n (int): Number to calculate the factorial of.

        Returns:
            str: JSON string of the result.
        """
        if n < 0:
            logger.error("Attempt to calculate factorial of a negative number")
            return json.dumps(
                {
                    "operation": "factorial",
                    "error": "Factorial of a negative number is undefined",
                }
            )
        result = math.factorial(n)
        logger.info(f"Calculating factorial of {n} to get {result}")
        return json.dumps({"operation": "factorial", "result": result})

    @register_tool(name="calculator_is_prime")
    def is_prime(self, n: int) -> str:
        """Check if a number is prime and return the result.

        Args:
            n (int): Number to check if prime.

        Returns:
            str: JSON string of the result.
        """
        if n <= 1:
            return json.dumps({"operation": "prime_check", "result": False})
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return json.dumps({"operation": "prime_check", "result": False})
        return json.dumps({"operation": "prime_check", "result": True})

    @register_tool(name="calculator_square_root")
    def square_root(self, n: float) -> str:
        """Calculate the square root of a number and return the result.

        Args:
            n (float): Number to calculate the square root of.

        Returns:
            str: JSON string of the result.
        """
        if n < 0:
            logger.error("Attempt to calculate square root of a negative number")
            return json.dumps(
                {
                    "operation": "square_root",
                    "error": "Square root of a negative number is undefined",
                }
            )

        result = math.sqrt(n)
        logger.info(f"Calculating square root of {n} to get {result}")
        return json.dumps({"operation": "square_root", "result": result})
