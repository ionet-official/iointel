from iointel.src.agent_methods.tools.agno.calculator import Calculator

# from agno.tools.calculator import CalculatorTools
import json
# import openai


def test_calculator_basic_arithmetic():
    calculator = Calculator()

    result = calculator.add(10, 5)
    assert "15" in result

    result = calculator.subtract(20, 5)
    assert "15" in result

    # Test multiplication
    result = calculator.multiply(10, 5)
    assert "50" in result

    result = calculator.multiply(2, 3)
    assert "6" in result

    # Test division
    result = calculator.divide(100, 4)
    assert "25" in result


def test_calculator_advanced_operations():
    calculator = Calculator()

    # Test exponentiation
    result = calculator.exponentiate(2, 3)
    assert "8" in result

    # Test square root
    result = calculator.square_root(16)
    assert "4" in result

    # Test factorial
    result = calculator.factorial(5)
    assert "120" in result

    # Test prime number check
    result = calculator.is_prime(17)
    assert json.loads(result)["result"] is True

    result = calculator.is_prime(4)
    assert json.loads(result)["result"] is False
