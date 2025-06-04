import os

import pytest
from iointel.src.agent_methods.tools.agno.calculator import Calculator
from pydantic_ai.models.openai import OpenAIModel
import json

from iointel.src.agents import Agent


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


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
async def test_calculator_with_agent():
    calculator = Calculator()
    agent = Agent(
        name="CalculatorAgent",
        instructions="""
        You are a calculator AI agent.
        Perform mathematical operations and provide answers in a clean string format (not json) and not a dictionary.
        """,
        tools=[calculator.add],
        model=OpenAIModel(model_name="gpt-4o-mini"),
    )

    result = await agent.run("calculate 2 + 3")
    assert "5" in result["result"]


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
async def test_calculator_with_agent_complex():
    """
    Test the agent's ability to:
      • Perform basic and multi-step calculations using calculator tools.
      • Correctly use single-argument and multi-argument tools.
      • Handle prime checks and nested operations.
    """

    calculator = Calculator()
    agent = Agent(
        name="MyAgent",
        instructions=(
            """
            You are a helpful agent. Use the calculator tools to perform mathematical operations.
            Available tools: add, subtract, multiply, divide, exponentiate, square_root, factorial, is_prime.
            """
        ),
        tools=[
            calculator.add,
            calculator.subtract,
            calculator.multiply,
            calculator.factorial,
            calculator.square_root,
            calculator.is_prime,
        ],
        model=OpenAIModel(model_name="gpt-4o-mini"),
    )

    # Test a basic calculation
    result = await agent.run("What is 10 - 6")
    assert "4" in str(result)

    # Test a multi-step calculation
    result = await agent.run(
        "What is (the factorial of 5 plus (the square root of 16), then multiply the result by 2?"
    )
    # factorial(5) = 120, sqrt(16) = 4, 120 + 4 = 124, 124 * 2 = 248
    assert "248" in str(result), f"Expected result to contain 248, got {result}"

    # Test a prime check
    result_prime = await agent.run("Is 17 a prime number?")
    assert "true" in str(result_prime).lower() or "yes" in str(result_prime).lower(), (
        f"Expected result to confirm 17 is prime, got {result_prime}"
    )
