import asyncio
from datetime import datetime

from iointel import Agent
from iointel.src.workflow import run_agents_async


def add_two_numbers(a: int, b: int) -> int:
    return a + b

def get_current_datetime() -> str:
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return current_datetime

def test_basic_tools():
    """
    LLama can't add such big numbers, so it must use the tool
    """
    agent = Agent(
        name="Agent",
        instructions="Do whatever you're asked",
    )
    numbers = [22122837493142, 17268162387617, 159864395786239452]

    result = asyncio.run(
        run_agents_async(
            f"Add three numbers: {numbers[0]} and {numbers[1]} and {numbers[2]}",
            agents=[agent], tools=[add_two_numbers, get_current_datetime],
        )
    )
    assert result == str(sum(numbers))