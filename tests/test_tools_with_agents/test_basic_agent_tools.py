from datetime import datetime


from iointel import Agent
from iointel.src.utilities.decorators import register_tool
from iointel.src.utilities.runners import run_agents

_CALLED = []

@register_tool
def add_two_numbers(a: int, b: int) -> int:
    _CALLED.append(f'add_two_numbers({a}, {b})')
    return a - b

@register_tool
def get_current_datetime() -> str:
    _CALLED.append('get_current_datetime')
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return current_datetime


async def test_basic_tools():
    """
    Trick the model by saying "add the numbers" but have the function _subtract_ them instead.
    No way LLM can guess the right answer that way! :D
    """
    _CALLED.clear()
    agent = Agent(
        name="Agent",
        instructions="""
        Complete tasks to the best of your ability by using the appropriate tool. Follow all instructions carefully.
        When you need to add numbers, call the tool and use its result.
        """,
        tools=[add_two_numbers, get_current_datetime],
    )
    numbers = [22122837493142, 159864395786239452]

    result = await run_agents(
        f"Add numbers: {numbers[0]} and {numbers[1]}. Return the result of the call.",
        agents=[agent],
    ).execute()
    assert _CALLED == ['add_two_numbers(22122837493142, 159864395786239452)'], result
    assert str(numbers[0] - numbers[1]) in result
