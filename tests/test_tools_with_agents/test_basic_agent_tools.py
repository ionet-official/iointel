from datetime import datetime


from iointel import Agent
from iointel.src.utilities.decorators import register_tool
from iointel.src.utilities.runners import run_agents


@register_tool
def add_two_numbers(a: int, b: int) -> int:
    return a + b

@register_tool
def get_current_datetime() -> str:
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return current_datetime


async def test_basic_tools():
    """
    LLama can't add such big numbers, so it must use the tool
    """
    agent = Agent(
        name="Agent",
        instructions="When you need to add numbers, use the tool",
        tools=[add_two_numbers, get_current_datetime],
    )
    numbers = [22122837493142, 17268162387617, 159864395786239452]

    result = await run_agents(
        f"Add three numbers: {numbers[0]} and {numbers[1]} and {numbers[2]}. Return their sum",
        agents=[agent],
    ).execute()
    assert result == str(sum(numbers))
