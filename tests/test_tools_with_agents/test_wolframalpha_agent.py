import marvin

from iointel import Agent
from iointel.src.agent_methods.tools.wolfram import query_wolfram_async


def get_wolfram_agent():
    return Agent(
        name="WolframAgent",
        instructions="""
            You are wolfram alfa AI agent.
            Use wolfram to do any calculations, and provide answers in correct format.
            """,
        tools=[query_wolfram_async],
    )


async def test_wolframalpha():
    agent = get_wolfram_agent()
    result = await marvin.run_async(
        "Find all solutions to this equation in REAL numbers: 13x^5-7x^4+3x^3+1=0. "
        "Return response in the following format: "
        "Solutions: X1,X2,X3,... "
        "Only provide solutions in REAL numbers, do not provide complex numbers in solutions. "
        "Format each solution as a float number with 2 floating digits. ",
        agents=[agent],
    )
    assert result is not None, "Expected a result from the agent run."
    assert "Solutions: -0.48" == result
