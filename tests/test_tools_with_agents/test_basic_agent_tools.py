from datetime import datetime
import asyncio
import os
from pydantic import BaseModel
import pytest
from pydantic_ai.models.openai import OpenAIModel

from iointel import Agent, LiberalToolAgent
from iointel.src.utilities.decorators import register_tool
from iointel.src.utilities.runners import run_agents
from iointel.src.agent_methods.agents.agents_factory import (
    create_agent,
    instantiate_agent_default,
)
from iointel.src.agent_methods.agents.tool_factory import instantiate_stateful_tool
from iointel.src.agent_methods.data_models.datamodels import AgentParams


_CALLED = []


@register_tool
def add_two_numbers(a: int, b: int) -> int:
    _CALLED.append(f"add_two_numbers({a}, {b})")
    return a - b


@register_tool
def get_current_datetime() -> str:
    _CALLED.append("get_current_datetime")
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
        tools=["add_two_numbers", get_current_datetime],
    )
    numbers = [22122837493142, 159864395786239452]

    result = await run_agents(
        f"Add numbers: {numbers[0]} and {numbers[1]}. Return the result of the call.",
        agents=[agent],
    ).execute()
    assert _CALLED == ["add_two_numbers(22122837493142, 159864395786239452)"], result
    assert str(numbers[0] - numbers[1]) in result


class StubTool(BaseModel):
    arg: str
    counter: dict[str, int] = {}

    @register_tool("whatever")
    def whatever(self):
        self.counter["whatever"] = self.counter.get("whatever", 0) + 1
        print(f"{self.counter=}")
        return f"foo, where arg={self.arg}"

    @register_tool("something")
    async def something(self):
        await asyncio.sleep(0.1)
        self.counter["something"] = self.counter.get("something", 0) + 1
        print(f"{self.counter=}")
        return f"bar, where arg={self.arg}"

    @register_tool
    async def more_whatever(self):
        await asyncio.sleep(0.1)
        self.counter["more_whatever"] = self.counter.get("more_whatever", 0) + 1
        print(f"{self.counter=}")
        return f"baz, where arg={self.arg}"


async def test_instancemethod_tool():
    tool = StubTool(arg="hello")
    agent = Agent(
        name="simple",
        instructions="Complete tasks to the best of your ability by using the appropriate tool. Follow all instructions carefully.",
        tools=[tool.whatever, tool.something],
    )
    result = await run_agents(
        "Call `whatever` tool exactly once and return its result", agents=[agent]
    ).execute()
    assert result
    result = await run_agents(
        "Call `something` tool exactly once and return its result", agents=[agent]
    ).execute()
    assert result
    assert len(tool.counter) == 2, "Both tools must have been called"


async def test_stateful_tool():
    agent = await create_agent(
        AgentParams(
            name="simple",
            instructions="Complete tasks to the best of your ability by using the appropriate tool. Follow all instructions carefully.",
            tools=[("StubTool-more_whatever", {"arg": "hey guys"})],
        )
    )
    result = await run_agents(
        "Call `StubTool-more_whatever` tool exactly once and return its result",
        agents=[agent],
    ).execute()
    assert result


def _custom_agent(params: AgentParams) -> Agent:
    return instantiate_agent_default(
        params.model_copy(
            update={"tools": [StubTool(arg="custom agent").more_whatever]}
        )
    )


def _custom_tool(tool, state_args: dict | None) -> BaseModel | None:
    return instantiate_stateful_tool(tool, dict(state_args, arg="custom tool"))


@pytest.mark.parametrize(
    "agent_creator,tool_creator,marker",
    [
        (None, None, "hey guys"),
        (instantiate_agent_default, None, "hey guys"),
        (instantiate_agent_default, instantiate_stateful_tool, "hey guys"),
        (None, instantiate_stateful_tool, "hey guys"),
        (_custom_agent, None, "custom agent"),
        (None, _custom_tool, "custom tool"),
    ],
)
async def test_custom_instantiators(agent_creator, tool_creator, marker):
    agent = await create_agent(
        AgentParams(
            name="simple",
            instructions="Complete tasks to the best of your ability by using the appropriate tool. Follow all instructions carefully.",
            tools=[("StubTool-more_whatever", {"arg": "hey guys"})],
        ),
        instantiate_agent=agent_creator,
        instantiate_tool=tool_creator,
    )
    assert marker in str(agent.tools[0].model_dump().get("fn_self")), (
        "Expected to have stateful tool arg"
    )


async def test_liberal_tool_agent():
    class NoIdea1(BaseModel):
        def weirdo(self) -> int:
            return 42

    agent = LiberalToolAgent(
        name="simple",
        instructions="Complete tasks to the best of your ability by using the appropriate tool. Follow all instructions carefully.",
        tools=[NoIdea1().weirdo],
    )
    assert "weirdo" in agent.tools[0].name


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
async def test_liberal_tool_agent_call():
    class NoIdea2(BaseModel):
        def weirdo(self) -> int:
            return 42

    agent = LiberalToolAgent(
        name="simple",
        instructions="Complete tasks to the best of your ability by using the appropriate tool. Follow all instructions carefully.",
        tools=[NoIdea2().weirdo],
        model=OpenAIModel(model_name="gpt-4o-mini"),
    )
    result = await agent.run(
        "execute `weirdo` tool and pass back its result", output_type=int
    )
    assert result["result"] == 42
