from iointel.src.utilities import tooling
import iointel.src.agent_methods.tools.agno.calculator  # noqa: F401
from iointel.src.agent_methods.tools.firecrawl import Crawler
from iointel.src.agent_methods.agents import tool_factory


async def test_simple_stateful_tool_init():
    tools = await tool_factory.resolve_tools(tools=["calculator_add"])
    assert tools[0].fn_self is not None, "Simple tool must be auto-initialised"


def test_show_default_args():
    result = tooling.show_tool_default_args()
    for lst in result.values():
        if "CRAWLER_API_KEY" in lst:
            break
    else:
        assert False, "Firecrawl key not listed"


async def test_fill_stateful_default():
    my_mapping = {"CRAWLER_API_KEY": "foo"}
    tooling.fill_tool_defaults(my_mapping)
    tools = await tool_factory.resolve_tools(tools=["Crawler-scrape_url"])
    fn_self = tools[0]._load_fn_self()
    assert isinstance(fn_self, Crawler)
    assert fn_self.api_key == "foo"


async def test_fill_stateful_dotenv(tmp_path):
    env = tmp_path / "test.env"
    env.write_text("CRAWLER_API_KEY=bar")

    tooling.fill_tool_defaults(env.absolute())
    tools = await tool_factory.resolve_tools(tools=["Crawler-scrape_url"])
    fn_self = tools[0]._load_fn_self()
    assert isinstance(fn_self, Crawler)
    assert fn_self.api_key == "bar"
