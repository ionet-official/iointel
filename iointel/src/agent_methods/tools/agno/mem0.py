from typing import Optional, Union, Dict
from agno.tools.mem0 import Mem0Tools as AgnoMem0Tools
from .common import make_base, wrap_tool
from agno.agent.agent import Agent


class Mem0(make_base(AgnoMem0Tools)):
    def _get_tool(self):
        return self.Inner(
            config=self.config_,
            api_key=self.api_key_,
            user_id=self.user_id_,
        )

    @wrap_tool("agno__mem0___get_user_id", AgnoMem0Tools._get_user_id)
    def _get_user_id(self, method_name: str, agent: Optional[Agent] = None) -> str:
        return self._get_user_id(self, method_name, agent)

    @wrap_tool("agno__mem0__add_memory", AgnoMem0Tools.add_memory)
    def add_memory(self, agent: Agent, content: Union[str, Dict[str, str]]) -> str:
        return self.add_memory(self, agent, content)

    @wrap_tool("agno__mem0__search_memory", AgnoMem0Tools.search_memory)
    def search_memory(self, agent: Agent, query: str) -> str:
        return self.search_memory(self, agent, query)

    @wrap_tool("agno__mem0__get_all_memories", AgnoMem0Tools.get_all_memories)
    def get_all_memories(self, agent: Agent) -> str:
        return self.get_all_memories(self, agent)

    @wrap_tool("agno__mem0__delete_all_memories", AgnoMem0Tools.delete_all_memories)
    def delete_all_memories(self, agent: Agent) -> str:
        return self.delete_all_memories(self, agent)
