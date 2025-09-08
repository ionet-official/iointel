from typing import Any, Optional, Union, Dict
from agno.tools.mem0 import Mem0Tools as AgnoMem0Tools
from iointel.src.agent_methods.tools.agno.common import make_base, wrap_tool
from agno.agent.agent import Agent
from pydantic import Field


class Mem0(make_base(AgnoMem0Tools)):
    config: Optional[Dict[str, Any]] = Field(default=None, frozen=True)
    api_key: Optional[str] = Field(default=None, frozen=True)
    user_id: Optional[str] = Field(default=None, frozen=True)

    def _get_tool(self):
        return self.Inner(
            config=self.config,
            api_key=self.api_key,
            user_id=self.user_id,
        )

    def _resolve_agent(self, agent_id: str) -> Agent:
        """
        Resolve agent from ID. This is a placeholder implementation.
        In a real system, this would look up the agent from a registry or database.
        """
        # TODO: Implement actual agent resolution from agent_id
        # For now, return a mock agent or raise an error
        raise NotImplementedError("Agent resolution not implemented yet")

    @wrap_tool("agno__mem0__add_memory", AgnoMem0Tools.add_memory)
    def add_memory(self, agent_id: str, content: Union[str, Dict[str, str]]) -> str:
        agent = self._resolve_agent(agent_id)
        return self._tool.add_memory(agent, content)

    @wrap_tool("agno__mem0__search_memory", AgnoMem0Tools.search_memory)
    def search_memory(self, agent_id: str, query: str) -> str:
        agent = self._resolve_agent(agent_id)
        return self._tool.search_memory(agent, query)

    @wrap_tool("agno__mem0__get_all_memories", AgnoMem0Tools.get_all_memories)
    def get_all_memories(self, agent_id: str) -> str:
        agent = self._resolve_agent(agent_id)
        return self._tool.get_all_memories(agent)

    @wrap_tool("agno__mem0__delete_all_memories", AgnoMem0Tools.delete_all_memories)
    def delete_all_memories(self, agent_id: str) -> str:
        agent = self._resolve_agent(agent_id)
        return self._tool.delete_all_memories(agent)
