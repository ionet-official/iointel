from agno.tools.zep import ZepTools as AgnoZepTools
from .common import make_base, wrap_tool


class Zep(make_base(AgnoZepTools)):
    def _get_tool(self):
        return self.Inner(
            session_id=self.session_id_,
            user_id=self.user_id_,
            api_key=self.api_key_,
            ignore_assistant_messages=self.ignore_assistant_messages_,
            add_zep_message=self.add_zep_message_,
            get_zep_memory=self.get_zep_memory_,
            search_zep_memory=self.search_zep_memory,
            instructions=self.instructions_,
            add_instructions=self.add_instructions_,
        )

    @wrap_tool("agno__zep__initialize", AgnoZepTools.initialize)
    def initialize(self) -> bool:
        return self._tool.initialize()

    @wrap_tool("agno__zep__add_zep_message", AgnoZepTools.add_zep_message)
    def add_zep_message(self, role: str, content: str) -> str:
        return self._tool.add_zep_message(role, content)

    @wrap_tool("agno__zep__get_zep_memory", AgnoZepTools.get_zep_memory)
    def get_zep_memory(self, memory_type: str = "context") -> str:
        return self._tool.get_zep_memory(memory_type)

    @wrap_tool("agno__zep__search_zep_memory", AgnoZepTools.search_zep_memory)
    def search_zep_memory(self, query: str, search_scope: str = "messages") -> str:
        return self._tool.search_zep_memory(query, search_scope)
