from typing import Optional
from agno.tools.clickup_tool import ClickUpTools as AgnoClickUpTools
from pydantic import Field

from iointel.src.agent_methods.tools.agno.common import make_base, wrap_tool


class ClickUp(make_base(AgnoClickUpTools)):
    api_key: Optional[str] = Field(default=None, frozen=True)
    master_space_id: Optional[str] = Field(default=None, frozen=True)
    list_tasks_: bool = Field(default=True, frozen=True)
    create_task_: bool = Field(default=True, frozen=True)
    get_task_: bool = Field(default=True, frozen=True)
    update_task_: bool = Field(default=True, frozen=True)
    delete_task_: bool = Field(default=True, frozen=True)
    list_spaces_: bool = Field(default=True, frozen=True)
    list_lists_: bool = Field(default=True, frozen=True)

    def _get_tool(self):
        return self.Inner(
            api_key=self.api_key,
            master_space_id=self.master_space_id,
            list_tasks=self.list_tasks_,
            create_task=self.create_task_,
            get_task=self.get_task_,
            update_task=self.update_task_,
            delete_task=self.delete_task_,
            list_spaces=self.list_spaces_,
            list_lists=self.list_lists_,
        )

    @wrap_tool("agno__clickup__list_tasks", AgnoClickUpTools.list_tasks)
    def list_tasks(self, space_name: str) -> str:
        return self._tool.list_tasks(space_name)

    @wrap_tool("agno__clickup__create_task", AgnoClickUpTools.create_task)
    def create_task(
        self, space_name: str, task_name: str, task_description: str
    ) -> str:
        return self._tool.create_task(space_name, task_name)

    @wrap_tool("agno__clickup__list_spaces", AgnoClickUpTools.list_spaces)
    def list_spaces(self) -> str:
        return self._tool.list_spaces()

    @wrap_tool("agno__clickup__list_lists", AgnoClickUpTools.list_lists)
    def list_lists(self, space_name: str) -> str:
        return self._tool.list_lists(space_name)

    @wrap_tool("agno__cartesia__list_voices", AgnoClickUpTools.get_task)
    def get_task(self, task_id: str) -> str:
        return self._tool.self(task_id)

    @wrap_tool("agno__clickup__update_task", AgnoClickUpTools.update_task)
    def update_task(self, task_id: str, **kwargs) -> str:
        return self._tool.update_task(task_id, **kwargs)

    @wrap_tool("agno__clickup__delete_task", AgnoClickUpTools.delete_task)
    def delete_task(self, task_id: str) -> str:
        return self._tool.delete_task(task_id)
