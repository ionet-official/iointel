from typing import Optional, List
from agno.tools.todoist import TodoistTools as AgnoTodoistTools
from .common import make_base, wrap_tool
from pydantic import Field


class Todoist(make_base(AgnoTodoistTools)):
    api_token: Optional[str] = Field(default=None, frozen=True)

    def _get_tool(self):
        return self.Inner(
            api_token=self.api_token,
            create_task=True,
            get_task=True,
            update_task=True,
            close_task=True,
            delete_task=True,
            get_active_tasks=True,
            get_projects=True,
        )

    @wrap_tool("agno__todoist__create_task", AgnoTodoistTools.create_task)
    def create_task(
        self,
        content: str,
        project_id: Optional[str] = None,
        due_string: Optional[str] = None,
        priority: Optional[int] = None,
        labels: Optional[List[str]] = None,
    ) -> str:
        return self._tool.create_task(content, project_id, due_string, priority, labels)

    @wrap_tool("agno__todoist__get_task", AgnoTodoistTools.get_task)
    def get_task(self, task_id: str) -> str:
        return self._tool.get_task(self, task_id)

    @wrap_tool("agno__todoist__update_task", AgnoTodoistTools.update_task)
    def update_task(
        self,
        task_id: str,
        content: Optional[str] = None,
        description: Optional[str] = None,
        labels: Optional[List[str]] = None,
        priority: Optional[int] = None,
        due_string: Optional[str] = None,
        due_date: Optional[str] = None,
        due_datetime: Optional[str] = None,
        due_lang: Optional[str] = None,
        assignee_id: Optional[str] = None,
        section_id: Optional[str] = None,
    ) -> str:
        return self._tool.update_task(
            task_id,
            content,
            description,
            labels,
            priority,
            due_string,
            due_date,
            due_datetime,
            due_lang,
            assignee_id,
            section_id,
        )

    @wrap_tool("agno__todoist__close_task", AgnoTodoistTools.close_task)
    def close_task(self, task_id: str) -> str:
        return self._tool.close_task(task_id)

    @wrap_tool("agno__todoist__delete_task", AgnoTodoistTools.delete_task)
    def delete_task(self, task_id: str) -> str:
        return self._tool.delete_task(task_id)

    @wrap_tool("agno__todoist__get_active_tasks", AgnoTodoistTools.get_active_tasks)
    def get_active_tasks(self) -> str:
        return self._tool.get_active_tasks()

    @wrap_tool("agno__todoist__get_projects", AgnoTodoistTools.get_projects)
    def get_projects(self) -> str:
        return self._tool.get_projects()
