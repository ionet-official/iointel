from typing import Any, Optional, List, Dict
from agno.tools.todoist import TodoistTools as AgnoTodoistTools
from .common import make_base, wrap_tool


class Todoist(make_base(AgnoTodoistTools)):
    def _get_tool(self):
        return self.Inner(
            api_token=self.api_token_,
            create_task=self.create_task_,
            get_task=self.get_task_,
            update_task=self.update_task_,
            close_task=self.close_task_,
            delete_task=self.delete_task_,
            get_active_tasks=self.get_active_tasks_,
            get_projects=self.get_projects,
        )

    @wrap_tool("agno__todoist___task_to_dict", AgnoTodoistTools._task_to_dict)
    def _task_to_dict(self, task: Any) -> Dict[str, Any]:
        return self._task_to_dict(self, task)

    @wrap_tool("agno__todoist__create_task", AgnoTodoistTools.create_task)
    def create_task(
        self,
        content: str,
        project_id: Optional[str] = None,
        due_string: Optional[str] = None,
        priority: Optional[int] = None,
        labels: Optional[List[str]] = None,
    ) -> str:
        return self.create_task(self, content, project_id, due_string, priority, labels)

    @wrap_tool("agno__todoist__get_task", AgnoTodoistTools.get_task)
    def get_task(self, task_id: str) -> str:
        return self.get_task(self, task_id)

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
        return self.update_task(
            self,
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
        return self.close_task(self, task_id)

    @wrap_tool("agno__todoist__delete_task", AgnoTodoistTools.delete_task)
    def delete_task(self, task_id: str) -> str:
        return self.delete_task(self, task_id)

    @wrap_tool("agno__todoist__get_active_tasks", AgnoTodoistTools.get_active_tasks)
    def get_active_tasks(self) -> str:
        return self.get_active_tasks(self)

    @wrap_tool("agno__todoist__get_projects", AgnoTodoistTools.get_projects)
    def get_projects(self) -> str:
        return self.get_projects(self)
