from typing import Optional
from agno.tools.linear import LinearTools as AgnoLinearTools
from .common import make_base, wrap_tool


class Linear(make_base(AgnoLinearTools)):
    def _get_tool(self):
        return self.Inner(
            get_user_details=self.get_user_details_,
            get_issue_details=self.get_issue_details_,
            create_issue=self.create_issue_,
            update_issue=self.update_issue_,
            get_user_assigned_issues=self.get_user_assigned_issues_,
            get_workflow_issues=self.get_workflow_issues_,
            get_high_priority_issues=self.get_high_priority_issues,
        )

    @wrap_tool("agno__linear___execute_query", AgnoLinearTools._execute_query)
    def _execute_query(self, query, variables=None):
        return self._execute_query(self, query, variables)

    @wrap_tool("agno__linear__get_user_details", AgnoLinearTools.get_user_details)
    def get_user_details(self) -> Optional[str]:
        return self.get_user_details(self)

    @wrap_tool("agno__linear__get_issue_details", AgnoLinearTools.get_issue_details)
    def get_issue_details(self, issue_id: str) -> Optional[str]:
        return self.get_issue_details(self, issue_id)

    @wrap_tool("agno__linear__create_issue", AgnoLinearTools.create_issue)
    def create_issue(
        self,
        title: str,
        description: str,
        team_id: str,
        project_id: str,
        assignee_id: str,
    ) -> Optional[str]:
        return self.create_issue(
            self, title, description, team_id, project_id, assignee_id
        )

    @wrap_tool("agno__linear__update_issue", AgnoLinearTools.update_issue)
    def update_issue(self, issue_id: str, title: Optional[str]) -> Optional[str]:
        return self.update_issue(self, issue_id, title)

    @wrap_tool(
        "agno__linear__get_user_assigned_issues",
        AgnoLinearTools.get_user_assigned_issues,
    )
    def get_user_assigned_issues(self, user_id: str) -> Optional[str]:
        return self.get_user_assigned_issues(self, user_id)

    @wrap_tool("agno__linear__get_workflow_issues", AgnoLinearTools.get_workflow_issues)
    def get_workflow_issues(self, workflow_id: str) -> Optional[str]:
        return self.get_workflow_issues(self, workflow_id)

    @wrap_tool(
        "agno__linear__get_high_priority_issues",
        AgnoLinearTools.get_high_priority_issues,
    )
    def get_high_priority_issues(self) -> Optional[str]:
        return self.get_high_priority_issues(self)
