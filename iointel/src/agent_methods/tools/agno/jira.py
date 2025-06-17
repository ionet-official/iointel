from typing import Optional
from agno.tools.jira import JiraTools as AgnoJiraTools
from .common import make_base, wrap_tool
from pydantic import Field


class Jira(make_base(AgnoJiraTools)):
    server_url: Optional[str] = Field(default=None, frozen=True)
    username: Optional[str] = Field(default=None, frozen=True)
    password: Optional[str] = Field(default=None, frozen=True)
    token: Optional[str] = Field(default=None, frozen=True)

    def _get_tool(self):
        return self.Inner(
            server_url=self.server_url,
            username=self.username,
            password=self.password,
            token=self.token,
        )

    @wrap_tool("agno__jira__get_issue", AgnoJiraTools.get_issue)
    def get_issue(self, issue_key: str) -> str:
        return self._tool.get_issue(self, issue_key)

    @wrap_tool("agno__jira__create_issue", AgnoJiraTools.create_issue)
    def create_issue(
        self, project_key: str, summary: str, description: str, issuetype: str = "Task"
    ) -> str:
        return self._tool.create_issue(
            self, project_key, summary, description, issuetype
        )

    @wrap_tool("agno__jira__search_issues", AgnoJiraTools.search_issues)
    def search_issues(self, jql_str: str, max_results: int = 50) -> str:
        return self._tool.search_issues(self, jql_str, max_results)

    @wrap_tool("agno__jira__add_comment", AgnoJiraTools.add_comment)
    def add_comment(self, issue_key: str, comment: str) -> str:
        return self._tool.add_comment(self, issue_key, comment)
