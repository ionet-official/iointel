from agno.tools.jira import JiraTools as AgnoJiraTools
from .common import make_base, wrap_tool


class Jira(make_base(AgnoJiraTools)):
    def _get_tool(self):
        return self.Inner(
            server_url=self.server_url_,
            username=self.username_,
            password=self.password_,
            token=self.token_,
        )

    @wrap_tool("agno__jira__get_issue", AgnoJiraTools.get_issue)
    def get_issue(self, issue_key: str) -> str:
        return self.get_issue(self, issue_key)

    @wrap_tool("agno__jira__create_issue", AgnoJiraTools.create_issue)
    def create_issue(
        self, project_key: str, summary: str, description: str, issuetype: str = "Task"
    ) -> str:
        return self.create_issue(self, project_key, summary, description, issuetype)

    @wrap_tool("agno__jira__search_issues", AgnoJiraTools.search_issues)
    def search_issues(self, jql_str: str, max_results: int = 50) -> str:
        return self.search_issues(self, jql_str, max_results)

    @wrap_tool("agno__jira__add_comment", AgnoJiraTools.add_comment)
    def add_comment(self, issue_key: str, comment: str) -> str:
        return self.add_comment(self, issue_key, comment)
