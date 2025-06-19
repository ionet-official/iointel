from typing import List, Optional
from agno.tools.github import GithubTools as AgnoGithubTools

from .common import make_base, wrap_tool
from pydantic import Field


class Github(make_base(AgnoGithubTools)):
    access_token: Optional[str] = Field(default=None, frozen=True)
    base_url: Optional[str] = Field(default=None, frozen=True)

    def _get_tool(self):
        return self.Inner(
            access_token=self.access_token,
            base_url=self.base_url,
            search_repositories=True,
            get_repository=True,
            get_pull_request=True,
            get_pull_request_changes=True,
            create_issue=True,
            create_repository=True,
            delete_repository=True,
            get_repository_languages=True,
            list_branches=True,
            get_pull_request_count=True,
            get_repository_stars=True,
            get_pull_requests=True,
            get_pull_request_comments=True,
            create_pull_request_comment=True,
            edit_pull_request_comment=True,
            get_pull_request_with_details=True,
            get_repository_with_stats=True,
            list_issues=True,
            get_issue=True,
            comment_on_issue=True,
            close_issue=True,
            reopen_issue=True,
            assign_issue=True,
            label_issue=True,
            list_issue_comments=True,
            edit_issue=True,
            create_pull_request=True,
            create_file=True,
            get_file_content=True,
            update_file=True,
            delete_file=True,
            get_directory_content=True,
            get_branch_content=True,
            create_branch=True,
            set_default_branch=True,
            search_code=True,
            search_issues_and_prs=True,
            create_review_request=True,
        )

    @wrap_tool("agno__github__authenticate", AgnoGithubTools.authenticate)
    def authenticate(self):
        return self._tool.authenticate()

    @wrap_tool("agno__github__search_repositories", AgnoGithubTools.search_repositories)
    def search_repositories(
        self,
        query: str,
        sort: str = "stars",
        order: str = "desc",
        page: int = 1,
        per_page: int = 30,
    ) -> str:
        return self._tool.search_repositories(query, sort, order, page, per_page)

    @wrap_tool("agno__github__list_repositories", AgnoGithubTools.list_repositories)
    def list_repositories(self) -> str:
        return self._tool.list_repositories()

    @wrap_tool("agno__github__create_repository", AgnoGithubTools.create_repository)
    def create_repository(
        self,
        name: str,
        private: bool = False,
        description: Optional[str] = None,
        auto_init: bool = False,
        organization: Optional[str] = None,
    ) -> str:
        return self._tool.create_repository(
            name, private, description, auto_init, organization
        )

    @wrap_tool("agno__github__get_repository", AgnoGithubTools.get_repository)
    def get_repository(self, repo_name: str) -> str:
        return self._tool.get_repository(repo_name)

    @wrap_tool(
        "agno__github__get_repository_languages",
        AgnoGithubTools.get_repository_languages,
    )
    def get_repository_languages(self, repo_name: str) -> str:
        return self._tool.get_repository_languages(repo_name)

    @wrap_tool(
        "agno__github__get_pull_request_count", AgnoGithubTools.get_pull_request_count
    )
    def get_pull_request_count(
        self,
        repo_name: str,
        state: str = "all",
        author: Optional[str] = None,
        base: Optional[str] = None,
        head: Optional[str] = None,
    ) -> str:
        return self._tool.get_pull_request_count(repo_name, state, author, base, head)

    @wrap_tool("agno__github__get_pull_request", AgnoGithubTools.get_pull_request)
    def get_pull_request(self, repo_name: str, pr_number: int) -> str:
        return self._tool.get_pull_request(repo_name, pr_number)

    @wrap_tool(
        "agno__github__get_pull_request_changes",
        AgnoGithubTools.get_pull_request_changes,
    )
    def get_pull_request_changes(self, repo_name: str, pr_number: int) -> str:
        return self._tool.get_pull_request_changes(repo_name, pr_number)

    @wrap_tool("agno__github__create_issue", AgnoGithubTools.create_issue)
    def create_issue(
        self, repo_name: str, title: str, body: Optional[str] = None
    ) -> str:
        return self._tool.create_issue(repo_name, title, body)

    @wrap_tool("agno__github__list_issues", AgnoGithubTools.list_issues)
    def list_issues(self, repo_name: str, state: str = "open", limit: int = 20) -> str:
        return self._tool.list_issues(repo_name, state, limit)

    @wrap_tool("agno__github__get_issue", AgnoGithubTools.get_issue)
    def get_issue(self, repo_name: str, issue_number: int) -> str:
        return self._tool.get_issue(repo_name, issue_number)

    @wrap_tool("agno__github__comment_on_issue", AgnoGithubTools.comment_on_issue)
    def comment_on_issue(
        self, repo_name: str, issue_number: int, comment_body: str
    ) -> str:
        return self._tool.comment_on_issue(repo_name, issue_number, comment_body)

    @wrap_tool("agno__github__close_issue", AgnoGithubTools.close_issue)
    def close_issue(self, repo_name: str, issue_number: int) -> str:
        return self._tool.close_issue(repo_name, issue_number)

    @wrap_tool("agno__github__reopen_issue", AgnoGithubTools.reopen_issue)
    def reopen_issue(self, repo_name: str, issue_number: int) -> str:
        return self._tool.reopen_issue(repo_name, issue_number)

    @wrap_tool("agno__github__assign_issue", AgnoGithubTools.assign_issue)
    def assign_issue(
        self, repo_name: str, issue_number: int, assignees: List[str]
    ) -> str:
        return self._tool.assign_issue(repo_name, issue_number, assignees)

    @wrap_tool("agno__github__label_issue", AgnoGithubTools.label_issue)
    def label_issue(self, repo_name: str, issue_number: int, labels: List[str]) -> str:
        return self._tool.label_issue(repo_name, issue_number, labels)

    @wrap_tool("agno__github__list_issue_comments", AgnoGithubTools.list_issue_comments)
    def list_issue_comments(self, repo_name: str, issue_number: int) -> str:
        return self._tool.list_issue_comments(repo_name, issue_number)

    @wrap_tool("agno__github__edit_issue", AgnoGithubTools.edit_issue)
    def edit_issue(
        self,
        repo_name: str,
        issue_number: int,
        title: Optional[str] = None,
        body: Optional[str] = None,
    ) -> str:
        return self._tool.edit_issue(repo_name, issue_number, title, body)

    @wrap_tool("agno__github__delete_repository", AgnoGithubTools.delete_repository)
    def delete_repository(self, repo_name: str) -> str:
        return self._tool.delete_repository(repo_name)

    @wrap_tool("agno__github__list_branches", AgnoGithubTools.list_branches)
    def list_branches(self, repo_name: str) -> str:
        return self._tool.list_branches(repo_name)

    @wrap_tool(
        "agno__github__get_repository_stars", AgnoGithubTools.get_repository_stars
    )
    def get_repository_stars(self, repo_name: str) -> str:
        return self._tool.get_repository_stars(repo_name)

    @wrap_tool("agno__github__get_pull_requests", AgnoGithubTools.get_pull_requests)
    def get_pull_requests(
        self,
        repo_name: str,
        state: str = "open",
        sort: str = "created",
        direction: str = "desc",
        limit: int = 50,
    ) -> str:
        return self._tool.get_pull_requests(repo_name, state, sort, direction, license)

    @wrap_tool(
        "agno__github__get_pull_request_comments",
        AgnoGithubTools.get_pull_request_comments,
    )
    def get_pull_request_comments(
        self, repo_name: str, pr_number: int, include_issue_comments: bool = True
    ) -> str:
        return self._tool.get_pull_request_comments(
            repo_name, pr_number, include_issue_comments
        )

    @wrap_tool(
        "agno__github__create_pull_request_comment",
        AgnoGithubTools.create_pull_request_comment,
    )
    def create_pull_request_comment(
        self,
        repo_name: str,
        pr_number: int,
        body: str,
        commit_id: str,
        path: str,
        position: int,
    ) -> str:
        return self._tool.create_pull_request_comment(
            repo_name, pr_number, body, commit_id, path, position
        )

    @wrap_tool(
        "agno__github__edit_pull_request_comment",
        AgnoGithubTools.edit_pull_request_comment,
    )
    def edit_pull_request_comment(
        self, repo_name: str, comment_id: int, body: str
    ) -> str:
        return self._tool.edit_pull_request_comment(repo_name, comment_id, body)

    @wrap_tool(
        "agno__github__get_pull_request_with_details",
        AgnoGithubTools.get_pull_request_with_details,
    )
    def get_pull_request_with_details(self, repo_name: str, pr_number: int) -> str:
        return self._tool.get_pull_request_with_details(repo_name, pr_number)

    @wrap_tool(
        "agno__github__get_repository_with_stats",
        AgnoGithubTools.get_repository_with_stats,
    )
    def get_repository_with_stats(self, repo_name: str) -> str:
        return self._tool.get_repository_with_stats(repo_name)

    @wrap_tool("agno__github__create_pull_request", AgnoGithubTools.create_pull_request)
    def create_pull_request(
        self,
        repo_name: str,
        title: str,
        body: str,
        head: str,
        base: str,
        draft: bool = False,
        maintainer_can_modify: bool = True,
    ) -> str:
        return self._tool.create_pull_request(
            repo_name, title, body, head, base, draft, maintainer_can_modify
        )

    @wrap_tool(
        "agno__github__create_review_request", AgnoGithubTools.create_review_request
    )
    def create_review_request(
        self,
        repo_name: str,
        pr_number: int,
        reviewers: List[str],
        team_reviewers: Optional[List[str]] = None,
    ) -> str:
        return self._tool.create_review_request(
            repo_name, pr_number, reviewers, team_reviewers
        )

    @wrap_tool("agno__github__create_file", AgnoGithubTools.create_file)
    def create_file(
        self,
        repo_name: str,
        path: str,
        content: str,
        message: str,
        branch: Optional[str] = None,
    ) -> str:
        return self._tool.create_file(repo_name, path, content, message, branch)

    @wrap_tool("agno__github__get_file_content", AgnoGithubTools.get_file_content)
    def get_file_content(
        self, repo_name: str, path: str, ref: Optional[str] = None
    ) -> str:
        return self._tool.get_file_content(repo_name, path, ref)

    @wrap_tool("agno__github__update_file", AgnoGithubTools.update_file)
    def update_file(
        self,
        repo_name: str,
        path: str,
        content: str,
        message: str,
        sha: str,
        branch: Optional[str] = None,
    ) -> str:
        return self._tool.update_file(repo_name, path, content, message, sha, branch)

    @wrap_tool("agno__github__delete_file", AgnoGithubTools.delete_file)
    def delete_file(
        self,
        repo_name: str,
        path: str,
        message: str,
        sha: str,
        branch: Optional[str] = None,
    ) -> str:
        return self._tool.delete_file(repo_name, path, message, sha, branch)

    @wrap_tool(
        "agno__github__get_directory_content", AgnoGithubTools.get_directory_content
    )
    def get_directory_content(
        self, repo_name: str, path: str, ref: Optional[str] = None
    ) -> str:
        return self._tool.get_directory_content(repo_name, path, ref)

    @wrap_tool("agno__github__get_branch_content", AgnoGithubTools.get_branch_content)
    def get_branch_content(self, repo_name: str, branch: str = "main") -> str:
        return self._tool.get_branch_content(repo_name, branch)

    @wrap_tool("agno__github__create_branch", AgnoGithubTools.create_branch)
    def create_branch(
        self, repo_name: str, branch_name: str, source_branch: Optional[str] = None
    ) -> str:
        return self._tool.create_branch(repo_name, branch_name, source_branch)

    @wrap_tool("agno__github__set_default_branch", AgnoGithubTools.set_default_branch)
    def set_default_branch(self, repo_name: str, branch_name: str) -> str:
        return self._tool.set_default_branch(repo_name, branch_name)

    @wrap_tool("agno__github__search_code", AgnoGithubTools.search_code)
    def search_code(
        self,
        query: str,
        language: Optional[str] = None,
        repo: Optional[str] = None,
        user: Optional[str] = None,
        path: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> str:
        return self._tool.search_code(query, language, repo, user, path, filename)

    @wrap_tool(
        "agno__github__search_issues_and_prs", AgnoGithubTools.search_issues_and_prs
    )
    def search_issues_and_prs(
        self,
        query: str,
        state: Optional[str] = None,
        type_filter: Optional[str] = None,
        repo: Optional[str] = None,
        user: Optional[str] = None,
        label: Optional[str] = None,
        sort: str = "created",
        order: str = "desc",
        page: int = 1,
        per_page: int = 30,
    ) -> str:
        return self._tool.search_issues_and_prs(
            query, state, type_filter, repo, user, label, sort, order, page, per_page
        )
