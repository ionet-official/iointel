from typing import Any, Optional
from agno.tools.google_bigquery import GoogleBigQueryTools as AgnoGoogleBigQueryTools
from .common import make_base, wrap_tool
from pydantic import Field


class GoogleBigQuery(make_base(AgnoGoogleBigQueryTools)):
    dataset: str = Field(frozen=True)
    project: Optional[str] = Field(default=None, frozen=True)
    location: Optional[str] = Field(default=None, frozen=True)
    list_tables_: Optional[bool] = Field(default=True, frozen=True)
    describe_table_: Optional[bool] = Field(default=True, frozen=True)
    run_sql_query_: Optional[bool] = Field(default=True, frozen=True)
    credentials: Optional[Any] = Field(default=None, frozen=True)

    def _get_tool(self):
        return self.Inner(
            dataset=self.dataset,
            project=self.project,
            location=self.location,
            list_tables=self.list_tables_,
            describe_table=self.describe_table_,
            run_sql_query=self.run_sql_query_,
            credentials=self.credentials,
        )

    @wrap_tool(
        "agno__google_bigquery__list_tables", AgnoGoogleBigQueryTools.list_tables
    )
    def list_tables(self) -> str:
        return self._tool.list_tables()

    @wrap_tool(
        "agno__google_bigquery__describe_table", AgnoGoogleBigQueryTools.describe_table
    )
    def describe_table(self, table_id: str) -> str:
        return self._tool.describe_table(table_id)

    @wrap_tool(
        "agno__google_bigquery__run_sql_query", AgnoGoogleBigQueryTools.run_sql_query
    )
    def run_sql_query(self, query: str) -> str:
        return self._tool.run_sql_query(query)
