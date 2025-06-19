from typing import Any, Dict, Optional, List
from agno.tools.sql import SQLTools as AgnoSQLTools
from .common import make_base, wrap_tool

from sqlalchemy import Engine
from pydantic import Field


class SQL(make_base(AgnoSQLTools)):
    db_url: Optional[str] = Field(default=None, frozen=True)
    db_engine: Optional[Engine] = Field(default=None, frozen=True)
    user: Optional[str] = Field(default=None, frozen=True)
    password: Optional[str] = Field(default=None, frozen=True)
    host: Optional[str] = Field(default=None, frozen=True)
    port: Optional[int] = Field(default=None, frozen=True)
    schema: Optional[str] = Field(default=None, frozen=True)
    dialect: Optional[str] = Field(default=None, frozen=True)
    tables: Optional[Dict[str, Any]] = Field(default=None, frozen=True)

    def _get_tool(self):
        return self.Inner(
            db_url=self.db_url,
            db_engine=self.db_engine,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            schema=self.schema,
            dialect=self.dialect,
            tables=self.tables,
            list_tables=True,
            describe_table=True,
            run_sql_query=True,
        )

    @wrap_tool("agno__sql__list_tables", AgnoSQLTools.list_tables)
    def list_tables(self) -> str:
        return self._tool.list_tables()

    @wrap_tool("agno__sql__describe_table", AgnoSQLTools.describe_table)
    def describe_table(self, table_name: str) -> str:
        return self._tool.describe_table(table_name)

    @wrap_tool("agno__sql__run_sql_query", AgnoSQLTools.run_sql_query)
    def run_sql_query(self, query: str, limit: Optional[int] = 10) -> str:
        return self._tool.run_sql_query(query, limit)

    @wrap_tool("agno__sql__run_sql", AgnoSQLTools.run_sql)
    def run_sql(self, sql: str, limit: Optional[int] = None) -> List[dict]:
        return self._tool.run_sql(sql, limit)
