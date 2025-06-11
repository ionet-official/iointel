from typing import Optional, List
from agno.tools.sql import SQLTools as AgnoSQLTools
from .common import make_base, wrap_tool


class SQL(make_base(AgnoSQLTools)):
    def _get_tool(self):
        return self.Inner(
            db_url=self.db_url_,
            db_engine=self.db_engine_,
            user=self.user_,
            password=self.password_,
            host=self.host_,
            port=self.port_,
            schema=self.schema_,
            dialect=self.dialect_,
            tables=self.tables_,
            list_tables=self.list_tables_,
            describe_table=self.describe_table_,
            run_sql_query=self.run_sql_query_,
        )

    @wrap_tool("agno__sql__list_tables", AgnoSQLTools.list_tables)
    def list_tables(self) -> str:
        return self.list_tables(self)

    @wrap_tool("agno__sql__describe_table", AgnoSQLTools.describe_table)
    def describe_table(self, table_name: str) -> str:
        return self.describe_table(self, table_name)

    @wrap_tool("agno__sql__run_sql_query", AgnoSQLTools.run_sql_query)
    def run_sql_query(self, query: str, limit: Optional[int] = 10) -> str:
        return self.run_sql_query(self, query, limit)

    @wrap_tool("agno__sql__run_sql", AgnoSQLTools.run_sql)
    def run_sql(self, sql: str, limit: Optional[int] = None) -> List[dict]:
        return self.run_sql(self, sql, limit)
