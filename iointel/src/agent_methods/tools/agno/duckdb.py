from typing import List, Optional, Tuple
from agno.tools.duckdb import DuckDbTools as AgnoDuckDbTools
import duckdb
from pydantic import Field

from .common import make_base, wrap_tool


class DuckDbTools(make_base(AgnoDuckDbTools)):
    db_path: Optional[str] = Field(default=None, frozen=True)
    connection_: Optional[duckdb.DuckDBPyConnection] = Field(default=None, frozen=True)
    init_commands: Optional[List] = Field(default=None, frozen=True)
    read_only: bool = Field(default=False, frozen=True)
    config: Optional[dict] = Field(default=None, frozen=True)
    run_queries: bool = Field(default=True, frozen=True)
    inspect_queries: bool = Field(default=False, frozen=True)
    create_tables: bool = Field(default=True, frozen=True)
    summarize_tables: bool = Field(default=True, frozen=True)
    export_tables: bool = Field(default=False, frozen=True)

    def _get_tool(self):
        return self.Inner(
            db_path=self.db_path,
            connection=self.connection_,
            init_commands=self.init_commands,
            read_only=self.read_only,
            config=self.config,
            run_queries=self.run_queries,
            inspect_queries=self.inspect_queries,
            create_tables=self.create_tables,
            summarize_tables=self.summarize_tables,
            export_tables=self.export_tables,
        )

    @property
    def get_connection(self) -> duckdb.DuckDBPyConnection:
        return self._tool.connection()

    @wrap_tool("agno__duckdb__show_tables", AgnoDuckDbTools.show_tables)
    def show_tables(self, show_tables: bool) -> str:
        return self._tool.show_tables(show_tables)

    @wrap_tool("agno__duckdb__describe_table", AgnoDuckDbTools.describe_table)
    def describe_table(self, table: str) -> str:
        return self._tool.describe_table(table)

    @wrap_tool("agno__duckdb__inspect_query", AgnoDuckDbTools.inspect_query)
    def inspect_query(self, query: str) -> str:
        return self._tool.inspect_query(query)

    @wrap_tool("agno__duckdb__run_query", AgnoDuckDbTools.run_query)
    def run_query(self, query: str) -> str:
        return self._tool.run_query(query)

    @wrap_tool("agno__duckdb__summarize_table", AgnoDuckDbTools.summarize_table)
    def summarize_table(self, table: str) -> str:
        return self._tool.summarize_table(table)

    @wrap_tool(
        "agno__duckdb__get_table_name_from_path",
        AgnoDuckDbTools.get_table_name_from_path,
    )
    def get_table_name_from_path(self, path: str) -> str:
        return self._tool.get_table_name_from_path(path)

    @wrap_tool(
        "agno__duckdb__create_table_from_path", AgnoDuckDbTools.create_table_from_path
    )
    def create_table_from_path(
        self, path: str, table: Optional[str] = None, replace: bool = False
    ) -> str:
        return self._tool.create_table_from_path(path, table, replace)

    @wrap_tool(
        "agno__duckdb__export_table_to_path", AgnoDuckDbTools.export_table_to_path
    )
    def export_table_to_path(
        self, table: str, format: Optional[str] = "PARQUET", path: Optional[str] = None
    ) -> str:
        return self._tool.export_table_to_path(table, format, path)

    @wrap_tool(
        "agno__duckdb__load_local_path_to_table",
        AgnoDuckDbTools.load_local_path_to_table,
    )
    def load_local_path_to_table(
        self, path: str, table: Optional[str] = None
    ) -> Tuple[str, str]:
        return self._tool.load_local_path_to_table(path, table)

    @wrap_tool(
        "agno__duckdb__load_local_csv_to_table", AgnoDuckDbTools.load_local_csv_to_table
    )
    def load_local_csv_to_table(
        self, path: str, table: Optional[str] = None, delimiter: Optional[str] = None
    ) -> Tuple[str, str]:
        return self._tool.load_local_csv_to_table(path, table, delimiter)

    @wrap_tool(
        "agno__duckdb__load_s3_path_to_table", AgnoDuckDbTools.load_s3_path_to_table
    )
    def load_s3_path_to_table(
        self, path: str, table: Optional[str] = None
    ) -> Tuple[str, str]:
        return self._tool.load_s3_path_to_table(path, table)

    @wrap_tool(
        "agno__duckdb__load_s3_csv_to_table", AgnoDuckDbTools.load_s3_csv_to_table
    )
    def load_s3_csv_to_table(
        self, path: str, table: Optional[str] = None, delimiter: Optional[str] = None
    ) -> Tuple[str, str]:
        return self._tool.load_s3_csv_to_table(path, table, delimiter)

    @wrap_tool("agno__duckdb__create_fts_index", AgnoDuckDbTools.create_fts_index)
    def create_fts_index(
        self, table: str, unique_key: str, input_values: list[str]
    ) -> str:
        return self._tool.create_fts_index(table, unique_key, input_values)

    @wrap_tool("agno__duckdb__full_text_search", AgnoDuckDbTools.full_text_search)
    def full_text_search(self, table: str, unique_key: str, search_text: str) -> str:
        return self._tool.full_text_search(table, unique_key, search_text)
