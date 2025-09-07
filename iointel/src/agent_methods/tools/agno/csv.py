from agno.tools.csv_toolkit import CsvTools as AgnoCsvTools

import json
from pathlib import Path
from typing import Any, Dict, Optional, List, Union

from iointel.src.agent_methods.tools.agno.common import make_base, wrap_tool


class Csv(make_base(AgnoCsvTools)):
    """CSV helper that exposes every CsvTools method as an OpenAI function‑calling
    tool while keeping all runtime parameters (csv paths, duckdb connection, etc.)
    on the specific instance.
    """

    csvs: Optional[List[Union[str, Path]]] = None
    row_limit: Optional[int] = None
    duckdb_connection: Optional[Any] = None
    duckdb_kwargs: Optional[Dict[str, Any]] = None

    def _get_tool(self):
        return self.Inner(
            csvs=self.csvs,
            row_limit=self.row_limit,
            read_csvs=True,
            list_csvs=True,
            query_csvs=True,
            read_column_names=True,
            duckdb_connection=self.duckdb_connection,
            duckdb_kwargs=self.duckdb_kwargs,
        )

    @wrap_tool("csv_list_csv_files", AgnoCsvTools.list_csv_files)
    def list_csv_files(self) -> List[str]:
        """Return a list with the *basename* (no extension) of every tracked CSV."""
        raw = self._tool.list_csv_files()
        return json.loads(raw)

    @wrap_tool("csv_read_csv_file", AgnoCsvTools.read_csv_file)
    def read_csv_file(self, csv_name: str, row_limit: Optional[int] = None) -> str:
        """Return the CSV rows as a JSON‑lines string. Optionally limit rows."""
        return self._tool.read_csv_file(csv_name, row_limit)

    @wrap_tool("csv_get_columns", AgnoCsvTools.get_columns)
    def get_columns(self, csv_name: str) -> str:
        """Return the column names of the given CSV as a JSON list."""
        return self._tool.get_columns(csv_name)

    @wrap_tool("csv_query_csv_file", AgnoCsvTools.query_csv_file)
    def query_csv_file(self, csv_name: str, sql_query: str) -> str:
        """Execute a SQL query (DuckDB dialect) against the chosen CSV and return the results as a JSON‑lines string."""
        return self._tool.query_csv_file(csv_name, sql_query)
