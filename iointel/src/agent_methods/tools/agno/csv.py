from functools import wraps
from pydantic import BaseModel
from agno.tools.csv_toolkit import CsvTools as AgnoCsvTools

from ..utils import register_tool
from .common import DisableAgnoRegistryMixin

import json
from pathlib import Path
from typing import Any, Dict, Optional, List, Union
# from functools import wraps

# from agno.tools.csv_toolkit import CsvTools as AgnoCsvTools
# from pydantic import BaseModel

# from ..utils import register_tool
# from .common import DisableAgnoRegistryMixin


class Csv(BaseModel, DisableAgnoRegistryMixin, AgnoCsvTools):
    """CSV helper that exposes every CsvTools method as an OpenAI function‑calling
    tool while keeping all runtime parameters (csv paths, duckdb connection, etc.)
    on the specific instance.
    """

    csvs: Optional[List[Union[str, Path]]] = None
    row_limit: Optional[int] = None
    read_csvs: bool = True
    list_csvs: bool = True
    query_csvs: bool = True
    read_column_names: bool = True
    duckdb_connection: Optional[Any] = None
    duckdb_kwargs: Optional[Dict[str, Any]] = None

    # --------------------------------------------------------------------- #
    # Constructor
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        csvs: Optional[List[Union[str, Path]]] = None,
        row_limit: Optional[int] = None,
        read_csvs: bool = True,
        list_csvs: bool = True,
        query_csvs: bool = True,
        read_column_names: bool = True,
        duckdb_connection: Optional[Any] = None,
        duckdb_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            csvs=csvs,
            row_limit=row_limit,
            read_csvs=read_csvs,
            list_csvs=list_csvs,
            query_csvs=query_csvs,
            read_column_names=read_column_names,
            duckdb_connection=duckdb_connection,
            duckdb_kwargs=duckdb_kwargs,
        )

    # --------------------------------------------------------------------- #
    # Tool‑exposed methods
    # --------------------------------------------------------------------- #

    @register_tool(name="csv_list_csv_files")
    @wraps(AgnoCsvTools.list_csv_files)
    def list_csv_files(self) -> List[str]:
        """Return a list with the *basename* (no extension) of every tracked CSV."""
        raw = super().list_csv_files()
        return json.loads(raw)

    @register_tool(name="csv_read_csv_file")
    @wraps(AgnoCsvTools.read_csv_file)
    def read_csv_file(self, csv_name: str, row_limit: Optional[int] = None) -> str:
        """Return the CSV rows as a JSON‑lines string. Optionally limit rows."""
        return super().read_csv_file(csv_name, row_limit)

    @register_tool(name="csv_get_columns")
    @wraps(AgnoCsvTools.get_columns)
    def get_columns(self, csv_name: str) -> str:
        """Return the column names of the given CSV as a JSON list."""
        return super().get_columns(csv_name)

    @register_tool(name="csv_query_csv_file")
    @wraps(AgnoCsvTools.query_csv_file)
    def query_csv_file(self, csv_name: str, sql_query: str) -> str:
        """Execute a SQL query (DuckDB dialect) against the chosen CSV and return the results as a JSON‑lines string."""
        return super().query_csv_file(csv_name, sql_query)
