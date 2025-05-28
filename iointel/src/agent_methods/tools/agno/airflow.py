from pathlib import Path
from typing import Optional, Union
from functools import wraps
from agno.tools.airflow import AirflowTools as AgnoAirflowTools

from ..utils import register_tool
from .common import DisableAgnoRegistryMixin


class Airflow(DisableAgnoRegistryMixin, AgnoAirflowTools):
    def __init__(self, dags_dir: Optional[Union[Path, str]] = None):
        super().__init__(dags_dir=dags_dir, save_dag=True, read_dag=True)

    @register_tool(name="airflow_save_dag_file")
    @wraps(AgnoAirflowTools.save_dag_file)
    def save_dag_file(self, contents: str, dag_file: str) -> str:
        return super().save_dag_file(contents=contents, dag_file=dag_file)

    @register_tool(name="airflow_read_dag_file")
    @wraps(AgnoAirflowTools.read_dag_file)
    def read_dag_file(self, dag_file: str) -> str:
        return super().read_dag_file(dag_file=dag_file)
