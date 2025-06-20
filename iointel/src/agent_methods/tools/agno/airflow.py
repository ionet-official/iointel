from pathlib import Path
from agno.tools.airflow import AirflowTools as AgnoAirflowTools

from .common import make_base, wrap_tool


class Airflow(make_base(AgnoAirflowTools)):
    dags_dir: Path | str | None = None

    def _get_tool(self):
        return self.Inner(dags_dir=self.dags_dir, save_dag=True, read_dag=True)

    @wrap_tool("airflow_save_dag_file", AgnoAirflowTools.save_dag_file)
    def save_dag_file(self, contents: str, dag_file: str) -> str:
        return self._tool.save_dag_file(contents=contents, dag_file=dag_file)

    @wrap_tool("airflow_read_dag_file", AgnoAirflowTools.read_dag_file)
    def read_dag_file(self, dag_file: str) -> str:
        return self._tool.read_dag_file(dag_file=dag_file)
