from typing import Any, Dict
from agno.tools.pandas import PandasTools as AgnoPandasTools
from .common import make_base, wrap_tool


class Pandas(make_base(AgnoPandasTools)):
    def _get_tool(self):
        return self.Inner()

    @wrap_tool(
        "agno__pandas__create_pandas_dataframe", AgnoPandasTools.create_pandas_dataframe
    )
    def create_pandas_dataframe(
        self,
        dataframe_name: str,
        create_using_function: str,
        function_parameters: Dict[str, Any],
    ) -> str:
        return self.create_pandas_dataframe(
            self, dataframe_name, create_using_function, function_parameters
        )

    @wrap_tool(
        "agno__pandas__run_dataframe_operation", AgnoPandasTools.run_dataframe_operation
    )
    def run_dataframe_operation(
        self, dataframe_name: str, operation: str, operation_parameters: Dict[str, Any]
    ) -> str:
        return self.run_dataframe_operation(
            self, dataframe_name, operation, operation_parameters
        )
