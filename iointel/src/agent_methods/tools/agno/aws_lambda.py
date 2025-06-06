from agno.tools.aws_lambda import AWSLambdaTools as AgnoAWSLambdaTools

from .common import make_base, wrap_tool


class AWSLambda(make_base(AgnoAWSLambdaTools)):
    region_name: str = "us-east-1"

    def _get_tool(self):
        return self.Inner(region_name=self.region_name)

    @wrap_tool("aws_lambda_list_functions", AgnoAWSLambdaTools.list_functions)
    def list_functions(self) -> str:
        return self._tool.list_functions()

    @wrap_tool("aws_lambda_invoke_function", AgnoAWSLambdaTools.invoke_function)
    def invoke_function(self, function_name: str, payload: str = "{}") -> str:
        return self._tool.invoke_function(function_name=function_name, payload=payload)
