from functools import wraps
from agno.tools.aws_lambda import AWSLambdaTools as AgnoAWSLambdaTools

from ..utils import register_tool
from .common import DisableAgnoRegistryMixin


class AWSLambda(DisableAgnoRegistryMixin, AgnoAWSLambdaTools):
    def __init__(self, region_name: str = "us-east-1"):
        super().__init__(region_name=region_name)

    @register_tool(name="aws_lambda_list_functions")
    @wraps(AgnoAWSLambdaTools.list_functions)
    def list_functions(self) -> str:
        return super().list_functions()

    @register_tool(name="aws_lambda_invoke_function")
    @wraps(AgnoAWSLambdaTools.invoke_function)
    def invoke_function(self, function_name: str, payload: str = "{}") -> str:
        return super().invoke_function(function_name=function_name, payload=payload)
