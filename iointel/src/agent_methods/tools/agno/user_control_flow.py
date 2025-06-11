from agno.tools.user_control_flow import (
    UserControlFlowTools as AgnoUserControlFlowTools,
)
from .common import make_base, wrap_tool


class UserControlFlow(make_base(AgnoUserControlFlowTools)):
    def _get_tool(self):
        return self.Inner(
            instructions=self.instructions_,
            add_instructions=self.add_instructions_,
        )

    @wrap_tool(
        "agno__usercontrolflow__get_user_input", AgnoUserControlFlowTools.get_user_input
    )
    def get_user_input(self, user_input_fields: list[dict]) -> str:
        return self.get_user_input(self, user_input_fields)
