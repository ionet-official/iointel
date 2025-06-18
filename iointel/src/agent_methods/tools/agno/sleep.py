from agno.tools.sleep import SleepTools as AgnoSleepTools
from .common import make_base, wrap_tool


class Sleep(make_base(AgnoSleepTools)):
    def _get_tool(self):
        return self.Inner()

    @wrap_tool("agno__sleep__sleep", AgnoSleepTools.sleep)
    def sleep(self, seconds: int) -> str:
        return self._tool.sleep(seconds)
