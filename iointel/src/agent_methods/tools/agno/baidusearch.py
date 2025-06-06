from agno.tools.baidusearch import BaiduSearchTools as AgnoBaiduSearchTools


from .common import make_base, wrap_tool


class BaiduSearch(make_base(AgnoBaiduSearchTools)):
    def _get_tool(self):
        return self.Inner()

    @wrap_tool("agno__baidu__make_request", AgnoBaiduSearchTools.baidu_search)
    def baidu_search(
        self, query: str, max_results: int = 5, language: str = "zh"
    ) -> str:
        return self._tool.baidu_search(query, max_results, language)
