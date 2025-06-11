from agno.tools.webbrowser import WebBrowserTools as AgnoWebBrowserTools
from .common import make_base, wrap_tool


class WebBrowser(make_base(AgnoWebBrowserTools)):
    def _get_tool(self):
        return self.Inner()

    @wrap_tool("agno__webbrowser__open_page", AgnoWebBrowserTools.open_page)
    def open_page(self, url: str, new_window: bool = False):
        return self.open_page(self, url, new_window)
