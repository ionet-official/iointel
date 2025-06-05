from agno.tools.yfinance import YFinanceTools as AgnoYFinanceTools

from .common import make_base, wrap_tool


class YFinance(make_base(AgnoYFinanceTools)):
    # base_dir: Path | None = None

    def _get_tool(self):
        return self.Inner()

    @wrap_tool("get_current_stock_price", AgnoYFinanceTools.get_current_stock_price)
    def get_current_stock_price(self, symbol: str) -> str:
        return self._tool.get_current_stock_price(symbol)

    @wrap_tool("get_company_info", AgnoYFinanceTools.get_company_info)
    def get_company_info(self, symbol: str) -> str:
        return self._tool.get_company_info(symbol)

    @wrap_tool(
        "get_historical_stock_prices", AgnoYFinanceTools.get_historical_stock_prices
    )
    def get_historical_stock_prices(
        self, symbol: str, period: str = "1mo", interval: str = "1d"
    ) -> str:
        return self._tool.get_historical_stock_prices(symbol, period, interval)

    @wrap_tool("get_stock_fundamentals", AgnoYFinanceTools.get_stock_fundamentals)
    def get_stock_fundamentals(self, symbol: str) -> str:
        return self._tool.get_stock_fundamentals(symbol)

    @wrap_tool("get_income_statements", AgnoYFinanceTools.get_income_statements)
    def get_income_statements(self, symbol: str) -> str:
        return self._tool.get_income_statements(symbol)

    @wrap_tool("get_key_financial_ratios", AgnoYFinanceTools.get_key_financial_ratios)
    def get_key_financial_ratios(self, symbol: str) -> str:
        return self._tool.get_key_financial_ratios(symbol)

    @wrap_tool(
        "get_analyst_recommendations", AgnoYFinanceTools.get_analyst_recommendations
    )
    def get_analyst_recommendations(self, symbol: str) -> str:
        return self._tool.get_analyst_recommendations(symbol)

    @wrap_tool("get_company_news", AgnoYFinanceTools.get_company_news)
    def get_company_news(self, symbol: str, num_stories: int = 3) -> str:
        return self._tool.get_company_news(symbol, num_stories)

    @wrap_tool("get_technical_indicators", AgnoYFinanceTools.get_technical_indicators)
    def get_technical_indicators(self, symbol: str, period: str = "3mo") -> str:
        return self._tool.get_technical_indicators(symbol, period)
