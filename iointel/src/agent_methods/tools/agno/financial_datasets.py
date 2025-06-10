from typing import Optional
from agno.tools.financial_datasets import (
    FinancialDatasetsTools as AgnoFinancialDatasetsTools,
)
from pydantic import Field
from .common import make_base, wrap_tool


class FinancialDatasetsTools(make_base(AgnoFinancialDatasetsTools)):
    api_key: Optional[str] = Field(default=None, frozen=True)
    enable_financial_statements: bool = Field(default=True, frozen=True)
    enable_company_info: bool = Field(default=True, frozen=True)
    enable_market_data: bool = Field(default=True, frozen=True)
    enable_ownership_data: bool = Field(default=True, frozen=True)
    enable_news: bool = Field(default=True, frozen=True)
    enable_sec_filings: bool = Field(default=True, frozen=True)
    enable_crypto: bool = Field(default=True, frozen=True)
    enable_search: bool = Field(default=True, frozen=True)

    def _get_tool(self):
        return self.Inner(
            api_key=self.api_key,
            enable_financial_statements=self.enable_financial_statements,
            enable_company_info=self.enable_company_info,
            enable_market_data=self.enable_market_data,
            enable_ownership_data=self.enable_ownership_data,
            enable_news=self.enable_news,
            enable_sec_filings=self.enable_sec_filings,
            enable_crypto=self.enable_crypto,
            enable_search=self.enable_search,
        )

    @wrap_tool(
        "agno__financial_datasets__get_income_statements",
        AgnoFinancialDatasetsTools.get_income_statements,
    )
    def get_income_statements(
        self, ticker: str, period: str = "annual", limit: int = 10
    ) -> str:
        return self._tool.get_income_statements(ticker, period, limit)

    @wrap_tool(
        "agno__financial_datasets__get_balance_sheets",
        AgnoFinancialDatasetsTools.get_balance_sheets,
    )
    def get_balance_sheets(
        self, ticker: str, period: str = "annual", limit: int = 10
    ) -> str:
        return self._tool.get_balance_sheets(ticker, period, limit)

    @wrap_tool(
        "agno__financial_datasets__get_cash_flow_statements",
        AgnoFinancialDatasetsTools.get_cash_flow_statements,
    )
    def get_cash_flow_statements(
        self, ticker: str, period: str = "annual", limit: int = 10
    ) -> str:
        return self._tool.get_cash_flow_statements(ticker, period, limit)

    @wrap_tool(
        "agno__financial_datasets__get_company_info",
        AgnoFinancialDatasetsTools.get_company_info,
    )
    def get_company_info(self, ticker: str) -> str:
        return self._tool.get_company_info(ticker)

    @wrap_tool(
        "agno__financial_datasets__get_crypto_prices",
        AgnoFinancialDatasetsTools.get_crypto_prices,
    )
    def get_crypto_prices(
        self, symbol: str, interval: str = "1d", limit: int = 100
    ) -> str:
        return self._tool.get_crypto_prices(symbol, interval, limit)

    @wrap_tool(
        "agno__financial_datasets__get_earnings",
        AgnoFinancialDatasetsTools.get_earnings,
    )
    def get_earnings(self, ticker: str, limit: int = 10) -> str:
        return self._tool.get_earnings(ticker, limit)

    @wrap_tool(
        "agno__financial_datasets__get_financial_metrics",
        AgnoFinancialDatasetsTools.get_financial_metrics,
    )
    def get_financial_metrics(self, ticker: str) -> str:
        return self._tool.get_financial_metrics(ticker)

    @wrap_tool(
        "agno__financial_datasets__get_insider_trades",
        AgnoFinancialDatasetsTools.get_insider_trades,
    )
    def get_insider_trades(self, ticker: str, limit: int = 50) -> str:
        return self._tool.get_insider_trades(ticker, limit)

    @wrap_tool(
        "agno__financial_datasets__get_institutional_ownership",
        AgnoFinancialDatasetsTools.get_institutional_ownership,
    )
    def get_institutional_ownership(self, ticker: str) -> str:
        return self._tool.get_institutional_ownership(ticker)

    @wrap_tool(
        "agno__financial_datasets__get_news", AgnoFinancialDatasetsTools.get_news
    )
    def get_news(self, ticker: Optional[str] = None, limit: int = 50) -> str:
        return self._tool.get_news(ticker, limit)

    @wrap_tool(
        "agno__financial_datasets__get_stock_prices",
        AgnoFinancialDatasetsTools.get_stock_prices,
    )
    def get_stock_prices(
        self, ticker: str, interval: str = "1d", limit: int = 100
    ) -> str:
        return self._tool.get_stock_prices(ticker, interval, limit)

    @wrap_tool(
        "agno__financial_datasets__search_tickers",
        AgnoFinancialDatasetsTools.search_tickers,
    )
    def search_tickers(self, query: str, limit: int = 10) -> str:
        return self._tool.search_tickers(query, limit)

    @wrap_tool(
        "agno__financial_datasets__get_sec_filings",
        AgnoFinancialDatasetsTools.get_sec_filings,
    )
    def get_sec_filings(
        self, ticker: str, form_type: Optional[str] = None, limit: int = 50
    ) -> str:
        return self._tool.get_sec_filings(ticker, form_type, limit)

    @wrap_tool(
        "agno__financial_datasets__get_segmented_financials",
        AgnoFinancialDatasetsTools.get_segmented_financials,
    )
    def get_segmented_financials(
        self, ticker: str, period: str = "annual", limit: int = 10
    ) -> str:
        return self._tool.get_segmented_financials(ticker, period, limit)
