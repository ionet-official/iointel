from os import getenv
from typing import Any, Dict, Optional

import requests

import logging

from ...utilities.decorators import register_tool


logger = logging.getLogger(__name__)



class FinancialDatasetsTools:
    def __init__(
        self,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the Financial Datasets Tools with feature flags.

        Args:
            api_key: API key for Financial Datasets API (optional, can be set via environment variable)
        """

        self.api_key: Optional[str] = api_key or getenv("FINANCIAL_DATASETS_API_KEY")
        if not self.api_key:
            logger.error("FINANCIAL_DATASETS_API_KEY not set. Please set the FINANCIAL_DATASETS_API_KEY environment variable."
            )

        self.base_url = "https://api.financialdatasets.ai"


    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> str:
        """
        Makes a request to the Financial Datasets API.

        Args:
            endpoint: API endpoint to call
            params: Query parameters for the request

        Returns:
            JSON response from the API
        """
        if not self.api_key:
            logger.error("No API key provided. Cannot make request.")
            return "API key not set"

        headers = {"X-API-KEY": self.api_key}
        url = f"{self.base_url}/{endpoint}"

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to {url}: {str(e)}")
            return f"Error making request to {url}: {str(e)}"

    # Financial Statements
    @register_tool(name="financial_datasets_get_income_statements")
    def get_income_statements(self, ticker: str, period: str = "annual", limit: int = 10) -> str:
        """
        Get income statements for a ticker.

        Args:
            ticker: Stock ticker symbol
            period: 'annual', 'quarterly', or 'ttm'
            limit: Number of statements to return

        Returns:
            Dictionary containing income statements
        """
        params = {"ticker": ticker, "period": period, "limit": limit}
        return self._make_request("financials/income-statements", params)

    @register_tool(name="financial_datasets_get_balance_sheets")
    def get_balance_sheets(self, ticker: str, period: str = "annual", limit: int = 10) -> str:
        """
        Get balance sheets for a ticker.

        Args:
            ticker: Stock ticker symbol
            period: 'annual', 'quarterly', or 'ttm'
            limit: Number of statements to return

        Returns:
            Dictionary containing balance sheets
        """
        params = {"ticker": ticker, "period": period, "limit": limit}
        return self._make_request("financials/balance-sheets", params)

    @register_tool(name="financial_datasets_get_cash_flow_statements")
    def get_cash_flow_statements(self, ticker: str, period: str = "annual", limit: int = 10) -> str:
        """
        Get cash flow statements for a ticker.

        Args:
            ticker: Stock ticker symbol
            period: 'annual', 'quarterly', or 'ttm'
            limit: Number of statements to return

        Returns:
            Dictionary containing cash flow statements
        """
        params = {"ticker": ticker, "period": period, "limit": limit}
        return self._make_request("financials/cash-flow-statements", params)

    # Other API endpoints from the documentation
    @register_tool(name="financial_datasets_get_company_info")
    def get_company_info(self, ticker: str) -> str:
        """
        Get company information for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary containing company information
        """
        params = {"ticker": ticker}
        return self._make_request("company", params)

    @register_tool(name="financial_datasets_get_crypto_prices")
    def get_crypto_prices(self, symbol: str, interval: str = "1d", limit: int = 100) -> str:
        """
        Get cryptocurrency prices.

        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC')
            interval: Price interval (e.g., '1d', '1h')
            limit: Number of price points to return

        Returns:
            Dictionary containing crypto prices
        """
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        return self._make_request("crypto/prices", params)

    @register_tool(name="financial_datasets_get_earnings")
    def get_earnings(self, ticker: str, limit: int = 10) -> str:
        """
        Get earnings data for a ticker.

        Args:
            ticker: Stock ticker symbol
            limit: Number of earnings reports to return

        Returns:
            Dictionary containing earnings data
        """
        params = {"ticker": ticker, "limit": limit}
        return self._make_request("earnings", params)

    @register_tool(name="financial_datasets_get_financial_metrics")
    def get_financial_metrics(self, ticker: str) -> str:
        """
        Get financial metrics for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary containing financial metrics
        """
        params = {"ticker": ticker}
        return self._make_request("financials/metrics", params)

    @register_tool(name="financial_datasets_get_insider_trades")
    def get_insider_trades(self, ticker: str, limit: int = 50) -> str:
        """
        Get insider trades for a ticker.

        Args:
            ticker: Stock ticker symbol
            limit: Number of trades to return

        Returns:
            Dictionary containing insider trades
        """
        params = {"ticker": ticker, "limit": limit}
        return self._make_request("insider-trades", params)

    @register_tool(name="financial_datasets_get_institutional_ownership")
    def get_institutional_ownership(self, ticker: str) -> str:
        """
        Get institutional ownership data for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary containing institutional ownership data
        """
        params = {"ticker": ticker}
        return self._make_request("institutional-ownership", params)

    @register_tool(name="financial_datasets_get_market_news")
    def get_news(self, ticker: Optional[str] = None, limit: int = 50) -> str:
        """
        Get market news, optionally filtered by ticker.

        Args:
            ticker: Stock ticker symbol (optional)
            limit: Number of news items to return

        Returns:
            Dictionary containing news items
        """
        params: Dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        return self._make_request("news", params)

    @register_tool(name="financial_datasets_get_stock_prices")
    def get_stock_prices(self, ticker: str, interval: str = "1d", limit: int = 100) -> str:
        """
        Get stock prices for a ticker.

        Args:
            ticker: Stock ticker symbol
            interval: Price interval (e.g., '1d', '1h')
            limit: Number of price points to return

        Returns:
            Dictionary containing stock prices
        """
        params = {"ticker": ticker, "interval": interval, "limit": limit}
        return self._make_request("prices", params)

    @register_tool(name="financial_datasets_search_tickers")
    def search_tickers(self, query: str, limit: int = 10) -> str:
        """
        Search for tickers based on a query.

        Args:
            query: Search query
            limit: Maximum number of results to return

        Returns:
            Dictionary containing search results
        """
        params = {"query": query, "limit": limit}
        return self._make_request("search", params)

    @register_tool(name="financial_datasets_get_sec_filings")
    def get_sec_filings(self, ticker: str, form_type: Optional[str] = None, limit: int = 50) -> str:
        """
        Get SEC filings for a ticker.

        Args:
            ticker: Stock ticker symbol
            form_type: Type of SEC form (e.g., '10-K', '10-Q')
            limit: Number of filings to return

        Returns:
            Dictionary containing SEC filings
        """
        params: Dict[str, Any] = {"ticker": ticker, "limit": limit}
        if form_type:
            params["form_type"] = form_type
        return self._make_request("sec-filings", params)

    @register_tool(name="financial_datasets_get_segmented_financials")
    def get_segmented_financials(self, ticker: str, period: str = "annual", limit: int = 10) -> str:
        """
        Get segmented financials for a ticker.

        Args:
            ticker: Stock ticker symbol
            period: 'annual' or 'quarterly'
            limit: Number of reports to return

        Returns:
            Dictionary containing segmented financials
        """
        params = {"ticker": ticker, "period": period, "limit": limit}
        return self._make_request("financials/segmented", params)