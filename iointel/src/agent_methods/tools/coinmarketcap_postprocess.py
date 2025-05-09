import re
from typing import Any

def format_coinmarketcap_result(query: str, tool_result: Any) -> Any:
    if tool_result is None:
        return "More information is needed to complete your request. Please provide additional details or clarify your task."

    q = query.lower()

    # 1. What year was bitcoin established at?
    if "bitcoin" in q and ("year" in q or "established" in q or "date" in q):
        # Try to extract from coin info
        data = tool_result.get("data")
        if data:
            for coin in data.values():
                if "date_added" in coin:
                    year = coin["date_added"].split("-")[0]
                    return year
        return "Unknown year"

    # 2. Top 10 currencies by capitalization
    if "top 10" in q and ("cryptocurrencies" in q or "currencies" in q) and "capitalization" in q:
        data = tool_result.get("data")
        if data and isinstance(data, list):
            names = [coin.get("name", "") for coin in data[:10]]
            return ",".join(names)
        return "No data"

    # 3. List all cryptos with symbol BTC
    if "symbol btc" in q or ("symbol" in q and "btc" in q):
        data = tool_result.get("data")
        if data:
            names = [coin.get("name", "") for coin in data.values()]
            return ",".join(names)
        return "No data"

    # 4. Capitalization or price as a float (current)
    if ("capitalization" in q or "market cap" in q or "price" in q) and "bitcoin" in q and "historical" not in q and "yesterday" not in q:
        data = tool_result.get("data")
        if data:
            # Try to get by slug or symbol
            for coin in data.values():
                quote = coin.get("quote", {})
                usd = quote.get("USD", {})
                if "market_cap" in usd:
                    return usd["market_cap"]
                if "price" in usd:
                    return usd["price"]
        return "No data"

    # 5. Historical price (e.g., yesterday at 12:00)
    if ("historical" in q or "yesterday" in q or "at" in q) and "price" in q and "bitcoin" in q:
        data = tool_result.get("data")
        if data and isinstance(data, list):
            # Try to get the first price
            for entry in data:
                quote = entry.get("quote", {})
                usd = quote.get("USD", {})
                if "price" in usd:
                    return usd["price"]
        return "No data"

    # Fallback: try to extract a float from the result
    if isinstance(tool_result, (int, float)):
        return tool_result
    if isinstance(tool_result, str):
        m = re.search(r"[0-9]+(\.[0-9]+)?", tool_result)
        if m:
            return m.group(0)
        return tool_result

    return str(tool_result) 