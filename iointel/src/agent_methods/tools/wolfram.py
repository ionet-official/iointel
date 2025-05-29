import os
import urllib.parse

import httpx

WOLFRAM_API_KEY = os.getenv("WOLFRAM_API_KEY")


def query_wolfram(query: str) -> str:
    if not WOLFRAM_API_KEY:
        raise RuntimeError("Wolfram API key is not set")
    prompt_escaped = urllib.parse.quote_plus(query)
    url = f"https://www.wolframalpha.com/api/v1/llm-api?appid={WOLFRAM_API_KEY}&input={prompt_escaped}"
    with httpx.Client() as client:
        response = client.get(url, timeout=10.0)
        response.raise_for_status()
        return response.text
