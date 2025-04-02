import os
import urllib.parse

import httpx

WOLFRAM_API_KEY = os.getenv("WOLFRAM_API_KEY")

async def query_wolfram_async(query: str) -> str:
    prompt_escaped = urllib.parse.quote_plus(query)
    url = f'https://www.wolframalpha.com/api/v1/llm-api?appid={WOLFRAM_API_KEY}&input={prompt_escaped}'
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=10.0)
        response.raise_for_status()
        return response.text
