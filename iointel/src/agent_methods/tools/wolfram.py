import os
import urllib.parse

import httpx
from iointel.src.utilities.decorators import register_tool
from pydantic import BaseModel

WOLFRAM_API_KEY = os.getenv("WOLFRAM_API_KEY")

class Wolfram(BaseModel):
    api_key: str

    def __init__(self, api_key: str|None = None):
        if not (api_key := api_key or WOLFRAM_API_KEY):
            raise RuntimeError("Wolfram API key is not set")
        super().__init__(api_key=api_key)

    @register_tool
    def query(self, query: str) -> str:
        prompt_escaped = urllib.parse.quote_plus(query)
        url = f"https://www.wolframalpha.com/api/v1/llm-api?appid={WOLFRAM_API_KEY}&input={prompt_escaped}"
        with httpx.Client() as client:
            response = client.get(url, timeout=10.0)
            response.raise_for_status()
            return response.text
