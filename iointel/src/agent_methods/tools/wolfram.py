import wolframalpha
from ...utilities.decorators import register_tool


class Wolfram:
    def __init__(self, api_key: str = None):
        if api_key is None:
            raise ValueError("please set the api_key")
        self.client = wolframalpha.Client(api_key)

    @register_tool(name="wolfram_query")
    def query(self, query):
        res = self.client.query(query)
        return next(res.results).text

    @register_tool(name="wolfram_query_async")
    async def query_async(self, query):
        res = await self.client.aquery(query)
        return next(res.results).text
