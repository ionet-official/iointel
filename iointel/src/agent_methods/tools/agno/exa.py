# from agno.tools.duckduckgo import DuckDuckGoTools as AgnoDuckDuckGoTools
from typing import List, Optional
from agno.tools.exa import ExaTools as AgnoExaTools
from .common import make_base, wrap_tool
from pydantic import Field


class Exa(make_base(AgnoExaTools)):
    search: bool = Field(default=True, frozen=True)
    get_contents_: bool = Field(default=True, frozen=True)
    find_similar_: bool = Field(default=True, frozen=True)
    answer: bool = Field(default=True, frozen=True)
    text_length_limit: int = Field(default=1000, frozen=True)
    highlights: bool = Field(default=True, frozen=True)
    summary: bool = Field(default=False, frozen=True)
    api_key: Optional[str] = Field(default=None, frozen=True)
    num_results: Optional[int] = Field(default=None, frozen=True)
    livecrawl: str = Field(default="always", frozen=True)
    start_crawl_date: Optional[str] = Field(default=None, frozen=True)
    end_crawl_date: Optional[str] = Field(default=None, frozen=True)
    start_published_date: Optional[str] = Field(default=None, frozen=True)
    end_published_date: Optional[str] = Field(default=None, frozen=True)
    use_autoprompt: Optional[bool] = Field(default=None, frozen=True)
    type: Optional[str] = Field(default=None, frozen=True)
    category: Optional[str] = Field(default=None, frozen=True)
    include_domains: Optional[List[str]] = Field(default=None, frozen=True)
    exclude_domains: Optional[List[str]] = Field(default=None, frozen=True)
    show_results: bool = Field(default=False, frozen=True)
    model: Optional[str] = Field(default=None, frozen=True)
    timeout: int = Field(default=30, frozen=True)

    def _get_tool(self):
        return self.Inner(
            search=self.search,
            get_contents=self.get_contents_,
            find_similar=self.find_similar_,
            answer=self.answer,
            text=True,
            text_length_limit=self.text_length_limit,
            highlights=self.highlights,
            summary=self.summary,
            api_key=self.api_key,
            num_results=self.num_results,
            livecrawl=self.livecrawl,
            start_crawl_date=self.start_crawl_date,
            end_crawl_date=self.end_crawl_date,
            start_published_date=self.start_published_date,
            end_published_date=self.end_published_date,
            use_autoprompt=self.use_autoprompt,
            type=self.type,
            category=self.category,
            include_domains=self.include_domains,
            exclude_domains=self.exclude_domains,
            show_results=self.show_results,
            model=self.model,
            timeout=self.timeout,
        )

    @wrap_tool("agno_exa_search_exa", AgnoExaTools.search_exa)
    def search_exa(
        self, query: str, num_results: int = 5, category: Optional[str] = None
    ) -> str:
        return self._tool.search_exa(query, num_results, category)

    @wrap_tool("agno_exa_get_contents", AgnoExaTools.get_contents)
    def get_contents(self, urls: list[str]) -> str:
        return self._tool.get_contents(urls)

    @wrap_tool("agno_exa_find_similar", AgnoExaTools.find_similar)
    def find_similar(self, url: str, num_results: int = 5) -> str:
        return self._tool.find_similar(url, num_results)

    @wrap_tool("agno_exa_exa_answer", AgnoExaTools.exa_answer)
    def exa_answer(self, query: str, text: bool = False) -> str:
        return self._tool.exa_answer(query, text)
