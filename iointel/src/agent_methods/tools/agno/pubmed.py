from typing import Any, Optional, List, Dict
from agno.tools.pubmed import PubmedTools as AgnoPubmedTools
from .common import make_base, wrap_tool
from xml.etree.ElementTree import Element


class Pubmed(make_base(AgnoPubmedTools)):
    def _get_tool(self):
        return self.Inner(
            email=self.email_,
            max_results=self.max_results_,
            results_expanded=self.results_expanded_,
        )

    @wrap_tool("agno__pubmed__fetch_pubmed_ids", AgnoPubmedTools.fetch_pubmed_ids)
    def fetch_pubmed_ids(self, query: str, max_results: int, email: str) -> List[str]:
        return self.fetch_pubmed_ids(self, query, max_results, email)

    @wrap_tool("agno__pubmed__fetch_details", AgnoPubmedTools.fetch_details)
    def fetch_details(self, pubmed_ids: List[str]) -> Element:
        return self.fetch_details(self, pubmed_ids)

    @wrap_tool("agno__pubmed__parse_details", AgnoPubmedTools.parse_details)
    def parse_details(self, xml_root: Element) -> List[Dict[str, Any]]:
        return self.parse_details(self, xml_root)

    @wrap_tool("agno__pubmed__search_pubmed", AgnoPubmedTools.search_pubmed)
    def search_pubmed(self, query: str, max_results: Optional[int] = 10) -> str:
        return self.search_pubmed(self, query, max_results)
