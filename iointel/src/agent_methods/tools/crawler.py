import asyncio
import uuid
import nest_asyncio
from firecrawl.firecrawl import FirecrawlApp
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar, Type
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
# Apply nest_asyncio to allow nested event loops (if needed)
nest_asyncio.apply()


class Crawler:
    """
    A wrapper class for the FirecrawlApp that provides methods for scraping,
    crawling, mapping, extracting, and watching crawl jobs.
    """

    def __init__(self, api_key: str, version: Optional[str] = None) -> None:
        """
        Initialize the Firecrawl app.

        Args:
            api_key (str): The API key for Firecrawl.
            version (Optional[str]): Optional API version.
        """
        if version:
            self.app: FirecrawlApp = FirecrawlApp(api_key=api_key, version=version)
        else:
            self.app: FirecrawlApp = FirecrawlApp(api_key=api_key)

    def scrape_url(
        self, url: str, options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Scrape a single URL.

        Args:
            url (str): The URL to scrape.
            options (Optional[Dict[str, Any]]): Optional scraping parameters.

        Returns:
            Dict[str, Any]: The scraping result.
        """
        if options is None:
            options = {}
        return self.app.scrape_url(url, options)

    def batch_scrape_urls(
        self, urls: List[str], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synchronously batch scrape multiple URLs.

        Args:
            urls (List[str]): List of URLs to scrape.
            params (Dict[str, Any]): Scraping parameters (e.g., formats).

        Returns:
            Dict[str, Any]: The batch scraping result.
        """
        return self.app.batch_scrape_urls(urls, params)

    async def async_batch_scrape_urls(
        self, urls: List[str], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Asynchronously batch scrape multiple URLs.

        Args:
            urls (List[str]): List of URLs to scrape.
            params (Dict[str, Any]): Scraping parameters.

        Returns:
            Dict[str, Any]: The asynchronous batch scraping result.
        """
        event_loop = asyncio.get_event_loop()
        return await event_loop.run_in_executor(None, self.app.async_batch_scrape_urls, urls, params)

    def crawl_url(
        self,
        url: str,
        crawl_params: Dict[str, Any],
        depth: int,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Synchronously crawl a website.

        Args:
            url (str): The URL to crawl.
            crawl_params (Dict[str, Any]): Crawl parameters (e.g., excludePaths).
            depth (int): Crawl depth.
            idempotency_key (Optional[str]): Optional idempotency key; if not provided one is generated.

        Returns:
            Dict[str, Any]: The crawl result.
        """
        if idempotency_key is None:
            idempotency_key = str(uuid.uuid4())
        return self.app.crawl_url(url, crawl_params, depth, idempotency_key)

    async def async_crawl_url(
        self,
        url: str,
        crawl_params: Dict[str, Any],
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Asynchronously crawl a website.

        Args:
            url (str): The URL to crawl.
            crawl_params (Dict[str, Any]): Crawl parameters.
            idempotency_key (Optional[str]): Optional idempotency key (default is empty string).

        Returns:
            Dict[str, Any]: The asynchronous crawl result.
        """
        if idempotency_key is None:
            idempotency_key = ""
        event_loop = asyncio.get_event_loop()
        return await event_loop.run_in_executor(None, self.app.async_crawl_url,
                                                url, crawl_params, idempotency_key)

    def scrape_urls(
        self, urls: List[str], extraction_schema: Type[T]
    ) -> List[Dict[str, Any]]:
        """
        Scrape a list of URLs concurrently using the crawler.scrape_url method.

        Each URL is scraped using the following parameters:
            - formats: ['json']
            - jsonOptions: containing a JSON schema derived from ExtractSchema

        Args:
            urls (List[str]): A list of URLs to scrape.

        Returns:
            List[Dict[str, Any]]: A list of the scrape results for the URLs that succeeded.
        """
        scrape_params: Dict[str, Any] = {
            "formats": ["json"],
            "jsonOptions": {"schema": extraction_schema.model_json_schema()},
        }

        results: List[Dict[str, Any]] = []

        # Use a ThreadPoolExecutor with a desired number of workers (e.g., 10)
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit a scrape task for each URL
            future_to_url = {
                executor.submit(self.scrape_url, url, scrape_params): url
                for url in urls
            }
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    print(f"Error scraping URL {url}: {exc}")
        return results

    def check_crawl_status(self, crawl_id: str) -> Dict[str, Any]:
        """
        Check the status of an ongoing crawl.

        Args:
            crawl_id (str): The crawl job's ID.

        Returns:
            Dict[str, Any]: The crawl status.
        """
        return self.app.check_crawl_status(crawl_id)

    def get_crawl_status(self, crawl_id: str) -> Dict[str, Any]:
        """
        Get the current status of a crawl.

        Args:
            crawl_id (str): The crawl job's ID.

        Returns:
            Dict[str, Any]: The current crawl status.
        """
        return self.app.get_crawl_status(crawl_id)

    def map_url(self, url: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map a website (e.g., for searching within the site).

        Args:
            url (str): The URL to map.
            options (Dict[str, Any]): Mapping options (e.g., search terms).

        Returns:
            Dict[str, Any]: The mapping result.
        """
        return self.app.map_url(url, options)

    def extract(
        self, urls: List[str], extraction_options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract information from one or more URLs using a defined schema.

        Args:
            urls (List[str]): List of URLs to extract data from.
            extraction_options (Dict[str, Any]): Extraction options including prompt and schema.

        Returns:
            Dict[str, Any]: The extraction result.
        """
        return self.app.extract(urls, extraction_options)

    async def crawl_and_watch(
        self,
        url: str,
        options: Dict[str, Any],
        event_listeners: Optional[Dict[str, Callable[[Dict[str, Any]], None]]] = None,
    ) -> Any:
        """
        Crawl a website and watch the crawl process via WebSockets.

        Args:
            url (str): The URL to crawl.
            options (Dict[str, Any]): Options for crawling (e.g., excludePaths, limit).
            event_listeners (Optional[Dict[str, Callable[[Dict[str, Any]], None]]]): A mapping of event names
                (e.g., "document", "error", "done") to callback functions.

        Returns:
            Any: The watcher object (after connection is established).
        """
        watcher = self.app.crawl_url_and_watch(url, options)
        if event_listeners:
            for event, handler in event_listeners.items():
                watcher.add_event_listener(event, handler)
        await watcher.connect()
        return watcher
