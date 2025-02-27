import logging

logger = logging.getLogger(__name__)

try:
    from newspaper import Article
except ImportError:
    raise ImportError("`newspaper3k` not installed. Please run `pip install newspaper3k lxml_html_clean`.")


class NewspaperTools:
    def __init__(self):
        pass

    def get_article_text(self, url: str) -> str:
        """Get the text of an article from a URL.

        Args:
            url (str): The URL of the article.

        Returns:
            str: The text of the article.
        """

        try:
            logger.debug(f"Reading news: {url}")
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except Exception as e:
            return f"Error getting article text from {url}: {e}"
