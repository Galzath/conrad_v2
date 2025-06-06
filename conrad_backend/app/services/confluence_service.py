from atlassian import Confluence
from bs4 import BeautifulSoup # For basic HTML to text conversion
from ..core.config import settings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfluenceService:
    def __init__(self):
        try:
            self.confluence = Confluence(
                url=settings.CONFLUENCE_URL,
                username=settings.CONFLUENCE_USERNAME,
                password=settings.CONFLUENCE_API_TOKEN, # API token is used as password
                cloud=True # Set to True if using Confluence Cloud
            )
            logger.info("Successfully connected to Confluence.")
        except Exception as e:
            logger.error(f"Failed to connect to Confluence: {e}")
            self.confluence = None

    def search_content(self, query: str, limit: int = 5) -> list[dict]:
        if not self.confluence:
            logger.error("Confluence client not initialized. Cannot perform search.")
            return []

        # Basic CQL query, can be expanded
        # Example: f'text ~ "{query}" AND type="page"'
        # For simplicity, we'll use a more direct query if possible or a broader one.
        # The atlassian-python-api's cql search can be complex with escaping.
        # We'll use content search which is more straightforward for keywords.

        cql = f'siteSearch ~ "{query}" AND type="page"' # A broader search
        # Alternatively, for more specific text matching:
        # cql = f'text ~ "{query}" AND type="page"'

        logger.info(f"Searching Confluence with CQL: {cql}")

        try:
            results = self.confluence.cql(cql, limit=limit, expand=None) # expand=None for now to keep it simple

            if results and 'results' in results:
                logger.info(f"Found {len(results['results'])} results from Confluence.")
                # Return a list of simplified result dicts
                return [
                    {
                        "id": result["content"]["id"],
                        "title": result["content"]["title"],
                        "url": f"{settings.CONFLUENCE_URL.rstrip('/')}/wiki{result['content']['_links']['webui']}"
                    }
                    for result in results['results']
                ]
            else:
                logger.info("No results found or unexpected response format.")
                return []
        except Exception as e:
            logger.error(f"Error during Confluence search: {e}")
            return []

    def get_page_content_by_id(self, page_id: str) -> str:
        if not self.confluence:
            logger.error("Confluence client not initialized. Cannot get page content.")
            return ""

        logger.info(f"Fetching content for page ID: {page_id}")
        try:
            # Expand body.storage to get the raw XML/HTML content
            page = self.confluence.get_page_by_id(page_id, expand="body.storage")
            if page and "body" in page and "storage" in page["body"] and "value" in page["body"]["storage"]:
                html_content = page["body"]["storage"]["value"]
                # Basic text extraction using BeautifulSoup
                soup = BeautifulSoup(html_content, "html.parser")
                text_content = soup.get_text(separator="\n", strip=True)
                logger.info(f"Successfully extracted text content for page ID: {page_id}")
                return text_content
            else:
                logger.warning(f"Could not find content in body.storage for page ID: {page_id}")
                return ""
        except Exception as e:
            logger.error(f"Error fetching or parsing page content for ID {page_id}: {e}")
            return ""

# Example Usage (for testing purposes, can be removed or commented out)
if __name__ == "__main__":
    # This part requires .env to be correctly set up in the parent directory
    # or environment variables to be available.
    # You might need to adjust path for config if running this file directly.
    # from conrad_backend.app.core.config import settings # Adjust import if necessary

    if not settings.CONFLUENCE_API_TOKEN or "YOUR_CONFLUENCE_API_TOKEN" in settings.CONFLUENCE_API_TOKEN:
        print("Confluence API token not configured. Please set it in .env")
    else:
        confluence_service = ConfluenceService()
        if confluence_service.confluence:
            search_query = "test" # Replace with a relevant search query for your Confluence
            print(f"Searching for: {search_query}")
            search_results = confluence_service.search_content(search_query, limit=3)

            if search_results:
                print(f"Found {len(search_results)} results:")
                for res in search_results:
                    print(f"  ID: {res['id']}, Title: {res['title']}, URL: {res['url']}")
                    if res.get("id"):
                        print(f"  Fetching content for page ID: {res['id']}...")
                        content = confluence_service.get_page_content_by_id(res["id"])
                        # print(f"  Content (first 200 chars): {content[:200]}...")
                        if content:
                             print(f"  Successfully retrieved content for {res['title']}.")
                        else:
                             print(f"  Could not retrieve content for {res['title']}.")

            else:
                print("No search results found.")
        else:
            print("Could not initialize ConfluenceService.")
