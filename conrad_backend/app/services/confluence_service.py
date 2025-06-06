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

    def search_content(self, query: str, space_keys: list[str] = None, limit: int = 5) -> list[dict]:
        if not self.confluence:
            logger.error("Confluence client not initialized. Cannot perform search.")
            return []

        # Sanitize the query to prevent CQL injection issues if not handled by the library
        # For text fields, quotes are important. If query contains quotes, they might need escaping
        # or rely on the library to handle it. Let's assume basic query for now.
        # A simple approach for quotes in query: if query contains ", escape them or remove.
        # However, text ~ "..." itself uses quotes.
        # The atlassian library's cql method should handle proper URL encoding of the CQL.

        # Ensure the query is wrapped in double quotes for exact phrase matching in text search
        # If query already has them, this might be redundant but often harmless.
        # Let's be careful not to double-wrap if user provides quotes.
        # A simple heuristic: if not (query.startswith('"') and query.endswith('"')):
        #    processed_query = f'"{query}"'
        # else:
        #    processed_query = query
        # For now, let's assume the input 'query' is the core phrase.

        processed_query = query.replace('"', '\\"') # Basic escaping of double quotes within the query term

        cql_parts = []
        if space_keys and len(space_keys) > 0:
            space_cql = " OR ".join([f'space = "{key.upper()}"' for key in space_keys])
            cql_parts.append(f"({space_cql})")

        # Using text search for better keyword relevance
        cql_parts.append(f'text ~ "{processed_query}"')
        cql_parts.append('type = "page"')

        cql = " AND ".join(cql_parts)

        logger.info(f"Searching Confluence with CQL: {cql}")

        try:
            # The `atlassian-python-api` handles `expand` as a list of strings.
            results = self.confluence.cql(cql, limit=limit, expand=None)

            if results and 'results' in results:
                logger.info(f"Found {len(results['results'])} results from Confluence.")
                return [
                    {
                        "id": result["content"]["id"],
                        "title": result["content"]["title"],
                        "url": f"{settings.CONFLUENCE_URL.rstrip('/')}/wiki{result['content']['_links']['webui']}"
                    }
                    for result in results['results']
                ]
            else:
                logger.info("No results found or unexpected response format from Confluence.")
                return []
        except Exception as e:
            logger.error(f"Error during Confluence search with CQL '{cql}': {e}")
            return []

    def get_page_content_by_id(self, page_id: str) -> str:
        if not self.confluence:
            logger.error("Confluence client not initialized. Cannot get page content.")
            return ""

        logger.info(f"Fetching content for page ID: {page_id}")
        try:
            page = self.confluence.get_page_by_id(page_id, expand="body.storage")
            if page and "body" in page and "storage" in page["body"] and "value" in page["body"]["storage"]:
                html_content = page["body"]["storage"]["value"]
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
    # This part requires .env to be correctly set up
    # from conrad_backend.app.core.config import settings

    if not settings.CONFLUENCE_API_TOKEN or "YOUR_CONFLUENCE_API_TOKEN" in settings.CONfluence_API_TOKEN:
        print("Confluence API token not configured. Please set it in .env")
    else:
        confluence_service = ConfluenceService()
        if confluence_service.confluence:
            # Test case 1: General query without space keys
            search_query_general = "test content"
            print(f"Searching for (general): '{search_query_general}'")
            search_results_general = confluence_service.search_content(search_query_general, limit=3)
            if search_results_general:
                print(f"Found {len(search_results_general)} general results:")
                for res in search_results_general:
                    print(f"  ID: {res['id']}, Title: {res['title']}, URL: {res['url']}")
            else:
                print("No general search results found.")
            print("----")

            # Test case 2: Query with specific space keys
            search_query_specific = "bridge process"
            # Replace "M2" and "SF" with actual space keys relevant to your Confluence for testing
            target_spaces = ["M2", "SF"]
            print(f"Searching for: '{search_query_specific}' in spaces: {target_spaces}")
            search_results_specific = confluence_service.search_content(search_query_specific, space_keys=target_spaces, limit=3)
            if search_results_specific:
                print(f"Found {len(search_results_specific)} specific results:")
                for res in search_results_specific:
                    print(f"  ID: {res['id']}, Title: {res['title']}, URL: {res['url']}")
            else:
                print("No specific search results found.")
            print("----")

            # Test case 3: Query that might contain quotes
            search_query_quotes = 'proceso de "alta"'
            print(f"Searching for (with quotes): '{search_query_quotes}'")
            search_results_quotes = confluence_service.search_content(search_query_quotes, limit=3)
            if search_results_quotes:
                print(f"Found {len(search_results_quotes)} results for query with quotes:")
                for res in search_results_quotes:
                    print(f"  ID: {res['id']}, Title: {res['title']}, URL: {res['url']}")
            else:
                print("No results found for query with quotes.")
            print("----")

        else:
            print("Could not initialize ConfluenceService.")
