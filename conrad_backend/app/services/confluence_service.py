from atlassian import Confluence
from bs4 import BeautifulSoup
from ..core.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfluenceService:
    def __init__(self):
        try:
            self.confluence = Confluence(
                url=settings.CONFLUENCE_URL,
                username=settings.CONFLUENCE_USERNAME,
                password=settings.CONFLUENCE_API_TOKEN,
                cloud=True
            )
            logger.info("Successfully connected to Confluence.")
        except Exception as e:
            logger.error(f"Failed to connect to Confluence: {e}")
            self.confluence = None

    def search_content(self, search_terms: dict, space_keys: list[str] = None, limit: int = 5) -> list[dict]:
        if not self.confluence:
            logger.error("Confluence client not initialized. Cannot perform search.")
            return []

        phrases = search_terms.get("phrases", [])
        keywords = search_terms.get("keywords", [])

        if not phrases and not keywords:
            logger.warning("No search phrases or keywords provided.")
            return []

        # --- CQL Construction ---
        cql_query_parts = []

        # 1. Term search (phrases first, then keywords)
        term_cql_parts = []
        if phrases:
            # Prioritize phrases, search for any of the phrases (OR logic for phrases)
            # Boost title matches for phrases
            phrase_queries = []
            for phrase in phrases:
                # Escape double quotes within the phrase itself for CQL
                escaped_phrase = phrase.replace('"', '\\"')
                phrase_queries.append(f'title ~ "{escaped_phrase}"^5 OR text ~ "{escaped_phrase}"')
            if phrase_queries:
                term_cql_parts.append(f"({' OR '.join(phrase_queries)})")

        if keywords:
            # If phrases were already added, keywords can be ANDed for refinement, or ORed for broader search.
            # For now, let's make keywords an alternative if phrases don't match, or a supplement.
            # Using OR for keywords to broaden the search if phrases are too specific or not present.
            # Boost title matches for keywords
            keyword_queries = []
            for keyword in keywords:
                escaped_keyword = keyword.replace('"', '\\"')
                keyword_queries.append(f'title ~ "{escaped_keyword}"^5 OR text ~ "{escaped_keyword}"')

            if keyword_queries:
                # If phrases are present, consider how to combine: AND for refinement, OR for more results.
                # Let's try OR for now to maximize chances of getting some results.
                # If term_cql_parts already has phrase conditions, this adds keyword conditions as an alternative set.
                if term_cql_parts: # If there were phrases
                     term_cql_parts.append(f"({' OR '.join(keyword_queries)})") # OR keywords as alternative
                else: # No phrases, just use keywords
                    term_cql_parts.append(f"({' OR '.join(keyword_queries)})")


        if not term_cql_parts:
            logger.info("No valid phrases or keywords to search for after processing.")
            return []

        # Join phrase and keyword parts. If both exist, they are currently ORed.
        # Example: (phrase_conditions) OR (keyword_conditions)
        cql_query_parts.append(f"({' OR '.join(term_cql_parts)})")


        # 2. Space key restriction
        if space_keys and len(space_keys) > 0:
            space_cql = " OR ".join([f'space = "{key.upper()}"' for key in space_keys])
            cql_query_parts.append(f"({space_cql})")

        # 3. Type restriction
        cql_query_parts.append('type = "page"')

        # Final CQL: Join all parts with AND
        cql = " AND ".join(cql_query_parts)

        logger.info(f"Searching Confluence with advanced CQL: {cql}")

        try:
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
        # ... (same as before)
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
    if not settings.CONFLUENCE_API_TOKEN or "YOUR_CONFLUENCE_API_TOKEN" in settings.CONFLUENCE_API_TOKEN:
        print("Confluence API token not configured. Please set it in .env")
    else:
        confluence_service = ConfluenceService()
        if confluence_service.confluence:
            print("Testing ConfluenceService with new search_content method...")

            test_search_terms_1 = {"phrases": ["alta de bridge"], "keywords": ["proceso", "configurar"]}
            target_spaces_1 = ["M2"]
            print(f"\nTest 1: Searching for terms: {test_search_terms_1}, spaces: {target_spaces_1}")
            results_1 = confluence_service.search_content(search_terms=test_search_terms_1, space_keys=target_spaces_1, limit=3)
            if results_1:
                print(f"Found {len(results_1)} results:")
                for res in results_1: print(f"  ID: {res['id']}, Title: {res['title']}")
            else:
                print("No results found for Test 1.")

            test_search_terms_2 = {"phrases": [], "keywords": ["información importante", "documentación"]}
            print(f"\nTest 2: Searching for terms: {test_search_terms_2} (no specific spaces)")
            results_2 = confluence_service.search_content(search_terms=test_search_terms_2, limit=3)
            if results_2:
                print(f"Found {len(results_2)} results:")
                for res in results_2: print(f"  ID: {res['id']}, Title: {res['title']}")
            else:
                print("No results found for Test 2.")

            test_search_terms_3 = {"phrases": ["configuración servidor", "base de datos"], "keywords": ["guía"]}
            target_spaces_3 = ["SGP", "IT"] # Replace with actual space keys
            print(f"\nTest 3: Searching for terms: {test_search_terms_3}, spaces: {target_spaces_3}")
            results_3 = confluence_service.search_content(search_terms=test_search_terms_3, space_keys=target_spaces_3, limit=3)
            if results_3:
                print(f"Found {len(results_3)} results:")
                for res in results_3: print(f"  ID: {res['id']}, Title: {res['title']}")
            else:
                print("No results found for Test 3.")
        else:
            print("Could not initialize ConfluenceService.")
