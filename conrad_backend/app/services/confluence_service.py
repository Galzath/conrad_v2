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

        phrases = search_terms.get("phrases", []) # Expected to be 0-2 distinct phrases
        keywords = search_terms.get("keywords", []) # Expected to be 0-5 distinct keywords

        if not phrases and not keywords:
            logger.warning("No search phrases or keywords provided.")
            return []

        cql_main_conditions = []

        if phrases:
            # If phrases are present, use them as the primary search condition.
            # AND them together if multiple phrases for high precision.
            phrase_cql_parts = []
            for phrase in phrases:
                escaped_phrase = phrase.replace('"', '\\"')
                # Search in title OR text for the phrase
                phrase_cql_parts.append(f'(title ~ "{escaped_phrase}" OR text ~ "{escaped_phrase}")')
            if phrase_cql_parts:
                cql_main_conditions.append(f"({' AND '.join(phrase_cql_parts)})")

        # If no phrases were used for the main condition, or as an alternative broaden search (currently not alternative, but additive if also keywords)
        # For now, if phrases are present, we rely on them. If not, we use keywords.
        # If we want to combine, the logic would be: (PHRASES) AND (KEYWORDS_ORed) or (PHRASES) OR (KEYWORDS_ORed)
        # Let's try: if phrases, use only phrases. If no phrases, use keywords.
        # This simplifies the query significantly.

        if not cql_main_conditions and keywords: # Only use keywords if no phrases were used
            keyword_cql_parts = []
            for keyword in keywords:
                escaped_keyword = keyword.replace('"', '\\"')
                # Search in title OR text for the keyword
                keyword_cql_parts.append(f'(title ~ "{escaped_keyword}" OR text ~ "{escaped_keyword}")')
            if keyword_cql_parts:
                # OR keywords together to find documents containing any of them
                cql_main_conditions.append(f"({' OR '.join(keyword_cql_parts)})")

        if not cql_main_conditions:
             logger.info("No valid search conditions from phrases or keywords.")
             return [] # No terms to search for

        # Now build the final query
        cql_parts = [f"({cql_main_conditions[0]})"] # Main phrase/keyword condition

        if space_keys and len(space_keys) > 0:
            space_cql = " OR ".join([f'space = "{key.upper()}"' for key in space_keys])
            cql_parts.append(f"({space_cql})")

        cql_parts.append('type = "page"')
        cql = " AND ".join(cql_parts)

        logger.info(f"Simplified CQL for Confluence: {cql}")

        try:
            results = self.confluence.cql(cql, limit=limit, expand=None)
            if results and 'results' in results:
                logger.info(f"Found {len(results['results'])} results.")
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
            logger.error(f"Error during Confluence search with CQL '{cql}': {e}")
            # If it's a parse error, this log will show the problematic CQL.
            return []

    def get_page_content_by_id(self, page_id: str) -> str:
        # ... (same as before)
        if not self.confluence: return ""
        try:
            page = self.confluence.get_page_by_id(page_id, expand="body.storage")
            if page and "body" in page and "storage" in page["body"] and "value" in page["body"]["storage"]:
                soup = BeautifulSoup(page["body"]["storage"]["value"], "html.parser")
                return soup.get_text(separator="\n", strip=True)
        except Exception as e:
            logger.error(f"Error fetching/parsing page {page_id}: {e}")
        return ""

# Example Usage (for testing purposes)
if __name__ == "__main__":
    if not settings.CONFLUENCE_API_TOKEN or "YOUR_CONFLUENCE_API_TOKEN" in settings.CONFLUENCE_API_TOKEN:
        print("Confluence API token not configured in .env")
    else:
        confluence_service = ConfluenceService()
        if confluence_service.confluence:
            print("Testing ConfluenceService with simplified CQL logic...")

            # Test 1: Prioritize phrases (ANDed)
            terms1 = {"phrases": ["alta de bridge", "proceso M2"], "keywords": ["configurar"]}
            spaces1 = ["M2"]
            print(f"\nTest 1: Terms: {terms1}, Spaces: {spaces1}")
            results1 = confluence_service.search_content(search_terms=terms1, space_keys=spaces1)
            if results1: print(f"Found {len(results1)} results for Test 1.")
            else: print("No results for Test 1.")

            # Test 2: Only keywords (ORed)
            terms2 = {"phrases": [], "keywords": ["salesforce", "beneficios", "manual"]}
            print(f"\nTest 2: Terms: {terms2} (no spaces)")
            results2 = confluence_service.search_content(search_terms=terms2)
            if results2: print(f"Found {len(results2)} results for Test 2.")
            else: print("No results for Test 2.")

            # Test 3: Single phrase, no keywords
            terms3 = {"phrases": ["como se edita un proceso"], "keywords": []}
            print(f"\nTest 3: Terms: {terms3} (problematic query from user)")
            results3 = confluence_service.search_content(search_terms=terms3)
            if results3: print(f"Found {len(results3)} results for Test 3.")
            else: print("No results for Test 3.")

            # Test 4: No phrases, few keywords, with space
            terms4 = {"phrases": [], "keywords": ["edita", "proceso"]}
            spaces4 = ["M2"] # Assuming M2 is a relevant space
            print(f"\nTest 4: Terms: {terms4}, Spaces: {spaces4}")
            results4 = confluence_service.search_content(search_terms=terms4, space_keys=spaces4)
            if results4: print(f"Found {len(results4)} results for Test 4.")
            else: print("No results for Test 4.")
        else:
            print("Could not initialize ConfluenceService.")
