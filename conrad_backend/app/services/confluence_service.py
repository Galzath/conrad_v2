import logging
from atlassian import Confluence
from bs4 import BeautifulSoup
import faiss
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from ..core.config import settings

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

        # Load semantic search components
        self.faiss_index = None
        self.chunk_metadata = []
        self.embedding_model_for_query = None

        try:
            if os.path.exists(settings.FAISS_INDEX_PATH) and os.path.exists(settings.CHUNKS_DATA_PATH):
                logger.info(f"Loading FAISS index from {settings.FAISS_INDEX_PATH}")
                self.faiss_index = faiss.read_index(settings.FAISS_INDEX_PATH)
                logger.info(f"FAISS index loaded. Index has {self.faiss_index.ntotal} vectors.")

                logger.info(f"Loading chunk metadata from {settings.CHUNKS_DATA_PATH}")
                with open(settings.CHUNKS_DATA_PATH, 'r', encoding='utf-8') as f:
                    self.chunk_metadata = json.load(f)
                logger.info(f"Chunk metadata loaded. {len(self.chunk_metadata)} chunks.")

                logger.info(f"Loading embedding model for queries: {settings.EMBEDDING_MODEL_NAME}")
                self.embedding_model_for_query = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
                logger.info("Embedding model for queries loaded successfully.")
            else:
                logger.warning("FAISS index or chunk metadata file not found. Semantic search capabilities will be limited/disabled.")
        except Exception as e:
            logger.error(f"Error loading FAISS index, chunk metadata, or embedding model: {e}", exc_info=True)
            self.faiss_index = None
            self.chunk_metadata = []
            self.embedding_model_for_query = None

    def _execute_cql_query(self, cql: str, limit: int) -> list[dict]:
        if not self.confluence:
            logger.error("Confluence client not initialized. Cannot execute CQL query.")
            return []
        try:
            logger.info(f"Executing Confluence CQL: {cql}")
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
                logger.info("No results found or unexpected response format from CQL query.")
                return []
        except Exception as e:
            logger.error(f"Error during Confluence search with CQL '{cql}': {e}")
            return []

    def search_content(self, search_terms: dict, space_keys: list[str] = None, limit: int = 5) -> list[dict]:
        if not self.confluence:
            logger.error("Confluence client not initialized. Cannot perform search.")
            return []

        original_phrases = search_terms.get("phrases", [])
        original_keywords = search_terms.get("keywords", [])

        if not original_phrases and not original_keywords:
            logger.warning("No search phrases or keywords provided.")
            return []

        # Common CQL parts
        space_cql_part = ""
        if space_keys and len(space_keys) > 0:
            space_cql = " OR ".join([f'space = "{key.upper()}"' for key in space_keys])
            space_cql_part = f"({space_cql})"

        type_cql_part = 'type = "page"'

        # Attempt 1: Phrase-based search (if phrases are provided)
        if original_phrases:
            phrase_cql_conditions = []
            for phrase in original_phrases:
                escaped_phrase = phrase.replace('"', '\\"') # Corrected escaping
                phrase_cql_conditions.append(f'(title ~ "{escaped_phrase}" OR text ~ "{escaped_phrase}")')

            if phrase_cql_conditions:
                main_condition = f"({' AND '.join(phrase_cql_conditions)})"
                cql_parts = [main_condition, type_cql_part]
                if space_cql_part:
                    cql_parts.insert(1, space_cql_part)
                cql_attempt1 = " AND ".join(cql_parts)

                results = self._execute_cql_query(cql_attempt1, limit)
                if results:
                    logger.info(f"Phrase search successful with {len(results)} results.")
                    return results
                else:
                    logger.info("Phrase search yielded no results. Attempting fallback keyword search.")
                    # Proceed to fallback keyword search logic below

        # Fallback or Primary Keyword Search:
        # This section is reached if:
        # 1. original_phrases were provided but yielded no results (fallback).
        # 2. No original_phrases were provided, so keywords are the primary search method.

        effective_keywords = set()
        # If falling back from phrases, derive keywords from them AND original keywords
        if original_phrases:
            for phrase in original_phrases:
                effective_keywords.update(phrase.lower().split())
            effective_keywords.update(k.lower() for k in original_keywords) # Add original keywords
        else: # Primary keyword search (no phrases given initially)
            effective_keywords.update(k.lower() for k in original_keywords)

        if not effective_keywords:
            logger.info("No effective keywords to search for after phrase fallback or initial check.")
            return []

        keyword_cql_conditions = []
        for keyword in effective_keywords:
            escaped_keyword = keyword.replace('"', '\\"') # Corrected escaping
            keyword_cql_conditions.append(f'(title ~ "{escaped_keyword}" OR text ~ "{escaped_keyword}")')

        if not keyword_cql_conditions:
            logger.info("No valid keyword CQL conditions generated.")
            return []

        # If original_phrases were present (meaning this is a fallback), keywords should be ORed for a broader search.
        # If no original_phrases, then this is a primary keyword search, also use OR.
        if original_phrases: # Fallback from phrases: OR keywords for broader search
            main_condition = f"({' OR '.join(keyword_cql_conditions)})"
        else: # Primary keyword search: OR keywords
            main_condition = f"({' OR '.join(keyword_cql_conditions)})"

        cql_parts = [main_condition, type_cql_part]
        if space_cql_part:
            cql_parts.insert(1, space_cql_part)
        cql_attempt2 = " AND ".join(cql_parts)

        results = self._execute_cql_query(cql_attempt2, limit)
        if results:
            logger.info(f"Keyword search successful with {len(results)} results.")
        else:
            logger.info("Keyword search also yielded no results.")
        return results

    def get_page_content_by_id(self, page_id: str) -> str:
        if not self.confluence:
            logger.error("Confluence client not initialized. Cannot get page content.")
            return ""
        try:
            page = self.confluence.get_page_by_id(page_id, expand="body.storage")
            if page and "body" in page and "storage" in page["body"] and "value" in page["body"]["storage"]:
                soup = BeautifulSoup(page["body"]["storage"]["value"], "html.parser")
                # Using separator="\n" for BeautifulSoup's get_text
                return soup.get_text(separator="\n", strip=True)
            logger.warning(f"Page {page_id} found but no content in expected format.")
            return ""
        except Exception as e:
            logger.error(f"Error fetching/parsing page {page_id}: {e}")
        return ""

    def get_available_spaces(self, limit: int = 200) -> list[dict]:
        if not self.confluence:
            logger.error("Confluence client not initialized. Cannot get available spaces.")
            return []
        try:
            logger.info(f"Fetching available Confluence spaces (limit: {limit}).")
            # Using space_type='global' to fetch standard spaces.
            # The limit is set to a high number, but pagination might be needed for instances with more spaces.
            response = self.confluence.get_all_spaces(space_type='global', start=0, limit=limit)

            spaces = []
            if response and 'results' in response:
                for space in response['results']:
                    spaces.append({
                        "id": space['key'],  # Space key is usually the ID needed for CQL
                        "text": space['name'] # Space name for display
                    })
                logger.info(f"Successfully fetched {len(spaces)} global spaces.")
            else:
                logger.warning("No spaces found or unexpected response format from get_all_spaces.")
            return spaces
        except Exception as e:
            logger.error(f"Error fetching available Confluence spaces: {e}", exc_info=True)
            return []

    def semantic_search_chunks(self, query_text: str, top_k: int = 5) -> list[dict]:
        if not self.faiss_index or not self.chunk_metadata or not self.embedding_model_for_query or self.faiss_index.ntotal == 0:
            logger.info("Semantic search components not available or index is empty. Skipping semantic search.")
            return []

        try:
            logger.info(f"Performing semantic search for query: '{query_text[:50]}...' with top_k={top_k}")
            query_embedding = self.embedding_model_for_query.encode([query_text], show_progress_bar=False)

            # FAISS search returns distances (D) and indices (I)
            # Ensure query_embedding is 2D for search
            if query_embedding.ndim == 1:
                query_embedding = np.expand_dims(query_embedding, axis=0)

            distances, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)

            results = []
            if indices.size == 0 or indices[0][0] == -1: # Check if any results found
                 logger.info("No relevant chunks found by semantic search.")
                 return []

            for i in range(len(indices[0])):
                idx = indices[0][i]
                if idx < 0 or idx >= len(self.chunk_metadata): # Invalid index
                    logger.warning(f"Semantic search returned invalid index {idx}. Skipping.")
                    continue

                chunk_meta = self.chunk_metadata[idx]
                # Score: FAISS L2 distance is lower is better. Convert to similarity (e.g., 1 / (1 + distance))
                # Or simply pass distance and let consumer decide. For now, pass distance.
                # A common practice is to use dot product for similarity if embeddings are normalized,
                # but IndexFlatL2 uses L2 distance.
                score = float(distances[0][i])

                results.append({
                    "chunk_id": idx, # The index in the chunk_metadata list
                    "page_id": chunk_meta.get("page_id"),
                    "title": chunk_meta.get("page_title"), # Ensure this key exists in your chunk_metadata
                    "url": chunk_meta.get("url"),
                    "text": chunk_meta.get("text"),
                    "context_hierarchy": chunk_meta.get("context_hierarchy"), # Pass along hierarchy
                    "score": score, # Raw L2 distance, lower is better
                    "search_method": "semantic"
                })
            logger.info(f"Semantic search found {len(results)} chunks.")
            return results
        except Exception as e:
            logger.error(f"Error during semantic search: {e}", exc_info=True)
            return []

# Example Usage (for testing purposes)
if __name__ == "__main__":
    if not settings.CONFLUENCE_API_TOKEN or "YOUR_CONFLUENCE_API_TOKEN" in settings.CONFLUENCE_API_TOKEN:
        print("Confluence API token not configured in .env")
    else:
        confluence_service = ConfluenceService()
        if confluence_service.confluence:
            print("Testing ConfluenceService with new fallback CQL logic...")

            # Test 1: Phrase search expected to work (or not, to test fallback)
            terms1 = {"phrases": ["documentacion tecnica de desarollo"], "keywords": ["extra"]}
            spaces1 = []
            print(f"\nTest 1 (Phrase with Fallback): Terms: {terms1}, Spaces: {spaces1}")
            results1 = confluence_service.search_content(search_terms=terms1, space_keys=spaces1, limit=3)
            if results1: print(f"Found {len(results1)} results for Test 1.")
            else: print("No results for Test 1.")
            for res in results1: print(f"  - {res['title']}")


            # Test 2: Only keywords (ORed - existing behavior for this case)
            terms2 = {"phrases": [], "keywords": ["salesforce", "manual"]}
            print(f"\nTest 2 (Keywords ORed): Terms: {terms2}")
            results2 = confluence_service.search_content(search_terms=terms2, limit=3)
            if results2: print(f"Found {len(results2)} results for Test 2.")
            else: print("No results for Test 2.")
            for res in results2: print(f"  - {res['title']}")

            # Test 3: Phrase that might exist
            terms3 = {"phrases": ["alta de bridge"], "keywords": []}
            spaces3 = ["M2"]
            print(f"\nTest 3 (Specific Phrase): Terms: {terms3}, Spaces: {spaces3}")
            results3 = confluence_service.search_content(search_terms=terms3, space_keys=spaces3, limit=3)
            if results3: print(f"Found {len(results3)} results for Test 3.")
            else: print("No results for Test 3.")
            for res in results3: print(f"  - {res['title']}")

            # Test 4: No phrases, few keywords (ORed), with space
            terms4 = {"phrases": [], "keywords": ["edita", "proceso"]}
            spaces4 = ["M2"]
            print(f"\nTest 4 (Keywords ORed with Space): Terms: {terms4}, Spaces: {spaces4}")
            results4 = confluence_service.search_content(search_terms=terms4, space_keys=spaces4, limit=3)
            if results4: print(f"Found {len(results4)} results for Test 4.")
            else: print("No results for Test 4.")
            for res in results4: print(f"  - {res['title']}")

            # Test 5: A phrase that likely won't exist, to test fallback ANDing of its components
            terms5 = {"phrases": ["nonexistent specific technical phrase"], "keywords": ["important"]}
            print(f"\nTest 5 (Non-existent Phrase Fallback): Terms: {terms5}")
            results5 = confluence_service.search_content(search_terms=terms5, limit=3)
            if results5: print(f"Found {len(results5)} results for Test 5.")
            else: print("No results for Test 5.")
            for res in results5: print(f"  - {res['title']}")

        else:
            print("Could not initialize ConfluenceService.")
