import logging
import json
import faiss
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import nltk # Added NLTK
from nltk.tokenize import sent_tokenize # Added sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity # Added cosine_similarity

from ..core.config import settings
from .confluence_service import ConfluenceService

logger = logging.getLogger(__name__)

class IndexingService:
    def __init__(self):
        try:
            self.confluence_service = ConfluenceService()
            if not self.confluence_service.confluence:
                raise ConnectionError("Failed to initialize ConfluenceService for IndexingService.")
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
            logger.info("Embedding model loaded successfully.")

            # NLTK 'punkt' tokenizer download
            try:
                nltk.data.find('tokenizers/punkt')
            except nltk.downloader.DownloadError:
                logger.info("NLTK 'punkt' tokenizer not found. Downloading...")
                nltk.download('punkt', quiet=True)
                logger.info("'punkt' tokenizer downloaded.")

        except Exception as e:
            logger.error(f"Error during IndexingService initialization: {e}", exc_info=True)
            self.confluence_service = None
            self.embedding_model = None

    def _get_text_from_element(self, element) -> str:
        texts = []
        if element.name == 'table':
            for row_idx, row in enumerate(element.find_all('tr', recursive=False)):
                cell_texts = []
                for cell in row.find_all(['td', 'th'], recursive=False):
                    cell_text = cell.get_text(separator=' ', strip=True)
                    cell_texts.append(cell_text)
                texts.append(" | ".join(cell_texts))
            return "\n".join(texts)

        # Handle Confluence code block macro <ac:structured-macro ac:name="code">
        if element.name == 'ac:structured-macro' and element.get('ac:name') == 'code':
            code_body = element.find('ac:plain-text-body', recursive=False)
            if code_body:
                return code_body.get_text(strip=False) # Preserve internal newlines
            # Fallback if structure is unexpected
            return element.get_text(separator='\n', strip=True)

        if element.name == 'pre': # Standard HTML code block
            return element.get_text(strip=False) # Preserve internal newlines

        # Handle lists
        if element.name == 'ul':
            for i, li in enumerate(element.find_all('li', recursive=False)):
                # Recursive call to handle complex list items that might contain other elements
                texts.append(f"- {self._get_text_from_element(li)}")
            return "\n".join(texts)
        if element.name == 'ol':
            for i, li in enumerate(element.find_all('li', recursive=False)):
                # Recursive call
                texts.append(f"{i+1}. {self._get_text_from_element(li)}")
            return "\n".join(texts)

        # General paragraph and other block elements, or text within list items
        if element.name in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div']: # Added div
            # For these, we want to concatenate text from direct children strings or simple inline tags
            # but avoid re-processing block elements like sub-lists or tables here.
            content_texts = []
            for child_node in element.children:
                if isinstance(child_node, str): # NavigableString
                    content_texts.append(child_node.strip())
                elif child_node.name and child_node.name not in ['ul', 'ol', 'table', 'pre', 'ac:structured-macro', 'div']:
                    # Get text from simple inline tags like strong, em, span, a, etc.
                    # Exclude block tags that are handled by their own rules or main loop.
                    content_texts.append(child_node.get_text(separator=' ', strip=True))
            return " ".join(filter(None, content_texts))

        if element.name == 'li': # Specifically for list items, process their direct content
            direct_text = []
            for child_node in element.children:
                if isinstance(child_node, str):
                    direct_text.append(child_node.strip())
                elif child_node.name and child_node.name not in ['ul', 'ol', 'table', 'pre', 'ac:structured-macro']:
                    # Get text from simple inline tags like strong, em, span, a etc.
                    direct_text.append(child_node.get_text(separator=' ', strip=True))
            return " ".join(filter(None, direct_text))


        # Fallback for other elements or direct text extraction if element is NavigableString
        if isinstance(element, str): # Should be NavigableString
            return element.strip()

        # Default get_text for unhandled tags if they are passed here directly.
        return element.get_text(separator=' ', strip=True)

    # Helper for semantic splitting
    def _split_text_semantically(self, text_block: str, similarity_threshold: float = 0.6, min_chunk_sentences: int = 2, max_chunk_sentences: int = 10) -> list[str]:
        try:
            sentences = sent_tokenize(text_block)
        except Exception as e:
            logger.warning(f"Sentence tokenization failed for a block: {e}. Falling back to character splitting for this block.")
            return [] # Signal to use character splitting

        if not sentences:
            return []

        if len(sentences) <= min_chunk_sentences: # Not enough sentences to split semantically
            return [" ".join(sentences)]

        try:
            sentence_embeddings = self.embedding_model.encode(sentences, show_progress_bar=False) # Turn off progress for internal calls
        except Exception as e:
            logger.warning(f"Failed to generate sentence embeddings for semantic splitting: {e}. Falling back to character splitting.")
            return []

        semantic_chunks = []
        current_chunk_sentences = []

        for i in range(len(sentences)):
            current_chunk_sentences.append(sentences[i])

            if i < len(sentences) - 1:
                # Calculate similarity between current sentence and the next
                sim = cosine_similarity(sentence_embeddings[i].reshape(1, -1), sentence_embeddings[i+1].reshape(1, -1))[0][0]

                # Check for split condition:
                # 1. Similarity drops below threshold
                # 2. Current chunk reaches max sentences (hard limit)
                if sim < similarity_threshold or len(current_chunk_sentences) >= max_chunk_sentences:
                    if len(current_chunk_sentences) >= min_chunk_sentences:
                        semantic_chunks.append(" ".join(current_chunk_sentences))
                        current_chunk_sentences = []
                    # If below min_chunk_sentences but similarity is low, it will be appended with the next segment or form a small chunk at the end.
            else: # Last sentence
                if current_chunk_sentences: # Append any remaining sentences
                    # If the last formed chunk is too small, try to merge with the previous one if possible
                    if semantic_chunks and len(current_chunk_sentences) < min_chunk_sentences :
                        semantic_chunks[-1] += " " + " ".join(current_chunk_sentences)
                    else:
                        semantic_chunks.append(" ".join(current_chunk_sentences))

        # Ensure all sentences are captured, even if the last chunk is small
        if not semantic_chunks and sentences: # If no chunks were formed at all but there were sentences
            return [" ".join(sentences)] # Return the whole block

        return semantic_chunks

    def extract_structured_chunks(self, page_id: str, page_title: str, page_url: str, page_html_content: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[dict]:
        '''
        Extracts structured chunks from Confluence page HTML.
        Attempts to break content by H1, H2, H3 headers first, then paragraphs, tables, lists, code blocks.
        Applies semantic splitting for long text blocks, then character splitting as fallback or for shorter blocks.
        '''
        chunks = []
        if not page_html_content:
            return chunks

        soup = BeautifulSoup(page_html_content, 'html.parser')
        main_content = soup.find(id='main-content') or soup.find(class_='wiki-content') or soup
        if not main_content: # If no main content area, use the whole body
            main_content = soup.body if soup.body else soup
            if not main_content: # If still no content, return empty
                logger.warning(f"Page ID {page_id} has no main_content or body tag. Skipping.")
                return chunks


        current_headings = [""] * 3
        processed_elements_for_text = set()

        for element in main_content.find_all(True, recursive=True):
            if element in processed_elements_for_text:
                continue

            context_str = f"{page_title}"
            current_section_text_content = None

            if element.name == 'h1':
                current_headings[0] = element.get_text(strip=True)
                current_headings[1] = ""
                current_headings[2] = ""
                current_section_text_content = current_headings[0]
                context_str += f" > {current_headings[0]}"
                processed_elements_for_text.add(element)

            elif element.name == 'h2':
                current_headings[1] = element.get_text(strip=True)
                current_headings[2] = ""
                current_section_text_content = current_headings[1]
                context_str = f"{page_title} > {current_headings[0]} > {current_headings[1]}" if current_headings[0] else f"{page_title} > {current_headings[1]}"
                processed_elements_for_text.add(element)

            elif element.name == 'h3':
                current_headings[2] = element.get_text(strip=True)
                current_section_text_content = current_headings[2]
                context_str = f"{page_title} > {current_headings[0]} > {current_headings[1]} > {current_headings[2]}" if current_headings[0] and current_headings[1] else f"{page_title} > {current_headings[1]} > {current_headings[2]}" if current_headings[1] else f"{page_title} > {current_headings[2]}"
                processed_elements_for_text.add(element)

            elif element.name in ['p', 'table', 'ul', 'ol', 'pre'] or \
                 (element.name == 'ac:structured-macro' and element.get('ac:name') == 'code') or \
                 (element.name == 'div' and any(cls in element.get('class', []) for cls in ['panel', 'infomail', 'code', 'codeMacro'])): # Added common div classes for code/panels

                current_section_text_content = self._get_text_from_element(element)

                temp_context_h = f"{page_title}"
                if current_headings[0]: temp_context_h += f" > {current_headings[0]}"
                if current_headings[1]: temp_context_h += f" > {current_headings[1]}"
                if current_headings[2]: temp_context_h += f" > {current_headings[2]}"
                context_str = temp_context_h

                for child_el in element.find_all(True, recursive=True):
                    processed_elements_for_text.add(child_el)
                processed_elements_for_text.add(element)

            if current_section_text_content and current_section_text_content.strip():
                text_to_split = current_section_text_content.strip()

                # Attempt semantic splitting for longer text blocks
                MIN_LEN_FOR_SEMANTIC = settings.MIN_TEXT_BLOCK_LENGTH_FOR_SEMANTIC_SPLIT
                # MAX_LEN_FOR_SEMANTIC is not used here to gate, but rather as a property of semantic splitter if implemented

                semantically_split_parts = []
                if len(text_to_split) > MIN_LEN_FOR_SEMANTIC:
                    logger.debug(f"Attempting semantic split for a block of length {len(text_to_split)}")
                    # Default similarity_threshold=0.6, min_chunk_sentences=2, max_chunk_sentences=10
                    semantically_split_parts = self._split_text_semantically(text_to_split)
                    if not semantically_split_parts:
                         logger.debug(f"Semantic splitting returned no parts or failed. Falling back to character split for this block.")

                if semantically_split_parts:
                    logger.debug(f"Successfully split block into {len(semantically_split_parts)} semantic chunks.")
                    for s_chunk_text in semantically_split_parts:
                        if s_chunk_text.strip(): # Ensure chunk is not empty
                            # Character splitting for semantic chunks if they are still too long
                            # This ensures that even after semantic splitting, chunks respect chunk_size.
                            start_index_semantic = 0
                            while start_index_semantic < len(s_chunk_text):
                                end_index_semantic = min(start_index_semantic + chunk_size, len(s_chunk_text))
                                chunk_text_to_add_semantic = s_chunk_text[start_index_semantic:end_index_semantic].strip()
                                if chunk_text_to_add_semantic:
                                    chunks.append({
                                        "page_id": page_id, "page_title": page_title, "url": page_url,
                                        "text": chunk_text_to_add_semantic, "context_hierarchy": context_str
                                    })
                                if end_index_semantic == len(s_chunk_text):
                                    break
                                start_index_semantic += (chunk_size - chunk_overlap)
                                if start_index_semantic >= len(s_chunk_text):
                                    break
                else: # Fallback to character splitting for the whole block
                    start_index = 0
                    while start_index < len(text_to_split):
                        end_index = min(start_index + chunk_size, len(text_to_split))
                        chunk_text_to_add = text_to_split[start_index:end_index].strip()

                        if chunk_text_to_add:
                            chunks.append({
                                "page_id": page_id,
                                "page_title": page_title,
                                "url": page_url,
                                "text": chunk_text_to_add,
                                "context_hierarchy": context_str
                            })

                        if end_index == len(text_to_split):
                            break
                        start_index += (chunk_size - chunk_overlap)
                        if start_index >= len(text_to_split):
                            break

        # Fallback: if no structured chunks were extracted, chunk the whole page text
        if not chunks:
            full_text = soup.get_text(separator="\n", strip=True)
            start_index = 0
            while start_index < len(full_text):
                end_index = min(start_index + chunk_size, len(full_text))
                chunk_text_to_add = full_text[start_index:end_index].strip()
                if chunk_text_to_add:
                    chunks.append({
                        "page_id": page_id,
                        "page_title": page_title,
                        "url": page_url,
                        "text": chunk_text_to_add,
                        "context_hierarchy": page_title # Default context
                    })
                if end_index == len(full_text):
                    break
                start_index += (chunk_size - chunk_overlap)
                if start_index >= len(full_text):
                    break

        logger.info(f"Extracted {len(chunks)} chunks from page ID {page_id} ({page_title})")
        return chunks

    def generate_embeddings(self, chunk_texts: list[str]):
        if not self.embedding_model:
            logger.error("Embedding model not initialized.")
            return []
        if not chunk_texts:
            logger.info("No chunk texts provided for embedding generation.")
            return []

        logger.info(f"Generating embeddings for {len(chunk_texts)} chunks...")
        try:
            embeddings = self.embedding_model.encode(chunk_texts, show_progress_bar=True)
            logger.info("Embeddings generated successfully.")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}", exc_info=True)
            return []

    def build_and_save_index(self):
        if not self.confluence_service or not self.embedding_model:
            logger.error("IndexingService not properly initialized (Confluence or Embedding Model missing). Cannot build index.")
            return

        all_chunks_metadata = []
        page_ids_to_index = set()

        # 1. Get Page IDs from specified spaces
        if not settings.CONFLUENCE_SPACE_KEYS_TO_INDEX:
            logger.warning("No CONFLUENCE_SPACE_KEYS_TO_INDEX configured. Indexing will be empty.")
            return

        for space_key in settings.CONFLUENCE_SPACE_KEYS_TO_INDEX:
            if not space_key or space_key == "YOUR_SPACE_KEY": # Skip empty or placeholder keys
                logger.warning(f"Skipping invalid or placeholder space key: '{space_key}'")
                continue
            try:
                # Assuming ConfluenceService has or will have a method to get all page IDs in a space.
                # For now, using search_content with a broad query to simulate this.
                # This part needs a robust way to list all pages in a space.
                # A placeholder: search for pages in the space. This might not get all pages.
                # The current search_content returns a list of dicts with 'id', 'title', 'url'.
                # Limit is just to avoid too many pages in this illustrative step.
                # In a real scenario, you'd paginate through all results.
                logger.info(f"Fetching pages for space: {space_key}...")
                pages_in_space = self.confluence_service.search_content(
                    search_terms={"keywords": [""], "phrases": []}, # Broad search
                    space_keys=[space_key],
                    limit=50 # Placeholder limit, ideally get all pages
                )

                if pages_in_space:
                    for page_summary in pages_in_space:
                        page_ids_to_index.add((page_summary['id'], page_summary['title'], page_summary['url']))
                    logger.info(f"Found {len(pages_in_space)} pages in space {space_key} (using search method).")
                else:
                    logger.info(f"No pages found via search in space {space_key}. Check space key and content.")

            except Exception as e:
                logger.error(f"Error fetching pages from space {space_key}: {e}", exc_info=True)

        if not page_ids_to_index:
            logger.warning("No page IDs found to index after checking all specified spaces.")
            # Create empty index and chunks file if they don't exist
            if not os.path.exists(settings.FAISS_INDEX_PATH) and self.embedding_model:
                placeholder_embedding = self.embedding_model.encode(["dummy text"])
                dimension = placeholder_embedding.shape[1]
                empty_index = faiss.IndexFlatL2(dimension)
                faiss.write_index(empty_index, settings.FAISS_INDEX_PATH)
                logger.info(f"Created empty FAISS index at {settings.FAISS_INDEX_PATH}")
            if not os.path.exists(settings.CHUNKS_DATA_PATH):
                with open(settings.CHUNKS_DATA_PATH, 'w') as f:
                    json.dump([], f)
                logger.info(f"Created empty chunks data file at {settings.CHUNKS_DATA_PATH}")
            return

        # 2. Fetch content, chunk, and collect metadata
        chunk_texts_for_embedding = []
        for page_id, page_title, page_url in list(page_ids_to_index): # Convert set to list for iteration
            try:
                logger.info(f"Processing page ID: {page_id} ({page_title})")
                # Fetching 'body.storage' for HTML content
                page_obj = self.confluence_service.confluence.get_page_by_id(page_id, expand="body.storage")
                if page_obj and "body" in page_obj and "storage" in page_obj["body"] and "value" in page_obj["body"]["storage"]:
                    page_html_content = page_obj["body"]["storage"]["value"]

                    extracted_page_chunks = self.extract_structured_chunks(page_id, page_title, page_url, page_html_content)
                    for chunk_data in extracted_page_chunks:
                        all_chunks_metadata.append(chunk_data)
                        chunk_texts_for_embedding.append(chunk_data['text'])
                else:
                    logger.warning(f"Could not retrieve HTML content for page ID {page_id}")
            except Exception as e:
                logger.error(f"Error processing page ID {page_id}: {e}", exc_info=True)

        if not all_chunks_metadata:
            logger.warning("No chunks were extracted from any page. Index will be empty.")
            # Create empty index and chunks file if they don't exist (similar to above)
            if not os.path.exists(settings.FAISS_INDEX_PATH) and self.embedding_model:
                placeholder_embedding = self.embedding_model.encode(["dummy text"])
                dimension = placeholder_embedding.shape[1]
                empty_index = faiss.IndexFlatL2(dimension)
                faiss.write_index(empty_index, settings.FAISS_INDEX_PATH)
                logger.info(f"Created empty FAISS index at {settings.FAISS_INDEX_PATH} (no chunks extracted).")
            if not os.path.exists(settings.CHUNKS_DATA_PATH):
                with open(settings.CHUNKS_DATA_PATH, 'w') as f:
                    json.dump([], f)
                logger.info(f"Created empty chunks data file at {settings.CHUNKS_DATA_PATH} (no chunks extracted).")
            return

        # 3. Generate embeddings
        embeddings = self.generate_embeddings(chunk_texts_for_embedding)

        if embeddings is None or len(embeddings) == 0:
            logger.error("No embeddings were generated. Cannot build FAISS index.")
            return

        embeddings_np = embeddings # Assuming encode returns a numpy array or compatible
        if not isinstance(embeddings_np, np.ndarray): # Ensure it's a numpy array
            embeddings_np = np.array(embeddings_np).astype('float32')
        if embeddings_np.ndim == 1: # Handle case of single embedding
             embeddings_np = np.expand_dims(embeddings_np, axis=0)


        dimension = embeddings_np.shape[1]
        logger.info(f"Dimension of embeddings: {dimension}")

        # 4. Create and save FAISS index
        try:
            logger.info(f"Building FAISS index with {len(all_chunks_metadata)} chunks...")
            index = faiss.IndexFlatL2(dimension) # Using L2 distance
            index.add(embeddings_np)
            faiss.write_index(index, settings.FAISS_INDEX_PATH)
            logger.info(f"FAISS index successfully built and saved to {settings.FAISS_INDEX_PATH} with {index.ntotal} vectors.")
        except Exception as e:
            logger.error(f"Error building or saving FAISS index: {e}", exc_info=True)
            return

        # 5. Save chunk metadata
        try:
            with open(settings.CHUNKS_DATA_PATH, 'w', encoding='utf-8') as f:
                json.dump(all_chunks_metadata, f, ensure_ascii=False, indent=4)
            logger.info(f"Chunk metadata successfully saved to {settings.CHUNKS_DATA_PATH}")
        except Exception as e:
            logger.error(f"Error saving chunk metadata: {e}", exc_info=True)

# Example of how this service might be triggered (e.g., from a CLI script or admin panel)
if __name__ == '__main__':
    # This is for direct execution testing of the indexing service
    # Ensure .env is loaded if running this directly and config relies on it.
    # from dotenv import load_dotenv
    # load_dotenv(dotenv_path='../../.env') # Adjust path as necessary

    logging.basicConfig(level=logging.INFO)
    logger.info("Starting indexing process directly...")

    # Check if API keys are placeholder
    if "YOUR_CONFLUENCE_API_TOKEN" in settings.CONFLUENCE_API_TOKEN or \
       "YOUR_CONFLUENCE_URL_HERE" in settings.CONFLUENCE_URL or \
       any("YOUR_SPACE_KEY" in key for key in settings.CONFLUENCE_SPACE_KEYS_TO_INDEX):
        logger.error("Confluence API token, URL, or Space Key is not configured. Please set them in your .env file (or directly in config for testing if not using .env for this script).")
        logger.error(f"CONFLUENCE_URL: {settings.CONFLUENCE_URL}")
        logger.error(f"CONFLUENCE_USERNAME: {settings.CONFLUENCE_USERNAME}")
        logger.error("CONFLUENCE_API_TOKEN: <hidden>" if settings.CONfluence_API_TOKEN else "None")
        logger.error(f"CONFLUENCE_SPACE_KEYS_TO_INDEX: {settings.CONFLUENCE_SPACE_KEYS_TO_INDEX}")

    else:
        # Need to import numpy for the main execution context if not already
        # import numpy as np # Already imported at the top
        # import os # Already imported at the top

        indexing_service = IndexingService()
        if indexing_service.confluence_service and indexing_service.embedding_model:
            indexing_service.build_and_save_index()
            logger.info("Indexing process completed.")

            # Verify files were created
            if os.path.exists(settings.FAISS_INDEX_PATH):
                logger.info(f"FAISS index file found at: {settings.FAISS_INDEX_PATH}")
                # Optionally load and check index.ntotal
                # index = faiss.read_index(settings.FAISS_INDEX_PATH)
                # logger.info(f"Loaded FAISS index has {index.ntotal} vectors.")
            else:
                logger.warning(f"FAISS index file NOT found at: {settings.FAISS_INDEX_PATH}")

            if os.path.exists(settings.CHUNKS_DATA_PATH):
                logger.info(f"Chunks data file found at: {settings.CHUNKS_DATA_PATH}")
                # Optionally load and check number of chunks
                # with open(settings.CHUNKS_DATA_PATH, 'r') as f:
                #     data = json.load(f)
                # logger.info(f"Chunks data file contains {len(data)} chunk entries.")
            else:
                logger.warning(f"Chunks data file NOT found at: {settings.CHUNKS_DATA_PATH}")
        else:
            logger.error("Failed to initialize IndexingService. Cannot run build_and_save_index.")
