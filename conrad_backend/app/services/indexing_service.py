import logging
import json
import faiss
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

from ..core.config import settings
from .confluence_service import ConfluenceService

logger = logging.getLogger(__name__)

# ... (El resto de la clase IndexingService, como _get_text_from_element, _split_text_semantically, etc., permanece igual) ...
# ... (Se omite por brevedad, ya que no hay cambios en esas funciones) ...
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

        if element.name == 'ac:structured-macro' and element.get('ac:name') == 'code':
            code_body = element.find('ac:plain-text-body', recursive=False)
            if code_body:
                return code_body.get_text(strip=False)
            return element.get_text(separator='\n', strip=True)

        if element.name == 'pre':
            return element.get_text(strip=False)

        if element.name == 'ul':
            for i, li in enumerate(element.find_all('li', recursive=False)):
                texts.append(f"- {self._get_text_from_element(li)}")
            return "\n".join(texts)
        if element.name == 'ol':
            for i, li in enumerate(element.find_all('li', recursive=False)):
                texts.append(f"{i+1}. {self._get_text_from_element(li)}")
            return "\n".join(texts)

        if element.name in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div']:
            content_texts = []
            for child_node in element.children:
                if isinstance(child_node, str):
                    content_texts.append(child_node.strip())
                elif child_node.name and child_node.name not in ['ul', 'ol', 'table', 'pre', 'ac:structured-macro', 'div']:
                    content_texts.append(child_node.get_text(separator=' ', strip=True))
            return " ".join(filter(None, content_texts))

        if element.name == 'li':
            direct_text = []
            for child_node in element.children:
                if isinstance(child_node, str):
                    direct_text.append(child_node.strip())
                elif child_node.name and child_node.name not in ['ul', 'ol', 'table', 'pre', 'ac:structured-macro']:
                    direct_text.append(child_node.get_text(separator=' ', strip=True))
            return " ".join(filter(None, direct_text))

        if isinstance(element, str):
            return element.strip()

        return element.get_text(separator=' ', strip=True)

    def _split_text_semantically(self, text_block: str, similarity_threshold: float = 0.6, min_chunk_sentences: int = 2, max_chunk_sentences: int = 10) -> list[str]:
        try:
            sentences = sent_tokenize(text_block)
        except Exception as e:
            logger.warning(f"Sentence tokenization failed for a block: {e}. Falling back to character splitting for this block.")
            return []

        if not sentences:
            return []

        if len(sentences) <= min_chunk_sentences:
            return [" ".join(sentences)]

        try:
            sentence_embeddings = self.embedding_model.encode(sentences, show_progress_bar=False)
        except Exception as e:
            logger.warning(f"Failed to generate sentence embeddings for semantic splitting: {e}. Falling back to character splitting.")
            return []

        semantic_chunks = []
        current_chunk_sentences = []

        for i in range(len(sentences)):
            current_chunk_sentences.append(sentences[i])

            if i < len(sentences) - 1:
                sim = cosine_similarity(sentence_embeddings[i].reshape(1, -1), sentence_embeddings[i+1].reshape(1, -1))[0][0]

                if sim < similarity_threshold or len(current_chunk_sentences) >= max_chunk_sentences:
                    if len(current_chunk_sentences) >= min_chunk_sentences:
                        semantic_chunks.append(" ".join(current_chunk_sentences))
                        current_chunk_sentences = []
            else:
                if current_chunk_sentences:
                    if semantic_chunks and len(current_chunk_sentences) < min_chunk_sentences :
                        semantic_chunks[-1] += " " + " ".join(current_chunk_sentences)
                    else:
                        semantic_chunks.append(" ".join(current_chunk_sentences))

        if not semantic_chunks and sentences:
            return [" ".join(sentences)]

        return semantic_chunks

    def extract_structured_chunks(self, page_id: str, page_title: str, page_url: str, page_html_content: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[dict]:
        chunks = []
        if not page_html_content:
            return chunks

        soup = BeautifulSoup(page_html_content, 'html.parser')
        main_content = soup.find(id='main-content') or soup.find(class_='wiki-content') or soup
        if not main_content:
            main_content = soup.body if soup.body else soup
            if not main_content:
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
                 (element.name == 'div' and any(cls in element.get('class', []) for cls in ['panel', 'infomail', 'code', 'codeMacro'])):

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
                MIN_LEN_FOR_SEMANTIC = settings.MIN_TEXT_BLOCK_LENGTH_FOR_SEMANTIC_SPLIT
                semantically_split_parts = []
                if len(text_to_split) > MIN_LEN_FOR_SEMANTIC:
                    logger.debug(f"Attempting semantic split for a block of length {len(text_to_split)}")
                    semantically_split_parts = self._split_text_semantically(text_to_split)
                    if not semantically_split_parts:
                         logger.debug(f"Semantic splitting returned no parts or failed. Falling back to character split for this block.")

                if semantically_split_parts:
                    logger.debug(f"Successfully split block into {len(semantically_split_parts)} semantic chunks.")
                    for s_chunk_text in semantically_split_parts:
                        if s_chunk_text.strip():
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
                else:
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
                        "context_hierarchy": page_title
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
            logger.error("IndexingService not properly initialized. Cannot build index.")
            return

        all_chunks_metadata = []
        page_ids_to_index = set()

        if not settings.CONFLUENCE_SPACE_KEYS_TO_INDEX:
            logger.warning("No CONFLUENCE_SPACE_KEYS_TO_INDEX configured. Indexing will be empty.")
            return

        for space_key in settings.CONFLUENCE_SPACE_KEYS_TO_INDEX:
            if not space_key or "YOUR_SPACE_KEY" in space_key:
                logger.warning(f"Skipping invalid or placeholder space key: '{space_key}'")
                continue
            
            ## MODIFICADO: Implementada paginación para obtener TODAS las páginas.
            try:
                logger.info(f"Fetching all pages for space: {space_key} using pagination...")
                all_pages_in_space = []
                start = 0
                limit = 50  # El tamaño del lote para cada llamada a la API
                
                while True:
                    logger.info(f"Fetching pages from space {space_key} with start={start}, limit={limit}")
                    
                    # Usamos el método directo de la librería atlassian-python-api
                    pages_batch = self.confluence_service.confluence.get_all_pages_from_space(
                        space_key, 
                        start=start, 
                        limit=limit,
                        expand=None # No necesitamos el cuerpo aquí
                    )
                    
                    if not pages_batch:
                        logger.info(f"No more pages found for space {space_key}. Finished pagination.")
                        break
                    
                    all_pages_in_space.extend(pages_batch)
                    start += limit

                if all_pages_in_space:
                    base_url = settings.CONFLUENCE_URL.rstrip('/')
                    for page_summary in all_pages_in_space:
                        page_url = f"{base_url}/wiki{page_summary['_links']['webui']}"
                        page_ids_to_index.add((page_summary['id'], page_summary['title'], page_url))
                    logger.info(f"Found a total of {len(all_pages_in_space)} pages in space {space_key}.")
                else:
                    logger.info(f"No pages found in space {space_key}.")

            except Exception as e:
                logger.error(f"Error fetching pages from space {space_key}: {e}", exc_info=True)
        
        # ... (El resto de la función para procesar, chunckear y guardar el índice permanece igual) ...
        # ... (Se omite por brevedad) ...
        if not page_ids_to_index:
            logger.warning("No page IDs found to index after checking all specified spaces.")
            # ... (Lógica de creación de índice vacío) ...
            return

        chunk_texts_for_embedding = []
        for page_id, page_title, page_url in list(page_ids_to_index):
            try:
                logger.info(f"Processing page ID: {page_id} ({page_title})")
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
        
        # ... (Resto del código para generar embeddings y guardar) ...
        if not all_chunks_metadata:
            logger.warning("No chunks were extracted. Index will be empty.")
            # ...
            return

        embeddings = self.generate_embeddings(chunk_texts_for_embedding)
        if embeddings is None or len(embeddings) == 0:
            logger.error("No embeddings were generated. Cannot build FAISS index.")
            return

        embeddings_np = np.array(embeddings).astype('float32')
        if embeddings_np.ndim == 1:
            embeddings_np = np.expand_dims(embeddings_np, axis=0)

        dimension = embeddings_np.shape[1]
        logger.info(f"Dimension of embeddings: {dimension}")

        try:
            logger.info(f"Building FAISS index with {len(all_chunks_metadata)} chunks...")
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_np)
            faiss.write_index(index, settings.FAISS_INDEX_PATH)
            logger.info(f"FAISS index successfully built and saved to {settings.FAISS_INDEX_PATH} with {index.ntotal} vectors.")
        except Exception as e:
            logger.error(f"Error building or saving FAISS index: {e}", exc_info=True)
            return

        try:
            with open(settings.CHUNKS_DATA_PATH, 'w', encoding='utf-8') as f:
                json.dump(all_chunks_metadata, f, ensure_ascii=False, indent=4)
            logger.info(f"Chunk metadata successfully saved to {settings.CHUNKS_DATA_PATH}")
        except Exception as e:
            logger.error(f"Error saving chunk metadata: {e}", exc_info=True)