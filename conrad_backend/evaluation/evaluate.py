import json
import logging
import os
import sys
import re
from collections import Counter

# Adjust path to import from app, assuming evaluate.py is in conrad_backend/evaluation/
# and the app directory is conrad_backend/app/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.services.confluence_service import ConfluenceService
from app.services.gemini_service import GeminiService
# Need to import or replicate helper functions from main.py
# For simplicity in this subtask, we'll assume they might be moved to a utils module later
# or we'll define simplified versions here if they are small.

# --- Replicated/Simplified Helper Functions (from main.py) ---
# It's better to refactor these into a shared utils module in a real scenario.

STOP_WORDS = set([
    "a", "un", "una", "el", "la", "los", "las", "de", "del", "en", "y", "o", "u", "que", "qué", "como", "cómo",
    "para", "por", "con", "sin", "sobre", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "should", "can", "could", "may", "might", "must", "the",
    "and", "or", "but", "if", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
    "to", "from", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
    "just", "don", "should", "now", "me", "i", "you", "he", "she", "it", "we", "they", "mi", "tu", "su"
])

def extract_search_terms_simplified(question: str) -> dict:
    normalized_question_for_keywords = question.lower()
    normalized_question_for_keywords = re.sub(r'[^\w\s-]', '', normalized_question_for_keywords)
    words = normalized_question_for_keywords.split()
    candidate_keywords = [word for word in words if word not in STOP_WORDS and len(word) > 2]
    # Simplified: no phrase extraction for this version of the eval script
    return {"keywords": candidate_keywords[:7], "phrases": []}

def extract_space_keys_from_query_simplified(question: str) -> list[str]:
    # Simplified: No space key extraction for this version of the eval script
    # In a real eval, this should be consistent with main.py
    return []

# score_paragraph can be copied if needed for keyword_page scoring from main.py's _build_context_from_search_results
def score_paragraph(paragraph_text: str, phrases: list[str], keywords: list[str]) -> int:
    score = 0
    para_lower = paragraph_text.lower()
    for phrase in phrases: # Phrases might be empty if using simplified extraction
        if phrase.lower() in para_lower:
            score += 10
    found_keywords_in_para = set()
    for keyword in keywords:
        if keyword.lower() in para_lower:
            found_keywords_in_para.add(keyword.lower())
    score += len(found_keywords_in_para) * 3
    return score

# --- End Helper Functions ---


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Metric Calculation Functions ---
def calculate_context_precision_at_k(retrieved_page_ids: list[str], expected_page_ids: list[str], k: int) -> float:
    if not retrieved_page_ids:
        return 0.0
    top_k_retrieved = retrieved_page_ids[:k]
    relevant_found = sum(1 for page_id in top_k_retrieved if page_id in expected_page_ids)
    return relevant_found / len(top_k_retrieved) if top_k_retrieved else 0.0

def calculate_context_recall_at_k(retrieved_page_ids: list[str], expected_page_ids: list[str], k: int) -> float:
    if not expected_page_ids:
        return 1.0 # Or 0.0, depending on definition. If nothing expected, all "expected" items found.
    if not retrieved_page_ids:
        return 0.0

    # Consider all retrieved items up to k, or all if fewer than k
    retrieved_set = set(retrieved_page_ids[:k])
    expected_set = set(expected_page_ids)

    relevant_found = len(retrieved_set.intersection(expected_set))
    return relevant_found / len(expected_set) if expected_set else 0.0

def calculate_mrr(retrieved_page_ids: list[str], expected_page_ids: list[str]) -> float:
    if not retrieved_page_ids:
        return 0.0
    for i, page_id in enumerate(retrieved_page_ids):
        if page_id in expected_page_ids:
            return 1.0 / (i + 1)
    return 0.0

def calculate_keyword_match_score(generated_answer_text: str, ideal_keywords: list[str]) -> float:
    if not ideal_keywords:
        return 1.0 # Or 0.0. If no keywords, all are "matched".
    if not generated_answer_text:
        return 0.0

    answer_lower = generated_answer_text.lower()
    matched_keywords = sum(1 for kw in ideal_keywords if kw.lower() in answer_lower)
    return matched_keywords / len(ideal_keywords) if ideal_keywords else 0.0
# --- End Metric Calculation Functions ---

def run_evaluation(test_set_path: str = "evaluation/test_set.json", N_SEMANTIC_RESULTS_EVAL = 5, N_CQL_RESULTS_EVAL = 3, K_FOR_METRICS = 5):
    logger.info("Starting evaluation process...")

    # Initialize services
    if "YOUR_CONFLUENCE_API_TOKEN" in settings.CONFLUENCE_API_TOKEN or \
       "YOUR_GEMINI_API_KEY" in settings.GEMINI_API_KEY or \
       "YOUR_CONFLUENCE_URL" in settings.CONFLUENCE_URL: # Added URL check
        logger.error("API keys or URL for Confluence or Gemini are not configured with actual values. Evaluation cannot proceed.")
        logger.error("Please set them in your .env file or environment variables.")
        return

    confluence_service = ConfluenceService()
    gemini_service = GeminiService()

    if not confluence_service.confluence or not gemini_service.model:
        logger.error("Failed to initialize Confluence or Gemini service. Aborting evaluation.")
        return

    if not confluence_service.faiss_index or not confluence_service.chunk_metadata or confluence_service.faiss_index.ntotal == 0:
        logger.warning("FAISS index or chunk metadata not loaded or index is empty in ConfluenceService. Retrieval quality will be affected.")
        # Allow to proceed to see generation if keyword search still works.

    try:
        with open(test_set_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Test set file not found at {test_set_path}")
        return
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {test_set_path}")
        return

    all_metrics = {
        "context_precision_at_k": [],
        "context_recall_at_k": [],
        "mrr": [],
        "keyword_match_score": []
    }

    for item in test_data:
        question_id = item["question_id"]
        question_text = item["question_text"]
        expected_page_ids = item["expected_confluence_page_ids"]
        ideal_answer_keywords = item["ideal_answer_keywords"]

        logger.info(f"--- Evaluating Question ID: {question_id} ---")
        logger.info(f"Question: {question_text}")

        # --- Simplified Orchestration (mimicking main.py) ---
        extracted_terms = extract_search_terms_simplified(question_text)
        # Space keys extraction is simplified to empty list for now
        extracted_space_keys = extract_space_keys_from_query_simplified(question_text)

        semantic_chunks = confluence_service.semantic_search_chunks(
            query_text=question_text,
            top_k=N_SEMANTIC_RESULTS_EVAL
        )

        keyword_pages = confluence_service.search_content(
            search_terms=extracted_terms,
            space_keys=extracted_space_keys, # Will be empty based on simplified function
            limit=N_CQL_RESULTS_EVAL
        )

        combined_results_for_eval = []
        processed_semantic_urls = set()
        retrieved_page_ids_for_metrics = [] # Store unique page_ids in order of retrieval importance

        # Process semantic chunks first
        for chunk in semantic_chunks:
            combined_results_for_eval.append(chunk) # Already chunk-like
            if chunk.get("page_id") and chunk.get("page_id") not in retrieved_page_ids_for_metrics:
                 retrieved_page_ids_for_metrics.append(chunk.get("page_id"))
            if chunk.get("url"):
                processed_semantic_urls.add(chunk.get("url"))

        # Process keyword pages
        for page in keyword_pages:
            if page.get("url") not in processed_semantic_urls:
                page_content_full = confluence_service.get_page_content_by_id(page['id'])
                if page_content_full:
                    # Treat keyword page as one chunk
                    combined_results_for_eval.append({
                        "text": page_content_full,
                        "url": page["url"],
                        "title": page["title"],
                        "score": 1000.0, # Default bad L2 for keyword results
                        "search_method": "keyword_page",
                        "context_hierarchy": page["title"],
                        "page_id": page['id'] # Corrected from page.get("page_id") to page['id'] for consistency
                    })
                    if page.get("id") and page.get("id") not in retrieved_page_ids_for_metrics: # Use page['id']
                         retrieved_page_ids_for_metrics.append(page.get("id"))

        # Sort combined_results for context building (lower L2 score is better)
        combined_results_for_eval.sort(key=lambda x: x.get("score", float('inf')))

        # --- Build Context (Simplified from main.py's _build_context_from_search_results) ---
        context_parts_for_gemini = []
        current_context_length = 0
        max_context_len_eval = 15000 # Should match main.py or be configurable

        # Re-score and prepare for context building
        scored_chunks_for_context_build = []
        for res_item in combined_results_for_eval:
            text = res_item.get("text")
            if not text: continue

            current_score = 0
            if res_item.get("search_method") == "semantic":
                l2 = res_item.get("score", float('inf'))
                current_score = 1.0 / (1.0 + l2) if l2 != float('inf') else 0
            elif res_item.get("search_method") == "keyword_page":
                current_score = score_paragraph(text, extracted_terms["phrases"], extracted_terms["keywords"])

            scored_chunks_for_context_build.append(
                (current_score, text, res_item.get("title", "N/A"), res_item.get("url", "N/A"), res_item.get("context_hierarchy", "N/A"))
            )

        scored_chunks_for_context_build.sort(key=lambda x: x[0], reverse=True) # Higher score is better

        for rel_score, chk_text, chk_title, chk_url, chk_hierarchy in scored_chunks_for_context_build:
            header = f"Contexto de '{chk_title}' (URL: {chk_url}, Jerarquía: '{chk_hierarchy}', Puntuación Relevancia: {rel_score:.4f}):\n"
            snippet = f"{header}{chk_text}\n---"
            if current_context_length + len(snippet) <= max_context_len_eval:
                context_parts_for_gemini.append(snippet)
                current_context_length += len(snippet)
            else: # Try to truncate
                remaining_space = max_context_len_eval - current_context_length
                if remaining_space > len(header) + 50:
                    truncated_text_len = remaining_space - (len(header) + len("\n---") + 3)
                    if truncated_text_len > 0:
                         context_parts_for_gemini.append(f"{header}{chk_text[:truncated_text_len]}...\n---")
                break

        final_context_str = "\n".join(context_parts_for_gemini)
        if not final_context_str:
            final_context_str = "No se encontró información relevante en Confluence para esta consulta."
        # --- End Build Context ---

        # --- Generate Answer ---
        ai_answer = gemini_service.generate_response(question_text, final_context_str)
        logger.info(f"AI Answer: {ai_answer[:200]}...")
        # --- End Generate Answer ---

        # --- Calculate Metrics ---
        # Note: retrieved_page_ids_for_metrics is built from both semantic and keyword results,
        # ordered by their original retrieval/combination order before re-sorting for context building.

        precision = calculate_context_precision_at_k(retrieved_page_ids_for_metrics, expected_page_ids, K_FOR_METRICS)
        recall = calculate_context_recall_at_k(retrieved_page_ids_for_metrics, expected_page_ids, K_FOR_METRICS)
        mrr_score = calculate_mrr(retrieved_page_ids_for_metrics, expected_page_ids)
        kw_match = calculate_keyword_match_score(ai_answer, ideal_answer_keywords)

        logger.info(f"Metrics for Q_ID {question_id}: Precision@{K_FOR_METRICS}={precision:.4f}, Recall@{K_FOR_METRICS}={recall:.4f}, MRR={mrr_score:.4f}, KeywordMatch={kw_match:.4f}")

        all_metrics["context_precision_at_k"].append(precision)
        all_metrics["context_recall_at_k"].append(recall)
        all_metrics["mrr"].append(mrr_score)
        all_metrics["keyword_match_score"].append(kw_match)
        logger.info("--- End Question Evaluation ---")

    # --- Calculate Average Metrics ---
    logger.info("========== Overall Evaluation Summary ==========")
    for metric_name, values in all_metrics.items():
        if values:
            avg_value = sum(values) / len(values)
            logger.info(f"Average {metric_name}: {avg_value:.4f}")
        else:
            logger.info(f"Average {metric_name}: N/A (no values)")
    logger.info("============================================")


if __name__ == "__main__":
    # Example: python -m evaluation.evaluate (if conrad_backend is in PYTHONPATH)
    # Or: python evaluate.py (if run from conrad_backend/ directory)

    # Ensure environment variables are loaded if .env file is used and script is run directly
    from dotenv import load_dotenv
    # Assuming .env is in the parent directory of 'evaluation' (i.e., conrad_backend/.env)
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        logger.info(f".env file loaded from {dotenv_path}")
    else:
        logger.info(".env file not found, relying on environment variables if set.")

    run_evaluation()
