from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
import logging
import re

from .schemas import UserQuestion, ChatResponse
from .services.confluence_service import ConfluenceService
from .services.gemini_service import GeminiService
from .core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Conrad Chatbot API",
    description="API for Conrad Chatbot using Confluence and Gemini",
    version="0.1.0"
)

confluence_service = ConfluenceService()
gemini_service = GeminiService()

KNOWN_SPACE_KEYWORDS = {"M2": "M2", "SGP": "SGP", "SF": "SF"}
SPACE_KEY_PATTERN = re.compile(r'\b(' + '|'.join(re.escape(key) for key in KNOWN_SPACE_KEYWORDS.keys()) + r')\b', re.IGNORECASE)

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

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup...")
    # ... (rest of startup is the same)
    if not confluence_service.confluence:
        logger.error("Confluence service failed to initialize.")
    if not gemini_service.model:
        logger.error("Gemini service failed to initialize.")
    logger.info("Confluence and Gemini services initialized (or attempted).")


def extract_space_keys_from_query(question: str) -> list[str]:
    # ... (same as before)
    found_keys = set()
    matches = SPACE_KEY_PATTERN.findall(question)
    for match in matches:
        for known_key in KNOWN_SPACE_KEYWORDS.keys():
            if match.upper() == known_key.upper():
                found_keys.add(KNOWN_SPACE_KEYWORDS[known_key])
                break
    if found_keys:
        logger.info(f"Extracted space keys: {list(found_keys)}")
    return list(found_keys)

def extract_search_terms(question: str) -> dict:
    # ... (same as before)
    normalized_question = question.lower()
    normalized_question = re.sub(r'[^\w\s-]', '', normalized_question)
    words = normalized_question.split()
    keywords = [word for word in words if word not in STOP_WORDS and len(word) > 2]
    potential_proper_nouns = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)+\b', question)
    phrases = set(potential_proper_nouns)
    for n in range(2, 4):
        for i in range(len(words) - n + 1):
            ngram_words = words[i:i+n]
            if not all(word in STOP_WORDS for word in ngram_words):
                phrases.add(" ".join(ngram_words))
    final_phrases = {p for p in phrases if len(p.split()) > 1 and len(p) > 3}
    final_keywords = set(keywords)
    for phrase in final_phrases:
        for word in phrase.split():
            if word in final_keywords:
                final_keywords.remove(word)
    return {
        "keywords": sorted(list(final_keywords), key=len, reverse=True),
        "phrases": sorted(list(final_phrases), key=lambda p: len(p.split()), reverse=True)
    }

# New helper function for scoring paragraphs
def score_paragraph(paragraph_text: str, phrases: list[str], keywords: list[str]) -> int:
    score = 0
    para_lower = paragraph_text.lower()

    # Score for phrase presence
    for phrase in phrases:
        if phrase.lower() in para_lower:
            score += 10  # Higher score for finding a whole phrase
            logger.debug(f"Paragraph scored +10 for phrase: '{phrase}'")

    # Score for unique keyword presence
    found_keywords_in_para = set()
    for keyword in keywords:
        if keyword.lower() in para_lower:
            found_keywords_in_para.add(keyword.lower())

    score += len(found_keywords_in_para) * 2 # Score for each unique keyword
    if found_keywords_in_para:
        logger.debug(f"Paragraph scored +{len(found_keywords_in_para)*2} for keywords: {found_keywords_in_para}")

    return score

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(user_question: UserQuestion = Body(...)):
    logger.info(f"Received question: {user_question.question}")

    if not confluence_service.confluence or not gemini_service.model:
        raise HTTPException(status_code=503, detail="One or more backend services are unavailable.")

    try:
        search_terms = extract_search_terms(user_question.question)
        extracted_space_keys = extract_space_keys_from_query(user_question.question)

        logger.info(f"Searching Confluence with terms: {search_terms}, space_keys: {extracted_space_keys}")

        search_results = confluence_service.search_content(
            search_terms=search_terms,
            space_keys=extracted_space_keys if extracted_space_keys else None,
            limit=5 # Fetch slightly more pages to have more paragraphs to choose from
        )

        context_for_gemini = "No specific information found in Confluence that matches your query well." # Default if nothing good is found
        source_urls = []

        if search_results:
            source_urls = [result['url'] for result in search_results] # All potential sources

            scored_paragraphs_with_source = [] # List of tuples: (score, paragraph_text, page_title, page_url)

            for result in search_results:
                page_id = result.get("id")
                if not page_id: continue

                page_content_full = confluence_service.get_page_content_by_id(page_id)
                if not page_content_full:
                    logger.warning(f"No content for page ID: {page_id}, title: {result.get('title')}")
                    continue

                page_title = result.get('title', 'Unknown Page')
                page_url = result.get('url', 'N/A')

                paragraphs = page_content_full.split('\n\n')
                if len(paragraphs) <= 1 and '\n' in page_content_full:
                    paragraphs = page_content_full.split('\n')

                for para_text in paragraphs:
                    if not para_text.strip(): continue # Skip empty paragraphs

                    # Use keywords and phrases from the specific search_terms extraction
                    score = score_paragraph(para_text, search_terms["phrases"], search_terms["keywords"])
                    if score > 0: # Only consider paragraphs that have some relevance
                        scored_paragraphs_with_source.append((score, para_text, page_title, page_url))

            if scored_paragraphs_with_source:
                # Sort all collected paragraphs by score, descending
                scored_paragraphs_with_source.sort(key=lambda x: x[0], reverse=True)

                logger.info(f"Found {len(scored_paragraphs_with_source)} relevant paragraphs with scores. Top 3 scores: {[s[0] for s in scored_paragraphs_with_source[:3]]}")

                selected_context_parts = []
                current_length = 0
                max_context_length = 15000 # Max characters for Gemini context

                for score, para_text, p_title, p_url in scored_paragraphs_with_source:
                    content_header = f"Context from '{p_title}' (URL: {p_url}, Relevance Score: {score}):\n"
                    full_snippet = f"{content_header}{para_text}\n---"

                    if current_length + len(full_snippet) <= max_context_length:
                        selected_context_parts.append(full_snippet)
                        current_length += len(full_snippet)
                    else:
                        # Try to fit a portion if the whole snippet is too long
                        remaining_space = max_context_length - current_length
                        if remaining_space > len(content_header) + 50: # Ensure there's enough space for header and some text
                            truncated_para_text_len = remaining_space - len(content_header) - len("\n---")
                            truncated_snippet = f"{content_header}{para_text[:truncated_para_text_len]}\n---"
                            selected_context_parts.append(truncated_snippet)
                            current_length += len(truncated_snippet)
                        break # Stop adding more paragraphs

                if selected_context_parts:
                    context_for_gemini = "\n".join(selected_context_parts)
                else: # Should not happen if scored_paragraphs_with_source was not empty, but as a safeguard
                    logger.warning("Relevant paragraphs found, but none could be fitted into context length.")
                    # Fallback to first page's initial content if all scored snippets are too long individually
                    if search_results:
                        first_page_id = search_results[0].get("id")
                        if first_page_id:
                            first_page_content = confluence_service.get_page_content_by_id(first_page_id)
                            if first_page_content:
                                context_for_gemini = f"Context from '{search_results[0].get('title', 'Unknown Page')}' (URL: {search_results[0].get('url', 'N/A')}):\n{first_page_content[:max_context_length*2//3]}\n---" # take a good chunk

            elif search_results: # Search results found, but no paragraphs scored > 0
                logger.info("No paragraphs scored > 0. Using initial content of the first search result as fallback.")
                # Fallback to initial content of the first result if no paragraphs scored positively
                first_page_id = search_results[0].get("id")
                if first_page_id:
                    first_page_content = confluence_service.get_page_content_by_id(first_page_id)
                    if first_page_content:
                         context_for_gemini = f"Context from '{search_results[0].get('title', 'Unknown Page')}' (URL: {search_results[0].get('url', 'N/A')}):\n{first_page_content[:max_context_length*2//3]}\n---"


        logger.info(f"Final context for Gemini (first 300 chars): {context_for_gemini[:300]}...")
        logger.info(f"Total context length for Gemini: {len(context_for_gemini)} chars.")

        ai_answer = gemini_service.generate_response(user_question.question, context_for_gemini)

        # ... (rest of the endpoint is the same)
        if ai_answer.startswith("Error:"):
            return ChatResponse(answer=ai_answer, source_urls=source_urls)
        return ChatResponse(answer=ai_answer, source_urls=source_urls)

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"An unexpected error occurred in /chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.get("/health", summary="Health Check")
async def health_check():
    # ... (same as before)
    confluence_status = "initialized" if confluence_service.confluence else "error_initializing"
    gemini_status = "initialized" if gemini_service.model else "error_initializing"
    if confluence_service.confluence and gemini_service.model:
        return {"status": "ok", "confluence_service": confluence_status, "gemini_service": gemini_status}
    else:
        return JSONResponse(status_code=503, content={"status": "error", "detail": "Services not fully available."})
