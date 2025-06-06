from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
import logging
import re
from collections import Counter

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
    if not confluence_service.confluence:
        logger.error("Confluence service failed to initialize.")
    if not gemini_service.model:
        logger.error("Gemini service failed to initialize.")
    logger.info("Confluence and Gemini services initialized (or attempted).")

def extract_space_keys_from_query(question: str) -> list[str]:
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
    # Normalize: lowercase and strip basic punctuation for keyword extraction
    normalized_question_for_keywords = question.lower()
    normalized_question_for_keywords = re.sub(r'[^\w\s-]', '', normalized_question_for_keywords) # allow hyphens

    words = normalized_question_for_keywords.split()

    # Extract individual Keywords: non-stop words, len > 2
    candidate_keywords = [
        word for word in words
        if word not in STOP_WORDS and len(word) > 2
    ]

    # Extract Phrases:
    # 1. Proper Nouns: Sequences of capitalized words from original question (allowing hyphens)
    #    This regex tries to capture multi-word capitalized phrases.
    proper_noun_phrases = re.findall(r'\b[A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+)+\b', question)

    # 2. N-grams (2-3 words) from normalized question, excluding those made of only stop words
    #    Using original words list (before lowercasing) to build ngrams to preserve original casing if needed,
    #    but comparison with stop words will be case-insensitive.
    #    Actually, using `words` (lowercase, punctuation-stripped) is better for ngram consistency.

    all_potential_phrases = set(proper_noun_phrases) # Start with proper nouns

    # Generate 2-word and 3-word phrases (bigrams and trigrams) from normalized words
    for n in range(2, 4):
        for i in range(len(words) - n + 1):
            ngram_words = words[i:i+n]
            # A phrase is valid if not all its words are stop words
            if not all(word in STOP_WORDS for word in ngram_words):
                # And if the phrase doesn't start or end with a stop word (heuristic for better phrases)
                if ngram_words[0] not in STOP_WORDS and ngram_words[-1] not in STOP_WORDS:
                    all_potential_phrases.add(" ".join(ngram_words))

    # Filter and select top phrases:
    # Prioritize longer phrases and ensure they are somewhat substantial.
    # Remove phrases that are substrings of other, longer selected phrases.

    # Sort by length (desc) then alphabetically to have some stability
    sorted_phrases = sorted(list(all_potential_phrases), key=lambda p: (len(p.split()), len(p)), reverse=True)

    selected_phrases = []
    max_phrases = 2 # Limit to max 2-3 phrases
    for phrase in sorted_phrases:
        is_substring = False
        for sel_phrase in selected_phrases:
            if phrase.lower() in sel_phrase.lower():
                is_substring = True
                break
        if not is_substring and len(phrase.split()) > 1 : # Ensure it's a multi-word phrase
             # Basic length check for the phrase itself too
            if len(phrase) > 3 : # e.g. avoid "a b" if a and b are single letters
                selected_phrases.append(phrase)
        if len(selected_phrases) >= max_phrases:
            break

    # Final Keywords: non-stop words that are NOT part of the *selected_phrases*
    final_keywords = set(candidate_keywords)
    for phrase_str in selected_phrases:
        # Split phrase into words and remove them from keywords
        # Use original phrase casing for splitting then lowercase for set operation
        phrase_words_lower = set(word.lower() for word in phrase_str.split())
        final_keywords.difference_update(phrase_words_lower)

    # Limit number of keywords too, e.g., top 5 by length or some other metric if too many
    final_keywords_list = sorted(list(final_keywords), key=len, reverse=True)[:5]

    search_terms_dict = {
        "keywords": final_keywords_list,
        "phrases": selected_phrases
    }
    logger.info(f"Refined - Extracted search terms: {search_terms_dict}")
    return search_terms_dict

def score_paragraph(paragraph_text: str, phrases: list[str], keywords: list[str]) -> int:
    # ... (same as before)
    score = 0
    para_lower = paragraph_text.lower()
    for phrase in phrases:
        if phrase.lower() in para_lower:
            score += 10
            logger.debug(f"Paragraph scored +10 for phrase: '{phrase}'")
    found_keywords_in_para = set()
    for keyword in keywords:
        if keyword.lower() in para_lower:
            found_keywords_in_para.add(keyword.lower())
    score += len(found_keywords_in_para) * 2
    if found_keywords_in_para:
        logger.debug(f"Paragraph scored +{len(found_keywords_in_para)*2} for keywords: {found_keywords_in_para}")
    return score

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(user_question: UserQuestion = Body(...)):
    # ... (The rest of chat_endpoint remains the same for now)
    # It already calls extract_search_terms and passes the result to confluence_service
    logger.info(f"Received question: {user_question.question}")

    if not confluence_service.confluence or not gemini_service.model:
        raise HTTPException(status_code=503, detail="One or more backend services are unavailable.")

    try:
        search_terms = extract_search_terms(user_question.question) # This is now the refined version
        extracted_space_keys = extract_space_keys_from_query(user_question.question)

        logger.info(f"Searching Confluence with refined terms: {search_terms}, space_keys: {extracted_space_keys}")

        search_results = confluence_service.search_content(
            search_terms=search_terms,
            space_keys=extracted_space_keys if extracted_space_keys else None,
            limit=5
        )

        context_for_gemini = "No specific information found in Confluence that matches your query well."
        source_urls = []

        if search_results:
            source_urls = [result['url'] for result in search_results]
            scored_paragraphs_with_source = []

            page_query_phrases = search_terms.get("phrases", [])
            page_query_keywords = search_terms.get("keywords", [])

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
                    if not para_text.strip(): continue

                    score = score_paragraph(para_text, page_query_phrases, page_query_keywords)
                    if score > 0:
                        scored_paragraphs_with_source.append((score, para_text, page_title, page_url))

            if scored_paragraphs_with_source:
                scored_paragraphs_with_source.sort(key=lambda x: x[0], reverse=True)
                logger.info(f"Found {len(scored_paragraphs_with_source)} relevant paragraphs. Top scores: {[s[0] for s in scored_paragraphs_with_source[:3]]}")

                selected_context_parts = []
                current_length = 0
                max_context_length = 15000

                for score, para_text, p_title, p_url in scored_paragraphs_with_source:
                    content_header = f"Context from '{p_title}' (URL: {p_url}, Relevance Score: {score}):\n"
                    full_snippet = f"{content_header}{para_text}\n---"

                    if current_length + len(full_snippet) <= max_context_length:
                        selected_context_parts.append(full_snippet)
                        current_length += len(full_snippet)
                    else:
                        remaining_space = max_context_length - current_length
                        if remaining_space > len(content_header) + 50:
                            truncated_para_text_len = remaining_space - len(content_header) - len("\n---")
                            truncated_snippet = f"{content_header}{para_text[:truncated_para_text_len]}\n---"
                            selected_context_parts.append(truncated_snippet)
                        break

                if selected_context_parts:
                    context_for_gemini = "\n".join(selected_context_parts)
                elif search_results:
                    first_page_content_fallback = confluence_service.get_page_content_by_id(search_results[0].get("id",""))
                    if first_page_content_fallback:
                        context_for_gemini = f"Context from '{search_results[0].get('title', '')}':\n{first_page_content_fallback[:max_context_length*2//3]}"

            elif search_results:
                logger.info("No paragraphs scored > 0. Using initial content of first result.")
                first_page_content_fallback = confluence_service.get_page_content_by_id(search_results[0].get("id",""))
                if first_page_content_fallback:
                     context_for_gemini = f"Context from '{search_results[0].get('title', '')}':\n{first_page_content_fallback[:max_context_length*2//3]}"

        logger.info(f"Final context for Gemini (first 300 chars): {context_for_gemini[:300]}...")
        ai_answer = gemini_service.generate_response(user_question.question, context_for_gemini)

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
    confluence_status = "initialized" if confluence_service.confluence else "error_initializing"
    gemini_status = "initialized" if gemini_service.model else "error_initializing"
    if confluence_service.confluence and gemini_service.model:
        return {"status": "ok", "confluence_service": confluence_status, "gemini_service": gemini_status}
    else:
        return JSONResponse(status_code=503, content={"status": "error", "detail": "Services not fully available."})
