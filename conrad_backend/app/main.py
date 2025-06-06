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

# Define known space acronyms/keys that might be mentioned in user queries
KNOWN_SPACE_KEYWORDS = {
    "M2": "M2",
    "SGP": "SGP",
    "SF": "SF",
}
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

# Definition of the missing function
def get_query_keywords(question: str) -> set[str]:
    # Simple keyword extraction: split by space, lowercase.
    words = re.split(r'\s+', question.lower())
    # Filter out very short words (e.g., len <= 2) and strip common punctuation.
    # This helps in getting more meaningful keywords.
    keywords = {word.strip(",.?!();:'\"") for word in words if len(word.strip(",.?!();:'\"")) > 2}
    logger.info(f"Extracted keywords from query: {keywords}")
    return keywords

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(user_question: UserQuestion = Body(...)):
    logger.info(f"Received question: {user_question.question}")

    if not confluence_service.confluence or not gemini_service.model:
        raise HTTPException(status_code=503, detail="One or more backend services are unavailable.")

    try:
        # Now get_query_keywords is defined before this call
        query_keywords = get_query_keywords(user_question.question)
        extracted_space_keys = extract_space_keys_from_query(user_question.question)

        logger.info(f"Searching Confluence with terms: {search_terms}, space_keys: {extracted_space_keys}")

        search_results = confluence_service.search_content(
            query=user_question.question, 
            space_keys=extracted_space_keys if extracted_space_keys else None,
            limit=3 
        )

        context_for_gemini = "No specific information found in Confluence that matches your query well." # Default if nothing good is found
        source_urls = []
        
        if search_results:
            logger.info(f"Found {len(search_results)} results. Processing for relevant context...")
            source_urls = [result['url'] for result in search_results]
            
            all_relevant_paragraphs = []
            current_length = 0
            max_context_length = 15000

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

                relevant_paragraphs_for_page = []
                for para in paragraphs:
                    # Ensure query_keywords is not empty before checking
                    if query_keywords and any(keyword in para.lower() for keyword in query_keywords):
                        relevant_paragraphs_for_page.append(para)
                
                page_context_contribution = ""
                if relevant_paragraphs_for_page:
                    logger.info(f"Found {len(relevant_paragraphs_for_page)} relevant paragraphs in '{page_title}'.")
                    page_context_contribution = "\n".join(relevant_paragraphs_for_page)
                else:
                    logger.info(f"No specific relevant paragraphs in '{page_title}'. Using initial part of content.")
                    page_context_contribution = page_content_full[:1500] 

                content_header = f"Context from '{page_title}' (URL: {page_url}):\n"
                
                if current_length + len(content_header) + len(page_context_contribution) > max_context_length:
                    remaining_chars = max_context_length - current_length - len(content_header)
                    if remaining_chars > 0:
                        all_relevant_paragraphs.append(f"{content_header}{page_context_contribution[:remaining_chars]}\n---")
                        current_length += len(content_header) + remaining_chars + len("\n---")
                    logger.info(f"Overall context length limit reached. Truncated content from {page_title}.")
                    break 
                else:
                    all_relevant_paragraphs.append(f"{content_header}{page_context_contribution}\n---")
                    current_length += len(content_header) + len(page_context_contribution) + len("\n---")

            if all_relevant_paragraphs:
                context_for_gemini = "\n".join(all_relevant_paragraphs)
            elif search_results: 
                context_for_gemini = "Found some documents in Confluence but could not extract relevant segments or content."
                logger.warning("Content extraction from Confluence results yielded no usable segments.")
        
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
