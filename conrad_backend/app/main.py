from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
import logging
import re # For regex-based keyword extraction

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
# These should ideally match the actual Confluence Space Keys.
# Making them uppercase as space keys are often uppercase.
KNOWN_SPACE_KEYWORDS = {
    "M2": "M2",
    "SGP": "SGP",
    "SF": "SF",
    # Add more mappings if needed, e.g., "marketing space" : "MKTG"
}
# A regex pattern to find these whole words (case-insensitive for detection)
# We'll extract the found word and then use the KNOWN_SPACE_KEYWORDS mapping.
# This pattern looks for whole words to avoid matching substrings (e.g., 'm2' in 'team2').
SPACE_KEY_PATTERN = re.compile(r'\b(' + '|'.join(re.escape(key) for key in KNOWN_SPACE_KEYWORDS.keys()) + r')\b', re.IGNORECASE)


@app.on_event("startup")
async def startup_event():
    logger.info("Application startup...")
    if not confluence_service.confluence:
        logger.error("Confluence service failed to initialize. Check Confluence settings and connectivity.")
    if not gemini_service.model:
        logger.error("Gemini service failed to initialize. Check Gemini API Key.")
    logger.info("Confluence and Gemini services initialized (or attempted).")
    logger.info(f"Confluence URL configured: {settings.CONFLUENCE_URL}")
    if "YOUR_CONFLUENCE_API_TOKEN" in settings.CONFLUENCE_API_TOKEN or not settings.CONFLUENCE_API_TOKEN:
        logger.warning("Confluence API token appears to be a placeholder or is not set.")
    if "YOUR_GEMINI_API_KEY" in settings.GEMINI_API_KEY or not settings.GEMINI_API_KEY:
        logger.warning("Gemini API key appears to be a placeholder or is not set.")

def extract_space_keys_from_query(question: str) -> list[str]:
    found_keys = set()
    # Find all occurrences of the keywords in the question using regex
    matches = SPACE_KEY_PATTERN.findall(question)
    for match in matches:
        # Normalize the found match to one of the keys in KNOWN_SPACE_KEYWORDS (e.g., "m2" -> "M2")
        # This handles case-insensitivity of detection but uses the canonical form for search.
        for known_key in KNOWN_SPACE_KEYWORDS.keys():
            if match.upper() == known_key.upper(): # Compare uppercase to handle different casing
                found_keys.add(KNOWN_SPACE_KEYWORDS[known_key]) # Add the canonical space key
                break

    if found_keys:
        logger.info(f"Extracted space keys from query: {list(found_keys)}")
    return list(found_keys)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(user_question: UserQuestion = Body(...)):
    logger.info(f"Received question: {user_question.question}")

    if not confluence_service.confluence or not gemini_service.model:
        raise HTTPException(status_code=503, detail="One or more backend services are unavailable.")

    try:
        query_keywords = get_query_keywords(user_question.question)
        extracted_space_keys = extract_space_keys_from_query(user_question.question)

        logger.info(f"Searching Confluence for: '{user_question.question}' with keys: {extracted_space_keys}, keywords: {query_keywords}")
        search_results = confluence_service.search_content(
            query=user_question.question, # Original query for broader search
            space_keys=extracted_space_keys if extracted_space_keys else None,
            limit=3
        )

        context_for_gemini = "No specific information found in Confluence regarding this query."
        source_urls = []

        if search_results:
            logger.info(f"Found {len(search_results)} results. Processing for relevant context...")
            source_urls = [result['url'] for result in search_results]

            all_relevant_paragraphs = []
            current_length = 0
            # Max length for combined relevant paragraphs before adding full content
            # This is for the "relevant snippets" part.
            max_relevant_snippet_length = 10000
            # Overall max length for Gemini
            max_context_length = 15000


            for result in search_results:
                page_id = result.get("id")
                if not page_id:
                    continue

                page_content_full = confluence_service.get_page_content_by_id(page_id)
                if not page_content_full:
                    logger.warning(f"No content found for page ID: {page_id}, title: {result.get('title')}")
                    continue

                page_title = result.get('title', 'Unknown Page')
                page_url = result.get('url', 'N/A')

                # Attempt to find relevant paragraphs
                paragraphs = page_content_full.split('\n\n') # Assuming double newline separation
                if len(paragraphs) <= 1 and '\n' in page_content_full: # Try single newline if double fails
                    paragraphs = page_content_full.split('\n')

                relevant_paragraphs_for_page = []
                for para in paragraphs:
                    if any(keyword in para.lower() for keyword in query_keywords):
                        relevant_paragraphs_for_page.append(para)

                page_context_contribution = ""
                if relevant_paragraphs_for_page:
                    logger.info(f"Found {len(relevant_paragraphs_for_page)} relevant paragraphs in '{page_title}'.")
                    page_context_contribution = "\n".join(relevant_paragraphs_for_page)
                else:
                    # Fallback: use first N chars of the page if no specific paragraphs found
                    logger.info(f"No specific relevant paragraphs in '{page_title}'. Using initial part of content.")
                    page_context_contribution = page_content_full[:1500] # Fallback to initial part

                # Add to overall context, respecting limits
                # Header for each page's contribution
                content_header = f"Context from '{page_title}' (URL: {page_url}):\n"

                # Check if adding this page's contribution exceeds overall max_context_length
                if current_length + len(content_header) + len(page_context_contribution) > max_context_length:
                    remaining_chars = max_context_length - current_length - len(content_header)
                    if remaining_chars > 0:
                        all_relevant_paragraphs.append(f"{content_header}{page_context_contribution[:remaining_chars]}\n---")
                        current_length += len(content_header) + remaining_chars + len("\n---")
                    logger.info(f"Overall context length limit reached. Truncated content from {page_title}.")
                    break # Stop adding content from further pages
                else:
                    all_relevant_paragraphs.append(f"{content_header}{page_context_contribution}\n---")
                    current_length += len(content_header) + len(page_context_contribution) + len("\n---")

            if all_relevant_paragraphs:
                context_for_gemini = "\n".join(all_relevant_paragraphs)
            elif search_results: # Search results found, but no content could be processed
                context_for_gemini = "Found some documents in Confluence but could not extract relevant segments or content."
                logger.warning("Content extraction from Confluence results yielded no usable segments.")

        logger.info(f"Final context for Gemini (first 300 chars): {context_for_gemini[:300]}...")
        logger.info(f"Total context length for Gemini: {len(context_for_gemini)} characters.")

        ai_answer = gemini_service.generate_response(user_question.question, context_for_gemini)

        if ai_answer.startswith("Error:"):
            logger.error(f"Gemini service returned an error: {ai_answer}")
            return ChatResponse(answer=ai_answer, source_urls=source_urls)

        return ChatResponse(answer=ai_answer, source_urls=source_urls)

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"An unexpected error occurred in /chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@app.get("/health", summary="Health Check", description="Returns the status of the API.")
async def health_check():
    confluence_status = "initialized" if confluence_service.confluence else "error_initializing"
    gemini_status = "initialized" if gemini_service.model else "error_initializing"

    if confluence_service.confluence and gemini_service.model:
        return {"status": "ok", "confluence_service": confluence_status, "gemini_service": gemini_status}
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "confluence_service": confluence_status, "gemini_service": gemini_status, "detail": "One or more critical services are not available."}
        )
