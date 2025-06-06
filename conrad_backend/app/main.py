from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
import logging

from .schemas import UserQuestion, ChatResponse
from .services.confluence_service import ConfluenceService
from .services.gemini_service import GeminiService
from .core.config import settings # To ensure config is loaded, though not directly used here

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Conrad Chatbot API",
    description="API for Conrad Chatbot using Confluence and Gemini",
    version="0.1.0"
)

# Initialize services
# These should be singletons or managed by a dependency injection system in a larger app
# For simplicity, we initialize them here.
confluence_service = ConfluenceService()
gemini_service = GeminiService()

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup...")
    if not confluence_service.confluence:
        logger.error("Confluence service failed to initialize. Check Confluence settings and connectivity.")
        # Depending on requirements, you might want to prevent startup or allow degraded functionality.
    if not gemini_service.model:
        logger.error("Gemini service failed to initialize. Check Gemini API Key.")
        # Depending on requirements, you might want to prevent startup or allow degraded functionality.
    logger.info("Confluence and Gemini services initialized (or attempted).")
    logger.info(f"Confluence URL configured: {settings.CONFLUENCE_URL}")
    if "YOUR_CONFLUENCE_API_TOKEN" in settings.CONFLUENCE_API_TOKEN or not settings.CONFLUENCE_API_TOKEN:
        logger.warning("Confluence API token appears to be a placeholder or is not set.")
    if "YOUR_GEMINI_API_KEY" in settings.GEMINI_API_KEY or not settings.GEMINI_API_KEY:
        logger.warning("Gemini API key appears to be a placeholder or is not set.")


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(user_question: UserQuestion = Body(...)):
    logger.info(f"Received question: {user_question.question}")

    if not confluence_service.confluence:
        logger.error("Confluence client not available.")
        raise HTTPException(status_code=503, detail="Confluence service is unavailable.")

    if not gemini_service.model:
        logger.error("Gemini model not available.")
        raise HTTPException(status_code=503, detail="Gemini AI service is unavailable.")

    try:
        # 1. Search Confluence
        logger.info(f"Searching Confluence for: {user_question.question}")
        search_results = confluence_service.search_content(user_question.question, limit=3) # Limit to 3 results for context

        if not search_results:
            logger.info("No relevant documents found in Confluence.")
            # Decide how to handle: directly tell Gemini or return a specific message
            # For now, let Gemini handle it with an empty context.
            context_for_gemini = "No specific information found in Confluence regarding this query."
            source_urls = []
        else:
            logger.info(f"Found {len(search_results)} results. Fetching content...")
            source_urls = [result['url'] for result in search_results]

            # 2. Extract content from Confluence pages
            # For simplicity, concatenate content from all found pages.
            # Consider token limits for Gemini.
            context_parts = []
            max_context_length = 15000 # Rough character limit to stay within Gemini token limits (e.g. 32k tokens ~ 120k chars, be conservative)
            current_length = 0

            for result in search_results:
                page_id = result.get("id")
                if page_id:
                    page_content = confluence_service.get_page_content_by_id(page_id)
                    if page_content:
                        if current_length + len(page_content) > max_context_length:
                            remaining_chars = max_context_length - current_length
                            if remaining_chars > 0: # Only add if there's space
                                context_parts.append(f"Content from {result.get('title', 'Unknown Page')}:\n{page_content[:remaining_chars]}\n---")
                            logger.info(f"Context length limit reached. Truncated content from {result.get('title')}.")
                            break
                        context_parts.append(f"Content from {result.get('title', 'Unknown Page')}:\n{page_content}\n---")
                        current_length += len(page_content) + len(f"Content from {result.get('title', 'Unknown Page')}:\n\n---")

            if not context_parts:
                context_for_gemini = "Found some documents in Confluence but could not extract their content."
                logger.warning("Content extraction from Confluence results failed.")
            else:
                context_for_gemini = "\n".join(context_parts)

        logger.info(f"Context for Gemini (first 300 chars): {context_for_gemini[:300]}...")

        # 3. Interact with Gemini
        logger.info("Sending request to Gemini service...")
        ai_answer = gemini_service.generate_response(user_question.question, context_for_gemini)

        if ai_answer.startswith("Error:"):
             # Propagate specific errors from Gemini service if needed, or return a generic one
            logger.error(f"Gemini service returned an error: {ai_answer}")
            # Return 200 with error in body, or use HTTPException for client/server errors
            return ChatResponse(answer=ai_answer, source_urls=source_urls)


        logger.info(f"Received answer from Gemini: {ai_answer[:100]}...")
        return ChatResponse(answer=ai_answer, source_urls=source_urls)

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions to let FastAPI handle them
        raise http_exc
    except Exception as e:
        logger.error(f"An unexpected error occurred in /chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@app.get("/health", summary="Health Check", description="Returns the status of the API.")
async def health_check():
    # Basic health check, can be expanded to check service connectivity
    # For now, just check if services were initialized (not if they are currently working)
    # More robust checks would ping Confluence/Gemini or check their status.
    confluence_status = "initialized" if confluence_service.confluence else "error_initializing"
    gemini_status = "initialized" if gemini_service.model else "error_initializing"

    if confluence_service.confluence and gemini_service.model:
        return {"status": "ok", "confluence_service": confluence_status, "gemini_service": gemini_status}
    else:
        # Return 503 if critical services are not initialized
        return JSONResponse(
            status_code=503,
            content={"status": "error", "confluence_service": confluence_status, "gemini_service": gemini_status, "detail": "One or more critical services are not available."}
        )

# To run the app (example, usually done via uvicorn command):
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
