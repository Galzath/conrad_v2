from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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

# --- CORS MIDDLEWARE CONFIGURATION ---
origins = ["*"] # Allows all origins - USE FOR LOCAL DEVELOPMENT ONLY

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
# --- END CORS MIDDLEWARE CONFIGURATION ---

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
    normalized_question_for_keywords = question.lower()
    normalized_question_for_keywords = re.sub(r'[^\w\s-]', '', normalized_question_for_keywords)
    words = normalized_question_for_keywords.split()
    candidate_keywords = [word for word in words if word not in STOP_WORDS and len(word) > 2]
    proper_noun_phrases = re.findall(r'\b[A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+)+\b', question)
    all_potential_phrases = set(proper_noun_phrases)
    for n in range(2, 5): # Include 4-grams
        for i in range(len(words) - n + 1):
            ngram_words = words[i:i+n]
            # Ensure the phrase is not all stop words and at least one word is not a stop word
            if not all(word in STOP_WORDS for word in ngram_words) and \
               any(word not in STOP_WORDS for word in ngram_words):
                all_potential_phrases.add(" ".join(ngram_words))
    sorted_phrases = sorted(list(all_potential_phrases), key=lambda p: (len(p.split()), len(p)), reverse=True)
    selected_phrases = []
    max_phrases = 4 # Increased max_phrases
    for phrase in sorted_phrases:
        is_substring = any(phrase.lower() in sel_phrase.lower() for sel_phrase in selected_phrases)
        if not is_substring and len(phrase.split()) > 1 and len(phrase) > 3:
            selected_phrases.append(phrase)
            if len(selected_phrases) >= max_phrases:
                break
    final_keywords = set(candidate_keywords)
    for phrase_str in selected_phrases:
        phrase_words_lower = set(word.lower() for word in phrase_str.split())
        final_keywords.difference_update(phrase_words_lower)
    final_keywords_list = sorted(list(final_keywords), key=len, reverse=True)[:7] # Increased keyword limit
    search_terms_dict = {"keywords": final_keywords_list, "phrases": selected_phrases}
    logger.info(f"Refined - Extracted search terms: {search_terms_dict}")
    return search_terms_dict

def score_paragraph(paragraph_text: str, phrases: list[str], keywords: list[str]) -> int:
    score = 0
    para_lower = paragraph_text.lower()
    for phrase in phrases:
        if phrase.lower() in para_lower:
            score += 10
    found_keywords_in_para = set()
    for keyword in keywords:
        if keyword.lower() in para_lower:
            found_keywords_in_para.add(keyword.lower())
    score += len(found_keywords_in_para) * 3 # Changed keyword score from 2 to 3
    # Removed debug logging from here for brevity, can be re-added if needed
    return score

# Imports for interactive questioning
import time
import uuid # Though create_new_session_id from state_manager should be used
from typing import List, Dict, Any, Optional # Ensure these are at the top if not already
from .core.state_manager import SessionData, save_session, get_session, delete_session, create_new_session_id

def _clean_title_for_options(title: str, query_terms_lower: List[str]) -> str:
    """Helper to generate a cleaner option text from a title."""
    # Remove common suffixes/prefixes (very basic example)
    common_suffixes = ["guide", "documentation", "overview", "document", "manual", "tutorial"]
    # Attempt to remove parts of the title that were in the original query to find distinguishing parts
    # This is a simplistic approach; more advanced NLP could be used.

    title_words = title.lower().split()
    # Remove query terms from title words to find distinguishing parts
    distinguishing_words = [word for word in title_words if word not in query_terms_lower and word not in STOP_WORDS]

    # Further clean up common document type words
    distinguishing_words = [word for word in distinguishing_words if word not in common_suffixes]

    if not distinguishing_words:
        # If all words were query terms or stop words, fall back to a part of the original title
        # or a simplified version. For now, take last few words if too long.
        # This part can be made more sophisticated.
        # For example, try to identify proper nouns or key terms not in the original query.
        # As a simple fallback, use a snippet of the title.
        # A very basic heuristic: if title contains ':', take text after last ':'
        if ':' in title:
            simplified_title = title.split(':')[-1].strip()
            if len(simplified_title) > 3: # ensure it's not just a letter or two
                 return simplified_title
        # Fallback to using the original title if no better simplification found or too short
        # Or, return a more structured part of the title if possible.
        # For now, let's just use a simple approach if no distinguishing words found:
        # Return the original title if it's not too generic by itself.
        # This part needs refinement.
        # A simple heuristic: take the last 3-4 words if the title is long.
        # For now, if no distinguishing words, use a simplified title or the original.
        # Let's try to be a bit more intelligent: if the title has a colon, often the part after is specific.
        parts = title.split(':')
        if len(parts) > 1 and len(parts[-1].strip()) > 3 : # Check if there's meaningful text after colon
            return parts[-1].strip().capitalize()

        # If still nothing good, return a capitalized version of first few distinguishing words if any
        # This part of the heuristic is tricky. The goal is to get a concise, descriptive option.
        # For now, let's keep it simple: if the above didn't yield a good result,
        # return the original title, hoping it's distinct enough.
        # This cleaning is the hardest part of the heuristic.
        # Let's try a simpler approach: remove query terms and stop words, then join remaining.
        # If that is empty, use the part of title after a colon if present.
        # If still empty, use the original title.

        # Let's refine the distinguishing words logic slightly.
        # Try to remove query terms from the title and see what's left.
        temp_title = title
        for term in query_terms_lower:
            temp_title = temp_title.lower().replace(term, "").strip()

        # Remove common punctuation that might be left over
        temp_title = re.sub(r'[:\-]+', ' ', temp_title).strip()
        temp_title = ' '.join(temp_title.split()) # Normalize spaces

        if len(temp_title) > 3 and len(temp_title.split()) <= 5: # Keep it concise
            return temp_title.capitalize()

        # Fallback: if title is "A B C" and query was "A", return "B C"
        # This is effectively covered by the replace logic above if query terms are single words.
        # For multi-word query terms, the simple replace might not be ideal.

        # Fallback to original title if cleaning doesn't produce a good result
        # Or try to extract capitalized words not in query as potential entities
        capitalized_words = [word for word in title.split() if word.istitle() and word.lower() not in query_terms_lower and word.lower() not in STOP_WORDS]
        if capitalized_words:
            return " ".join(capitalized_words)

        return title # Fallback if no better option found by cleaning

    return " ".join(distinguishing_words).capitalize()


def generate_clarification_from_results(
    search_results: List[Dict[str, Any]],
    original_query_terms: List[str]
) -> Optional[Dict[str, Any]]:

    if not search_results or len(search_results) < 2:
        return None

    query_terms_lower = [term.lower() for term in original_query_terms]

    candidate_options = []
    titles_processed = set() # To avoid processing the exact same title multiple times if results are redundant

    for result in search_results:
        title = result.get("title")
        page_id = result.get("id")

        if not title or not page_id or title in titles_processed:
            continue

        titles_processed.add(title)

        # Simplified: try to extract a distinguishing part of the title.
        # This is a placeholder for a more robust "theme extraction".
        # The _clean_title_for_options is a starting point.
        # We want the option text to be what's *different* or *specific* about this title.

        option_text = _clean_title_for_options(title, query_terms_lower)

        # Ensure option_text is reasonably unique and not just a generic term
        # This check needs to be more robust. For now, check if it's not empty and not too short.
        if option_text and len(option_text) > 2 and option_text.lower() not in query_terms_lower:
            # Check for uniqueness among already added options (by text)
            is_duplicate_option_text = any(opt['text'].lower() == option_text.lower() for opt in candidate_options)
            if not is_duplicate_option_text:
                 candidate_options.append({"id": page_id, "text": option_text, "original_title": title})

    # Filter options to ensure they are distinct enough.
    # The previous check for duplicate option_text helps.
    # Now, ensure we have a good number of options.

    if not candidate_options:
        return None

    # Simplistic way to ensure diversity: if many options have very similar text, group them or pick best.
    # For now, we rely on the _clean_title_for_options and the duplicate check.

    # Limit to 2-4 distinct options as per requirement
    # If we have many candidates, try to pick the most distinct ones.
    # This part could also involve scoring options based on how distinct they are.
    # For now, just take the first few unique ones if many are generated.

    final_options = []
    seen_option_texts = set()
    for option in candidate_options:
        if len(final_options) < 4:
            # The check for duplicate option text was already done during candidate_options.append
            # but if _clean_title_for_options generates identical text for different original titles,
            # we need to ensure we don't offer identical choices.
            # The earlier check `is_duplicate_option_text` should handle this for `option_text`.
            final_options.append({"id": option["id"], "text": option["text"]}) # Keep only id and text for final output
        else:
            break

    if len(final_options) >= 2 and len(final_options) <= 4:
        # Create a summary of original query terms for the question text
        # Take first 2-3 terms for brevity
        query_context = " ".join(original_query_terms[:3])
        if len(original_query_terms) > 3:
            query_context += "..."

        question_text = f"Your query about '{query_context}' returned information on different topics. Which specific area are you interested in?"

        return {
            "question_text": question_text,
            "options": final_options
        }

    return None


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(user_question: UserQuestion = Body(...)):
    logger.info(f"Received payload: {user_question.model_dump_json(indent=2)}")

    if not confluence_service.confluence or not gemini_service.model:
        logger.error("Backend services unavailable: Confluence or Gemini failed to initialize.")
        raise HTTPException(status_code=503, detail="Uno o más servicios de backend no están disponibles.")

    max_context_length = 15000
    current_session_id = user_question.session_id
    proceed_as_new_question = True # Flag to control flow

    try: # ADDED TRY HERE
        # A. Handling Follow-up Request (User's answer to clarification)
        if current_session_id and (user_question.clarification_response or user_question.selected_option_id):
            logger.info(f"Handling follow-up for session ID: {current_session_id}")
            retrieved_session_data = get_session(current_session_id)

            if retrieved_session_data and retrieved_session_data.conversation_step == "awaiting_clarification_response":
                logger.info(f"Valid session found for {current_session_id}. Processing clarification response.")
                proceed_as_new_question = False # We are processing a clarification response

                original_question_text = retrieved_session_data.original_question
                search_results_from_session = retrieved_session_data.initial_search_results

                refined_search_results = []

                if user_question.selected_option_id:
                    logger.info(f"User selected option ID: {user_question.selected_option_id}")
                    for result in search_results_from_session:
                        if result.get("id") == user_question.selected_option_id:
                            refined_search_results.append(result)
                            logger.info(f"Prioritizing result based on selected_option_id: {result.get('title')}")
                            break # Found the selected option
                # else if user_question.clarification_response:
                    # TODO: Implement free-text clarification response handling (more complex)
                    # For now, if no selected_option_id, we might just use all initial_search_results
                    # or try a naive filtering based on the text.
                    # logger.info(f"User provided free-text clarification: {user_question.clarification_response}")
                    # This part is complex: requires re-evaluating results or new search.
                    # For this version, we'll fall back to using initial results if no option selected.

                if not refined_search_results: # Fallback if selected_option_id didn't yield a specific result
                    logger.warning(f"No specific result refined for session {current_session_id}, using all initial results from session.")
                    refined_search_results = search_results_from_session

                # Assemble context from these (potentially refined) search_results
                search_terms_from_session = retrieved_session_data.extracted_terms
                context_for_gemini, unique_source_urls = _build_context_from_search_results(
                    refined_search_results,
                    search_terms_from_session.get("phrases", []),
                    search_terms_from_session.get("keywords", []),
                    max_context_length
                )

                if not context_for_gemini or context_for_gemini.startswith("No se encontró información"):
                     logger.warning(f"Context for Gemini is empty or default after clarification for session {current_session_id}.")
                     # Provide a more specific message if context is still poor after clarification
                     context_for_gemini = "Aunque intentamos aclarar, no se encontró información específica en Confluence para su consulta refinada."


                ai_answer = gemini_service.generate_response(original_question_text, context_for_gemini)
                delete_session(current_session_id)
                logger.info(f"Session {current_session_id} deleted after processing clarification.")
                return ChatResponse(answer=ai_answer, source_urls=unique_source_urls, session_id=None) # Session is complete

            else:
                if retrieved_session_data:
                    logger.warning(f"Session {current_session_id} found, but not awaiting clarification (step: {retrieved_session_data.conversation_step}). Treating as new question.")
                    delete_session(current_session_id) # Clean up invalid state session
                else:
                    logger.warning(f"Session ID {current_session_id} provided by user, but no active session found. Treating as new question.")
                # proceed_as_new_question remains True

        # B. Handling New Question (or if follow-up processing decided to treat as new)
        if proceed_as_new_question:
            logger.info(f"Processing as new question: {user_question.question}")
            extracted_terms = extract_search_terms(user_question.question)
            extracted_space_keys = extract_space_keys_from_query(user_question.question)
            
            logger.info(f"Searching Confluence with terms: {extracted_terms}, space_keys: {extracted_space_keys}")
            initial_search_results = confluence_service.search_content(
                search_terms=extracted_terms,
                space_keys=extracted_space_keys if extracted_space_keys else None,
                limit=10 # Increased limit slightly for better clarification options
            )

            if initial_search_results:
                # Attempt to generate clarification questions
                combined_query_terms = list(extracted_terms.get("keywords", [])) + list(extracted_terms.get("phrases", []))
                clarification_details = generate_clarification_from_results(initial_search_results, combined_query_terms)

                if clarification_details:
                    new_session_id = create_new_session_id()
                    logger.info(f"Clarification needed. Creating new session: {new_session_id}")

                    session_to_save = SessionData(
                        original_question=user_question.question,
                        extracted_terms=extracted_terms, # Store the dict directly
                        initial_search_results=initial_search_results, # Store raw results
                        clarification_question_asked=clarification_details["question_text"],
                        clarification_options_provided=clarification_details["options"],
                        conversation_step="awaiting_clarification_response",
                        timestamp=time.time()
                    )
                    save_session(new_session_id, session_to_save)

                    return ChatResponse(
                        answer="", # No direct answer when clarification is needed
                        session_id=new_session_id,
                        needs_clarification=True,
                        clarification_question_text=clarification_details["question_text"],
                        clarification_options=clarification_details["options"]
                    )

            # If no clarification needed, or no search results, proceed to generate response directly
            context_for_gemini, unique_source_urls = _build_context_from_search_results(
                initial_search_results if initial_search_results else [],
                extracted_terms.get("phrases", []),
                extracted_terms.get("keywords", []),
                max_context_length
            )
            
            ai_answer = gemini_service.generate_response(user_question.question, context_for_gemini)
            # No session_id in response if it's a direct answer to a new question without clarification
            return ChatResponse(answer=ai_answer, source_urls=unique_source_urls, session_id=None)

    # Fallback for any logic error, though ideally all paths are covered.
    except HTTPException as http_exc: # ALIGNED EXCEPTION BLOCK
        raise http_exc # Re-raise to let FastAPI handle it
    except Exception as e: # ALIGNED EXCEPTION BLOCK
        logger.error(f"An unexpected error occurred in /chat: {e}", exc_info=True)
        # If a session was created for clarification but an error occurred before returning, delete it.
        # This is a bit broad; might need more specific error handling if a session_id was generated in this try block.
        # For now, this is a general cleanup.
        # if 'new_session_id' in locals() and get_session(new_session_id):
        #     delete_session(new_session_id)
        #     logger.info(f"Session {new_session_id} deleted due to an error during processing.")
        raise HTTPException(status_code=500, detail="Error interno del servidor.")

# Helper function to consolidate context building logic (refactored from original)
def _build_context_from_search_results(
    search_results: List[Dict[str, Any]],
    query_phrases: List[str],
    query_keywords: List[str],
    max_context_length: int
) -> tuple[str, List[str]]:

    context_for_gemini = "No se encontró información específica en Confluence que coincida bien con su consulta." # Default
    unique_source_urls = []

    if not search_results:
        return context_for_gemini, unique_source_urls

    raw_source_urls = [result['url'] for result in search_results if result.get('url')]
    unique_source_urls = sorted(list(dict.fromkeys(raw_source_urls)))

    scored_paragraphs_with_source = []
    for result in search_results:
        page_id = result.get("id")
        if not page_id: continue

        page_content_full = confluence_service.get_page_content_by_id(page_id)
        if not page_content_full:
            logger.warning(f"No content for page ID: {page_id}, title: {result.get('title')}")
            continue

        page_title = result.get('title', 'Página Desconocida')
        page_url = result.get('url', 'N/A')
        paragraphs = page_content_full.split('\n\n')
        if len(paragraphs) <= 1 and '\n' in page_content_full:
            paragraphs = page_content_full.split('\n')

        for para_text in paragraphs:
            if not para_text.strip(): continue
            score = score_paragraph(para_text, query_phrases, query_keywords)
            if score > 0:
                scored_paragraphs_with_source.append((score, para_text, page_title, page_url))

    if scored_paragraphs_with_source:
        scored_paragraphs_with_source.sort(key=lambda x: x[0], reverse=True)
        logger.info(f"Found {len(scored_paragraphs_with_source)} relevant paragraphs. Top scores: {[s[0] for s in scored_paragraphs_with_source[:3]]}")

        selected_context_parts = []
        current_length = 0

        for score, para_text, p_title, p_url in scored_paragraphs_with_source:
            content_header = f"Contexto de '{p_title}' (URL: {p_url}, Puntuación de Relevancia: {score}):\n"
            full_snippet = f"{content_header}{para_text}\n---"

            if current_length + len(full_snippet) <= max_context_length:
                selected_context_parts.append(full_snippet)
                current_length += len(full_snippet)
            else:
                remaining_space = max_context_length - current_length
                if remaining_space > len(content_header) + 50: # Ensure space for header and some text
                    truncated_para_text_len = remaining_space - (len(content_header) + len("\n---") + 3) # +3 for "..."
                    if truncated_para_text_len > 0:
                        truncated_snippet = f"{content_header}{para_text[:truncated_para_text_len]}...\n---"
                        selected_context_parts.append(truncated_snippet)
                break

        if selected_context_parts:
            context_for_gemini = "\n".join(selected_context_parts)
        # Fallback if no scored paragraphs fit but results exist (covered by initial default for context_for_gemini)
        # If selected_context_parts is empty but search_results were there, and scored_paragraphs_with_source was also empty
        # this means no paragraph scored > 0. The initial default message for context_for_gemini will be used.

    # Fallback if no paragraphs scored > 0 from any document, but search_results existed.
    # Use content from the very first search result as a last resort.
    elif search_results: # scored_paragraphs_with_source was empty
        logger.info("No paragraphs scored > 0 from any search result. Using initial content of first search result as fallback.")
        first_result_id = search_results[0].get("id")
        if first_result_id:
            first_page_content_fallback = confluence_service.get_page_content_by_id(first_result_id)
            if first_page_content_fallback:
                context_for_gemini = f"Contexto de '{search_results[0].get('title', 'Título Desconocido')}':\n{first_page_content_fallback[:max_context_length*2//3]}"

    return context_for_gemini, unique_source_urls

@app.get("/health", summary="Health Check")
async def health_check():
    confluence_status = "initialized" if confluence_service.confluence else "error_initializing"
    gemini_status = "initialized" if gemini_service.model else "error_initializing"
    if confluence_service.confluence and gemini_service.model:
        return {"status": "ok", "confluence_service": confluence_status, "gemini_service": gemini_status}
    else:
        return JSONResponse(
            status_code=503, 
            content={
                "status": "error", 
                "confluence_service": confluence_status, 
                "gemini_service": gemini_status, 
                "detail": "Uno o más servicios de backend no están completamente disponibles." # Translated
            }
        )