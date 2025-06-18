from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import re
from collections import Counter
import time # Ensure time is imported

from .schemas import UserQuestion, ChatResponse
from .services.confluence_service import ConfluenceService
from .services.gemini_service import GeminiService
from .core.config import settings

# Fallback settings values if not defined in core.config
if not hasattr(settings, 'MAX_CONTEXT_LENGTH_GEMINI'):
    settings.MAX_CONTEXT_LENGTH_GEMINI = 15000
if not hasattr(settings, 'N_SEMANTIC_RESULTS'):
    settings.N_SEMANTIC_RESULTS = 5
if not hasattr(settings, 'N_CQL_RESULTS'):
    settings.N_CQL_RESULTS = 3
if not hasattr(settings, 'K_CANDIDATES_FOR_RERANKING'):
    settings.K_CANDIDATES_FOR_RERANKING = 25
# MAX_RESULTS_FOR_CONTEXT calculation
if hasattr(settings, 'K_CANDIDATES_FOR_RERANKING'):
    if not hasattr(settings, 'MAX_RESULTS_FOR_CONTEXT'):
        settings.MAX_RESULTS_FOR_CONTEXT = settings.K_CANDIDATES_FOR_RERANKING + 10
else:
    if not hasattr(settings, 'MAX_RESULTS_FOR_CONTEXT'):
        settings.MAX_RESULTS_FOR_CONTEXT = 30

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

# Ensure List, Dict, Any, Optional are imported if not already at the top
from typing import List, Dict, Any, Optional # This line is fine
from sentence_transformers import CrossEncoder
import numpy as np # numpy is used for cross_encoder scores potentially

# (Existing service imports and logger setup remain here)
# ...

# --- Initialize Services and Models ---
# (Existing ConfluenceService and GeminiService initializations remain here)
# ...

# Load Cross-Encoder Model
# Moved this section up to be before helper functions, as it's an initialization step.
# logger.info("Loading Cross-Encoder model...") # Already exists
# try:
#     cross_encoder_model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
#     cross_encoder = CrossEncoder(cross_encoder_model_name)
#     logger.info(f"Cross-Encoder model '{cross_encoder_model_name}' loaded successfully.")
# except Exception as e:
#     logger.error(f"Failed to load Cross-Encoder model '{cross_encoder_model_name}': {e}", exc_info=True)
#     cross_encoder = None
# The above block is effectively moved, so no direct replacement here, but ensuring the new structure is okay.
# The original block is fine where it is if not explicitly moved by other changes.
# For now, I will assume this block stays and the new chat endpoint is inserted later.
# The full overwrite attempted to move it. A targeted diff won't.

logger.info("Loading Cross-Encoder model...")
try:
    cross_encoder_model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    cross_encoder = CrossEncoder(cross_encoder_model_name)
    logger.info(f"Cross-Encoder model '{cross_encoder_model_name}' loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Cross-Encoder model '{cross_encoder_model_name}': {e}", exc_info=True)
    cross_encoder = None

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

# logger is defined globally at the top of main.py

def _clean_title_for_options(title: str, query_terms_lower: List[str]) -> str:
    """Helper to generate a cleaner option text from a title."""
    logger.debug(f"_clean_title_for_options: Input title='{title}', query_terms_lower={query_terms_lower}")
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

        cleaned_text_being_returned = title # Fallback if no better option found by cleaning
        logger.debug(f"_clean_title_for_options: Output cleaned_text='{cleaned_text_being_returned}'")
        return cleaned_text_being_returned

    cleaned_text_being_returned = " ".join(distinguishing_words).capitalize()
    logger.debug(f"_clean_title_for_options: Output cleaned_text='{cleaned_text_being_returned}'")
    return cleaned_text_being_returned


def generate_clarification_from_results(
    search_results_for_clarification: List[Dict[str, Any]], # List of ALL initial chunks/results
    original_query_text: str, # Changed from original_query_terms
    gemini_service_instance: GeminiService # Pass the instance
) -> Optional[Dict[str, Any]]:
    logger.debug(f"generate_clarification_from_results: Input search_results_count={len(search_results_for_clarification)}")

    if not search_results_for_clarification or len(search_results_for_clarification) < 1:
        logger.debug("generate_clarification_from_results: Not enough results to generate clarification options.")
        return None

    unique_pages_for_clarification = []
    seen_page_ids_for_clarification = set()
    for res_chunk in search_results_for_clarification:
        page_id = res_chunk.get("page_id")
        if page_id and res_chunk.get("title") and page_id not in seen_page_ids_for_clarification:
            unique_pages_for_clarification.append({
                "id": page_id,
                "title": res_chunk.get("title"),
                "url": res_chunk.get("url") # Keep url if needed, though not directly used for option text
            })
            seen_page_ids_for_clarification.add(page_id)

    if len(unique_pages_for_clarification) < 2: # Need at least 2 distinct pages for clarification
       logger.debug("Not enough unique pages from results to generate clarification.")
       return None

    candidate_options = []
    # Limit the number of pages to generate summaries for, to avoid too many LLM calls
    MAX_PAGES_FOR_SUMMARY_CLARIFICATION = 4
    for page_candidate in unique_pages_for_clarification[:MAX_PAGES_FOR_SUMMARY_CLARIFICATION]:
        page_id = page_candidate["id"]
        page_title = page_candidate["title"]

        chunks_for_this_page = [
            chk.get("text", "") for chk in search_results_for_clarification
            if chk.get("page_id") == page_id and chk.get("text")
        ]

        if not chunks_for_this_page:
            logger.warning(f"No text chunks found for page ID {page_id} ('{page_title}') for summary generation.")
            option_text = page_title # Fallback to title
        else:
            # Concatenate top 1 or 2 chunks (or more, up to a limit)
            # Assuming search_results_for_clarification is already somewhat relevance-sorted for these top chunks.
            # Or, if not sorted, it takes the first few found.
            # Let's take up to ~500 words for summary context.
            context_text_for_summary = ""
            char_limit_for_summary_context = 1500 # Approx 500 words
            for chunk_text in chunks_for_this_page:
                if len(context_text_for_summary) + len(chunk_text) > char_limit_for_summary_context:
                    needed = char_limit_for_summary_context - len(context_text_for_summary)
                    if needed > 50: # Add partial if space allows
                         context_text_for_summary += "\n" + chunk_text[:needed] + "..."
                    break
                context_text_for_summary += "\n" + chunk_text
            context_text_for_summary = context_text_for_summary.strip()

            logger.info(f"Generating summary for page ID {page_id} ('{page_title}') for clarification.")
            summary = gemini_service_instance.generate_clarification_summary(original_query_text, context_text_for_summary)
            option_text = summary if summary and "Error" not in summary and "no disponible" not in summary and "No se pudo" not in summary else page_title

        candidate_options.append({"id": page_id, "text": option_text, "original_title": page_title})

        # Stop if we have enough good options, even if more unique_pages_for_clarification exist
        if len(candidate_options) >= 4:
            break

    final_options = []
    seen_option_texts = set()
    for option in candidate_options:
        if len(final_options) < 4: # Max 4 distinct text options
            # Ensure option text is not just the title if summary failed badly or was identical to title
            # And ensure it's somewhat unique to avoid identical clarification choices
            option_display_text = option['text']
            if option_display_text.lower() == option['original_title'].lower() and option_display_text != option['original_title']:
                 option_display_text = option['original_title'] # Prefer original casing if identical after lowercasing

            if option_display_text.lower() not in seen_option_texts:
                final_options.append({"id": option["id"], "text": option_display_text})
                seen_option_texts.add(option_display_text.lower())
        else:
            break

    if len(final_options) < 2: # Still need at least 2 distinct options
        logger.debug("generate_clarification_from_results: Not enough distinct final options after summary generation.")
        return None

    logger.debug(f"generate_clarification_from_results: Final options selected: {final_options}")

    if len(final_options) >= 2 and len(final_options) <= 4: # Check again, as some summaries might have failed
        # Use original_query_text for context in the clarification question
        query_context_words = original_query_text.split()
        query_context = " ".join(query_context_words[:7]) # First 7 words
        if len(query_context_words) > 7:
            query_context += "..."

        question_text = f"Tu consulta sobre '{query_context}' arrojó información sobre diferentes temas. ¿En qué área específica estás interesado/a?"

        clarification_dict_to_return = {
            "question_text": question_text,
            "options": final_options
        }
        logger.debug(f"generate_clarification_from_results: Returning clarification_details: {clarification_dict_to_return}")
        return clarification_dict_to_return

    logger.debug("generate_clarification_from_results: No clarification generated, returning None (final options count not met).")
    return None


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(user_question: UserQuestion = Body(...)):
    logger.info(f"Received payload: {user_question.model_dump_json(indent=2)}")

    if not confluence_service.confluence or not gemini_service.model:
        logger.error("Backend services unavailable: Confluence or Gemini failed to initialize.")
        raise HTTPException(status_code=503, detail="Uno o más servicios de backend no están disponibles.")

    max_context_length = settings.MAX_CONTEXT_LENGTH_GEMINI
    current_session_id = user_question.session_id
    original_query_from_user = user_question.question

    session_data: Optional[SessionData] = None
    if current_session_id:
        session_data = get_session(current_session_id)
        if not session_data:
            logger.warning(f"Session ID {current_session_id} provided, but no active session found. Treating as new.")
            current_session_id = None

    try:
        if session_data and user_question.selected_option_id:
            logger.info(f"Processing selection for session {current_session_id}, option: {user_question.selected_option_id}, type: {session_data.clarification_type}")
            original_query_for_processing = session_data.original_question

            if session_data.clarification_type == "space_selection":
                session_data.selected_space_key = user_question.selected_option_id
                session_data.conversation_step = "processing_space_selection"
                logger.info(f"Session {current_session_id}: Space '{session_data.selected_space_key}' selected.")
                save_session(current_session_id, session_data)

            elif session_data.clarification_type == "document_selection":
                logger.info(f"Session {current_session_id}: Document ID '{user_question.selected_option_id}' selected for answer.")
                search_results_from_session = session_data.initial_search_results or []
                refined_search_results = [res for res in search_results_from_session if res.get("page_id") == user_question.selected_option_id]
                if not refined_search_results:
                    logger.warning(f"Selected document {user_question.selected_option_id} not found in session results for {current_session_id}. Using all.")
                    refined_search_results = search_results_from_session
                context_for_gemini, unique_source_urls = _build_context_from_search_results(
                    refined_search_results, session_data.extracted_terms.get("phrases", []),
                    session_data.extracted_terms.get("keywords", []), max_context_length)
                ai_answer = gemini_service.generate_response(original_query_for_processing, context_for_gemini)
                remaining_topics_options = []
                if session_data.initial_search_results:
                    seen_page_ids = {user_question.selected_option_id}
                    for item in session_data.initial_search_results:
                        if item.get("page_id") and item.get("page_id") not in seen_page_ids:
                            remaining_topics_options.append({"id": item["page_id"], "text": item.get("title", "Tema relacionado")})
                            seen_page_ids.add(item["page_id"])
                            if len(remaining_topics_options) >= 3: break
                if remaining_topics_options:
                    session_data.clarification_type = "related_topics_followup"
                    session_data.conversation_step = "awaiting_related_topic_choice"
                    session_data.clarification_options_provided = remaining_topics_options
                    save_session(current_session_id, session_data)
                    return ChatResponse(
                        answer=ai_answer, source_urls=unique_source_urls, session_id=current_session_id,
                        needs_clarification=True,
                        clarification_question_text="Aquí está tu respuesta. ¿Te gustaría explorar alguno de estos temas relacionados o hacer una nueva pregunta?",
                        clarification_options=[{"id": opt["id"], "text": opt["text"]} for opt in remaining_topics_options] + [{"id": "new_query", "text": "Hacer una nueva pregunta"}],
                        clarification_type="related_topics_followup")
                else:
                    delete_session(current_session_id)
                    return ChatResponse(answer=ai_answer, source_urls=unique_source_urls, session_id=None)

            elif session_data.clarification_type == "broaden_search_scope":
                selected_broaden_option = user_question.selected_option_id
                if selected_broaden_option == "search_all":
                    session_data.selected_space_key = None
                    session_data.conversation_step = "processing_broad_search"
                    logger.info(f"Session {current_session_id}: Broadening search to all spaces.")
                    save_session(current_session_id, session_data)
                elif selected_broaden_option == "select_other_space":
                    if not session_data.available_spaces:
                         session_data.available_spaces = [{"id": sp["id"], "text": sp["text"]} for sp in confluence_service.get_available_spaces()]
                    session_data.clarification_type = "space_selection"
                    session_data.conversation_step = "awaiting_space_selection"
                    session_data.selected_space_key = None
                    save_session(current_session_id, session_data)
                    return ChatResponse(
                        answer="", needs_clarification=True, session_id=current_session_id,
                        clarification_question_text="Por favor, elige el espacio de Confluence donde te gustaría buscar:",
                        clarification_options=[{"id": sp["id"], "text": sp["text"]} for sp in session_data.available_spaces or []],
                        clarification_type="space_selection")
                elif selected_broaden_option == "new_query":
                    delete_session(current_session_id)
                    current_session_id = None
                    session_data = None
                else: # "cancel" or unknown
                    delete_session(current_session_id)
                    return ChatResponse(answer="Entendido. Si tienes otra pregunta, no dudes en consultarme.", session_id=None)
            
            elif session_data.clarification_type == "related_topics_followup":
                if user_question.selected_option_id == "new_query":
                    delete_session(current_session_id)
                    current_session_id = None
                    session_data = None
                else:
                    page_id_for_related_topic = user_question.selected_option_id
                    logger.info(f"Session {current_session_id}: User selected related topic ID '{page_id_for_related_topic}'.")
                    page_content = confluence_service.get_page_content_by_id(page_id_for_related_topic)
                    page_title = "este tema"
                    page_url = ""
                    if session_data.clarification_options_provided:
                        for opt in session_data.clarification_options_provided:
                            if opt["id"] == page_id_for_related_topic:
                                page_title = opt["text"]; break
                    if session_data.initial_search_results:
                        for res in session_data.initial_search_results:
                            if res.get("page_id") == page_id_for_related_topic:
                                page_url = res.get("url", "")
                                if page_title == "este tema" and res.get("title"): page_title = res.get("title")
                                break
                    context_for_gemini = f"Contexto de la página '{page_title}' (URL: {page_url if page_url else 'N/A'}):\n{page_content}"
                    ai_answer = gemini_service.generate_response(f"Proporciona información detallada sobre {page_title}", context_for_gemini)
                    delete_session(current_session_id)
                    return ChatResponse(answer=ai_answer, source_urls=[page_url] if page_url else [], session_id=None)
            else:
                logger.warning(f"Session {current_session_id} in unknown state: {session_data.clarification_type}, {session_data.conversation_step}. Treating as new.")
                if current_session_id: delete_session(current_session_id)
                current_session_id = None; session_data = None

        if not current_session_id or \
           (session_data and session_data.conversation_step in ["processing_space_selection", "processing_broad_search"]):
            if not current_session_id:
                logger.info(f"Processing as new question: {original_query_from_user}")
                current_session_id = create_new_session_id()
                available_spaces_raw = confluence_service.get_available_spaces()
                available_spaces_for_session = [{"id": sp["id"], "text": sp["text"]} for sp in available_spaces_raw]

                session_data = SessionData(
                    original_question=original_query_from_user, extracted_terms={}, initial_search_results=[],
                    available_spaces=available_spaces_for_session, clarification_type="space_selection",
                    conversation_step="awaiting_space_selection", timestamp=time.time())
                save_session(current_session_id, session_data)
                return ChatResponse(
                    answer="", needs_clarification=True, session_id=current_session_id,
                    clarification_question_text="Hola! Soy Conrad. Para ayudarte mejor, por favor, elige el espacio de Confluence donde te gustaría buscar:",
                    clarification_options=available_spaces_for_session, clarification_type="space_selection")

            original_query_for_processing = session_data.original_question
            current_selected_space_key = session_data.selected_space_key
            logger.info(f"Session {current_session_id}: Performing search. Query: '{original_query_for_processing}', Space: {current_selected_space_key if current_selected_space_key else 'ALL'}")
            extracted_terms = extract_search_terms(original_query_for_processing)
            session_data.extracted_terms = extracted_terms
            semantic_chunks = confluence_service.semantic_search_chunks(query_text=original_query_for_processing, top_k=settings.N_SEMANTIC_RESULTS)
            keyword_pages = confluence_service.search_content(
                search_terms=extracted_terms, space_keys=[current_selected_space_key] if current_selected_space_key else None,
                limit=settings.N_CQL_RESULTS)

            combined_results_for_ranking = []
            processed_page_urls_from_semantic = set()
            for chunk in semantic_chunks:
                combined_results_for_ranking.append({
                    "text": chunk["text"], "url": chunk["url"], "title": chunk["title"], "score": chunk["score"],
                    "search_method": "semantic", "context_hierarchy": chunk.get("context_hierarchy", chunk["title"]),
                    "page_id": chunk.get("page_id")})
                if chunk["url"]: processed_page_urls_from_semantic.add(chunk["url"])
            for page in keyword_pages:
                if page["url"] not in processed_page_urls_from_semantic:
                    page_content_full = confluence_service.get_page_content_by_id(page['id'])
                    if page_content_full:
                        combined_results_for_ranking.append({
                            "text": page_content_full, "url": page["url"], "title": page["title"], "score": 1000.0,
                            "search_method": "keyword_page", "context_hierarchy": page["title"], "page_id": page['id']})
            combined_results_for_ranking.sort(key=lambda x: x.get("score", float('inf')))
            reranked_search_results = combined_results_for_ranking
            if cross_encoder and combined_results_for_ranking:
                MAX_TEXT_LEN_FOR_CROSS_ENCODER = 2000
                pairs = []
                candidates_for_reranking = combined_results_for_ranking[:settings.K_CANDIDATES_FOR_RERANKING]
                for res in candidates_for_reranking:
                    text_for_reranking = res.get("text","")
                    if len(text_for_reranking) > MAX_TEXT_LEN_FOR_CROSS_ENCODER:
                        text_for_reranking = text_for_reranking[:MAX_TEXT_LEN_FOR_CROSS_ENCODER]
                    pairs.append((original_query_for_processing, text_for_reranking))
                if pairs:
                    try:
                        scores = cross_encoder.predict(pairs, show_progress_bar=False)
                        for idx, res in enumerate(candidates_for_reranking):
                            res["score"] = float(scores[idx]); res["search_method"] = "hybrid_reranked"
                        candidates_for_reranking.sort(key=lambda x: x.get("score", float('-inf')), reverse=True)
                        reranked_search_results = candidates_for_reranking + combined_results_for_ranking[settings.K_CANDIDATES_FOR_RERANKING:]
                    except Exception as e_rerank: logger.error(f"Cross-Encoder error: {e_rerank}", exc_info=True)
            
            current_initial_search_results = reranked_search_results[:settings.MAX_RESULTS_FOR_CONTEXT]
            session_data.initial_search_results = current_initial_search_results
            save_session(current_session_id, session_data)

            if not current_initial_search_results:
                logger.info(f"Session {current_session_id}: No results (Space: {current_selected_space_key}). Offering broaden.")
                session_data.conversation_step = "awaiting_broaden_search_choice"
                session_data.clarification_type = "broaden_search_scope"
                save_session(current_session_id, session_data)
                clarif_q_text = f"No encontré resultados para '{original_query_for_processing[:30]}...' en '{current_selected_space_key if current_selected_space_key else 'Todos los espacios'}'. ¿Quieres probar otras opciones?"
                clarif_opts = []
                if current_selected_space_key : clarif_opts.append({"id": "search_all", "text": "Buscar en todos los espacios"})
                # Corrected: broaden options after no results in a specific space
                clarif_opts.extend([
                    {"id": "select_other_space", "text": "Elegir otro espacio"},
                    {"id": "new_query", "text": "Hacer nueva consulta"}, # Allow new query
                    {"id": "cancel", "text": "No, gracias"}
                ])
                # Corrected logic for "already searched all"
                # Check if current_selected_space_key is None (meaning "search_all" was attempted)
                # AND the conversation_step that led here was "processing_broad_search" (or about to be "awaiting_broaden_search_choice" from such a state)
                if not current_selected_space_key:
                     clarif_q_text = f"No encontré resultados para '{original_query_for_processing[:30]}...' incluso buscando en todos los espacios. ¿Nueva consulta o elegir espacio?"
                     clarif_opts = [
                         {"id": "new_query", "text": "Nueva consulta"},
                         {"id": "select_other_space", "text": "Elegir un espacio específico"},
                         {"id": "cancel", "text": "No, gracias"}
                     ]
                return ChatResponse(answer="", needs_clarification=True, session_id=current_session_id,
                    clarification_question_text=clarif_q_text, clarification_options=clarif_opts, clarification_type="broaden_search_scope")

            clarification_details = generate_clarification_from_results(current_initial_search_results, original_query_for_processing, gemini_service)
            if clarification_details:
                logger.info(f"Session {current_session_id}: Document clarification needed.")
                session_data.conversation_step = "awaiting_clarification_response"
                session_data.clarification_type = "document_selection"
                session_data.clarification_question_asked = clarification_details["question_text"]
                session_data.clarification_options_provided = clarification_details["options"]
                save_session(current_session_id, session_data)
                return ChatResponse(answer="", needs_clarification=True, session_id=current_session_id,
                    clarification_question_text=clarification_details["question_text"],
                    clarification_options=clarification_details["options"], clarification_type="document_selection")

            logger.info(f"Session {current_session_id}: Proceeding to direct answer.")
            context_for_gemini, unique_source_urls = _build_context_from_search_results(
                current_initial_search_results, extracted_terms.get("phrases", []),
                extracted_terms.get("keywords", []), max_context_length)
            ai_answer = gemini_service.generate_response(original_query_for_processing, context_for_gemini)
            delete_session(current_session_id)
            return ChatResponse(answer=ai_answer, source_urls=unique_source_urls, session_id=None)
        else:
            logger.error(f"Session {current_session_id if current_session_id else 'None'} in unhandled state: {session_data.conversation_step if session_data else 'No session data'}. Error.")
            if current_session_id: delete_session(current_session_id)
            raise HTTPException(status_code=500, detail="Error de estado de la conversación.")
    except HTTPException as http_exc:
        logger.error(f"HTTPException in /chat for session {current_session_id if current_session_id else 'None'}: {http_exc.detail}", exc_info=True)
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error in /chat for session {current_session_id if current_session_id else 'None'}: {e}", exc_info=True)
        if current_session_id: delete_session(current_session_id)
        raise HTTPException(status_code=500, detail="Error interno del servidor durante el chat.")

# Helper function to consolidate context building logic
def _build_context_from_search_results(
    search_results: List[Dict[str, Any]], # Each dict is now a chunk or a page treated as chunk
    query_phrases: List[str],
    query_keywords: List[str],
    max_context_length: int
) -> tuple[str, List[str]]:

    context_for_gemini = "No se encontró información específica en Confluence que coincida bien con su consulta." # Default

    if not search_results:
        return context_for_gemini, []

    # New logic for handling chunks:
    scored_chunks_with_source = []
    for chunk_data in search_results:
        chunk_text = chunk_data.get("text")
        if not chunk_text:
            continue

        relevance_score = 0
        search_method = chunk_data.get("search_method")
        current_score = chunk_data.get("score")

        if search_method == "hybrid_reranked":
            relevance_score = current_score  # This score is already "higher is better"
        elif search_method == "semantic":
            # Convert L2 distance (lower is better) to a similarity-like score (higher is better)
            relevance_score = 1.0 / (1.0 + current_score) if current_score != float('inf') else 0
        elif search_method == "keyword_page":
            # This case applies if a keyword page was NOT reranked (e.g., beyond K_CANDIDATES_FOR_RERANKING)
            # Its 'score' might be the default high L2 (1000.0).
            # We should score it using keywords for a "higher is better" relevance.
            relevance_score = score_paragraph(chunk_text, query_phrases, query_keywords)
        else: # Fallback for any other cases or if score/method is missing
            relevance_score = 0.0

        scored_chunks_with_source.append(
            (relevance_score,
             chunk_text,
             chunk_data.get("title", "Fuente Desconocida"),
             chunk_data.get("url", "N/A"),
             chunk_data.get("context_hierarchy", chunk_data.get("title", "N/A"))
            )
        )

    # Sort by final relevance score (higher is better for all types now)
    scored_chunks_with_source.sort(key=lambda x: x[0], reverse=True)

    selected_context_parts = []
    current_length = 0
    final_selected_source_urls = set()

    if scored_chunks_with_source:
        logger.info(f"Found {len(scored_chunks_with_source)} relevant chunks/pages. Top scores: {[s[0] for s in scored_chunks_with_source[:3]]}")

        for score, chk_text, chk_title, chk_url, chk_hierarchy in scored_chunks_with_source:
            # Use context_hierarchy if available and different from title, otherwise just title
            display_title = chk_title
            if chk_hierarchy and chk_hierarchy.lower() != chk_title.lower():
                display_title = f"{chk_title} (Sección: {chk_hierarchy})"

            content_header = f"Contexto de '{display_title}' (URL: {chk_url}, Puntuación Relevancia: {score:.4f}):\n"
            full_snippet = f"{content_header}{chk_text}\n---"

            if current_length + len(full_snippet) <= max_context_length:
                selected_context_parts.append(full_snippet)
                current_length += len(full_snippet)
                if chk_url != "N/A":
                    final_selected_source_urls.add(chk_url)
            else:
                remaining_space = max_context_length - current_length
                if remaining_space > len(content_header) + 50:
                    truncated_para_text_len = remaining_space - (len(content_header) + len("\n---") + 3)
                    if truncated_para_text_len > 0:
                        truncated_snippet = f"{content_header}{chk_text[:truncated_para_text_len]}...\n---"
                        selected_context_parts.append(truncated_snippet)
                        if chk_url != "N/A":
                           final_selected_source_urls.add(chk_url)
                break

        if selected_context_parts:
            context_for_gemini = "\n".join(selected_context_parts)

    # Fallback if no context built but search_results (chunks) existed.
    # This means chunks might not have scored well or other issues.
    # The initial default message for context_for_gemini will be used in such cases.
    # If selected_context_parts is empty, the default "No se encontró..." message is used.

    return context_for_gemini, sorted(list(final_selected_source_urls))


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