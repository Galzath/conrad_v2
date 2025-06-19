# conrad_backend/app/main.py

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
import time

from sentence_transformers import CrossEncoder
import numpy as np

from .schemas import UserQuestion, ChatResponse, ClarificationOption
from .services.confluence_service import ConfluenceService
from .services.gemini_service import GeminiService
from .core.config import settings
from .core.state_manager import SessionData, save_session, get_session, delete_session, create_new_session_id

# --- Configuración de Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Inicialización de la Aplicación y Servicios ---
app = FastAPI(
    title="Conrad Chatbot API",
    description="API para Conrad Chatbot usando Confluence y Gemini",
    version="0.2.1" # Version bump por fix de CORS
)

# --- Configuración de CORS ---
origins = settings.CORS_ORIGINS.split(",")
logger.info(f"Configurando CORS para los siguientes orígenes: {origins}")

## MODIFICADO: Ajustes para manejar correctamente las peticiones OPTIONS (pre-vuelo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos, crucial para OPTIONS
    allow_headers=["*"],  # Permitir todas las cabeceras
)

# --- Inicialización de Servicios y Modelos ---
confluence_service = ConfluenceService()
gemini_service = GeminiService()

try:
    logger.info(f"Cargando Cross-Encoder model: {settings.CROSS_ENCODER_MODEL_NAME}")
    cross_encoder = CrossEncoder(settings.CROSS_ENCODER_MODEL_NAME)
    logger.info("Cross-Encoder model cargado exitosamente.")
except Exception as e:
    logger.error(f"Error al cargar Cross-Encoder model: {e}", exc_info=True)
    cross_encoder = None

# --- Constantes y Helpers de Procesamiento de Texto ---
# (Las funciones extract_search_terms y otras utilidades se mantienen aquí)
STOP_WORDS = set([...]) # Omitido por brevedad
KNOWN_SPACE_KEYWORDS = {"M2": "M2", "SGP": "SGP", "SF": "SF"}
SPACE_KEY_PATTERN = re.compile(r'\b(' + '|'.join(re.escape(key) for key in KNOWN_SPACE_KEYWORDS.keys()) + r')\b', re.IGNORECASE)

def extract_search_terms(question: str) -> dict:
    # ... Lógica sin cambios ...
    # Omitido por brevedad
    normalized_question_for_keywords = question.lower()
    normalized_question_for_keywords = re.sub(r'[^\w\s-]', '', normalized_question_for_keywords)
    words = normalized_question_for_keywords.split()
    candidate_keywords = [word for word in words if word not in STOP_WORDS and len(word) > 2]
    proper_noun_phrases = re.findall(r'\b[A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+)+\b', question)
    all_potential_phrases = set(proper_noun_phrases)
    for n in range(2, 5):
        for i in range(len(words) - n + 1):
            ngram_words = words[i:i+n]
            if not all(word in STOP_WORDS for word in ngram_words) and any(word not in STOP_WORDS for word in ngram_words):
                all_potential_phrases.add(" ".join(ngram_words))
    sorted_phrases = sorted(list(all_potential_phrases), key=lambda p: (len(p.split()), len(p)), reverse=True)
    selected_phrases = []
    max_phrases = 4
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
    final_keywords_list = sorted(list(final_keywords), key=len, reverse=True)[:7]
    search_terms_dict = {"keywords": final_keywords_list, "phrases": selected_phrases}
    logger.info(f"Términos de búsqueda extraídos: {search_terms_dict}")
    return search_terms_dict


# --- SECCIÓN DE LÓGICA DE NEGOCIO REFACTORIZADA ---

## NUEVO: Función dedicada a la búsqueda y ranking
def _perform_retrieval_and_ranking(query: str, space_key: Optional[str]) -> List[Dict[str, Any]]:
    """Realiza búsqueda híbrida (semántica + keyword) y re-rankea los resultados."""
    extracted_terms = extract_search_terms(query)
    
    # Búsqueda
    semantic_chunks = confluence_service.semantic_search_chunks(query_text=query, top_k=settings.N_SEMANTIC_RESULTS)
    keyword_pages = confluence_service.search_content(
        search_terms=extracted_terms, 
        space_keys=[space_key] if space_key else None,
        limit=settings.N_CQL_RESULTS
    )

    # Combinación de resultados
    combined_results = []
    processed_urls = set()
    for chunk in semantic_chunks:
        combined_results.append({**chunk, "search_method": "semantic"})
        if chunk.get("url"): processed_urls.add(chunk["url"])
            
    for page in keyword_pages:
        if page["url"] not in processed_urls:
            content = confluence_service.get_page_content_by_id(page['id'])
            if content:
                combined_results.append({
                    "text": content, "url": page["url"], "title": page["title"], 
                    "score": 1000.0, "search_method": "keyword_page", "page_id": page['id'],
                    "context_hierarchy": page["title"]
                })

    # Re-ranking con CrossEncoder
    if not cross_encoder or not combined_results:
        return combined_results

    pairs = [(query, res.get("text", "")[:2000]) for res in combined_results[:settings.K_CANDIDATES_FOR_RERANKING]]
    if not pairs:
        return combined_results
        
    try:
        scores = cross_encoder.predict(pairs, show_progress_bar=False)
        for idx, res in enumerate(combined_results[:len(scores)]):
            res["score"] = float(scores[idx])
        
        # Ordenar por el nuevo score descendente
        combined_results.sort(key=lambda x: x.get("score", float('-inf')), reverse=True)
        logger.info(f"Resultados re-rankeados. Top 3 scores: {[r.get('score', 0) for r in combined_results[:3]]}")
    except Exception as e:
        logger.error(f"Error durante el re-ranking con Cross-Encoder: {e}", exc_info=True)

    return combined_results

## NUEVO: Función para generar opciones de clarificación
def _generate_clarification_options(search_results: List[Dict[str, Any]], query: str) -> Optional[Dict[str, Any]]:
    """Genera opciones de clarificación si los resultados son ambiguos."""
    if len(search_results) < 2:
        return None

    unique_pages = {}
    for res in search_results:
        page_id = res.get("page_id")
        if page_id and page_id not in unique_pages:
            unique_pages[page_id] = res.get("title", "Título desconocido")

    if len(unique_pages) < 2:
        return None

    options = []
    # MODIFICADO: Estrategia más simple y rápida sin llamar a Gemini para cada opción
    for page_id, title in list(unique_pages.items())[:4]: # Limitar a 4 opciones
        # Una heurística simple para limpiar el título podría ir aquí
        cleaned_title = title.split(':')[-1].strip().capitalize()
        options.append({"id": page_id, "text": cleaned_title})
    
    if len(options) >= 2:
        question_text = f"Tu consulta sobre '{query[:40]}...' parece relacionada con varios temas. ¿Cuál te interesa más?"
        return {"question_text": question_text, "options": options, "type": "document_selection"}
    
    return None

## NUEVO: Función para construir el contexto para Gemini
def _build_context_for_llm(search_results: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """Construye el string de contexto y la lista de URLs de las fuentes."""
    context_parts = []
    source_urls = set()
    current_length = 0

    for res in search_results[:settings.MAX_RESULTS_FOR_CONTEXT]:
        text = res.get("text", "")
        title = res.get("title", "N/A")
        url = res.get("url", "N/A")
        score = res.get("score", 0.0)
        
        header = f"Contexto de '{title}' (URL: {url}, Relevancia: {score:.4f}):\n"
        snippet = f"{header}{text}\n---\n"
        
        if current_length + len(snippet) > settings.MAX_CONTEXT_LENGTH_GEMINI:
            break
            
        context_parts.append(snippet)
        current_length += len(snippet)
        if url != "N/A":
            source_urls.add(url)
            
    if not context_parts:
        return "No se encontró información relevante en Confluence.", []
        
    return "\n".join(context_parts), sorted(list(source_urls))

# --- Endpoint Principal de Chat ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(user_question: UserQuestion = Body(...)):
    """Punto de entrada principal para todas las interacciones de chat."""
    logger.info(f"Payload recibido: {user_question.model_dump_json(indent=2)}")

    session_id = user_question.session_id
    session_data = get_session(session_id) if session_id else None

    if not session_data:
        # --- Flujo de una nueva conversación ---
        session_id = create_new_session_id()
        available_spaces = [{"id": sp["id"], "text": sp["text"]} for sp in confluence_service.get_available_spaces()]
        
        session_data = SessionData(
            original_question=user_question.question,
            conversation_step="awaiting_space_selection",
            available_spaces=available_spaces,
            timestamp=time.time(),
            # Otros campos se inicializan a None/vacío por defecto en Pydantic
            extracted_terms={},
            initial_search_results=[]
        )
        save_session(session_id, session_data)
        
        return ChatResponse(
            session_id=session_id,
            needs_clarification=True,
            clarification_question_text="¡Hola! Soy Conrad. Para ayudarte mejor, elige el espacio de Confluence donde buscar:",
            clarification_options=available_spaces,
            clarification_type="space_selection"
        )

    # --- Flujo de una conversación existente (respuesta a clarificación) ---
    original_query = session_data.original_question
    
    # 1. Usuario seleccionó un espacio
    if session_data.conversation_step == "awaiting_space_selection" and user_question.selected_option_id:
        space_key = user_question.selected_option_id
        session_data.selected_space_key = space_key
        
        search_results = _perform_retrieval_and_ranking(original_query, space_key)
        session_data.initial_search_results = search_results
        
        if not search_results:
            # Ofrecer buscar en todos los espacios o reintentar
            clarif_opts = [{"id": "search_all", "text": f"Buscar '{original_query[:30]}...' en TODOS los espacios"}]
            # ... (se puede añadir más lógica aquí) ...
            return ChatResponse(session_id=session_id, needs_clarification=True, clarification_question_text=f"No encontré nada en el espacio '{space_key}'. ¿Quieres que busque en todos los espacios?", clarification_options=clarif_opts, clarification_type="broaden_search")

        # 2. Comprobar si se necesita clarificación de documento
        clarification = _generate_clarification_options(search_results, original_query)
        if clarification:
            session_data.conversation_step = "awaiting_document_clarification"
            save_session(session_id, session_data)
            return ChatResponse(session_id=session_id, needs_clarification=True, **clarification)
            
        # 3. Si no se necesita clarificación, generar respuesta final
        context, urls = _build_context_for_llm(search_results)
        answer = gemini_service.generate_response(original_query, context)
        delete_session(session_id)
        return ChatResponse(answer=answer, source_urls=urls)

    # 4. Usuario seleccionó un documento para clarificar
    elif session_data.conversation_step == "awaiting_document_clarification" and user_question.selected_option_id:
        selected_page_id = user_question.selected_option_id
        # Filtrar resultados para enfocarse en la página seleccionada
        focused_results = [res for res in session_data.initial_search_results if res.get("page_id") == selected_page_id]
        
        context, urls = _build_context_for_llm(focused_results or session_data.initial_search_results)
        answer = gemini_service.generate_response(original_query, context)
        delete_session(session_id)
        return ChatResponse(answer=answer, source_urls=urls)

    # ... se pueden añadir más manejadores para otros pasos de la conversación ...

    # Fallback por si el estado es inconsistente
    delete_session(session_id)
    raise HTTPException(status_code=400, detail="Estado de conversación inválido o respuesta inesperada.")

@app.get("/health", summary="Health Check")
async def health_check():
    # ... Lógica sin cambios ...
    confluence_status = "initialized" if confluence_service.confluence else "error_initializing"
    gemini_status = "initialized" if gemini_service.model else "error_initializing"
    if confluence_status == "initialized" and gemini_status == "initialized":
        return {"status": "ok", "confluence_service": confluence_status, "gemini_service": gemini_status}
    else:
        return JSONResponse(status_code=503, content={"status": "error", "details": "Uno o más servicios no están disponibles."})