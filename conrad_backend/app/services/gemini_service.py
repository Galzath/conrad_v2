import google.generativeai as genai
from ..core.config import settings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        try:
            if not settings.GEMINI_API_KEY or "YOUR_GEMINI_API_KEY" in settings.GEMINI_API_KEY:
                logger.error("Gemini API Key not configured. Please set it in .env or environment variables.")
                self.model = None
                raise ValueError("Gemini API Key not configured.")

            genai.configure(api_key=settings.GEMINI_API_KEY)

            # Set the model permanently based on user selection from the list.
            # The previous step (listing models) used 'gemini-1.5-flash-latest' as the trial model
            # after the list was printed. User confirms this choice.
            model_name_to_use = 'gemini-1.5-flash-latest'

            self.model = genai.GenerativeModel(model_name_to_use)
            logger.info(f"Successfully configured Gemini API and initialized the model: '{model_name_to_use}'.")

        except Exception as e:
            # This will catch errors if 'gemini-1.5-flash-latest' is still problematic
            # (e.g., not found for v1beta, permissions, etc.)
            logger.error(f"Failed to initialize Gemini Service with model '{model_name_to_use}': {e}")
            self.model = None

    def generate_response(self, user_question: str, confluence_context: str) -> str:
        if not self.model:
            logger.error("Gemini model not initialized. Cannot generate response.")
            return "Error: The AI model is currently unavailable. Please try again later."

        # For logging, use the model name that was attempted during initialization.
        # If self.model exists, it means initialization with model_name_to_use (in __init__) was successful.
        # However, self.model object itself might not directly expose the "friendly name" like "gemini-1.5-flash-latest"
        # if it resolves to a more specific versioned name.
        # The log in __init__ is the most reliable for what was *intended*.
        # For this call log, it's okay to assume the intended model is what's active.

        prompt = f"""
Eres **Conrad**, un asistente virtual experto en la base de conocimientos de nuestra empresa (Confluence). Tu objetivo principal es ayudar a los usuarios respondiendo sus preguntas basándote **estrictamente** en la información proporcionada de Confluence.

**[USER_QUESTION]**
{user_question}

**[CONFLUENCE_CONTEXT_START]**
A continuación, se presenta el contexto recuperado de Confluence. Cada fragmento puede tener indicada su página y URL de origen.

{confluence_context}
**[CONFLUENCE_CONTEXT_END]**

**[INSTRUCTIONS]**
1.  Responde a la pregunta del usuario utilizando **única y exclusivamente** la información contenida en el **[CONFLUENCE_CONTEXT_START]** y **[CONFLUENCE_CONTEXT_END]**.
2.  **No inventes información, no completes por intuición y no uses conocimientos externos.**
3.  Si encuentras información relevante de múltiples fragmentos, sintetízala en una respuesta coherente y concisa.
4.  Cuando sea posible y relevante, menciona la fuente de tu información (por ejemplo, el título de la página de Confluence o su URL) de forma natural en tu respuesta. (Ej: "Según la página 'Guía de Incorporación'...")
5.  Si después de analizar el contexto proporcionado, determinas que la información **no es suficiente** para responder de manera precisa a la pregunta del usuario, debes indicarlo claramente. Responde con: "No he encontrado información suficiente en Confluence para responder a tu pregunta." Si puedes, añade sobre qué tema específico no encontraste información, basándote en la pregunta del usuario. (Ej: "...sobre la configuración avanzada del módulo X.")
6.  Si la pregunta es un saludo, una despedida, o una conversación trivial no relacionada con una búsqueda de información, responde de manera cortés y breve sin intentar buscar en Confluence.

**Respuesta de Conrad:**
"""

        logger.info(f"Sending prompt to Gemini (using configured model): {prompt[:500]}...")

        try:
            response = self.model.generate_content(prompt)

            if response and response.parts:
                generated_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
                logger.info("Successfully received response from Gemini.")
                return generated_text if generated_text else "No response text found."
            elif response and hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                logger.warning(f"Gemini API call did not return content. Prompt Feedback: {response.prompt_feedback}")
                return f"Error: The request was blocked by the AI for the following reason: {response.prompt_feedback}"
            else:
                logger.warning(f"Gemini API call did not return any content or parts. Response: {response}")
                return "Error: No response was generated by the AI model."

        except Exception as e:
            logger.error(f"Error during Gemini API call: {e}")
            if hasattr(e, 'response') and e.response:
                 logger.error(f"Gemini API Error Response Details: {e.response}")
            return "Error: An unexpected issue occurred while trying to communicate with the AI model."

    def generate_clarification_summary(self, user_query: str, page_chunk_texts: str, max_length_chars: int = 150) -> str:
        if not self.model:
            logger.error("Gemini model not initialized. Cannot generate clarification summary.")
            return "No se pudo generar un resumen." # Fallback text

        prompt = f"""
Eres un asistente encargado de ayudar a clarificar una búsqueda de información.
Un usuario ha realizado la siguiente consulta:
"{user_query}"

Basándote **únicamente** en el siguiente texto extraído de una página de Confluence, genera un resumen conciso de 1 o 2 frases (máximo {max_length_chars} caracteres) que describa de qué trata el texto y por qué podría ser relevante para la consulta del usuario. El objetivo es ayudar al usuario a decidir si esta página es la que busca. No añadas información externa.

Texto de la página:
---
{page_chunk_texts}
---

Resumen conciso (1-2 frases, max {max_length_chars} caracteres):
"""
        # Ensure page_chunk_texts is not excessively long for this specific prompt.
        MAX_CONTEXT_FOR_SUMMARY_PROMPT = 3000
        if len(prompt) > MAX_CONTEXT_FOR_SUMMARY_PROMPT:
            excess = len(prompt) - MAX_CONTEXT_FOR_SUMMARY_PROMPT
            chars_to_cut_from_context = excess + 200
            if chars_to_cut_from_context < len(page_chunk_texts):
                 page_chunk_texts_truncated = page_chunk_texts[:-chars_to_cut_from_context] + "..."
                 prompt = f"""
Eres un asistente encargado de ayudar a clarificar una búsqueda de información.
Un usuario ha realizado la siguiente consulta:
"{user_query}"

Basándote **únicamente** en el siguiente texto extraído de una página de Confluence, genera un resumen conciso de 1 o 2 frases (máximo {max_length_chars} caracteres) que describa de qué trata el texto y por qué podría ser relevante para la consulta del usuario. El objetivo es ayudar al usuario a decidir si esta página es la que busca. No añadas información externa.

Texto de la página:
---
{page_chunk_texts_truncated}
---

Resumen conciso (1-2 frases, max {max_length_chars} caracteres):
"""
            else:
                 logger.warning(f"Context for summary is too long and cannot be meaningfully truncated for prompt. Prompt length: {len(prompt)}")


        logger.info(f"Sending prompt to Gemini for clarification summary (query: '{user_query[:30]}...'): {prompt[:300]}...")

        try:
            response = self.model.generate_content(prompt)

            if response and response.parts:
                generated_text = "".join(part.text for part in response.parts if hasattr(part, 'text')).strip()
                logger.info(f"Successfully received clarification summary from Gemini: {generated_text}")
                return generated_text[:max_length_chars] if generated_text else "Información relevante (resumen no disponible)."
            elif response and hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                logger.warning(f"Gemini API call for summary did not return content. Prompt Feedback: {response.prompt_feedback}")
                return f"Error al generar resumen (bloqueo): {response.prompt_feedback}"
            else:
                logger.warning(f"Gemini API call for summary did not return any content or parts. Response: {response}")
                return "Resumen no disponible."
        except Exception as e:
            logger.error(f"Error during Gemini API call for summary: {e}", exc_info=True)
            return "Error al generar resumen."

# Example Usage (for testing purposes)
if __name__ == "__main__":
    if not settings.GEMINI_API_KEY or "YOUR_GEMINI_API_KEY" in settings.GEMINI_API_KEY:
        print("Gemini API key not configured. Please set it in .env")
    else:
        gemini_service = GeminiService() # Model initialization logged here
        if gemini_service.model:
            print(f"GeminiService initialized (model intended: 'gemini-1.5-flash-latest').")
            test_question = "¿Cómo creo una página en Confluence?"
            test_context = "Context from 'Crear Páginas' (URL: example.com/crear, Relevance Score: 10):\nPara crear una página en Confluence, ve al espacio deseado, haz clic en el botón 'Crear' en la parte superior y selecciona 'Página'. Luego, puedes añadir un título y contenido.\n---"

            print(f"Testing Gemini with question: {test_question}")
            answer = gemini_service.generate_response(test_question, test_context)
            print(f"Gemini's Answer:\n{answer}")
        else:
            print("Could not initialize GeminiService. Check logs for errors.")
