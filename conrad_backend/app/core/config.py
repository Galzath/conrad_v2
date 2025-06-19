import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # --- Configuración de Conexiones ---
    CONFLUENCE_URL: str = os.getenv("CONFLUENCE_URL", "YOUR_CONFLUENCE_URL_HERE")
    CONFLUENCE_USERNAME: str = os.getenv("CONFLUENCE_USERNAME", "YOUR_CONFLUENCE_USERNAME")
    CONFLUENCE_API_TOKEN: str = os.getenv("CONFLUENCE_API_TOKEN", "YOUR_CONFLUENCE_API_TOKEN")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")

    # --- Configuración de Modelos y Búsqueda ---
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    CROSS_ENCODER_MODEL_NAME: str = os.getenv("CROSS_ENCODER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    ## NUEVO: Parámetros de búsqueda y ranking movidos desde main.py
    N_SEMANTIC_RESULTS: int = int(os.getenv("N_SEMANTIC_RESULTS", 5))
    N_CQL_RESULTS: int = int(os.getenv("N_CQL_RESULTS", 5)) # Aumentado ligeramente para más candidatos
    K_CANDIDATES_FOR_RERANKING: int = int(os.getenv("K_CANDIDATES_FOR_RERANKING", 25))
    MAX_RESULTS_FOR_CONTEXT: int = int(os.getenv("MAX_RESULTS_FOR_CONTEXT", 10)) # Top N resultados después de re-rankear

    # --- Configuración de Indexación ---
    CONFLUENCE_SPACE_KEYS_TO_INDEX: list[str] = os.getenv("CONFLUENCE_SPACE_KEYS_TO_INDEX", "YOUR_SPACE_KEY").split(",")
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "faiss_index.idx")
    CHUNKS_DATA_PATH: str = os.getenv("CHUNKS_DATA_PATH", "chunks_data.json")

    # --- Configuración del Procesamiento de Texto ---
    MAX_CONTEXT_LENGTH_GEMINI: int = int(os.getenv("MAX_CONTEXT_LENGTH_GEMINI", 15000))
    MIN_TEXT_BLOCK_LENGTH_FOR_SEMANTIC_SPLIT: int = int(os.getenv("MIN_TEXT_BLOCK_LENGTH_FOR_SEMANTIC_SPLIT", "300"))

    # --- Configuración de Seguridad y Red ---
    ## NUEVO: Orígenes CORS configurables para mayor seguridad
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "http://localhost,http://127.0.0.1")

settings = Settings()