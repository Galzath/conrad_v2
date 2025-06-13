import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    CONFLUENCE_URL: str = os.getenv("CONFLUENCE_URL", "YOUR_CONFLUENCE_URL_HERE")
    CONFLUENCE_USERNAME: str = os.getenv("CONFLUENCE_USERNAME", "YOUR_CONFLUENCE_USERNAME")
    CONFLUENCE_API_TOKEN: str = os.getenv("CONFLUENCE_API_TOKEN", "YOUR_CONFLUENCE_API_TOKEN")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
    VECTOR_DB_CHOICE: str = os.getenv("VECTOR_DB_CHOICE", "FAISS")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    CONFLUENCE_SPACE_KEYS_TO_INDEX: list[str] = os.getenv("CONFLUENCE_SPACE_KEYS_TO_INDEX", "YOUR_SPACE_KEY").split(",")
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "faiss_index.idx")
    CHUNKS_DATA_PATH: str = os.getenv("CHUNKS_DATA_PATH", "chunks_data.json")
    MAX_TEXT_BLOCK_LENGTH_FOR_SEMANTIC_SPLIT: int = int(os.getenv("MAX_TEXT_BLOCK_LENGTH_FOR_SEMANTIC_SPLIT", "2500")) # Characters
    MIN_TEXT_BLOCK_LENGTH_FOR_SEMANTIC_SPLIT: int = int(os.getenv("MIN_TEXT_BLOCK_LENGTH_FOR_SEMANTIC_SPLIT", "300")) # Characters
    # Add any other configurations here

settings = Settings()
