import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    CONFLUENCE_URL: str = os.getenv("CONFLUENCE_URL", "YOUR_CONFLUENCE_URL_HERE")
    CONFLUENCE_USERNAME: str = os.getenv("CONFLUENCE_USERNAME", "YOUR_CONFLUENCE_USERNAME")
    CONFLUENCE_API_TOKEN: str = os.getenv("CONFLUENCE_API_TOKEN", "YOUR_CONFLUENCE_API_TOKEN")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
    # Add any other configurations here

settings = Settings()
