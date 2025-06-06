from pydantic import BaseModel
from typing import Optional

class UserQuestion(BaseModel):
    question: str
    # Optional session_id for more advanced state management later
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    # Optional fields for providing source/context information
    source_urls: Optional[list[str]] = None
    debug_info: Optional[dict] = None
