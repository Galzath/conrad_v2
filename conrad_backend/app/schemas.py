from pydantic import BaseModel
from typing import Optional, List # Added List

# New model for clarification options
class ClarificationOption(BaseModel):
    id: str
    text: str

class UserQuestion(BaseModel):
    question: str
    session_id: Optional[str] = None
    clarification_response: Optional[str] = None # User's textual answer to a clarification
    selected_option_id: Optional[str] = None    # ID of the option selected by the user for clarification

class ChatResponse(BaseModel):
    answer: str
    session_id: Optional[str] = None # To send back to the client
    needs_clarification: Optional[bool] = False # Flag if clarification is needed
    clarification_question_text: Optional[str] = None # The clarification question itself
    clarification_options: Optional[List[ClarificationOption]] = None # List of options for the user
    clarification_type: Optional[str] = None
    available_spaces: Optional[List[ClarificationOption]] = None # For listing Confluence spaces
    remaining_topics: Optional[List[ClarificationOption]] = None # For suggesting related pages after an answer
    # Optional fields for providing source/context information
    source_urls: Optional[list[str]] = None # Kept for consistency, using list generic for Python 3.9+
    debug_info: Optional[dict] = None
