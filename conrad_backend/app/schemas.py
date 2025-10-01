"""
Pydantic schemas for the Conrad Chatbot API.

This module defines the data models for API requests and responses,
ensuring that data conforms to the expected structure and types.
"""
from pydantic import BaseModel
from typing import Optional, List

class ClarificationOption(BaseModel):
    """
    Represents a single choice in a clarification prompt.

    Attributes:
        id: A unique identifier for the option (e.g., a Confluence page ID or space key).
        text: The user-facing text to display for this option.
    """
    id: str
    text: str

class UserQuestion(BaseModel):
    """
    Represents an incoming request from a user.

    Attributes:
        question: The primary question or message from the user.
        session_id: An optional ID to maintain conversation state across multiple requests.
        clarification_response: The user's textual answer to a clarification question (if any).
        selected_option_id: The ID of a `ClarificationOption` selected by the user.
    """
    question: str
    session_id: Optional[str] = None
    clarification_response: Optional[str] = None
    selected_option_id: Optional[str] = None

class ChatResponse(BaseModel):
    """
    Represents a response sent back to the user.

    This model can represent a final answer, a prompt for clarification,
    or other types of interaction required by the conversation flow.

    Attributes:
        answer: The primary text response from the chatbot.
        session_id: The session ID to be used in subsequent requests.
        needs_clarification: A boolean flag indicating if the user needs to
                             provide more information.
        clarification_question_text: The question to ask the user for clarification.
        clarification_options: A list of `ClarificationOption` objects for the user to choose from.
        clarification_type: A string indicating the type of clarification needed (e.g., "space_selection").
        available_spaces: A list of available Confluence spaces, presented as clarification options.
        remaining_topics: A list of suggested related topics to explore after an answer is given.
        source_urls: A list of source URLs used to generate the answer.
        debug_info: Optional dictionary for debugging information.
    """
    answer: Optional[str] = None
    session_id: Optional[str] = None
    needs_clarification: Optional[bool] = False
    clarification_question_text: Optional[str] = None
    clarification_options: Optional[List[ClarificationOption]] = None
    clarification_type: Optional[str] = None
    available_spaces: Optional[List[ClarificationOption]] = None
    remaining_topics: Optional[List[ClarificationOption]] = None
    source_urls: Optional[list[str]] = None
    debug_info: Optional[dict] = None
