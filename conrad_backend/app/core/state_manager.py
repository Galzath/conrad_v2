import time
import uuid
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

# Global dictionary to store active sessions
active_sessions: Dict[str, 'SessionData'] = {}

class SessionData(BaseModel):
    original_question: str
    # Using Dict[str, List[str]] for extracted_terms to match previous usage,
    # e.g., {"keywords": ["kw1", "kw2"], "phrases": ["phrase1"]}
    extracted_terms: Dict[str, List[str]]
    # Using List[Dict[str, Any]] for search results, allowing flexibility for various keys like id, title, url
    initial_search_results: List[Dict[str, Any]]
    available_spaces: Optional[List[Dict[str, str]]] = None # Spaces offered for selection
    selected_space_key: Optional[str] = None      # Space key chosen by user
    clarification_type: Optional[str] = None      # Type of current clarification (e.g., space_selection)
    clarification_question_asked: Optional[str] = None
    # Using List[Dict[str, str]] for clarification options as specified, e.g., [{"id": "opt1", "text": "Option 1"}]
    clarification_options_provided: Optional[List[Dict[str, str]]] = None
    conversation_step: str  # e.g., "awaiting_clarification_response", "processing_complete"
    timestamp: float       # Store creation/update time for potential cleanup

def create_new_session_id() -> str:
    """Generates a new unique session ID."""
    return str(uuid.uuid4())

def save_session(session_id: str, data: SessionData) -> None:
    """
    Saves the provided SessionData object into the active_sessions dictionary.
    The timestamp in the data will be updated upon saving.
    """
    data.timestamp = time.time() # Update timestamp on save
    active_sessions[session_id] = data
    print(f"Session saved: {session_id}, Step: {data.conversation_step}") # Basic logging

def get_session(session_id: str) -> Optional[SessionData]:
    """
    Retrieves SessionData from active_sessions.
    Returns None if session_id is not found.
    """
    session = active_sessions.get(session_id)
    if session:
        print(f"Session retrieved: {session_id}, Step: {session.conversation_step}") # Basic logging
    else:
        print(f"Session not found: {session_id}") # Basic logging
    return session

def delete_session(session_id: str) -> None:
    """Removes a session from active_sessions if it exists."""
    if session_id in active_sessions:
        del active_sessions[session_id]
        print(f"Session deleted: {session_id}") # Basic logging
    else:
        print(f"Session not found for deletion: {session_id}") # Basic logging

def cleanup_old_sessions(max_age_seconds: int = 3600) -> None:
    """
    Iterates through active_sessions and removes any session older than
    max_age_seconds based on its timestamp.
    """
    current_time = time.time()
    sessions_to_delete = [
        session_id
        for session_id, data in active_sessions.items()
        if (current_time - data.timestamp) > max_age_seconds
    ]

    deleted_count = 0
    for session_id in sessions_to_delete:
        delete_session(session_id)
        deleted_count +=1

    if deleted_count > 0:
        print(f"Cleaned up {deleted_count} old sessions.")
    else:
        print("No old sessions to clean up.")

if __name__ == '__main__':
    # Example Usage (for testing this module directly)
    print("Testing State Manager...")

    # Create a session
    session_id_1 = create_new_session_id()
    session_data_1 = SessionData(
        original_question="What is Python?",
        extracted_terms={"keywords": ["python"], "phrases": ["what is python"]},
        initial_search_results=[{"id": "doc1", "title": "Python Intro", "url": "http://example.com/python"}],
        conversation_step="initial_query",
        timestamp=time.time() # Initial timestamp
    )
    save_session(session_id_1, session_data_1)

    # Retrieve a session
    retrieved_session = get_session(session_id_1)
    if retrieved_session:
        print(f"Retrieved original question: {retrieved_session.original_question}")

    # Update session (e.g., moving to clarification)
    if retrieved_session:
        retrieved_session.conversation_step = "awaiting_clarification"
        retrieved_session.clarification_question_asked = "Which version of Python?"
        save_session(session_id_1, retrieved_session) # This will update the timestamp

    # Test cleanup (set a short max_age for testing)
    print(f"Number of active sessions before cleanup: {len(active_sessions)}")
    # Make session_id_1 look old for testing cleanup
    if session_id_1 in active_sessions:
         active_sessions[session_id_1].timestamp = time.time() - 7000 # Make it older than 1 hour

    cleanup_old_sessions(max_age_seconds=3600) # Default 1 hour
    print(f"Number of active sessions after cleanup: {len(active_sessions)}")

    retrieved_after_cleanup = get_session(session_id_1)
    if not retrieved_after_cleanup:
        print(f"Session {session_id_1} was correctly cleaned up.")

    # Delete a session explicitly
    session_id_2 = create_new_session_id()
    session_data_2 = SessionData(
        original_question="What is Java?",
        extracted_terms={"keywords": ["java"], "phrases": ["what is java"]},
        initial_search_results=[],
        conversation_step="initial_query",
        timestamp=time.time()
    )
    save_session(session_id_2, session_data_2)
    print(f"Number of active sessions: {len(active_sessions)}")
    delete_session(session_id_2)
    print(f"Number of active sessions after specific delete: {len(active_sessions)}")

    # Test getting a non-existent session
    get_session("non_existent_id")
    delete_session("non_existent_id")

    print("State Manager test complete.")
