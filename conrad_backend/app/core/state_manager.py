"""
In-memory session management for multi-turn conversations.

This module provides a simple in-memory key-value store to manage the state
of ongoing conversations. It is not suitable for production environments
with multiple server instances but works well for single-instance deployments.
"""
import time
import uuid
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

# Global dictionary to store active sessions
active_sessions: Dict[str, 'SessionData'] = {}


class SessionData(BaseModel):
    """
    Represents the state of a single user conversation session.

    Attributes:
        original_question: The initial question that started the conversation.
        extracted_terms: Keywords and phrases extracted from the original question.
        initial_search_results: The first set of search results retrieved.
        available_spaces: A list of Confluence spaces offered to the user.
        selected_space_key: The space key the user selected.
        clarification_type: The type of clarification currently being handled.
        clarification_question_asked: The text of the clarification question asked.
        clarification_options_provided: The options provided for clarification.
        conversation_step: The current stage of the conversation flow (e.g., 'awaiting_space_selection').
        timestamp: The epoch timestamp of the last update to this session.
    """
    original_question: str
    extracted_terms: Dict[str, List[str]]
    initial_search_results: List[Dict[str, Any]]
    available_spaces: Optional[List[Dict[str, str]]] = None
    selected_space_key: Optional[str] = None
    clarification_type: Optional[str] = None
    clarification_question_asked: Optional[str] = None
    clarification_options_provided: Optional[List[Dict[str, str]]] = None
    conversation_step: str
    timestamp: float


def create_new_session_id() -> str:
    """
    Generates a new, unique session ID.

    Returns:
        A string representing a new UUID version 4.
    """
    return str(uuid.uuid4())


def save_session(session_id: str, data: SessionData) -> None:
    """
    Saves or updates a session in the in-memory store.

    The session's timestamp is automatically updated to the current time
    upon saving.

    Args:
        session_id: The unique ID for the session.
        data: The `SessionData` object to save.
    """
    data.timestamp = time.time()
    active_sessions[session_id] = data
    print(f"Session saved: {session_id}, Step: {data.conversation_step}")


def get_session(session_id: str) -> Optional[SessionData]:
    """
    Retrieves a session from the in-memory store.

    Args:
        session_id: The ID of the session to retrieve.

    Returns:
        The `SessionData` object if found, otherwise None.
    """
    session = active_sessions.get(session_id)
    if session:
        print(f"Session retrieved: {session_id}, Step: {session.conversation_step}")
    else:
        print(f"Session not found: {session_id}")
    return session


def delete_session(session_id: str) -> None:
    """
    Removes a session from the in-memory store.

    Args:
        session_id: The ID of the session to delete.
    """
    if session_id in active_sessions:
        del active_sessions[session_id]
        print(f"Session deleted: {session_id}")
    else:
        print(f"Session not found for deletion: {session_id}")


def cleanup_old_sessions(max_age_seconds: int = 3600) -> None:
    """
    Removes sessions that have not been updated for a specified duration.

    This function is useful for preventing the in-memory store from growing
    indefinitely with stale sessions.

    Args:
        max_age_seconds: The maximum age of a session in seconds. Sessions
                         older than this will be removed.
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
