import unittest
import time
import uuid # For type checking if needed, though create_new_session_id uses it internally.

# Assuming conrad_backend is in the python path or discoverable
from conrad_backend.app.core.state_manager import (
    SessionData,
    create_new_session_id,
    save_session,
    get_session,
    delete_session,
    cleanup_old_sessions,
    active_sessions # For direct manipulation in tests
)

class TestStateManager(unittest.TestCase):

    def setUp(self):
        """Clear active_sessions before each test."""
        active_sessions.clear()

    def tearDown(self):
        """Clear active_sessions after each test to ensure independence."""
        active_sessions.clear()

    def test_create_new_session_id(self):
        """Test if session ID creation is working and generates unique IDs."""
        sid1 = create_new_session_id()
        sid2 = create_new_session_id()
        self.assertIsInstance(sid1, str)
        self.assertNotEqual(sid1, sid2)
        # A simple check for UUID format (36 characters with hyphens)
        self.assertEqual(len(sid1), 36)

    def _create_sample_session_data(self, original_question="Test question") -> SessionData:
        return SessionData(
            original_question=original_question,
            extracted_terms={"keywords": ["test"], "phrases": ["test phrase"]},
            initial_search_results=[{"id": "doc1", "title": "Test Doc", "url": "http://example.com"}],
            conversation_step="initial_query",
            timestamp=time.time()
        )

    def test_save_and_get_session(self):
        """Test saving and retrieving a session."""
        session_id = "test_sess_123"
        data_to_save = self._create_sample_session_data()

        save_session(session_id, data_to_save)

        retrieved_data = get_session(session_id)
        self.assertIsNotNone(retrieved_data)
        self.assertIsInstance(retrieved_data, SessionData)
        self.assertEqual(retrieved_data.original_question, data_to_save.original_question)
        self.assertEqual(retrieved_data.extracted_terms, data_to_save.extracted_terms)
        self.assertEqual(retrieved_data.conversation_step, data_to_save.conversation_step)
        # Timestamp is updated on save, so it might be slightly different than data_to_save.timestamp
        self.assertAlmostEqual(retrieved_data.timestamp, time.time(), delta=1.0)

        # Test getting a non-existent session
        non_existent_data = get_session("non_existent_id")
        self.assertIsNone(non_existent_data)

    def test_delete_session(self):
        """Test deleting a session."""
        session_id = "test_sess_to_delete"
        data_to_save = self._create_sample_session_data()
        save_session(session_id, data_to_save)

        # Ensure it's there
        self.assertIsNotNone(get_session(session_id))

        delete_session(session_id)
        self.assertIsNone(get_session(session_id))

        # Test deleting a non-existent session (should not raise error)
        try:
            delete_session("non_existent_for_delete")
        except Exception as e:
            self.fail(f"Deleting non-existent session raised an error: {e}")


    def test_save_session_updates_timestamp(self):
        """Test that saving a session updates its timestamp."""
        session_id = "test_sess_timestamp"
        data = self._create_sample_session_data()
        original_timestamp = data.timestamp

        time.sleep(0.01) # Ensure time moves forward a bit
        save_session(session_id, data)

        saved_data = get_session(session_id)
        self.assertIsNotNone(saved_data)
        self.assertNotEqual(saved_data.timestamp, original_timestamp)
        self.assertTrue(saved_data.timestamp > original_timestamp)

    def test_cleanup_old_sessions(self):
        """Test cleanup of old sessions."""
        max_age_seconds = 60  # 1 minute for testing

        # Session 1: Old
        session_id_old = "sess_old"
        data_old = self._create_sample_session_data("Old question")
        # Manually set timestamp to be older than max_age
        data_old.timestamp = time.time() - (max_age_seconds * 2)
        active_sessions[session_id_old] = data_old # Directly add to bypass save_session's timestamp update for this test

        # Session 2: New
        session_id_new = "sess_new"
        data_new = self._create_sample_session_data("New question")
        data_new.timestamp = time.time() - 10 # Recent
        active_sessions[session_id_new] = data_new

        # Session 3: Also new, saved via save_session
        session_id_new_saved = "sess_new_saved"
        data_new_saved = self._create_sample_session_data("New saved question")
        save_session(session_id_new_saved, data_new_saved) # Will have current timestamp

        self.assertEqual(len(active_sessions), 3)

        cleanup_old_sessions(max_age_seconds=max_age_seconds)

        self.assertEqual(len(active_sessions), 2) # Old one should be gone
        self.assertIsNone(get_session(session_id_old))
        self.assertIsNotNone(get_session(session_id_new))
        self.assertIsNotNone(get_session(session_id_new_saved))

    def test_cleanup_no_old_sessions(self):
        """Test cleanup when no sessions are old."""
        max_age_seconds = 60

        session_id_1 = "sess_recent_1"
        data_1 = self._create_sample_session_data()
        save_session(session_id_1, data_1) # Current timestamp

        session_id_2 = "sess_recent_2"
        data_2 = self._create_sample_session_data()
        # Manually make it recent but not super new
        data_2.timestamp = time.time() - (max_age_seconds / 2)
        active_sessions[session_id_2] = data_2

        self.assertEqual(len(active_sessions), 2)
        cleanup_old_sessions(max_age_seconds=max_age_seconds)
        self.assertEqual(len(active_sessions), 2) # No sessions should be deleted

if __name__ == '__main__':
    unittest.main()
