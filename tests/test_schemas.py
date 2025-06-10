import unittest
from typing import List, Dict, Any
from pydantic import ValidationError

# Assuming conrad_backend is in the python path or discoverable
from conrad_backend.app.schemas import UserQuestion, ChatResponse, ClarificationOption

class TestSchemas(unittest.TestCase):

    def test_user_question_instantiation(self):
        """Test UserQuestion can be instantiated with all fields, including new ones."""
        data_full = {
            "question": "What is Python?",
            "session_id": "sess_123",
            "clarification_response": "Version 3",
            "selected_option_id": "opt_python3"
        }
        user_q = UserQuestion(**data_full)
        self.assertEqual(user_q.question, data_full["question"])
        self.assertEqual(user_q.session_id, data_full["session_id"])
        self.assertEqual(user_q.clarification_response, data_full["clarification_response"])
        self.assertEqual(user_q.selected_option_id, data_full["selected_option_id"])

    def test_user_question_defaults(self):
        """Test UserQuestion defaults for new optional fields."""
        user_q = UserQuestion(question="Hello")
        self.assertEqual(user_q.question, "Hello")
        self.assertIsNone(user_q.session_id)
        self.assertIsNone(user_q.clarification_response)
        self.assertIsNone(user_q.selected_option_id)

    def test_clarification_option_instantiation(self):
        """Test ClarificationOption instantiation."""
        option = ClarificationOption(id="opt1", text="Option 1")
        self.assertEqual(option.id, "opt1")
        self.assertEqual(option.text, "Option 1")

    def test_chat_response_instantiation_full(self):
        """Test ChatResponse with all fields, including new clarification fields."""
        clarif_opts = [
            ClarificationOption(id="opt1", text="Choice A"),
            ClarificationOption(id="opt2", text="Choice B")
        ]
        data_full = {
            "answer": "This is the final answer.",
            "session_id": "sess_456",
            "needs_clarification": True,
            "clarification_question_text": "Which choice?",
            "clarification_options": clarif_opts,
            "source_urls": ["http://example.com/source1"],
            "debug_info": {"key": "value"}
        }
        chat_res = ChatResponse(**data_full)
        self.assertEqual(chat_res.answer, data_full["answer"])
        self.assertEqual(chat_res.session_id, data_full["session_id"])
        self.assertTrue(chat_res.needs_clarification)
        self.assertEqual(chat_res.clarification_question_text, data_full["clarification_question_text"])
        self.assertEqual(len(chat_res.clarification_options), 2)
        self.assertEqual(chat_res.clarification_options[0].id, "opt1")
        self.assertEqual(chat_res.source_urls, data_full["source_urls"])
        self.assertEqual(chat_res.debug_info, data_full["debug_info"])

    def test_chat_response_defaults(self):
        """Test ChatResponse defaults for new optional fields."""
        chat_res = ChatResponse(answer="A direct answer.")
        self.assertEqual(chat_res.answer, "A direct answer.")
        self.assertIsNone(chat_res.session_id)
        self.assertFalse(chat_res.needs_clarification) # Default should be False
        self.assertIsNone(chat_res.clarification_question_text)
        self.assertIsNone(chat_res.clarification_options)
        self.assertIsNone(chat_res.source_urls)
        self.assertIsNone(chat_res.debug_info)

    def test_chat_response_clarification_options_validation(self):
        """Test ChatResponse with list of dicts for clarification_options (should work with Pydantic)."""
        # Pydantic v2 should automatically convert dicts to ClarificationOption if types match
        clarif_opts_dicts = [
            {"id": "opt1", "text": "Choice A"},
            {"id": "opt2", "text": "Choice B"}
        ]
        chat_res = ChatResponse(
            answer="",
            needs_clarification=True,
            clarification_options=clarif_opts_dicts
        )
        self.assertIsNotNone(chat_res.clarification_options)
        self.assertEqual(len(chat_res.clarification_options), 2)
        self.assertIsInstance(chat_res.clarification_options[0], ClarificationOption)
        self.assertEqual(chat_res.clarification_options[0].id, "opt1")

        # Test with invalid structure for an option
        with self.assertRaises(ValidationError):
            ChatResponse(
                answer="",
                needs_clarification=True,
                clarification_options=[{"id_wrong": "opt1", "text": "Choice A"}] # 'id_wrong' instead of 'id'
            )

if __name__ == '__main__':
    unittest.main()
