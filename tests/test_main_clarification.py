import pytest
from typing import List, Dict, Any, Optional

# Since we are testing a private function, we need to import it directly.
# This is often done for complex helper functions to ensure their logic is sound.
from conrad_backend.app.main import _generate_clarification_options

# --- Test Data Fixtures ---

@pytest.fixture
def sample_query() -> str:
    """A sample user query."""
    return "How do I fix my connection issue?"

@pytest.fixture
def single_search_result() -> List[Dict[str, Any]]:
    """Search results with only one item."""
    return [
        {"page_id": "101", "title": "Networking: Common connection problems"}
    ]

@pytest.fixture
def non_distinct_search_results() -> List[Dict[str, Any]]:
    """Search results from the same page, but different chunks."""
    return [
        {"page_id": "101", "title": "Networking: Common connection problems", "text": "First chunk..."},
        {"page_id": "101", "title": "Networking: Common connection problems", "text": "Second chunk..."}
    ]

@pytest.fixture
def distinct_search_results() -> List[Dict[str, Any]]:
    """Search results from multiple distinct pages."""
    return [
        {"page_id": "101", "title": "Guide: Solving Connection Issues", "text": "..." },
        {"page_id": "102", "title": "Permissions: Why you can't connect", "text": "..." },
        {"page_id": "103", "title": "Server Status Page", "text": "..." },
        {"page_id": "101", "title": "Guide: Solving Connection Issues", "text": "another chunk from the same page" }
    ]

@pytest.fixture
def five_distinct_search_results() -> List[Dict[str, Any]]:
    """Search results from five distinct pages to test option limiting."""
    return [
        {"page_id": "101", "title": "Title One"},
        {"page_id": "102", "title": "Title Two"},
        {"page_id": "103", "title": "Title Three"},
        {"page_id": "104", "title": "Title Four"},
        {"page_id": "105", "title": "Title Five"}
    ]

# --- Test Cases ---

def test_no_clarification_for_no_results(sample_query):
    """
    Verify that no clarification is generated when there are no search results.
    """
    clarification = _generate_clarification_options([], sample_query)
    assert clarification is None

def test_no_clarification_for_single_result(single_search_result, sample_query):
    """
    Verify that no clarification is generated for only one search result.
    """
    clarification = _generate_clarification_options(single_search_result, sample_query)
    assert clarification is None

def test_no_clarification_for_non_distinct_results(non_distinct_search_results, sample_query):
    """
    Verify no clarification is generated if all results are from the same page.
    """
    clarification = _generate_clarification_options(non_distinct_search_results, sample_query)
    assert clarification is None

def test_clarification_generated_for_distinct_results(distinct_search_results, sample_query):
    """
    Verify clarification is generated correctly for multiple distinct pages.
    """
    clarification = _generate_clarification_options(distinct_search_results, sample_query)

    assert clarification is not None
    assert "question_text" in clarification
    assert "options" in clarification
    assert "type" in clarification
    assert clarification["type"] == "document_selection"

    options = clarification["options"]
    assert len(options) == 3  # Three unique page_ids

    # Check that the option IDs match the unique page IDs
    option_ids = {opt["id"] for opt in options}
    assert option_ids == {"101", "102", "103"}

    # Check the text cleaning logic: `title.split(':')[-1].strip().capitalize()`
    option_texts = {opt['text'] for opt in options}
    expected_texts = {
        "Solving connection issues",      # From "Guide: Solving Connection Issues"
        "Why you can't connect", # From "Permissions: Why you can't connect"
        "Server status page"              # From "Server Status Page"
    }
    assert option_texts == expected_texts

def test_clarification_limits_to_four_options(five_distinct_search_results, sample_query):
    """
    Verify that if more than 4 distinct pages are found, options are limited to 4.
    """
    clarification = _generate_clarification_options(five_distinct_search_results, sample_query)

    assert clarification is not None
    assert len(clarification["options"]) == 4

    option_ids = {opt["id"] for opt in clarification["options"]}
    assert option_ids == {"101", "102", "103", "104"}