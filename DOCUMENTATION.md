# Conrad Chatbot Backend Documentation

## Project Overview

The Conrad Chatbot Backend serves as the server-side logic for Conrad, an intelligent chatbot. Its primary purpose is to connect to a Confluence knowledge base and leverage Google's Gemini Pro API to provide users with accurate and contextually relevant answers to their queries.

The core functionality includes:
- Receiving a user's question through an API endpoint.
- Searching the configured Confluence instance for pages and content relevant to the question.
- Extracting and preparing the retrieved information as context.
- Sending the user's question along with the prepared context to the Gemini Pro API.
- Returning the AI-generated answer and any relevant source Confluence page URLs to the user.

## Backend Functionality

The backend is built using FastAPI and is structured into several key components to handle its operations:

### Core Components

-   **`app/main.py`**: This is the entry point of the FastAPI application. It defines the API endpoints (`/chat`, `/health`), initializes and orchestrates the different services (Confluence and Gemini), and handles incoming HTTP requests and responses.
-   **`app/services/confluence_service.py`**: This service module is responsible for all interactions with the Confluence wiki. It handles connecting to the Confluence API, performing content searches based on user queries, and fetching and parsing the content of specific Confluence pages.
-   **`app/services/gemini_service.py`**: This service module manages the communication with Google's Gemini Pro API. It takes the user's question and the context retrieved from Confluence, formats it into a suitable prompt, sends it to the Gemini API, and retrieves the generated textual response.
-   **`app/core/config.py`**: This module handles the application's configuration. It loads sensitive information like API keys (Confluence, Gemini) and Confluence URL from environment variables (typically stored in a `.env` file), making them available to the rest of the application.
-   **`app/schemas.py`**: This file defines the Pydantic models used for data validation and serialization/deserialization. These models ensure that the data exchanged through API requests and responses conforms to a predefined structure (e.g., `UserQuestion` for incoming questions, `ChatResponse` for outgoing answers).

### API Endpoints

The backend exposes `/chat` (POST) and `/health` (GET) endpoints. For detailed information on request/response structures and usage, please refer to the [API Endpoints section in the main README.md](README.md#api-endpoints).

## Technologies Used

The Conrad Chatbot Backend leverages several key Python technologies and libraries:

-   **FastAPI**: A modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints. It's used to create the `/chat` and `/health` endpoints.
-   **Uvicorn**: An ASGI (Asynchronous Server Gateway Interface) server, used to run the FastAPI application. It provides a high-performance way to serve asynchronous Python web applications.
-   **Pydantic**: A library for data validation and settings management using Python type annotations. It's used in `schemas.py` to define the structure of API request and response bodies and in `core/config.py` for managing settings.
-   **python-dotenv**: A module used to load environment variables from a `.env` file into the application's environment. This is crucial for managing sensitive data like API keys without hardcoding them.
-   **atlassian-python-api**: The official Atlassian Python client library used to interact with the Confluence API. This library facilitates searching for content and retrieving page details from the Confluence instance.
-   **google-generativeai**: The official Google Python SDK for the Gemini API. This library is used to send prompts (user question + Confluence context) to the Gemini Pro model and receive the generated responses.
-   **BeautifulSoup4**: A Python library for pulling data out of HTML and XML files. It's used by the `confluence_service.py` to parse and clean the HTML content retrieved from Confluence pages, extracting plain text for use as context.

## Dependencies

The project relies on the following Python packages (as listed in `requirements.txt`):

-   **`fastapi`**: Core framework for building the API.
-   **`uvicorn[standard]`**: ASGI server for running the FastAPI application. The `[standard]` option includes recommended extras like `httptools` for faster parsing.
-   **`requests`**: A simple HTTP library; may be a sub-dependency or used for other HTTP interactions.
-   **`google-generativeai`**: SDK for interacting with Google's Gemini API.
-   **`python-dotenv`**: For loading environment variables from a `.env` file.
-   **`atlassian-python-api`**: Client library for Confluence API interaction.
-   **`pydantic`**: Used for data validation, serialization, and settings management.
-   **`beautifulsoup4`**: For parsing HTML content from Confluence pages.

Raw `requirements.txt` content:
```
fastapi
uvicorn[standard]
requests
google-generativeai
python-dotenv
atlassian-python-api
pydantic
beautifulsoup4
```

## Setup and Running

For detailed instructions on setting up the development environment, installing dependencies, and running the application, please refer to the main [README.md](README.md) file.

### Key Configuration: Environment Variables

A crucial part of the setup is configuring the environment variables. The application expects a `.env` file in the `conrad_backend` root directory. This file should contain the following variables:

-   `CONFLUENCE_URL`: The base URL of your Confluence instance (e.g., `https://your-domain.atlassian.net/wiki`).
-   `CONFLUENCE_USERNAME`: Your Confluence username (usually your email address).
-   `CONFLUENCE_API_TOKEN`: Your Confluence API token. This is used for authenticating API requests to Confluence.
-   `GEMINI_API_KEY`: Your API key for Google's Gemini API. This is required to authenticate requests to the Gemini service for response generation.

Ensure these variables are correctly set in your `.env` file before attempting to run the application. Refer to the `README.md` for guidance on creating the `.env` file (e.g., by copying from an `.env.example` if one is provided).

## Basic Testing (Conceptual)

While full tests are not yet implemented, here's an outline of tests that should be developed:

### Unit Tests

-   **`app.core.config`**:
    -   Test that settings are loaded correctly from environment variables.
    -   Test default values if environment variables are not set (if applicable).
-   **`app.schemas`**:
    -   Test Pydantic model validation (e.g., `UserQuestion` requires a `question` field).
-   **`app.services.confluence_service`**:
    -   Mock `atlassian.Confluence` and `BeautifulSoup`.
    -   Test `__init__` for successful connection and connection failure.
    -   Test `search_content` with various mock API responses (successful search, no results, API error).
    -   Test `get_page_content_by_id` with various mock API responses (successful content retrieval, page not found, API error, different content formats).
    -   Test HTML parsing logic.
-   **`app.services.gemini_service`**:
    -   Mock `google.generativeai.GenerativeModel`.
    -   Test `__init__` for successful client initialization and API key error.
    -   Test `generate_response` with various mock API responses (successful generation, API error, blocked prompt).
    -   Test prompt formatting.

### Integration Tests

-   **`app.main` (API Endpoints)**:
    -   Use `TestClient` from FastAPI.
    -   Test the `/chat` endpoint:
        -   Mock `ConfluenceService` and `GeminiService` to control their behavior.
        -   Test successful flow: question -> Confluence search -> Gemini response.
        -   Test scenarios:
            -   Confluence finds no documents.
            -   Confluence finds documents, but content extraction fails.
            -   Gemini returns an error.
            -   Confluence service unavailable.
            -   Gemini service unavailable.
        -   Test request validation (e.g., missing `question` field).
    -   Test the `/health` endpoint under different service availability conditions.
-   **Service Integration**:
    -   Potentially, tests that verify the interaction *between* `ConfluenceService` and `GeminiService` via `main.py`, but with external APIs (Confluence, Gemini) mocked at a lower level (e.g., using `httpx` mocks if `requests` or `aiohttp` were used directly, or patching the respective SDK methods).

This conceptual outline provides a starting point for developing a comprehensive test suite.
