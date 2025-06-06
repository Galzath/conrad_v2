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

The backend exposes the following API endpoints:

-   **`POST /chat`**
    -   **Description**: This is the primary endpoint for interacting with the chatbot. It receives a user's question, orchestrates the search for relevant context in Confluence, generates an answer using the Gemini API, and returns the answer along with URLs of the Confluence pages used as sources.
    -   **Request Body** (`application/json`):
        ```json
        {
            "question": "Your question here"
        }
        ```
    -   **Response Body** (`application/json`):
        ```json
        {
            "answer": "The AI-generated answer.",
            "source_urls": ["url_to_confluence_page_1", "url_to_confluence_page_2"]
        }
        ```

-   **`GET /health`**
    -   **Description**: This endpoint provides a health check of the API. It can be used to monitor the status of the application and its core services (Confluence and Gemini).
    -   **Response Body** (`application/json` - on success):
        ```json
        {
            "status": "ok",
            "confluence_service": "initialized",
            "gemini_service": "initialized"
        }
        ```
    -   **Response Body** (`application/json` - on error, e.g., if a service failed to initialize):
        ```json
        {
            "status": "error",
            "confluence_service": "error_initializing", // example status
            "gemini_service": "initialized",      // example status
            "detail": "One or more critical services are not available."
        }
        ```

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

The project relies on the following Python packages, as listed in `requirements.txt`:

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

### Key Dependency Roles:

-   **`fastapi`**: Core framework for building the API.
-   **`uvicorn[standard]`**: ASGI server for running the FastAPI application. The `[standard]` option includes recommended extras like `httptools` for faster parsing and `websockets` support (though not directly used in this project's current scope).
-   **`requests`**: A simple, yet elegant HTTP library for Python. While `atlassian-python-api` and `google-generativeai` handle their own HTTP communications, `requests` might be included as a dependency of one of these, or it could be available for other potential HTTP interactions.
-   **`google-generativeai`**: SDK for interacting with Google's Gemini API.
-   **`python-dotenv`**: For loading environment variables from a `.env` file.
-   **`atlassian-python-api`**: Client library for Confluence API interaction.
-   **`pydantic`**: Used for data validation, serialization, and settings management.
-   **`beautifulsoup4`**: For parsing HTML content from Confluence pages.

## Setup and Running

For detailed instructions on setting up the development environment, installing dependencies, and running the application, please refer to the main `README.md` file located in the `conrad_backend` root directory.

### Key Configuration: Environment Variables

A crucial part of the setup is configuring the environment variables. The application expects a `.env` file in the `conrad_backend` root directory. This file should contain the following variables:

-   `CONFLUENCE_URL`: The base URL of your Confluence instance (e.g., `https://your-domain.atlassian.net/wiki`).
-   `CONFLUENCE_USERNAME`: Your Confluence username (usually your email address).
-   `CONFLUENCE_API_TOKEN`: Your Confluence API token. This is used for authenticating API requests to Confluence.
-   `GEMINI_API_KEY`: Your API key for Google's Gemini API. This is required to authenticate requests to the Gemini service for response generation.

Ensure these variables are correctly set in your `.env` file before attempting to run the application. Refer to the `README.md` for guidance on creating the `.env` file (e.g., by copying from an `.env.example` if one is provided).
