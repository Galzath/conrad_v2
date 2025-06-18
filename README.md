# Conrad Chatbot Backend

This project is the backend for Conrad, a chatbot that uses Confluence as a knowledge base and Google's Gemini API for generating responses.

## Project Structure

```
conrad_backend/
├── app/
│   ├── __init__.py
│   ├── main.py             # FastAPI app, API endpoints, orchestration
│   ├── schemas.py          # Pydantic models for request/response
│   ├── services/
│   │   ├── __init__.py
│   │   ├── confluence_service.py # Logic for Confluence API interaction
│   │   └── gemini_service.py     # Logic for Gemini API interaction
│   └── core/
│       ├── __init__.py
│       └── config.py         # Configuration loading (API keys, URLs)
├── .env                    # Environment variables (API keys, etc.) - NOT versioned
├── .gitignore              # Specifies intentionally untracked files
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    # git clone <repository_url>
    # cd conrad_backend
    ```

2.  **Create a Python virtual environment:**
    It's recommended to use a virtual environment to manage project dependencies.
    ```bash
    python -m venv venv
    ```
    Activate the virtual environment:
    -   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    -   On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    Create a `.env` file in the `conrad_backend` root directory by copying the example or creating a new one.
    ```bash
    cp .env.example .env  # If you create an .env.example
    ```
    Or create `.env` manually and add the following, replacing placeholder values with your actual credentials:
    ```env
    CONFLUENCE_URL="https://your-domain.atlassian.net/wiki"
    CONFLUENCE_USERNAME="your_email@example.com"
    CONFLUENCE_API_TOKEN="your_confluence_api_token"
    GEMINI_API_KEY="your_gemini_api_key"
    ```
    -   `CONFLUENCE_URL`: The base URL of your Confluence instance.
    -   `CONFLUENCE_USERNAME`: Your Confluence username (usually email).
    -   `CONFLUENCE_API_TOKEN`: Your Confluence API token. [How to create a Confluence API token](https://support.atlassian.com/atlassian-account/docs/manage-api-tokens-for-your-atlassian-account/).
    -   `GEMINI_API_KEY`: Your Google Gemini API key. [How to get a Gemini API key](https://ai.google.dev/tutorials/python_quickstart#set_up_your_api_key).

## Running the Application

1.  **Ensure your virtual environment is activated.**
2.  **Run the FastAPI application using Uvicorn:**
    ```bash
    uvicorn app.main:app --reload
    ```
    -   `--reload` enables auto-reloading when code changes, useful for development.

3.  The API will typically be available at `http://127.0.0.1:8000`.
    -   You can access the API documentation (Swagger UI) at `http://127.0.0.1:8000/docs`.
    -   The health check endpoint is at `http://127.0.0.1:8000/health`.

## API Endpoints

-   **`POST /chat`**:
    -   **Description**: Receives a user's question, searches Confluence for relevant context, uses Gemini to generate an answer, and returns the answer.
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

-   **`GET /health`**:
    -   **Description**: Provides a health check of the API and its services.
    -   **Response Body** (`application/json`):
        ```json
        {
            "status": "ok", // or "error"
            "confluence_service": "initialized", // or "error_initializing"
            "gemini_service": "initialized" // or "error_initializing"
        }
        ```
