# Conrad Chatbot

Conrad is a sophisticated chatbot designed to integrate seamlessly with your browser. It acts as an intelligent assistant, leveraging the knowledge stored in a Confluence database and the natural language processing power of Google's Gemini API to answer user queries.

This repository contains the complete source code for both the Conrad Chrome Extension (frontend) and the FastAPI-based server (backend).

## Features

-   **Conversational AI**: Engage in natural conversations to get information.
-   **Confluence Knowledge Base**: Directly queries and utilizes your existing Confluence pages as its source of truth.
-   **Semantic Search**: Uses vector embeddings (FAISS) and re-ranking (Cross-Encoders) to find the most relevant information, even if the keywords don't match exactly.
-   **Multi-Turn Conversations**: Remembers the context of your questions to handle clarifications (e.g., "Which of these documents are you interested in?").
-   **Chrome Extension Interface**: A user-friendly popup interface that's accessible from any webpage.
-   **Configurable**: Easily point the chatbot to your Confluence instance and spaces through a simple options page.
-   **Dark Mode**: For comfortable viewing in any lighting condition.

## Architecture

The Conrad system consists of two main components:

1.  **Chrome Extension (Frontend)**: The user-facing part of the application, built with HTML, CSS, and JavaScript. It provides the chat popup, options page, and communicates with the backend.
2.  **FastAPI Server (Backend)**: A Python-based server that orchestrates the entire process. It receives requests from the extension, queries the `ConfluenceService`, performs semantic search and ranking, generates responses using the `GeminiService`, and manages conversation state.

## Project Structure

```
.
├── conrad_backend/
│   ├── app/
│   │   ├── core/               # Core components like config and state management
│   │   ├── services/           # Business logic (Confluence, Gemini, Indexing)
│   │   ├── main.py             # FastAPI app, endpoints, and orchestration
│   │   └── schemas.py          # Pydantic models for API requests/responses
│   ├── requirements.txt        # Python dependencies
│   └── ...
├── images/                     # Icons for the Chrome Extension
├── background.js               # Chrome Extension background script
├── content_script.js           # Chrome Extension content script
├── manifest.json               # Chrome Extension manifest
├── options.html & options.js   # Extension configuration page
├── popup.html & popup.js       # Extension chat interface
└── README.md                   # This file
```

---

## Setup and Installation

Follow these steps to get Conrad up and running on your local machine.

### Prerequisites

-   Python 3.9+
-   Google Chrome (or any Chromium-based browser)

### 1. Backend Setup

First, set up the FastAPI server which powers the chatbot's logic.

**a. Clone the Repository**
```bash
git clone <repository_url>
cd <repository_directory>
```

**b. Create and Activate a Python Virtual Environment**
It's highly recommended to use a virtual environment.
```bash
# In the root directory of the project
python -m venv venv
```
Activate it:
-   **Windows**: `.\venv\Scripts\activate`
-   **macOS/Linux**: `source venv/bin/activate`

**c. Install Dependencies**
```bash
# From the conrad_backend directory
pip install -r conrad_backend/requirements.txt
```

**d. Configure Environment Variables**
Create a `.env` file inside the `conrad_backend` directory. You can copy the structure from `.env.example` if one is provided, or create it from scratch.

Fill it with your credentials and desired settings:
```env
# --- Connection Settings ---
CONFLUENCE_URL="https://your-domain.atlassian.net/wiki"
CONFLUENCE_USERNAME="your_email@example.com"
CONFLUENCE_API_TOKEN="your_confluence_api_token"
GEMINI_API_KEY="your_gemini_api_key"

# --- Indexing Settings ---
# Comma-separated list of Confluence space keys to index
CONFLUENCE_SPACE_KEYS_TO_INDEX="SPACE1,SPACE2,DOCS"

# --- Search & Model Settings ---
# (Optional) You can override the default model names and search parameters
# EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2"
# CROSS_ENCODER_MODEL_NAME="cross-encoder/ms-marco-MiniLM-L-6-v2"
# N_SEMANTIC_RESULTS=5
# N_CQL_RESULTS=5
# K_CANDIDATES_FOR_RERANKING=25
# MAX_RESULTS_FOR_CONTEXT=10

# --- CORS Settings ---
# (Optional) Origins to allow for CORS. Defaults are usually fine for local dev.
# CORS_ORIGINS="http://localhost,http://127.0.0.1"
```

**e. Build the Search Index**
This is a critical one-time step. The indexing service will scan your specified Confluence spaces, chunk the content, and create a local FAISS vector index for semantic search.

Run the indexing script from the project's root directory:
```bash
python -m conrad_backend.app.services.indexing_service
```
This process may take some time depending on the amount of content in your Confluence spaces. It will create `faiss_index.idx` and `chunks_data.json` files in your project root.

**f. Run the Backend Server**
From the project's root directory:
```bash
uvicorn conrad_backend.app.main:app --reload
```
The backend API will now be running at `http://127.0.0.1:8000`. You can verify it by visiting the interactive documentation at `http://127.0.0.1:8000/docs`.

### 2. Frontend (Chrome Extension) Setup

Now, load the extension into your browser.

**a. Open Chrome Extensions Page**
Navigate to `chrome://extensions` in your Chrome browser.

**b. Enable Developer Mode**
In the top-right corner, toggle the "Developer mode" switch to **On**.

**c. Load the Extension**
Click the "Load unpacked" button that appears. In the file dialog, select the **root directory** of this project (the one containing `manifest.json`).

The Conrad Chatbot icon should now appear in your browser's toolbar.

## Usage

1.  **Configure the Extension**:
    -   Click the Conrad icon in your toolbar and go to the settings/options page (or right-click the icon and select "Options").
    -   Enter the required API keys and URLs. These settings are stored locally in your browser's storage.
    -   **Note**: The settings on this page are for the *extension itself* and may be separate from the backend's `.env` file, allowing for flexibility. For local development, you can use the same values.

2.  **Start Chatting**:
    -   Click the Conrad icon in your browser toolbar to open the chat popup.
    -   If configured correctly, you'll be greeted by Conrad. Ask it a question related to your Confluence knowledge base!

## API Endpoints

The backend exposes the following endpoints:

-   `POST /chat`
    -   **Description**: The main endpoint for chat interactions. It manages the conversation flow, including initial questions, clarifications, and final answers.
    -   **Request Body**: `UserQuestion` schema (see `schemas.py`)
    -   **Response Body**: `ChatResponse` schema (see `schemas.py`)

-   `GET /health`
    -   **Description**: A health check endpoint to verify that the backend and its services (Confluence, Gemini) are initialized correctly.
    -   **Response**: A JSON object with the status of each service.