document.addEventListener('DOMContentLoaded', () => {
    console.log("Popup DOMContentLoaded: Script starting.");
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const chatMessages = document.getElementById('chat-messages');
    const loadingIndicator = document.getElementById('loading-indicator');
    const errorMessage = document.getElementById('error-message');
    const clearChatButton = document.getElementById('clear-chat-button'); // Added constant

    let originalUserQuery = ""; // Declare originalUserQuery
    let chatHistory = []; // Initialize chatHistory array

    // Apply Dark Mode theme based on preference
    chrome.storage.local.get(['darkMode'], (items) => {
        if (chrome.runtime.lastError) {
            console.error("Error loading dark mode setting:", chrome.runtime.lastError.message);
            return;
        }
        if (items.darkMode) {
            document.body.classList.add('dark-mode');
        } else {
            document.body.classList.remove('dark-mode');
        }
    });

    // Saves the current chat history to sessionStorage
    function saveChatHistory() {

        console.log("saveChatHistory: Attempting to save. Current chatHistory:", JSON.parse(JSON.stringify(chatHistory))); // Deep copy for logging
        try {
            const jsonHistory = JSON.stringify(chatHistory);
            sessionStorage.setItem('conradChatHistory', jsonHistory);
            console.log("saveChatHistory: Successfully saved to sessionStorage. JSON:", jsonHistory);
        } catch (e) {
            console.error("saveChatHistory: Error saving chat history to sessionStorage:", e, "chatHistory state:", chatHistory);
        }
    }

    // Renders a message to the chat DOM
    function renderMessage(text, sender, sourceUrls = []) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender === 'user' ? 'user-message' : 'conrad-message');

        const textNode = document.createElement('div');
        textNode.textContent = text;
        messageDiv.appendChild(textNode);

        if (sender === 'conrad' && sourceUrls && sourceUrls.length > 0) {
            const sourcesDiv = document.createElement('div');
            sourcesDiv.className = 'source-urls';
            sourceUrls.forEach((url, index) => {
                const link = document.createElement('a');
                link.href = url;
                link.textContent = `Fuente ${index + 1}`;
                link.target = '_blank';
                sourcesDiv.appendChild(link);
            });
            messageDiv.appendChild(sourcesDiv);
        }

        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Adds a message to chat history, renders it, and saves history
    function addMessageToChat(text, sender, sourceUrls = []) {
        console.log("addMessageToChat: Adding message:", {text, sender});
        chatHistory.push({ text, sender, sourceUrls });
        console.log("addMessageToChat: chatHistory after push:", JSON.parse(JSON.stringify(chatHistory)));
        renderMessage(text, sender, sourceUrls);
        saveChatHistory();

        if (sender === 'user') {
            messageInput.value = '';
            messageInput.focus();
        }
    }

    // Loads chat history from sessionStorage and renders it
    function loadChatHistory() {
        console.log("loadChatHistory: Attempting to load history.");
        const storedHistory = sessionStorage.getItem('conradChatHistory');
        console.log("loadChatHistory: Raw data from sessionStorage:", storedHistory);

        if (storedHistory) {
            try {
                const parsedHistory = JSON.parse(storedHistory);
                console.log("loadChatHistory: Parsed history from sessionStorage:", parsedHistory);
                if (Array.isArray(parsedHistory)) {
                    chatHistory = parsedHistory;
                    console.log("loadChatHistory: Global chatHistory array populated:", JSON.parse(JSON.stringify(chatHistory)));

                    // Clear existing messages before rendering loaded history
                    chatMessages.innerHTML = ''; // Clear display before rendering loaded history

                    chatHistory.forEach(message => {
                        renderMessage(message.text, message.sender, message.sourceUrls);
                    });
                    console.log("loadChatHistory: Rendered messages from loaded history.");
                } else {
                    console.error("loadChatHistory: Stored chat history is not an array. Resetting chatHistory.", parsedHistory);
                    chatHistory = [];
                    chatMessages.innerHTML = ''; // Ensure display is also cleared
                }
            } catch (e) {
                console.error("loadChatHistory: Error parsing chat history from sessionStorage:", e, "Raw data:", storedHistory);
                chatHistory = [];
                chatMessages.innerHTML = ''; // Ensure display is also cleared
            }
        } else {
            console.log("loadChatHistory: No history found in sessionStorage.");
            // chatHistory remains empty, which is the default.
            // Display should also be empty if no history.
            chatMessages.innerHTML = '';
        }
    }

    loadChatHistory();

    // Implements the UI for clarification questions
    function displayClarificationUI(questionText, options, sessionId, originalQuery) {
        console.log("displayClarificationUI called with:", questionText, options, sessionId, originalQuery);

        addMessageToChat(questionText, 'conrad'); // Display the clarification question

        // Create and display clarification options
        const optionsContainerId = 'clarification-options-container';
        let optionsContainer = document.getElementById(optionsContainerId);
        if (!optionsContainer) {
            optionsContainer = document.createElement('div');
            optionsContainer.id = optionsContainerId;
            chatMessages.appendChild(optionsContainer); // Append to chat messages for visibility
        }
        optionsContainer.innerHTML = ''; // Clear previous options

        options.forEach(option => {
            const button = document.createElement('button');
            button.textContent = option.text;
            button.dataset.id = option.id;
            button.classList.add('clarification-option'); // For styling
            button.addEventListener('click', () => {
                addMessageToChat(`Tu selección: "${option.text.substring(0, 50)}..."`, 'user');
                handleClarificationSelection(option.id, sessionId, originalQuery);
                optionsContainer.innerHTML = ''; // Clear options after selection
            });
            optionsContainer.appendChild(button);
        });

        // Disable user input field
        const userInputField = document.getElementById('message-input');
        if (userInputField) {
            userInputField.disabled = true;
        }
    }

    // Handles the user's selection from the clarification options
    async function handleClarificationSelection(selectedOptionId, sessionId, originalQuery) {
        console.log("handleClarificationSelection called with:", selectedOptionId, sessionId, originalQuery);
        // Optional: Show what was selected to the user.
        // addMessageToChat('Procesando tu selección...', 'user'); // Using a generic message // MODIFIED: This line is removed

        const userInputField = document.getElementById('message-input'); // Use actual ID
        // Input field is already disabled by displayClarificationUI. It will be re-enabled in finally.

        // Show loading indicator for processing selection
        loadingIndicator.textContent = "Procesando selección...";
        loadingIndicator.style.display = 'block';
        errorMessage.style.display = 'none';


        try {
            const response = await fetch('http://127.0.0.1:8000/chat', { // Use actual backend URL
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: originalQuery,
                    session_id: sessionId,
                    selected_option_id: selectedOptionId
                })
            });

            if (!response.ok) {
                // Try to get error message from backend if available
                let errorMsg = `HTTP error! Status: ${response.status}`;
                try {
                    const errorData = await response.json();
                    // Prefer 'answer' if available and looks like an error message, else 'detail'
                    errorMsg = errorData.answer && typeof errorData.answer === 'string' ? errorData.answer : (errorData.detail || errorMsg);
                } catch (e) { /* ignore if error response is not json */ }
                throw new Error(errorMsg);
            }

            const data = await response.json();
            // Assuming this response is always a direct answer as per requirements
            if (data.needs_clarification === true) {
                // This case should ideally not happen if backend honors clarification selection
                console.warn("Received needs_clarification=true after selecting an option.");
                displayClarificationUI(data.clarification_question_text, data.clarification_options, data.session_id, originalQuery);
            } else {
                displayDirectAnswer(data.answer, data.source_urls);
            }

        } catch (error) {
            console.error('Error fetching clarification follow-up:', error);
            // Display error in the chat
            addMessageToChat(`Lo siento, hubo un error procesando tu selección: ${error.message}`, 'conrad');
            // Also display in the dedicated error message area if it makes sense
            errorMessage.textContent = `Error procesando selección: ${error.message}`;
            errorMessage.classList.add('error-style');
            errorMessage.style.display = 'block';

        } finally {
            loadingIndicator.style.display = 'none'; // Hide loading indicator
            if (userInputField) {
                userInputField.disabled = false; // Re-enable input
                userInputField.focus(); // Focus on input field
            }
            // Ensure clarification options container is cleared (click handler in displayClarificationUI should also do this)
            const optionsContainer = document.getElementById('clarification-options-container');
            if (optionsContainer) {
                optionsContainer.innerHTML = '';
            }
        }
    }

    // Displays a direct answer from Conrad in the chat.
    function displayDirectAnswer(answerText, sourceUrls) {
        // Log for debugging (optional)
        console.log("Displaying direct answer:", answerText, "Sources:", sourceUrls);

        // Use the existing function to add the message and sources to the chat.
        // addMessageToChat handles rendering of the answer and clickable source URLs.
        addMessageToChat(answerText, 'conrad', sourceUrls);

        // Defensive cleanup of clarification UI, in case it wasn't cleared properly.
        const optionsContainer = document.getElementById('clarification-options-container');
        if (optionsContainer) {
            optionsContainer.innerHTML = '';
        }

        // Re-enabling the input field is handled by the calling functions
        // (handleSendMessage or handleClarificationSelection) in their respective finally blocks,
        // or not disabled at all in the case of a direct flow through handleSendMessage.
    }

    async function handleSendMessage() {
        const inputText = messageInput.value.trim();
        if (inputText === '') {
            messageInput.focus();
            return;
        }

        originalUserQuery = inputText; // Assign trimmed input to originalUserQuery
        addMessageToChat(inputText, 'user');

        loadingIndicator.textContent = "Conrad está escribiendo...";
        loadingIndicator.style.display = 'block';
        errorMessage.style.display = 'none';
        errorMessage.textContent = '';

        chrome.storage.local.get(['confluenceUrl', 'geminiKey'], async (items) => {
            try {
                if (!items.confluenceUrl || !items.geminiKey) {
                    errorMessage.textContent = 'Configuración incompleta. Por favor, ve a Configuración.';
                    errorMessage.classList.add('error-style');
                    errorMessage.style.display = 'block';
                    loadingIndicator.style.display = 'none';
                    messageInput.focus();
                    return;
                }

                const response = await fetch('http://127.0.0.1:8000/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: inputText }),
                });

                loadingIndicator.style.display = 'none';
                const data = await response.json();

                if (response.ok) {
                    if (data.needs_clarification === true) {
                        displayClarificationUI(data.clarification_question_text, data.clarification_options, data.session_id, originalUserQuery);
                    } else {
                        displayDirectAnswer(data.answer, data.source_urls);
                    }
                } else {
                    // Try to get error message from backend if available
                    let errorMsg = `Error del servidor: ${response.status}`;
                    try {
                        const errorData = await response.json();
                        // Prefer 'answer' if available and looks like an error message, else 'detail'
                        errorMsg = errorData.answer && typeof errorData.answer === 'string' ? errorData.answer : (errorData.detail || errorMsg);
                    } catch (e) { /* ignore if error response is not json */ }
                    throw new Error(errorMsg);
                }
            } catch (error) {
                console.error("Error in handleSendMessage:", error);
                loadingIndicator.style.display = 'none';
                
                const displayErrorMessage = error.message || 'Error de red o backend no accesible.';
                errorMessage.textContent = displayErrorMessage;
                errorMessage.classList.add('error-style');
                errorMessage.style.display = 'block';
                // Add error message to chat for visibility
                addMessageToChat(`Lo siento, ocurrió un error: ${displayErrorMessage}`, 'conrad');
            } finally {
                if (document.activeElement !== messageInput) {
                    messageInput.focus();
                }
            }
        });
    }

    sendButton.addEventListener('click', handleSendMessage);
    messageInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            handleSendMessage();
        }
    });

    if (chatHistory.length === 0) {
        chrome.storage.local.get(['confluenceUrl', 'geminiKey'], (items) => {
            if (chrome.runtime.lastError) {
                console.error("Error loading settings for initial message:", chrome.runtime.lastError.message);
                return;
            }
            const isConfigured = items.confluenceUrl && items.geminiKey;
            if (isConfigured) {
                addMessageToChat("¡Hola! Soy Conrad, tu asistente virtual para la biblioteca de Confluence. ¿En qué te puedo ayudar hoy?", 'conrad');
            } else {
                errorMessage.textContent = "Configuración incompleta. Por favor, ve a Configuración.";
                errorMessage.classList.remove('error-style');
                errorMessage.style.color = '';
                errorMessage.style.backgroundColor = '';
                errorMessage.style.display = 'block';
            }
        });
    }

    messageInput.focus();

    messageInput.addEventListener('input', () => {
        const currentMessage = errorMessage.textContent;
        if (currentMessage === "Configuración incompleta. Por favor, ve a Configuración." && !errorMessage.classList.contains('error-style')) {
            errorMessage.style.display = 'none';
            errorMessage.textContent = '';
        }
    });

    // Event listener for the Clear Chat button
    if (clearChatButton) {
        clearChatButton.addEventListener('click', () => {
            chatHistory = []; // Clear the global JS array
            sessionStorage.removeItem('conradChatHistory'); // Clear from sessionStorage
            chatMessages.innerHTML = ''; // Clear the DOM display
            console.log("Chat history cleared.");

            // Reset error message display before showing potential new message
            errorMessage.style.display = 'none';
            errorMessage.textContent = '';
            errorMessage.classList.remove('error-style');

            // Optionally, re-display initial welcome message or config status
            chrome.storage.local.get(['confluenceUrl', 'geminiKey'], (items) => {
                if (chrome.runtime.lastError) {
                    console.error("Error loading settings for initial message after clear:", chrome.runtime.lastError.message);
                    return;
                }
                const isConfigured = items.confluenceUrl && items.geminiKey;
                if (isConfigured) {
                    // Call addMessageToChat to add welcome and save it (as it's a new session start)
                    addMessageToChat("¡Hola! Soy Conrad, tu asistente virtual para la biblioteca de Confluence. ¿En qué te puedo ayudar hoy?", 'conrad');
                } else {
                    errorMessage.textContent = "Configuración incompleta. Por favor, ve a Configuración.";
                    // Ensure neutral styling for config message
                    errorMessage.classList.remove('error-style');
                    errorMessage.style.color = '';
                    errorMessage.style.backgroundColor = '';
                    errorMessage.style.display = 'block';
                }
            });
            messageInput.focus();
        });
    }
});