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

// Saves the current chat history to chrome.storage.session
function saveChatHistory() {
    console.log("saveChatHistory: Attempting to save to chrome.storage.session. Current chatHistory:", JSON.parse(JSON.stringify(chatHistory)));
    try {
        // chatHistory is an array of objects. JSON.stringify will handle it.
        chrome.storage.session.set({ conradChatHistory: chatHistory }, () => {
            if (chrome.runtime.lastError) {
                console.error("saveChatHistory: Error saving to chrome.storage.session:", chrome.runtime.lastError.message, "chatHistory state:", JSON.parse(JSON.stringify(chatHistory)));
            } else {
                console.log("saveChatHistory: Successfully saved to chrome.storage.session. chatHistory item count:", chatHistory.length);
            }
        });
    } catch (e) {
        // This catch is primarily for errors if chatHistory itself is in a state that causes JSON.stringify to fail (unlikely for simple objects)
        // or any other synchronous error before the async call.
        console.error("saveChatHistory: Synchronous error before calling chrome.storage.session.set:", e, "chatHistory state:", JSON.parse(JSON.stringify(chatHistory)));
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

    // Loads chat history from chrome.storage.session and renders it
function loadChatHistory() {
    console.log("loadChatHistory: Attempting to load history from chrome.storage.session.");

    chrome.storage.session.get(['conradChatHistory'], (items) => {
        if (chrome.runtime.lastError) {
            console.error("loadChatHistory: Error loading from chrome.storage.session:", chrome.runtime.lastError.message);
            chatHistory = []; // Reset chat history on error
            chatMessages.innerHTML = ''; // Ensure display is cleared
            handleHistoryLoadingFinished(); // Call finish handler even on error
            return;
        }

        const retrievedHistory = items.conradChatHistory;
        // No JSON.parse needed as chrome.storage stores objects directly if they were set as objects.
        console.log("loadChatHistory: Data retrieved from chrome.storage.session. Item count:", retrievedHistory ? retrievedHistory.length : 'undefined');

        if (retrievedHistory && Array.isArray(retrievedHistory)) {
            chatHistory = retrievedHistory;
            console.log("loadChatHistory: Global chatHistory array populated. Item count:", chatHistory.length);

            chatMessages.innerHTML = ''; // Clear display before rendering loaded history

            chatHistory.forEach(message => {
                if (typeof message === 'object' && message !== null && 'text' in message && 'sender' in message) {
                    renderMessage(message.text, message.sender, message.sourceUrls);
                } else {
                    console.warn("loadChatHistory: Invalid message object in stored history:", message);
                }
            });
            console.log("loadChatHistory: Rendered messages from loaded history.");
        } else {
            if (retrievedHistory) { // It exists but is not an array or null
                console.error("loadChatHistory: Stored chat history is not a valid array. Resetting chatHistory.", retrievedHistory);
            } else { // undefined or null
                console.log("loadChatHistory: No history found or history is null in chrome.storage.session.");
            }
            chatHistory = []; // Reset or initialize chat history
            chatMessages.innerHTML = ''; // Ensure display is also cleared
        }
        handleHistoryLoadingFinished(); // Call finish handler after processing
    });
}

// This function consolidates what happens after chat history is loaded or initialized
function handleHistoryLoadingFinished() {
    console.log("handleHistoryLoadingFinished: Chat history loading process complete. Current history length:", chatHistory.length);

    // Logic to display initial welcome message if chat is empty
    if (chatHistory.length === 0) {
        // This part depends on other variables like 'errorMessage' being available in this scope
        // Ensure 'errorMessage' and 'addMessageToChat' are accessible.
        // They are defined in the outer scope of DOMContentLoaded, so they should be.
        chrome.storage.local.get(['confluenceUrl', 'geminiKey'], (settingsItems) => {
            if (chrome.runtime.lastError) {
                console.error("handleHistoryLoadingFinished: Error loading settings for initial message:", chrome.runtime.lastError.message);
                return;
            }
            const isConfigured = settingsItems.confluenceUrl && settingsItems.geminiKey;
            if (isConfigured) {
                addMessageToChat("¡Hola! Soy Conrad, tu asistente virtual para la biblioteca de Confluence. ¿En qué te puedo ayudar hoy?", 'conrad');
            } else {
                // Assuming 'errorMessage' is the DOM element for error messages
                if (errorMessage) {
                    errorMessage.textContent = "Configuración incompleta. Por favor, ve a Configuración.";
                    errorMessage.classList.remove('error-style'); // Ensure no error style for this message
                    errorMessage.style.color = ''; // Reset potential inline styles
                    errorMessage.style.backgroundColor = ''; // Reset potential inline styles
                    errorMessage.style.display = 'block';
                } else {
                    console.error("handleHistoryLoadingFinished: errorMessage DOM element not found.");
                }
            }
        });
    }

    // Ensure the message input is focused
    // Assuming 'messageInput' is the DOM element for the message input field
    if (messageInput) {
        messageInput.focus();
    } else {
        console.error("handleHistoryLoadingFinished: messageInput DOM element not found.");
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

        // Ensure chatMessages is scrolled to the bottom to show the new options
        if (chatMessages) {
            chatMessages.scrollTop = chatMessages.scrollHeight;
            console.log("Scrolled chatMessages to bottom after adding clarification options.");
        }

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

    // The initial message logic is now inside handleHistoryLoadingFinished,
    // so the block that was here is removed.
    // messageInput.focus() is also called inside handleHistoryLoadingFinished.

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
            console.log("Clear Chat button clicked.");
            chatHistory = []; // Clear the global JS array
            chatMessages.innerHTML = ''; // Clear the DOM display immediately

            // Reset error message display (can be done early)
            if (errorMessage) { // Ensure errorMessage element exists
                errorMessage.style.display = 'none';
                errorMessage.textContent = '';
                errorMessage.classList.remove('error-style');
            }

            chrome.storage.session.remove('conradChatHistory', () => {
                if (chrome.runtime.lastError) {
                    console.error("Error removing chat history from chrome.storage.session:", chrome.runtime.lastError.message);
                } else {
                    console.log("Chat history successfully removed from chrome.storage.session.");
                }
                // After removal (or attempted removal), re-run the logic for an empty chat.
                // handleHistoryLoadingFinished will check chatHistory (which is now empty)
                // and display the appropriate initial message and focus input.
                handleHistoryLoadingFinished();
            });
        });
    }
});