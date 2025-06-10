document.addEventListener('DOMContentLoaded', () => {
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const chatMessages = document.getElementById('chat-messages');
    const loadingIndicator = document.getElementById('loading-indicator');
    const errorMessage = document.getElementById('error-message');

    // Apply Dark Mode theme based on preference
    chrome.storage.local.get(['darkMode'], (items) => {
        if (chrome.runtime.lastError) {
            console.error("Error loading dark mode setting:", chrome.runtime.lastError.message);
            // Default to light mode or do nothing, letting CSS defaults apply
            return;
        }
        const isDarkMode = !!items.darkMode;
        if (isDarkMode) {
            document.body.classList.add('dark-mode');
        } else {
            document.body.classList.remove('dark-mode'); // Ensure it's removed if not set or false
        }
    });

    function addMessageToChat(text, sender, sourceUrls = []) {
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
                link.textContent = `Fuente ${index + 1}`; // Localized "Source"
                link.target = '_blank';
                sourcesDiv.appendChild(link);
            });
            messageDiv.appendChild(sourcesDiv);
        }

        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        if (sender === 'user') {
            messageInput.value = '';
            messageInput.focus(); // Return focus after sending
        }
    }

    async function handleSendMessage() {
        const inputText = messageInput.value.trim();
        if (inputText === '') {
            messageInput.focus(); // Focus if trying to send empty
            return;
        }

        addMessageToChat(inputText, 'user'); // This will also focus input due to its internal logic

        loadingIndicator.textContent = "Conrad está escribiendo..."; // Localized loading text
        loadingIndicator.style.display = 'block';
        errorMessage.style.display = 'none';
        errorMessage.textContent = '';

        chrome.storage.local.get(['confluenceUrl', 'geminiKey'], async (items) => {
            try {
                if (!items.confluenceUrl || !items.geminiKey) {
                    errorMessage.textContent = 'Configuración incompleta. Por favor, ve a Configuración.'; // Localized config missing
                    errorMessage.classList.add('error-style');
                    errorMessage.style.display = 'block';
                    loadingIndicator.style.display = 'none';
                    messageInput.focus(); // Focus on config error
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
                    addMessageToChat(data.answer, 'conrad', data.source_urls);
                } else {
                    errorMessage.textContent = `Error: ${data.detail || response.statusText || 'Unknown error from backend'}`;
                    errorMessage.classList.add('error-style');
                    errorMessage.style.display = 'block';
                }
            } catch (error) {
                console.error("Fetch error:", error);
                loadingIndicator.style.display = 'none';
                errorMessage.textContent = 'Error de red o backend de Conrad no accesible.'; // Localized network error
                errorMessage.classList.add('error-style');
                errorMessage.style.display = 'block';
            } finally {
                messageInput.focus(); // Ensure focus after API call attempt
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

    // Initial logic for welcome/configuration messages
    chrome.storage.local.get(['confluenceUrl', 'geminiKey'], (items) => {
        if (chrome.runtime.lastError) {
            console.error("Error loading settings for initial message:", chrome.runtime.lastError.message);
            // Potentially show a generic error, but for now, do nothing specific if storage fails here
            return;
        }

        const isConfigured = items.confluenceUrl && items.geminiKey;
        const isChatEmpty = chatMessages.children.length === 0;

        if (isChatEmpty) {
            if (isConfigured) {
                // Settings ARE CONFIGURED and chat is empty: Show welcome bubble
                addMessageToChat("¡Hola! Soy Conrad, tu asistente virtual para la biblioteca de Confluence. ¿En qué te puedo ayudar hoy?", 'conrad');
                errorMessage.style.display = 'none'; // Ensure no other message is showing
                errorMessage.textContent = '';
            } else {
                // Settings ARE MISSING and chat is empty: Show "Configuración incompleta..." in errorMessage (neutral style)
                errorMessage.textContent = "Configuración incompleta. Por favor, ve a Configuración.";
                errorMessage.classList.remove('error-style'); // Ensure neutral styling
                errorMessage.style.color = ''; // Reset direct styles if any
                errorMessage.style.backgroundColor = '';
                errorMessage.style.display = 'block';
            }
        }
        // If chat is NOT empty, no automatic message is shown by DOMContentLoaded.
    });

    messageInput.focus(); // Set initial focus

    // Clear "Configuración incompleta..." from errorMessage if user starts typing
    messageInput.addEventListener('input', () => {
        const currentMessage = errorMessage.textContent;
        // Only clear the specific neutral message from initial load
        if (currentMessage === "Configuración incompleta. Por favor, ve a Configuración." && !errorMessage.classList.contains('error-style')) {
            errorMessage.style.display = 'none';
            errorMessage.textContent = '';
        }
    });
});
