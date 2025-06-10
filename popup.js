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
                link.textContent = `Source ${index + 1}`;
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
        loadingIndicator.style.display = 'block';
        errorMessage.style.display = 'none';
        errorMessage.textContent = '';

        chrome.storage.local.get(['confluenceUrl', 'geminiKey'], async (items) => {
            try {
                if (!items.confluenceUrl || !items.geminiKey) {
                    errorMessage.textContent = 'Configuration missing. Please set Confluence URL and Gemini API Key in Settings.';
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
                errorMessage.textContent = 'Network error or Conrad backend not reachable. Ensure it is running on http://127.0.0.1:8000.';
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

    // Initial check for settings and set focus
    chrome.storage.local.get(['confluenceUrl', 'geminiKey'], (items) => {
        if (!items.confluenceUrl || !items.geminiKey) {
            if (chatMessages.children.length === 0) { // Only show if chat is empty
                errorMessage.textContent = "¡Hola! Soy Conrad, tu asistente virtual para la biblioteca de Confluence. ¿En qué te puedo ayudar hoy?";
                errorMessage.classList.remove('error-style'); // Ensure neutral styling
                // Clearing direct styles just in case, though class removal should be enough
                errorMessage.style.color = '';
                errorMessage.style.backgroundColor = '';
                errorMessage.style.display = 'block';
            }
        } else {
            // Optional: If settings ARE found and chat is empty, one could add a different welcome message
            // For now, we only show a special message if settings are MISSING.
            if (chatMessages.children.length === 0) {
                 // Example: addMessageToChat("Conrad is ready. Ask me anything!", "conrad");
            }
        }
    });

    messageInput.focus(); // Set initial focus

    // Clear welcome/config message if user starts typing
    messageInput.addEventListener('input', () => {
        const currentMessage = errorMessage.textContent;
        if (currentMessage.startsWith('¡Hola! Soy Conrad') || currentMessage.startsWith('Configuration missing')) {
            errorMessage.style.display = 'none';
            errorMessage.textContent = '';
        }
    });
});
