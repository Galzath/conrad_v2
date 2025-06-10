// Ensure DOM elements are queried after DOMContentLoaded
let darkModeToggle;

function applyDarkModePreference(isDarkMode) {
    if (isDarkMode) {
        document.body.classList.add('dark-mode');
    } else {
        document.body.classList.remove('dark-mode');
    }
}

function saveOptions(e) {
    e.preventDefault();
    const confluenceUrl = document.getElementById('confluence-url').value;
    const confluenceUsername = document.getElementById('confluence-username').value;
    const confluenceToken = document.getElementById('confluence-token').value;
    const geminiKey = document.getElementById('gemini-key').value;
    const isDarkMode = darkModeToggle.checked; // Get dark mode state

    const status = document.getElementById('status-message');
    status.textContent = ''; // Clear previous messages
    status.classList.remove('success-style', 'error-style');

    // Basic validation for API keys
    if (!confluenceUrl || !geminiKey) {
        status.textContent = 'Por favor, completa los campos requeridos: URL de Confluence y Clave API de Gemini.'; // Localized validation
        status.classList.add('error-style');
        setTimeout(() => {
            status.textContent = '';
            status.classList.remove('error-style');
        }, 5000);
        return;
    }

    chrome.storage.local.set({
        confluenceUrl: confluenceUrl,
        confluenceUsername: confluenceUsername,
        confluenceToken: confluenceToken,
        geminiKey: geminiKey,
        darkMode: isDarkMode // Save dark mode preference
    }, () => {
        if (chrome.runtime.lastError) {
            console.error("Error saving settings:", chrome.runtime.lastError);
            status.textContent = 'Error al guardar. Por favor, revisa los campos.'; // Localized save error
            status.classList.add('error-style');
        } else {
            status.textContent = '¡Configuración guardada con éxito!'; // Localized save success
            status.classList.add('success-style');
            // Apply dark mode immediately after saving, if changed
            applyDarkModePreference(isDarkMode);
        }
        setTimeout(() => {
            status.textContent = '';
            status.classList.remove('success-style', 'error-style');
        }, 3000);
    });
}

function loadOptions() {
    // Get references to all settings fields, including the new dark mode toggle
    const confluenceUrlInput = document.getElementById('confluence-url');
    const confluenceUsernameInput = document.getElementById('confluence-username');
    const confluenceTokenInput = document.getElementById('confluence-token');
    const geminiKeyInput = document.getElementById('gemini-key');
    // darkModeToggle is already globally available after DOMContentLoaded

    const status = document.getElementById('status-message');
    status.textContent = '';
    status.classList.remove('success-style', 'error-style');

    chrome.storage.local.get(['confluenceUrl', 'confluenceUsername', 'confluenceToken', 'geminiKey', 'darkMode'], (items) => {
        if (chrome.runtime.lastError) {
            console.error("Error loading settings:", chrome.runtime.lastError);
            status.textContent = 'Error al cargar la configuración.'; // Localized load error
            status.classList.add('error-style');
            return;
        }

        confluenceUrlInput.value = items.confluenceUrl || '';
        confluenceUsernameInput.value = items.confluenceUsername || '';
        confluenceTokenInput.value = items.confluenceToken || '';
        geminiKeyInput.value = items.geminiKey || '';

        const isDarkMode = !!items.darkMode;
        darkModeToggle.checked = isDarkMode;
        applyDarkModePreference(isDarkMode);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    // Initialize darkModeToggle after DOM is loaded
    darkModeToggle = document.getElementById('dark-mode-toggle');

    loadOptions(); // Load all options, including dark mode state

    document.getElementById('settings-form').addEventListener('submit', saveOptions);

    if(darkModeToggle) { // Ensure toggle exists before adding listener
        darkModeToggle.addEventListener('change', () => {
            const isDarkMode = darkModeToggle.checked;
            applyDarkModePreference(isDarkMode);
            // Note: The preference is saved when the main "Save Settings" button is clicked.
            // If immediate save on toggle change is desired, that's a separate small change.
            // For now, it changes visually and is saved with other settings.
        });
    }
});
