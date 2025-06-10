function saveOptions(e) {
    e.preventDefault();
    const confluenceUrl = document.getElementById('confluence-url').value;
    const confluenceUsername = document.getElementById('confluence-username').value;
    const confluenceToken = document.getElementById('confluence-token').value;
    const geminiKey = document.getElementById('gemini-key').value;

    const status = document.getElementById('status-message');
    status.textContent = ''; // Clear previous messages
    status.classList.remove('success-style', 'error-style');


    // Basic validation
    if (!confluenceUrl || !geminiKey) {
        status.textContent = 'Error: Confluence URL and Gemini API Key are required.';
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
        geminiKey: geminiKey
    }, () => {
        if (chrome.runtime.lastError) {
            console.error("Error saving settings:", chrome.runtime.lastError);
            status.textContent = 'Error saving settings: ' + chrome.runtime.lastError.message;
            status.classList.add('error-style');
        } else {
            status.textContent = 'Settings saved successfully!';
            status.classList.add('success-style');
        }
        setTimeout(() => {
            status.textContent = '';
            status.classList.remove('success-style', 'error-style');
        }, 3000);
    });
}

function loadOptions() {
    chrome.storage.local.get(['confluenceUrl', 'confluenceUsername', 'confluenceToken', 'geminiKey'], (items) => {
        const status = document.getElementById('status-message');
        status.textContent = ''; // Clear previous messages
        status.classList.remove('success-style', 'error-style');

        if (chrome.runtime.lastError) {
            console.error("Error loading settings:", chrome.runtime.lastError);
            status.textContent = 'Error loading settings: ' + chrome.runtime.lastError.message;
            status.classList.add('error-style');
            return;
        }
        document.getElementById('confluence-url').value = items.confluenceUrl || '';
        document.getElementById('confluence-username').value = items.confluenceUsername || '';
        document.getElementById('confluence-token').value = items.confluenceToken || '';
        document.getElementById('gemini-key').value = items.geminiKey || '';
    });
}

document.addEventListener('DOMContentLoaded', loadOptions);
document.getElementById('settings-form').addEventListener('submit', saveOptions);
