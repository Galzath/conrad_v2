// background.js

// Listen for clicks on the browser action icon (Manifest V3)
chrome.action.onClicked.addListener((tab) => {
  // Ensure the tab has an ID, which is necessary for chrome.tabs.sendMessage
  if (tab.id) {
    console.log(`Conrad Sidebar: Action icon clicked for tab ${tab.id}. Sending TOGGLE_SIDEBAR message.`);
    chrome.tabs.sendMessage(tab.id, { type: "TOGGLE_SIDEBAR" })
      .then(response => {
        // chrome.runtime.lastError is checked for issues like "No receiving end"
        if (chrome.runtime.lastError) {
          console.error("Conrad Sidebar: Error sending TOGGLE_SIDEBAR message:", chrome.runtime.lastError.message);
          // This error often means the content script isn't active on the page or isn't listening.
          // For example, if the page is a chrome:// URL where content scripts don't run,
          // or if the extension was just installed/updated and the page hasn't been reloaded.
        } else {
          // Optional: handle response from content script if it sends one
          console.log("Conrad Sidebar: TOGGLE_SIDEBAR message sent. Response from content script:", response);
        }
      })
      .catch(error => {
        // This catch is for promise rejections from sendMessage itself (e.g. tab closed before message sent)
        console.error("Conrad Sidebar: Exception sending TOGGLE_SIDEBAR message:", error);
      });
  } else {
    console.error("Conrad Sidebar: Tab ID is missing, cannot send TOGGLE_SIDEBAR message.");
  }
});

// Optional: Log when the extension is installed or updated for debugging
chrome.runtime.onInstalled.addListener(details => {
    if (details.reason === chrome.runtime.OnInstalledReason.INSTALL) {
        console.log("Conrad Sidebar: Extension Installed.");
    } else if (details.reason === chrome.runtime.OnInstalledReason.UPDATE) {
        console.log("Conrad Sidebar: Extension Updated.");
        // Could also be used to set up initial settings in storage if needed.
    }
});

console.log("Conrad Sidebar: background.js service worker started.");
