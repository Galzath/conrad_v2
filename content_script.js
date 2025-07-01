// content_script.js
console.log("Conrad Sidebar: content_script.js loaded.");

let sidebarIframe = null;
let isSidebarVisible = false;
const SIDEBAR_IFRAME_ID = 'conrad-sidebar-iframe';
const SIDEBAR_WIDTH = '360px'; // Configurable width

function ensureSidebarExists() {
    if (document.getElementById(SIDEBAR_IFRAME_ID)) {
        sidebarIframe = document.getElementById(SIDEBAR_IFRAME_ID);
        console.log("Conrad Sidebar: Iframe already exists.");
        return true; // Already exists
    }

    console.log("Conrad Sidebar: Creating iframe...");
    sidebarIframe = document.createElement('iframe');
    sidebarIframe.id = SIDEBAR_IFRAME_ID;
    sidebarIframe.src = chrome.runtime.getURL('sidebar.html');

    // Style the iframe itself
    sidebarIframe.style.position = 'fixed';
    sidebarIframe.style.top = '0';
    sidebarIframe.style.right = '0';
    sidebarIframe.style.width = SIDEBAR_WIDTH;
    sidebarIframe.style.height = '100%';
    sidebarIframe.style.border = 'none';
    sidebarIframe.style.zIndex = '2147483647'; // Max z-index
    sidebarIframe.style.boxShadow = '-2px 0px 10px rgba(0,0,0,0.15)';
    sidebarIframe.style.display = 'none'; // Initially hidden
    sidebarIframe.style.transform = `translateX(${SIDEBAR_WIDTH})`; // Start off-screen to the right
    sidebarIframe.style.transition = 'transform 0.3s ease-in-out, opacity 0.3s ease-in-out';
    sidebarIframe.style.opacity = '0'; // Start fully transparent for fade-in effect

    document.body.appendChild(sidebarIframe);
    console.log("Conrad Sidebar: Iframe created and appended to body.");

    // Allow some time for iframe to potentially load its content,
    // though visibility is controlled by transform/opacity.
    // First toggle will make it visible.
    return true;
}

function toggleSidebar() {
    if (!ensureSidebarExists()) { // Creates iframe if it doesn't exist
      console.error("Conrad Sidebar: Could not create or find sidebar iframe.");
      return Promise.reject("Sidebar iframe could not be initialized.");
    }

    if (isSidebarVisible) {
        console.log("Conrad Sidebar: Hiding sidebar.");
        sidebarIframe.style.transform = `translateX(${SIDEBAR_WIDTH})`;
        sidebarIframe.style.opacity = '0';
        // Consider setting display to none after transition for performance, if needed
        // setTimeout(() => { sidebarIframe.style.display = 'none'; }, 300); // Match transition duration
    } else {
        console.log("Conrad Sidebar: Showing sidebar.");
        sidebarIframe.style.display = 'block'; // Make it part of layout before transform
        // Force a reflow before applying the transform for the transition to work on first show
        void sidebarIframe.offsetHeight;

        // Defer style changes that trigger transition to ensure display:block is processed
        setTimeout(() => {
            if (sidebarIframe) { // Ensure sidebarIframe still exists in this async callback
                sidebarIframe.style.transform = 'translateX(0%)';
                sidebarIframe.style.opacity = '1';

        }, 10); // 10ms delay
    }
    isSidebarVisible = !isSidebarVisible;
    console.log("Conrad Sidebar: Visibility toggled. New state:", isSidebarVisible);
    return Promise.resolve({ sidebarVisible: isSidebarVisible });
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    console.log("Conrad Sidebar: Message received in content_script:", request);
    if (request.type === "TOGGLE_SIDEBAR") {
        toggleSidebar()
            .then(response => sendResponse(response))
            .catch(error => sendResponse({ error: error.toString() }));
        return true; // Indicates that the response will be sent asynchronously
    }
    // Handle other message types if any
    return false; // No async response from this listener path
});

// Optional: Attempt to create the sidebar on script load but keep it hidden.
// This might make the first toggle appear slightly faster.
// ensureSidebarExists();
// However, it's generally better to create it on demand to avoid injecting
// the iframe on pages where the user might never use it.
// The current on-demand creation in toggleSidebar is fine.

console.log("Conrad Sidebar: content_script.js event listeners set up.");
