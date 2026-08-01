// Minimal service worker whose only job is to satisfy the browser's
// installability requirement for "Add to Home Screen". It does NOT cache
// pages for offline use: this app needs a live connection to the Streamlit
// server to run the ML models, so a fully offline mode isn't meaningful
// here — the app just plain won't work without a network connection,
// installed or not.
self.addEventListener("install", () => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("fetch", (event) => {
  event.respondWith(fetch(event.request));
});
