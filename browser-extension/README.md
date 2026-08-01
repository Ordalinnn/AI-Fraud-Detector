# AI Fraud Detector — Browser Extension

A standalone Chrome/Edge extension version of the AI Fraud Detector. It runs
entirely on your device — no server, no network request, no data collection.

## What it does

- **Right-click any selected text** on a webpage → "Check '...' for fraud/scam"
  → a notification pops up with the risk level.
- **Click the extension icon** to open a popup where you can paste any text
  (SMS, email, message) and analyze it manually.
- The last analysis (from either method) is remembered and shown when you
  reopen the popup.

## How it detects fraud

This extension does **not** use the trained ML ensemble (Logistic Regression /
Random Forest / Gradient Boosting) from the main Streamlit web app — those
models require Python/scikit-learn, which can't run in a browser extension.

Instead, `detector.js` ports the same keyword/domain heuristics (urgency,
secrecy, money, threat, reward, pressure words; suspicious domains and TLDs;
known-brand impersonation) and combines them with hand-tuned weights and the
same rule-based "boost" logic used in the main app. It's a lighter, offline
approximation — not a 1:1 match of the full web app's accuracy — but it needs
no server and never sends your text anywhere.

## Installing it locally (developer mode)

1. Open `chrome://extensions` (or `edge://extensions` in Edge).
2. Turn on **Developer mode** (top-right toggle).
3. Click **Load unpacked** and select this `browser-extension` folder.
4. The AI Fraud Detector icon appears in your toolbar — pin it for easy access.

## Files

- `manifest.json` — Manifest V3 extension config.
- `detector.js` — the fraud-scoring logic (shared by the popup and the
  right-click context menu).
- `background.js` — service worker: registers the right-click menu and shows
  notifications.
- `popup.html` / `popup.css` / `popup.js` — the toolbar popup UI.
- `icons/` — extension icons (reused from the main app's logo).
