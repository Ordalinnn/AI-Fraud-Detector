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
- **Language switcher** (🇰🇿 KZ / 🇷🇺 RU / 🇬🇧 EN) in the popup header — your
  choice is saved and also applied to notifications and the right-click
  menu label.

## How it detects fraud

This extension does **not** run the full trained ML ensemble (Logistic
Regression + Random Forest + Gradient Boosting + TF-IDF/Naive Bayes) from the
main Streamlit web app as-is — those models require Python/scikit-learn,
which can't run in a browser extension.

`detector.js` ports the same keyword/domain heuristics (urgency, secrecy,
money, threat, reward, pressure words; suspicious domains and TLDs;
known-brand impersonation) and combines them with hand-tuned weights and the
same rule-based "boost" logic used in the main app. It also blends in the
real Logistic Regression ensemble member's trained coefficients — a
StandardScaler + LogisticRegression pipeline reduces to a plain dot product
and sigmoid, so it's cheap to port and doesn't need scikit-learn at runtime
(see `scripts/extract_lr_coefficients.py`, which regenerates the embedded
coefficients from `app.py`'s real training data). That component is given a
deliberately minority weight, since evaluated on its own — without the other
3 ensemble members averaging it out — it over-scores some plain
conversational messages. Random Forest, Gradient Boosting, and the TF-IDF/NB
member aren't portable this way, so this remains a lighter, offline
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
  right-click context menu). Returns language-agnostic keys (risk level,
  reason keys, domain flag keys) rather than hardcoded text, so the same
  result can be rendered in any supported language.
- `i18n.js` — KZ/RU/EN translations, shared by `popup.js` and `background.js`,
  plus helpers for reading/writing the saved language preference.
- `background.js` — service worker: registers the right-click menu and shows
  notifications, both in the currently selected language.
- `popup.html` / `popup.css` / `popup.js` — the toolbar popup UI, including
  the language switcher.
- `icons/` — extension icons at their declared 16/48/128px sizes, generated
  from the same square master icon as the main app's PWA icon
  (`static/icon.png`) via `scripts/generate_icons.py`.
