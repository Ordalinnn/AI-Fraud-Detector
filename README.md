# 🔐 AI Fraud Detector

An AI-powered prototype that detects fraud and scam patterns in SMS messages, chat messages, and call transcripts — in Kazakh, Russian, and English. Built with Streamlit and scikit-learn as an educational/science-project demonstration of applied machine learning for consumer safety.

**Live app:** deployed via Streamlit Community Cloud.
**Not a certified security tool.** See [Limitations](#limitations--disclaimer) below.

---

## Table of contents

- [What it does](#what-it-does)
- [How it works](#how-it-works)
- [Features](#features)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Running the tests](#running-the-tests)
- [Mobile app (PWA)](#mobile-app-pwa)
- [Browser extension](#browser-extension)
- [Dataset & models](#dataset--models)
- [Limitations & disclaimer](#limitations--disclaimer)
- [Privacy](#privacy)

---

## What it does

You paste in a suspicious message (or upload a file, or a whole spreadsheet of messages), and the app tells you:

- **How likely it is to be fraud** (a 0–100% risk score)
- **Why** — which specific words, phrases, or link patterns triggered the score
- **What to do about it** — plain-language safety advice
- **How confident the system is** — whether its four underlying models agree or disagree with each other

It's designed to be understandable by anyone, not just technical users: a plain-language "how to use this" guide, a color legend, and tooltips explain the interface without requiring any ML background.

## How it works

```
Input text
    │
    ▼
extract_features()  →  20 hand-engineered features
    │                   (urgency/secrecy/money/threat word counts,
    │                    link presence, domain reputation, brand
    │                    impersonation, text statistics...)
    │
    ├──▶ Logistic Regression  ─┐
    ├──▶ Random Forest         ├─▶ average probability
    ├──▶ Gradient Boosting    ─┘
    │
    └──▶ TF-IDF + Naive Bayes  (learns word patterns directly
         (trained on raw text)  from the training text itself,
                                 not just the hand-picked keyword lists)
    │
    ▼
Ensemble average  +  rule-based boost (capped at +30%)
    │                (fires on realistic combinations, e.g.
    │                 "link + request for a code" or "brand
    │                 impersonation", not single features alone)
    ▼
Final risk score → risk level (Low / Suspicious / High / Critical)
```

The first three models are trained on **hand-engineered features** (keyword counts, domain checks). The fourth model (TF-IDF + Naive Bayes) is trained directly on the **raw text**, so it can pick up on fraud-indicative language patterns the hand-curated keyword lists don't explicitly cover. Averaging all four, rather than trusting a single model, is what the in-app "model agreement" indicator is measuring — when the four disagree strongly, the UI flags the result as lower-confidence.

The pure detection logic (feature extraction, rule boosting, domain/brand checks, risk levels, text highlighting) lives in [`fraud_logic.py`](fraud_logic.py), completely independent of Streamlit, so it's unit-testable and reusable — the browser extension's `detector.js` is a JavaScript port of the same logic. UI strings for all three languages live in [`translations.py`](translations.py), which validates that KZ/RU/EN all define the same set of keys before the app ever renders a page.

## Features

**Analysis modes**
- SMS / message text, call transcript, `.txt` file upload, or batch CSV (many messages at once, with a summary chart and CSV export)

**Explainability**
- Highlighted trigger words directly in your input text
- A feature-contribution table (which words moved the Logistic Regression score, and by how much)
- A feature-importance table (what the Random Forest model weighs most)
- "Words the model learned" — the strongest fraud/safe-associated words the TF-IDF model picked up from the training data itself
- Domain analysis tab: flags suspicious keywords, TLDs, long/numeric domains, and brand impersonation (e.g. `kaspi-login.xyz` flagged as impersonating `kaspi.kz`, while the real `kaspi.kz` is *not* flagged)

**Trust & feedback**
- A model-agreement indicator (do all 4 models agree, or should you double-check manually?)
- "Was this result accurate?" feedback buttons, logged locally for future dataset improvements
- A Methodology / Model Card section: dataset composition, model architecture, honest limitations, and privacy notes

**Everyday usability**
- Fully bilingual+ UI: 🇰🇿 Kazakh, 🇷🇺 Russian, 🇬🇧 English, switchable without losing your typed text or settings
- A plain-language "how to use this" guide and color legend for non-technical users
- History of past checks (persisted across restarts) with CSV export
- Export any result as a TXT report or JSON file

**Beyond the browser tab**
- Installable as a home-screen app on mobile (PWA)
- A standalone browser extension: right-click any selected text on any webpage to check it, entirely on-device, no server call

## Project structure

```
app.py                        Streamlit app: UI, model training, all pages
fraud_logic.py                Pure detection logic (no Streamlit dependency) — feature
                               extraction, rule boosting, domain/brand checks, risk levels,
                               text highlighting. Imported by app.py, covered by tests/.
translations.py                KZ/RU/EN UI strings. Validates at import time that all three
                               languages have identical key sets, so a missing translation
                               fails loudly at startup instead of silently as a runtime
                               KeyError the first time a user picks that language.
tests/test_fraud_logic.py     Pytest suite for fraud_logic.py
tests/test_translations.py    Pytest suite for translations.py
.github/workflows/tests.yml   Runs the test suite on every push/PR

static/
  manifest.json                Web App Manifest (name, icon, theme) for "Add to Home Screen"
  sw.js                        Minimal service worker (installability only, no offline caching)
  icon.png                     App icon

browser-extension/            Standalone Chrome/Edge extension (Manifest V3)
  detector.js                   JS port of fraud_logic.py's rule-based logic
  i18n.js                       KZ/RU/EN translations shared by the popup and background script
  background.js                 Right-click "check for fraud" + notifications
  popup.html / .js / .css       Toolbar popup UI
  manifest.json                 Extension manifest
  README.md                     Extension-specific install/usage instructions

.streamlit/config.toml        Forces a light theme (so text stays readable regardless of the
                               visitor's OS dark/light preference) and enables static file serving
requirements.txt               Runtime dependencies (streamlit, scikit-learn, pandas, numpy)
requirements-dev.txt           Adds pytest for running the test suite
```

## Getting started

Requires Python 3.9+.

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Running the tests

```bash
pip install -r requirements-dev.txt
```

```bash
pytest tests/ -v
```

The suite tests `fraud_logic.py` directly (no Streamlit session needed) — fraud/safe detection, brand-impersonation vs. real domains, the rule-boost cap, risk-level boundaries, and HTML-escaping in the text highlighter (an XSS-safety check, not just a happy-path test). It also runs automatically via GitHub Actions on every push and pull request to `main`.

## Mobile app (PWA)

The site is installable to a phone's home screen, opening full-screen with no browser address bar:

- **Android (Chrome):** open the site → ⋮ menu → **Add to Home screen**
- **iPhone/iPad (Safari — must be Safari, not an in-app browser like Telegram's):** open the site → Share icon → **Add to Home Screen**

It still needs an internet connection to reach the server — installing it skips the browser UI, it doesn't enable offline use.

## Browser extension

A self-contained Chrome/Edge extension in [`browser-extension/`](browser-extension/) — see its own [README](browser-extension/README.md) for install steps. Runs entirely on-device: your text is never sent anywhere, and it works even without the main web app running. It uses hand-tuned rule-based scoring (not the trained ML ensemble, since scikit-learn models can't run in a browser) as a lighter, offline approximation of the full app.

## Dataset & models

- **262 hand-written training examples** (150 fraud, 112 safe) in Kazakh, Russian, and English, covering 23+ scam categories: bank/card fraud, delivery/prize scams, fake government notices, relative-in-trouble scams, investment/job offers, tech support scams, romance scams, crypto scams, QR code scams, charity scams, subscription/trial scams, SIM swap, social media account recovery, tax refunds, inheritance scams, global brand impersonation (PayPal, Amazon, Apple, Google), business email compromise, AI-generated deepfake investment videos, fake "verify you're human" checks, unpaid-toll SMS scams, AI voice-cloning family emergency scams, crypto romance ("pig butchering") scams, bank "call us back" schemes, NFC tap scams, and landlord impersonation.
  Written from real, current fraud patterns reported by the [FTC](https://consumer.ftc.gov/scams), the [FBI's 2025 IC3 Internet Crime Report](https://www.ic3.gov/AnnualReport/Reports/2025_IC3Report.pdf), the [FCC's toll-scam advisory](https://www.fcc.gov/consumer-governmental-affairs/how-spot-and-avoid-toll-road-payment-scam-texts), reporting on 2026 AI voice-cloning scams, and Kazakhstan/Russia-specific banking fraud coverage (e.g. fake Kaspi.kz couriers demanding SMS codes, Halyk Bank bonus-phishing SMS, Sberbank "call-back" and NFC-tap schemes) — written as new representative example sentences inspired by the documented patterns, not copied from any source.
- **4-model ensemble**: Logistic Regression, Random Forest, and Gradient Boosting (scikit-learn, trained on the 20 hand-engineered features), plus TF-IDF + Multinomial Naive Bayes (trained on raw text).
- Full breakdown — including exact per-model cross-validated accuracy and a fuller discussion of limitations — is available inside the running app under **"📋 Methodology & Model Card."**

## Limitations & disclaimer

This is an educational prototype, not a certified security product:

- The training dataset is small (a few hundred examples), so generalization to entirely new scam wording is limited.
- Cross-validated accuracy may look inflated because many training examples are similar to each other.
- It only analyzes text — it can't detect voice, video, or real-time deception cues.
- **Always verify suspicious messages through an official source. Never share codes, passwords, CVVs, or card numbers based on any single tool's output, including this one.**

## Privacy

- Analyzed text is never sent to a third-party server. The Streamlit app stores your session's history locally on the server it's running on (`history.json`), not in the git repository (it's in `.gitignore`).
- The browser extension does all analysis on-device — text never leaves your browser.
- Data submitted via the feedback buttons is used only to help improve the model.
