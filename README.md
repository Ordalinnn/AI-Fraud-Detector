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
- [Configuration](#configuration)
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
- **How confident the system is** — whether its five underlying models agree or disagree with each other

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
    ├──▶ Logistic Regression   ─┐
    ├──▶ Random Forest          │
    ├──▶ Gradient Boosting      ├─▶ logistic-regression STACKER
    ├──▶ TF-IDF + Naive Bayes   │   (learns how much to trust each model
    ├──▶ Char n-grams + LR      │    from out-of-fold predictions grouped
    └──▶ rule-based boost      ─┘    by template)
                                     │
                                     ▼
              Final risk = calibrated probability → risk level
```

The first three models read **hand-engineered features** (keyword counts, domain checks); TF-IDF + Naive Bayes reads **words**; the character n-gram model reads **pieces of words**, so Kazakh/Russian endings (*картаңыз / карту / карта*) and obfuscated spellings (*k0d*) still match. The rule boost (fires on realistic combinations such as "link + request for a code") is a sixth input.

**Why a stacker, not an average** (changed after the NIS Taraz jury review, 09.10.2026): measured on the same 282 held-out messages, the old plain average of 4 models (82.3% accuracy) was *worse* than its own best member, TF-IDF + NB (86.2%) — the three keyword-feature models outvoted it, and this held in all 15 folds of a repeated grouped cross-validation. The stacker learns the weights instead (the text models get the largest), and because it is fitted with log-loss its output is a calibrated probability: "80% risk" now means about 80% of such messages really are scams. Full evidence: [`reports/jury_evaluation/REPORT_RU.md`](reports/jury_evaluation/REPORT_RU.md).

The pure detection logic (feature extraction, rule boosting, domain/brand checks, risk levels, text highlighting) lives in [`fraud_logic.py`](fraud_logic.py), completely independent of Streamlit, so it's unit-testable and reusable — the browser extension's `detector.js` is a JavaScript port of the same logic. UI strings for all three languages live in [`translations.py`](translations.py), which validates that KZ/RU/EN all define the same set of keys before the app ever renders a page.

## Features

**Analysis modes**
- SMS / message text, call transcript, `.txt` file upload, or batch CSV (many messages at once, with a summary chart and CSV export)
- **Voice / call recording**: record a call on speakerphone with the browser microphone, or upload a voice message or recording (WhatsApp/Telegram `.ogg`/`.opus`, `.mp3`, `.m4a`, `.wav`...). Whisper (via [faster-whisper](https://github.com/SYSTRAN/faster-whisper), model `small` by default, override with the `WHISPER_MODEL` environment variable) turns it into text locally on the server, in Kazakh, Russian or English; the transcript goes into the normal text box to be checked and analysed. Audio is never sent to a third party and is not stored. For Kazakh it switches to a Kazakh fine-tune of Whisper ([akuzdeuov/whisper-base.kk](https://huggingface.co/akuzdeuov/whisper-base.kk), trained on ISSAI KSC2; converted by `.github/workflows/build-kazakh-model.yml` and downloaded on first use with a pinned SHA-256); Russian/English get a short vocabulary hint (bank names, "код из СМС"...) and quiet recordings are loudness-normalised. Measure it on your own recordings with `python scripts/eval_speech.py` (see [`data/voice/README.md`](data/voice/README.md)). Phones don't let ordinary apps record or listen to calls automatically (Android 10+ reserves call audio for system apps; iOS gives none), which is why the user records the call themselves - see [`speech.py`](speech.py).

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
- **Optional Deep AI Analysis**: an opt-in button that sends the message to Claude (Anthropic's API) for a genuine reasoning-based second opinion — not keyword matching, actual language understanding of tactics and intent. Unlike the rest of the app, this specific feature sends text to a third-party server, so it's clearly labeled, rate-limited to one call per 15 seconds per session, and only runs when you click it. Requires an `ANTHROPIC_API_KEY` — see [Configuration](#configuration) below; without one, the button explains that plainly instead of disappearing, and the rest of the app is unaffected.

**Everyday usability**
- Fully bilingual+ UI: 🇰🇿 Kazakh, 🇷🇺 Russian, 🇬🇧 English, switchable without losing your typed text or settings
- A plain-language "how to use this" guide and color legend for non-technical users
- History of past checks, scoped to your own browser session (persisted across restarts, never visible to other visitors) with CSV export
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
fraud_model.py                The 5 base models + stacker, hyperparameters, template grouping,
                               real-message loaders. Shared by app.py, the evaluation and tests.
fraud_eval.py                 Offline evaluation toolkit (dataset loader, metrics, McNemar,
                               cluster bootstrap, calibration, threshold costs).
scripts/run_jury_evaluation.py  Reproduces every number in reports/jury_evaluation/.
scripts/prepare_external_data.py  Splits the real UCI SMS corpus into train sample / holdout.
data/external/                Real UCI SMS: full corpus, 400-message train sample, holdout.
data/real/                    Template for the team's own collected KZ/RU real messages.
reports/jury_evaluation/      Results, charts and the defense report (Russian).
tests/test_fraud_model.py     Pytest suite for fraud_model.py / fraud_eval.py
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

## Configuration

Everything works out of the box with no configuration. The one optional piece is **Deep AI Analysis** (see [Features](#features)), which needs an Anthropic API key:

1. Get a key at [console.anthropic.com](https://console.anthropic.com/settings/keys).
2. Copy [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) to `.streamlit/secrets.toml` and paste your key in. That file is gitignored — never commit a real key.
3. On Streamlit Community Cloud, don't upload a secrets file at all — set `ANTHROPIC_API_KEY` under your app's **Settings → Secrets** in the dashboard instead.

Without a key configured, the button simply explains it isn't set up yet — nothing else in the app is affected.

## Running the tests

```bash
pip install -r requirements-dev.txt
```

```bash
pytest tests/ -v
```

The suite tests `fraud_logic.py` directly (no Streamlit session needed) — fraud/safe detection, brand-impersonation vs. real domains, the rule-boost cap, risk-level boundaries, and HTML-escaping in the text highlighter (an XSS-safety check, not just a happy-path test). It also runs automatically via GitHub Actions on every push and pull request to `main`.

The browser extension's heuristic scorer (`browser-extension/detector.js`) has its own suite, run separately since it's plain JS, not Python:

```bash
cd browser-extension
node --test tests/
```

It also runs automatically via GitHub Actions alongside the Python tests.

## Mobile app (PWA)

The site is installable to a phone's home screen, opening full-screen with no browser address bar:

- **Android (Chrome):** open the site → ⋮ menu → **Add to Home screen**
- **iPhone/iPad (Safari — must be Safari, not an in-app browser like Telegram's):** open the site → Share icon → **Add to Home Screen**

It still needs an internet connection to reach the server — installing it skips the browser UI, it doesn't enable offline use.

## Browser extension

A self-contained Chrome/Edge extension in [`browser-extension/`](browser-extension/) — see its own [README](browser-extension/README.md) for install steps. Runs entirely on-device: your text is never sent anywhere, and it works even without the main web app running. It blends hand-tuned rule-based scoring with the real trained Logistic Regression member's coefficients (ported to plain JS via `scripts/extract_lr_coefficients.py` — a dot product + sigmoid, cheap to run without scikit-learn) at a deliberately minority weight; the Random Forest, Gradient Boosting, and TF-IDF/Naive Bayes ensemble members aren't portable to a browser, so this remains a lighter, offline approximation of the full app, not a 1:1 port of its accuracy.

## Dataset & models

- **1,398 hand-written training examples** (686 fraud, 712 safe; 672 Russian, 605 English, 121 Kazakh), covering 24+ scam categories: bank/card fraud, delivery/prize scams, fake government notices, relative-in-trouble scams, investment/job offers, tech support scams, romance scams, crypto scams, QR code scams, charity scams, subscription/trial scams, SIM swap, social media account recovery, tax refunds, inheritance scams, global brand impersonation (PayPal, Amazon, Apple, Google), business email compromise, AI-generated deepfake investment videos, fake "verify you're human" checks, unpaid-toll SMS scams, AI voice-cloning family emergency scams, crypto romance ("pig butchering") scams, bank "call us back" schemes, NFC tap scams, landlord impersonation, and marketplace fake-verification scams (a fake buyer on Avito/OLX/Telegram asks the seller to pay for a bogus "item check" before a meetup, then disappears).
  The base examples were written from real, current fraud patterns reported by the [FTC](https://consumer.ftc.gov/scams), the [FBI's 2025 IC3 Internet Crime Report](https://www.ic3.gov/AnnualReport/Reports/2025_IC3Report.pdf), the [FCC's toll-scam advisory](https://www.fcc.gov/consumer-governmental-affairs/how-spot-and-avoid-toll-road-payment-scam-texts), reporting on 2026 AI voice-cloning scams, and Kazakhstan/Russia-specific banking fraud coverage (e.g. fake Kaspi.kz couriers demanding SMS codes, Halyk Bank bonus-phishing SMS, Sberbank "call-back" and NFC-tap schemes). The dataset was then scaled up by systematically varying real bank/service names across each template (e.g. the same "your card is blocked" pattern adapted to Kaspi, Halyk Bank, Sberbank, Tinkoff, VTB, and others) — disclosed here as a standard data-scaling technique, not as independently sourced incidents for each line. See the [Limitations](#limitations--disclaimer) section for what this means for generalization.
- **+400 real SMS** (300 legitimate, 100 spam) from the public [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (CC BY 4.0), added because a model trained only on written examples falsely flagged 12–18% of ordinary real messages. The other half of that corpus (2,574 messages, split by template group) is never trained on and serves as an independent real-message test — see [`scripts/prepare_external_data.py`](scripts/prepare_external_data.py).
- **Stacked 5-model ensemble**: Logistic Regression, Random Forest, Gradient Boosting (on the 20 hand-engineered features), TF-IDF + Multinomial Naive Bayes and character n-grams + Logistic Regression (on raw text), combined by a logistic-regression stacker — see [`fraud_model.py`](fraud_model.py).
- **Evaluation**: every number in the app's sidebar comes from one template-grouped held-out test, the same for every model. The full jury evaluation (same-test comparison with McNemar tests, per-language metrics, never-seen scam types, real messages, calibration and threshold costs) is reproducible with `python scripts/run_jury_evaluation.py`; results and charts are in [`reports/jury_evaluation/`](reports/jury_evaluation/). To test on your own collected messages: `python scripts/run_jury_evaluation.py --real data/real/your_file.csv` (columns `text,label[,language]`, template in [`data/real/`](data/real/)).

## Limitations & disclaimer

This is an educational prototype, not a certified security product:

- Many training examples were created by swapping the bank/service name into the same template, so the model may have learned those specific names more than genuinely new phrasing — see [Dataset & models](#dataset--models) above.
- Ordinary cross-validation accuracy would look inflated because many training examples are very similar to each other, so all reported numbers use a template-grouped split instead.
- The real-message test is English only (UCI corpus). Kazakh is the weakest language (121 examples) and romance/investment scams are the hardest never-seen types — see the jury report for exact numbers.
- It only analyzes text — it can't detect voice, video, or real-time deception cues.
- **Always verify suspicious messages through an official source. Never share codes, passwords, CVVs, or card numbers based on any single tool's output, including this one.**

## Privacy

- Analyzed text is never sent to a third-party server, except when you explicitly click **Deep AI Analysis** (see [Features](#features) and [Configuration](#configuration) above) — that's the one deliberate exception, and it's opt-in per message.
- History is **per-browser-session**, not shared across visitors: each session gets its own file under `session_data/` (named with a random session id), so on a multi-user deployment no one can see another visitor's analyzed messages. Nothing in `session_data/` is committed to the git repository (it's in `.gitignore`).
- The browser extension does all analysis on-device — text never leaves your browser.
- Data submitted via the "Was this result accurate?" feedback buttons is aggregated into a single shared `feedback.json` (also gitignored) used only to help improve the model — unlike history, this is intentionally cross-session since its whole purpose is building a better training set, and it's never displayed back to any user.
- CSV exports (batch results and history) neutralize spreadsheet-formula-injection payloads (e.g. text starting with `=`, `+`, `-`, `@`) so opening an export in Excel/Sheets can't execute a formula from analyzed message text.
