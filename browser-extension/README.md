# AI Fraud Detector — Browser Extension

A standalone Chrome/Edge extension version of the AI Fraud Detector. It runs
entirely on your device — no server, no network request, no data collection.

## What it does

- **Automatic checks in WhatsApp Web, Telegram Web and Gmail.** While one of
  these is open, the extension watches for new incoming messages and puts a
  red label **"⚠️ Looks like fraud · 87% · why?"** next to a suspicious one.
  Click it to see why: the words that raised the risk (also highlighted in
  yellow inside the message), the warning signs, what is wrong with the
  links, and what to do. Your own outgoing
  messages are not checked. Can be switched off in the popup.
- **Right-click any selected text** on any other webpage → "Check '...' for
  fraud/scam" → a notification pops up with the risk level.
- **Click the extension icon** to open a popup where you can paste any text
  (SMS, email, message) and analyze it manually.
- **"The model is wrong"**: after a check in the popup and as "Not fraud" in
  the in-page panel. "Not fraud" takes effect at once: the red label turns
  into a quiet note and that exact message is not flagged again on this
  computer (until the marks are cleared in the popup). Reports stay in
  chrome.storage; the popup downloads them as CSV in the team's
  real-message format (`data/real/README.md`).
- **Disguised text** ("к0д", Latin look-alike letters, "к о д ы", invisible
  characters) is undone before the model reads it (`normalize.js`, same
  algorithm as the site and the Android app) and shown as a warning sign.
- **Language switcher** (🇰🇿 KZ / 🇷🇺 RU / 🇬🇧 EN) in the popup header. Until
  you pick one, the browser's language is used.

Everything runs inside the extension. Message text goes from the page to the
extension's own background worker and only a number comes back; there is no
server and no network request.

## How it detects fraud

The extension runs **the website's own model**, so the same text gets the
same percentage and the same verdict on the site and in the extension:

- the 5 base models — logistic regression, random forest and gradient
  boosting on the 25 rule features, word TF-IDF + Naive Bayes, and character
  n-grams + logistic regression;
- the stacker that combines them with the rule boost.

`scripts/export_extension_model.py` trains exactly what `app.py` trains
(same data, same order, same settings) and writes every number into
`ensemble_model.json` (~3 MB); `ensemble.js` computes the same formulas in
JavaScript, and `detector.js` computes the same 25 features as
`fraud_logic.py` (including Python's Unicode rules for digits, spaces, word
boundaries and rounding).

How we know they are the same:

- `tests/ensemble.test.js` — on ~4,500 texts (the whole training set, real
  SMS, disguised scams, tricky Unicode) every feature is identical to
  Python, every model's probability and the final one differ from
  scikit-learn by < 10⁻⁹, and the shown percentage is the same text;
- `tests/test_extension_ensemble.py` (Python) — the exported model is
  trained on today's training set, a model trained now gives the same
  numbers, and the real `app.py`, run headless, shows those numbers.

Both the site (under the result) and the popup (at the bottom) show
**"model xxxxxxxx"** — a fingerprint of the training set. Same id = same
model. When the dataset changes, re-run the export script; the tests fail
until you do.

`detector.js` also adds the human-readable part: reasons (urgency, codes,
money, threats…) and link/domain checks (look-alike brands, risky TLDs).
If the model file ever fails to load, it falls back to its older rule-based
score, so the extension keeps working.

Results can still differ for one reason only: the text itself. On WhatsApp
Web, Telegram Web and Gmail the extension reads the message from the page,
which may not be exactly what you paste into the site. The site's threshold
slider (default 0.5) changes the site's verdict, not the percentage; the
extension always uses 0.5.

## Automatic checks: how and limits

`content.js` is injected only on `web.whatsapp.com`, `web.telegram.org` and
`mail.google.com` (no other site permissions). It uses a `MutationObserver`,
debounced, finds message elements by CSS selectors, scores new texts in
batches and caches results (WhatsApp redraws messages when you scroll).

- These web apps change their HTML from time to time. If labels stop
  appearing, the selectors in `SITES` in `content.js` need updating; the
  right-click check and the popup keep working regardless.
- Threshold 0.5 — the "Normal" sensitivity of the Android app (catches ~93 %
  of scams in the jury test, ~1 false alarm per 100 real ordinary SMS).
- Messages shorter than 12 characters are skipped.

## Installing it locally (developer mode)

1. Open `chrome://extensions` (or `edge://extensions` in Edge).
2. Turn on **Developer mode** (top-right toggle).
3. Click **Load unpacked** and select this `browser-extension` folder.
4. The AI Fraud Detector icon appears in your toolbar — pin it for easy access.

## Files

- `manifest.json` — Manifest V3 extension config.
- `ensemble_model.json` + `ensemble.js` — the website's model (5 models +
  stacker), run on the device.
- `scorer.js` — the character n-gram model inside it (also the Android app's model).
- `normalize.js` — undoes disguises before the model and the rules read text.
- `mistakes.js` — "the model is wrong" reports and their CSV export.
- `content.js` / `content.css` — automatic checks in WhatsApp Web, Telegram
  Web and Gmail, and the red label.
- `detector.js` — the 25 rule features, reasons, domain checks and the fallback scoring logic (shared by the popup and the
  right-click context menu). Returns language-agnostic keys (risk level,
  reason keys, domain flag keys) rather than hardcoded text, so the same
  result can be rendered in any supported language.
- `i18n.js` — KZ/RU/EN translations, shared by `popup.js` and `background.js`,
  plus helpers for reading/writing the saved language preference.
- `background.js` — service worker: loads the model, scores texts for
  `content.js`, registers the right-click menu and shows notifications.
- `popup.html` / `popup.css` / `popup.js` — the toolbar popup UI, including
  the language switcher.
- `icons/` — extension icons at their declared 16/48/128px sizes, generated
  from the same square master icon as the main app's PWA icon
  (`static/icon.png`) via `scripts/generate_icons.py`.
