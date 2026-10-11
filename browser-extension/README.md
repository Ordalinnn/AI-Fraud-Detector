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
- **Language switcher** (🇰🇿 KZ / 🇷🇺 RU / 🇬🇧 EN) in the popup header. Until
  you pick one, the browser's language is used.

Everything runs inside the extension. Message text goes from the page to the
extension's own background worker and only a number comes back; there is no
server and no network request.

## How it detects fraud

`scorer.js` runs **the same model as the Android app**: character n-grams
(scikit-learn's `char_wb`, 2–5 characters) + TF-IDF + logistic regression,
the strongest single member of the web app's ensemble (93.6 % on the jury's
282-message test). The model file `char_model.tsv` (20 000 n-grams, ~0.9 MB)
is exported by `scripts/export_mobile_model.py` together with the Android
copy. `tests/scorer.test.js` checks that the JavaScript gives the same
probabilities as scikit-learn on ~400 texts (difference < 10⁻⁶) and that the
file is byte-for-byte the Android one.

`detector.js` adds the human-readable part: reasons (urgency, codes, money,
threats…) and link/domain checks (look-alike brands, risky TLDs). If the
model file ever fails to load, it falls back to its older rule-based score,
so the extension keeps working.

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
- `scorer.js` + `char_model.tsv` — the on-device ML model (same as Android).
- `content.js` / `content.css` — automatic checks in WhatsApp Web, Telegram
  Web and Gmail, and the red label.
- `detector.js` — reasons, domain checks and the fallback scoring logic (shared by the popup and the
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
