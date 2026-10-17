# Design System Master File

> **LOGIC:** When building a specific page, first check `design-system/ai-fraud-detector/pages/[page-name].md`.
> If that file exists, its rules **override** this Master file.
> If not, strictly follow the rules below.

---

**Project:** AI Fraud Detector
**Updated:** 2026-10-16. Built with ui-ux-pro-max guidance, edited to match what is actually implemented.
**Category:** Security / Fintech tool
**Direction (2026-10-17 reskin):** "Graphite & Sky": graphite hero, sky-700 `#0369a1` primary (5.9:1 on white), tighter radii 6/10/14/18, IBM Plex Sans for headings (Montserrat dropped). Values below the table header may be stale; `:root` in app.py wins.
**Previous direction:** "Trust, rich": the navy/cobalt/teal brand with gradient depth, colour-coded icon tiles and layered cards.
**Source of truth for values:** the `:root` block in `app.py`'s UI STYLE section. Keep this file and that block in sync.

> **Rejected generator suggestions:** a dark gold `#F59E0B` + purple `#8B5CF6` palette, and later "Claymorphism" with Varela Round / Nunito Sans (playful, aimed at children's products, weak Kazakh Cyrillic). Reasons: the existing brand is navy/cobalt/teal, `.streamlit/config.toml` forces a light theme, and the generator's own anti-patterns forbid AI purple gradients and playful design for this category.

---

## Color Palette (light theme only)

| Role | Hex | CSS Variable |
|------|-----|--------------|
| Primary (buttons, links, active tab) | `#1d4ed8` | `--clr-cobalt` (= `primaryColor` in config.toml) |
| Primary hover | `#1e40af` | `--clr-cobalt-hover` |
| Brand dark | `#0a1a44` / `#12307a` | `--clr-navy` / `--clr-navy-2` |
| Cyan / teal accents | `#0e7490` / `#0f766e` | `--clr-cyan` / `--clr-teal` |
| Accent on dark | `#5eead4` | `--clr-teal-soft` |
| Brand gradient (buttons, accent bars, progress) | `135deg #1d4ed8 → #2563eb → #0e7490` | `--grad-brand` |
| Hero gradient | `125deg #081436 → #0f2a6b → #1d4ed8 → #0e7490` | `--grad-hero` |
| Text | `#0f172a` | `--clr-ink` |
| Text secondary | `#334155` | `--clr-ink-2` |
| Text muted (≥ 7:1 on white) | `#475569` | `--clr-muted` |
| Page background | `#f3f6fb` + soft blue/teal radial glows | `--clr-bg` |
| Surface / card | `#ffffff` | `--clr-surface` |
| Border | `#e3e9f2` | `--clr-border` |
| Focus ring | `#2563eb` | `--clr-focus` |

**Icon tones** (`.tone-*`: tinted background + darker same-hue icon, all ≥ 4.5:1): blue, teal, cyan, amber, orange, rose, emerald, indigo, sky, slate. Use one tone per meaning and don't repeat a tone next to itself.

**Risk levels** (always icon + text label + colour, never colour alone):

| Level | Background | Text | Accent / icon tile |
|-------|-----------|------|-------------|
| Low | `#ecfdf3` | `#166534` | `#16a34a` (shield-check) |
| Suspicious | `#fefce8` | `#854d0e` | `#ca8a04` (shield-alert) |
| High | `#fff3e6` | `#9a3412` | `#ea580c` (shield-alert) |
| Critical | `#ffecec` | `#991b1b` | `#dc2626` (siren) |

## Typography

- **Headings, big numbers:** Montserrat 600/700/800 (`--font-head`)
- **Body:** IBM Plex Sans 400/500/600/700
- **Mono (domains, model tags):** IBM Plex Mono
- All three cover the Russian and Kazakh Cyrillic letters (ә ғ қ ң ө ұ ү һ і). Check any new font for this before adopting it.
- **Scale:** 12 (labels, uppercase +0.06em) / 13–14 (meta) / 15–16 (body) / 20 (section title) / 26–34 (metric and verdict values) / 44 (hero title)
- Inputs 16px (prevents iOS zoom). Numbers use `tabular-nums`.

## Radius, Shadow

| Token | Value | Usage |
|-------|-------|-------|
| `--radius-sm` | 10px | Buttons, badges, list rows |
| `--radius-md` | 14px | Inputs, alerts, meter, forms |
| `--radius-lg` | 20px | Cards, expanders, verdict |
| `--radius-xl` | 28px | Hero |
| `--shadow-sm / md / lg` | layered, `rgba(15,23,42,…)` | rest / card / hover |
| `--shadow-brand` | `0 8px 22px rgba(29,78,216,.28)` | primary buttons |

Spacing follows a 4/8px rhythm.

---

## Components

- **Hero:** `--grad-hero`, masked 34px grid, drifting teal glow, and the canvas particle network (the `HERO PARTICLE EFFECT` script, which already respects reduced motion). Left: logo, kicker, title ("AI" in teal-cyan gradient), subtitle, icon badges. Right: a glowing shield with pulsing rings above a glass "readiness" panel with model tags (LR · RF · GB · TF-IDF · CHAR · STACK). Bottom: 4 icon stats (examples, categories, accuracy, languages). Under 640px the right-hand visual is hidden so the input box comes sooner.
- **Cards:** two kinds. `.glass-card` is for content rendered in a single `st.markdown` call. `st.container(key="card_*")` is for cards that contain widgets; never open and close a `<div>` across separate `st.markdown` calls, because Streamlit doesn't nest them. Cards are white with a 1px border, `--radius-lg` and `--shadow-md`, plus a 4px `--grad-brand` bar across the top (the deep-analysis card uses a teal gradient).
- **Section title:** `section_title(icon, text, tone)` produces a 40px tone tile followed by Montserrat 20/700 text.
- **Primary button:** `--grad-brand`, white 600 text, min-height 46px, `--shadow-brand`. On hover the gradient slides and the button lifts 1px.
- **Secondary button** (downloads): white, cobalt text, `#bcd0f5` border.
- **Metric card:** `metric_card(icon, tone, label, value)`, with a tone tile, uppercase label, Montserrat value, a soft tinted corner glow, and lift on hover.
- **Percentages:** always format with `risk_percent()` (one decimal, capped at 99.9), so the cards, verdict and meter show the same number.
- **Verdict banner:** a gradient tint, 6px left accent and a 56px solid icon tile, followed by a kicker, the level label and the score %. It has `role="status" aria-atomic="true"`. Its label comes from the current language (`T[level]`), so it changes when the language is switched.
- **Risk meter:** a card with a 14px green→red track, pointer, dashed threshold line, and a text legend giving the score and threshold. It has `role="img"` with an aria-label.
- **Features panel:** 10 rows, each with a 34px tone tile and a label.
- **Sidebar:** dark navy with blue and teal glows. The selected language pill uses a gradient. Model accuracies (held-out test) are shown as labelled gradient bars, with the final stacked model highlighted.
- **Empty state:** a dashed box with a blue→teal tint, a search icon and a one-sentence hint.
- **Icons:** Lucide via `icon(name)`, using 2px strokes and `aria-hidden` (always shown beside text). Don't add emoji as structural icons in new UI.

---

## Motion

- Durations: `--dur-fast` .2s (hover and state changes), `--dur-med` .4s (entrances).
- Entrances: `fade-slide-up` (8px) on the hero, cards, metric cards and verdict.
- Decorative loops are allowed **only in the hero**: the glow drift (16s), the shield ring pulse (3.2s) and the particle network. Nothing loops anywhere else.
- `prefers-reduced-motion: reduce` collapses all motion and removes the loops entirely.
- Spinners only for work that isn't near-instant (batch CSV, Deep AI Analysis).

---

## Anti-Patterns (Do NOT Use)

- ❌ Purple/pink AI gradients, playful or "clay" styling
- ❌ Emojis as structural icons; use `icon()`
- ❌ Colour as the only signal for risk
- ❌ Text contrast below 4.5:1 (e.g. `#94a3b8` on white)
- ❌ Invisible focus states
- ❌ Raw hex values in new components; use the `:root` tokens and tone classes
- ❌ HTML wrappers split across `st.markdown` calls
- ❌ Hard-coded English in the UI; every string goes through `translations.py` (KZ/RU/EN)
- ❌ Hard-coded facts that can drift, such as model count, dataset size or model list. Derive them from code where possible, and update all three languages and the README together.

---

## Pre-Delivery Checklist

- [ ] Uses `:root` tokens and tone classes
- [ ] No emojis used as icons
- [ ] Touch targets ≥ 44px; `cursor: pointer` on clickables
- [ ] Text contrast ≥ 4.5:1; status shown with icon + text + colour
- [ ] Visible keyboard focus
- [ ] `prefers-reduced-motion` respected
- [ ] Tested at 375px, 768px, 1024px, 1440px; no horizontal scroll
- [ ] New strings added to KZ, RU and EN; facts match the code (5 models + stacker: LR, RF, GB, TF-IDF/NB, char n-gram LR; accuracy = held-out `ENSEMBLE_ACCURACY`)
