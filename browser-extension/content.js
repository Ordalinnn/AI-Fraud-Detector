// Automatic checks inside WhatsApp Web, Telegram Web and Gmail.
//
// Watches the page for new incoming messages, sends their text to the
// extension's background worker (scored there by the on-device model; nothing
// leaves the computer) and puts a red "⚠️ Looks like fraud, 87%" label next to
// suspicious ones. Can be switched off in the popup ("autoscan").
//
// Selectors are deliberately loose and several per site: these web apps change
// their markup often. If one stops matching, the right-click check and the
// popup still work.

(function (root) {
  "use strict";

  const THRESHOLD = 0.5;      // same as "Normal" sensitivity on Android
  const MIN_CHARS = 12;       // "ok", "👍", "Привет!" are never scams
  const MAX_CHARS = 4000;
  const DEBOUNCE_MS = 400;
  const BATCH = 40;
  const CHECKED = "afdChecked"; // dataset key -> data-afd-checked

  // Outgoing messages are checked too: a forwarded scam gets the label, and
  // the easiest way to try the extension is to send yourself a scam text.
  // `skip` (optional): ancestors whose messages must not be checked.
  // `where`: put the label after the text (chat bubbles) or before it (long emails).
  const SITES = [
    { host: "web.whatsapp.com",
      selector: "div.copyable-text[data-pre-plain-text]",
      textOf: (el) => (el.querySelector("span.selectable-text") || el).innerText || el.textContent,
      skip: null,
      where: "afterend" },
    { host: "web.telegram.org",
      selector: ".text-content, .translatable-message",
      skip: null,
      where: "afterend" },
    { host: "mail.google.com",
      selector: "div.a3s",
      skip: null,
      where: "beforebegin" },
  ];

  function siteFor(hostname) {
    return SITES.find((s) => hostname === s.host || hostname.endsWith("." + s.host)) || null;
  }

  function textOf(site, el) {
    const t = site.textOf ? site.textOf(el) : (el.innerText || el.textContent);
    return String(t || "").trim().slice(0, MAX_CHARS);
  }

  // New, not yet checked message elements under `scope`.
  function findCandidates(site, scope) {
    const out = [];
    const all = [];
    if (scope.matches && scope.matches(site.selector)) all.push(scope);
    if (scope.querySelectorAll) all.push(...scope.querySelectorAll(site.selector));
    for (const el of all) {
      if (el.dataset[CHECKED] !== undefined) continue;
      if (el.closest(".afd-box")) continue;
      // a match nested inside another match is the same message
      if (el.parentElement && el.parentElement.closest(site.selector)) continue;
      if (site.skip && el.closest(site.skip)) { el.dataset[CHECKED] = "own"; continue; }
      const text = textOf(site, el);
      if (text.length < MIN_CHARS) continue; // may still be loading; look again later
      out.push({ el, text });
    }
    return out;
  }

  const WEB_APP_URL = "https://ai-fraud-detector-p6jkftqrssphfcsxxepipo.streamlit.app/";

  // Yellow highlight of the suspicious words inside the message itself. Uses
  // the CSS Custom Highlight API: no change to WhatsApp's/Telegram's own DOM
  // (they redraw messages and would fight any <mark> we insert).
  function highlightWords(doc, el, words) {
    const win = doc.defaultView;
    if (!win || !win.CSS || !win.CSS.highlights || typeof win.Highlight !== "function") return;
    const ranges = [];
    const wanted = (words || []).map((w) => w.toLowerCase()).filter((w) => w.length >= 2);
    const walker = doc.createTreeWalker(el, win.NodeFilter.SHOW_TEXT);
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      const low = n.data.toLowerCase();
      for (const w of wanted) {
        for (let i = low.indexOf(w); i >= 0; i = low.indexOf(w, i + w.length)) {
          const r = doc.createRange();
          r.setStart(n, i);
          r.setEnd(n, i + w.length);
          ranges.push(r);
        }
      }
    }
    const h = win.CSS.highlights.get("afd-words") || new win.Highlight();
    for (const r of ranges) h.add(r);
    win.CSS.highlights.set("afd-words", h);
  }

  function clearHighlights(doc) {
    const win = doc.defaultView;
    if (win && win.CSS && win.CSS.highlights) win.CSS.highlights.delete("afd-words");
  }

  function mk(doc, tag, cls, text) {
    const e = doc.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  // "Why is it suspicious?" panel shown when the label is clicked.
  function makeDetails(doc, result, strings, text) {
    const d = mk(doc, "div", "afd-details");
    d.hidden = true;
    d.appendChild(mk(doc, "div", "afd-details-title", strings.detailsTitle));

    if ((result.words || []).length) {
      const row = mk(doc, "div", "afd-row");
      row.appendChild(mk(doc, "span", "afd-row-label", strings.badgeWords + " "));
      for (const w of result.words) row.appendChild(mk(doc, "span", "afd-chip", w));
      d.appendChild(row);
    }

    if ((result.reasons || []).length) {
      d.appendChild(mk(doc, "div", "afd-row-label", strings.detailsReasons));
      const ul = mk(doc, "ul", "afd-list");
      // the first few are enough; link problems get their own section below
      for (const key of result.reasons.slice(0, 6)) ul.appendChild(mk(doc, "li", null, strings.reasonLabels[key] || key));
      d.appendChild(ul);
    }

    const flagged = (result.domains || []).filter((x) => x.flags.length);
    if (flagged.length) {
      d.appendChild(mk(doc, "div", "afd-row-label", strings.detailsLinks));
      const ul = mk(doc, "ul", "afd-list");
      for (const x of flagged) {
        const flags = x.flags.map((f) => (f.severity === "critical" ? "🔴 " : "⚠️ ") +
          (strings.domainFlagLabels[f.labelKey] || f.labelKey).replace("{brand}", f.brand || ""));
        ul.appendChild(mk(doc, "li", null, `${x.domain} — ${flags.join(", ")}`));
      }
      d.appendChild(ul);
    }

    d.appendChild(mk(doc, "div", "afd-advice", strings.badgeAdvice));

    // "Not fraud": saved on this computer only (mistakes.js), exportable from the popup
    if (root.Mistakes) {
      const btn = mk(doc, "button", "afd-mistake", strings.mistakeFlaggedBtn);
      btn.type = "button";
      btn.addEventListener("click", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        btn.disabled = true;
        root.Mistakes.add({ text, predictedFraud: true, prob: result.prob, source: doc.location ? doc.location.hostname : "page" })
          .then(() => { btn.textContent = strings.mistakeThanks; })
          .catch(() => { btn.disabled = false; });
      });
      d.appendChild(btn);
    }

    const a = mk(doc, "a", "afd-link", strings.detailsOpenSite);
    a.href = WEB_APP_URL + "?text=" + encodeURIComponent(text.slice(0, 1500));
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    a.title = strings.detailsOpenSiteNote;
    d.appendChild(a);
    return d;
  }

  // The red label + its (hidden) details, in one box next to the message.
  function makeBadge(doc, result, strings, text, msgEl) {
    const box = mk(doc, "div", "afd-box");
    const b = mk(doc, "span", "afd-badge", `⚠️ ${strings.badgeLabel} · ${result.pct}% · ${strings.badgeWhy} ▾`);
    b.setAttribute("role", "button");
    b.setAttribute("tabindex", "0");
    b.setAttribute("aria-expanded", "false");
    b.title = strings.badgeClickHint;
    const details = makeDetails(doc, result, strings, text || "");
    const toggle = (ev) => {
      // don't let WhatsApp/Telegram treat the click as "select/open message"
      ev.preventDefault();
      ev.stopPropagation();
      details.hidden = !details.hidden;
      b.setAttribute("aria-expanded", String(!details.hidden));
      b.textContent = `⚠️ ${strings.badgeLabel} · ${result.pct}% · ${strings.badgeWhy} ${details.hidden ? "▾" : "▴"}`;
      clearHighlights(doc);
      if (!details.hidden && msgEl) highlightWords(doc, msgEl, result.words);
    };
    b.addEventListener("click", toggle);
    b.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") toggle(ev); });
    for (const t of ["mousedown", "mouseup", "pointerdown", "pointerup", "dblclick"]) {
      box.addEventListener(t, (ev) => ev.stopPropagation());
    }
    details.addEventListener("click", (ev) => { if (ev.target.tagName !== "A") ev.stopPropagation(); });
    box.appendChild(b);
    box.appendChild(details);
    return box;
  }

  function placeBadge(site, el, badge) {
    // re-rendered message: drop our previous label first
    const prev = site.where === "afterend" ? el.nextElementSibling : el.previousElementSibling;
    if (prev && prev.classList.contains("afd-box")) prev.remove();
    el.insertAdjacentElement(site.where, badge);
  }

  // A scanner for one document. `score(texts) -> Promise<[{prob,pct,words}]>`
  // is injected so tests can run it without Chrome.
  function createScanner({ doc, site, score, strings }) {
    const cache = new Map(); // text -> result; WhatsApp re-renders on scroll
    let timer = null;
    let observer = null;
    let running = false;
    let busy = false;
    let again = false;

    async function scan() {
      if (!running) return;
      if (busy) { again = true; return; }
      busy = true;
      try {
        const items = findCandidates(site, doc.body || doc);
        const unknown = [...new Set(items.map((i) => i.text).filter((t) => !cache.has(t)))];
        for (let i = 0; i < unknown.length; i += BATCH) {
          const chunk = unknown.slice(i, i + BATCH);
          const results = await score(chunk);
          if (!Array.isArray(results)) return;
          chunk.forEach((t, k) => cache.set(t, results[k]));
        }
        if (cache.size > 2000) cache.clear();
        if (!running) return;
        for (const { el, text } of items) {
          const r = cache.get(text);
          if (!r) continue;
          el.dataset[CHECKED] = r.prob >= THRESHOLD ? "fraud" : "ok";
          if (r.prob >= THRESHOLD) placeBadge(site, el, makeBadge(doc, r, strings(), text, el));
        }
      } finally {
        busy = false;
        if (again) { again = false; schedule(); }
      }
    }

    function schedule() {
      if (!running) return;
      clearTimeout(timer);
      timer = setTimeout(() => { scan().catch(stopOnDeadExtension); }, DEBOUNCE_MS);
    }

    function stopOnDeadExtension(e) {
      // "Extension context invalidated": the extension was updated or removed
      // while this tab stayed open. Stop quietly; a page reload re-injects us.
      stop();
      if (!/context invalidated/i.test(String(e && e.message))) console.warn("AI Fraud Detector:", e);
    }

    function start() {
      if (running) return;
      running = true;
      const Observer = doc.defaultView.MutationObserver;
      observer = new Observer(schedule);
      observer.observe(doc.body || doc.documentElement, { childList: true, subtree: true, characterData: true });
      schedule();
    }

    function stop() {
      running = false;
      clearTimeout(timer);
      if (observer) observer.disconnect();
      observer = null;
    }

    // Remove all labels and forget what was checked (autoscan off, language change).
    function reset() {
      for (const b of doc.querySelectorAll(".afd-box")) b.remove();
      clearHighlights(doc);
      for (const el of doc.querySelectorAll("[data-afd-checked]")) delete el.dataset[CHECKED];
    }

    function status() {
      return {
        host: site.host,
        running,
        checked: doc.querySelectorAll("[data-afd-checked]").length,
        flagged: doc.querySelectorAll(".afd-badge").length,
      };
    }

    return { start, stop, reset, scan, schedule, status, get running() { return running; } };
  }

  function boot() {
    const site = siteFor(location.hostname);
    if (!site) return;
    let lang = root.I18N.browserLang();
    const strings = () => root.I18N.strings(lang);
    const score = (texts) => new Promise((resolve, reject) => {
      try {
        chrome.runtime.sendMessage({ type: "score", texts }, (res) => {
          if (chrome.runtime.lastError) reject(new Error(chrome.runtime.lastError.message));
          else resolve(res);
        });
      } catch (e) { reject(e); }
    });
    const scanner = createScanner({ doc: document, site, score, strings });

    chrome.storage.local.get({ autoscan: true, lang }, (data) => {
      lang = data.lang || lang;
      if (data.autoscan !== false) scanner.start();
    });
    // The popup asks "is autoscan working on this tab?"
    chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
      if (msg && msg.type === "status") sendResponse(scanner.status());
    });
    chrome.storage.onChanged.addListener((changes, area) => {
      if (area !== "local") return;
      if (changes.lang) lang = changes.lang.newValue || lang;
      if (changes.autoscan) {
        if (changes.autoscan.newValue === false) { scanner.stop(); scanner.reset(); }
        else { scanner.reset(); scanner.start(); }
      } else if (changes.lang && scanner.running) {
        scanner.reset();
        scanner.schedule();
      }
    });
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { SITES, siteFor, findCandidates, createScanner, THRESHOLD, MIN_CHARS };
  } else if (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.id) {
    boot();
  }
})(typeof self !== "undefined" ? self : globalThis);
