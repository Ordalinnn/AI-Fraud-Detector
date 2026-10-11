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
      if (el.closest(".afd-badge")) continue;
      // a match nested inside another match is the same message
      if (el.parentElement && el.parentElement.closest(site.selector)) continue;
      if (site.skip && el.closest(site.skip)) { el.dataset[CHECKED] = "own"; continue; }
      const text = textOf(site, el);
      if (text.length < MIN_CHARS) continue; // may still be loading; look again later
      out.push({ el, text });
    }
    return out;
  }

  function makeBadge(doc, result, strings) {
    const b = doc.createElement("span");
    b.className = "afd-badge";
    b.setAttribute("role", "note");
    b.textContent = `⚠️ ${strings.badgeLabel} · ${result.pct}%`;
    const words = (result.words || []).length ? `${strings.badgeWords} ${result.words.join(", ")}\n` : "";
    b.title = words + strings.badgeAdvice;
    return b;
  }

  function placeBadge(site, el, badge) {
    // re-rendered message: drop our previous label first
    const prev = site.where === "afterend" ? el.nextElementSibling : el.previousElementSibling;
    if (prev && prev.classList.contains("afd-badge")) prev.remove();
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
          if (r.prob >= THRESHOLD) placeBadge(site, el, makeBadge(doc, r, strings()));
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
      for (const b of doc.querySelectorAll(".afd-badge")) b.remove();
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
