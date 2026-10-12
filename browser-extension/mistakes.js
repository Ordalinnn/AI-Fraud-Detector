// "This is a mistake" reports, kept ONLY on this computer (chrome.storage).
//
// The user marks a wrong verdict (a scam the model missed, or an ordinary
// message it flagged); the popup can export all reports as a CSV in exactly
// the format of data/real/team_real_messages_TEMPLATE.csv, so the team can
// collect them from testers and check / retrain the model on real messages
// (python scripts/import_mistakes.py). Nothing is sent anywhere
// automatically.

(function (root) {
  "use strict";

  const KEY = "mistakes";
  const MAX = 500;
  const MAX_TEXT = 2000;
  const KAZAKH_ONLY = /[әғқңөұүһіӘҒҚҢӨҰҮҺІ]/;
  const CYRILLIC = /[а-яёА-ЯЁ]/g;
  const LATIN = /[a-zA-Z]/g;

  // Same rule as fraud_eval.detect_language.
  function detectLanguage(text) {
    if (KAZAKH_ONLY.test(text)) return "kk";
    const cyr = (text.match(CYRILLIC) || []).length;
    const lat = (text.match(LATIN) || []).length;
    if (!cyr && !lat) return "other";
    return cyr >= lat ? "ru" : "en";
  }

  // `predictedFraud`: what the model said; the report means "the opposite is
  // true", so the CSV label is the corrected one.
  function makeEntry({ text, predictedFraud, prob, source }) {
    const t = String(text || "").trim().slice(0, MAX_TEXT);
    return {
      text: t,
      label: predictedFraud ? 0 : 1,
      language: detectLanguage(t),
      source: `${source || "extension"}: model said ${predictedFraud ? "fraud" : "safe"} ${Math.round((prob || 0) * 100)}%`,
      date: new Date().toISOString().slice(0, 10),
    };
  }

  // Newest first, one entry per text (a second report replaces the first).
  function addTo(list, entry) {
    if (!entry.text) return list;
    return [entry, ...(list || []).filter((e) => e.text !== entry.text)].slice(0, MAX);
  }

  const CSV_COLUMNS = ["text", "label", "language", "scam_type", "source", "date_collected"];

  function csvCell(v) {
    let s = String(v == null ? "" : v);
    // spreadsheet formula injection guard
    if (/^[=+\-@\t\r]/.test(s)) s = "'" + s;
    return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  }

  function toCsv(list) {
    const rows = [CSV_COLUMNS.join(",")];
    for (const e of list || []) {
      rows.push([e.text, e.label, e.language, "", e.source, e.date].map(csvCell).join(","));
    }
    return rows.join("\n") + "\n";
  }

  // chrome.storage wrappers (popup and content scripts)
  function list() {
    return new Promise((resolve) => chrome.storage.local.get({ [KEY]: [] }, (d) => resolve(d[KEY] || [])));
  }
  async function add(args) {
    const next = addTo(await list(), makeEntry(args));
    await new Promise((resolve) => chrome.storage.local.set({ [KEY]: next }, resolve));
    return next.length;
  }
  function clear() {
    return new Promise((resolve) => chrome.storage.local.set({ [KEY]: [] }, resolve));
  }

  const api = { detectLanguage, makeEntry, addTo, toCsv, list, add, clear, CSV_COLUMNS };
  root.Mistakes = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof self !== "undefined" ? self : globalThis);
