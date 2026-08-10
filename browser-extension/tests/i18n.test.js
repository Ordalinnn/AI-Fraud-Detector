"use strict";

// Minimal test suite for i18n.js using Node's built-in test runner, mirroring
// detector.test.js's approach: i18n.js is a classic script (attaches itself
// to `self.I18N`), so `self` is aliased to `globalThis` before requiring it.
const test = require("node:test");
const assert = require("node:assert/strict");

globalThis.self = globalThis;
globalThis.chrome = { storage: { local: { get: () => {}, set: () => {} } } };
const { TRANSLATIONS, DEFAULT_LANG, validateTranslations, collectKeyPaths } = require("../i18n.js");

test("the real TRANSLATIONS object (already validated at load time) has full en/ru/kz key parity", () => {
  assert.doesNotThrow(() => validateTranslations(TRANSLATIONS));
});

test("collectKeyPaths() flattens nested objects into dotted paths", () => {
  const paths = collectKeyPaths({ a: "x", b: { c: "y", d: "z" } }, "");
  assert.deepEqual(paths.sort(), ["a", "b.c", "b.d"]);
});

test("validateTranslations() throws naming the language and the exact missing top-level key", () => {
  const broken = {
    en: { appTitle: "Title", footer: "Footer" },
    ru: { appTitle: "Title" } // missing "footer"
  };
  assert.throws(() => validateTranslations(broken), /ru is missing: \["footer"\]/);
});

test("validateTranslations() also catches missing NESTED keys (e.g. inside reasonLabels)", () => {
  const broken = {
    en: { reasonLabels: { hasLink: "a", urgentCount: "b" } },
    ru: { reasonLabels: { hasLink: "a" } } // missing "urgentCount"
  };
  assert.throws(() => validateTranslations(broken), /ru is missing: \["reasonLabels\.urgentCount"\]/);
});

test("validateTranslations() does not throw when all languages already match", () => {
  const ok = {
    en: { a: "1", nested: { x: "1" } },
    ru: { a: "2", nested: { x: "2" } }
  };
  assert.doesNotThrow(() => validateTranslations(ok));
});

test("DEFAULT_LANG is a real language present in TRANSLATIONS", () => {
  assert.ok(TRANSLATIONS[DEFAULT_LANG]);
});

test("every language has the three risk-label keys used by detector.js's riskStyle() levelKeys", () => {
  for (const lang of Object.keys(TRANSLATIONS)) {
    const risk = TRANSLATIONS[lang].riskLabels;
    for (const level of ["low", "mid", "high", "critical"]) {
      assert.ok(risk[level], `${lang}.riskLabels.${level} is missing`);
    }
  }
});
