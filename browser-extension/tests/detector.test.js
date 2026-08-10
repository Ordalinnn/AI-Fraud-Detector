"use strict";

// Minimal test suite for detector.js using Node's built-in test runner
// (node:test / node:assert) so no extra dependency or build step is
// needed — run with `node --test` from browser-extension/.
//
// detector.js is a classic script (not a module): it attaches itself to
// `self.FraudDetector` for compatibility with the MV3 service worker /
// popup context. Node doesn't define `self` globally, so we alias it to
// `globalThis` before requiring the script, exactly as a browser worker
// would provide it.
const test = require("node:test");
const assert = require("node:assert/strict");

globalThis.self = globalThis;
require("../detector.js");
const { FraudDetector } = globalThis;

test("flags an obvious scam message as fraud", () => {
  const result = FraudDetector.analyzeText(
    "Срочно! Ваша карта заблокирована, отправьте код подтверждения на http://kaspi-verify.xyz"
  );
  assert.ok(result.isFraud);
  assert.ok(result.prob > 0.5);
  assert.ok(result.reasons.length > 0);
});

test("does not flag an ordinary safe message", () => {
  const result = FraudDetector.analyzeText("Привет, как дела? Встретимся завтра в 9:00.");
  assert.equal(result.isFraud, false);
});

test("detects brand impersonation via homoglyph domain", () => {
  const result = FraudDetector.analyzeText("confirm your account at http://kаspi.kz/login now");
  const impersonated = result.domainInfo.some((d) =>
    d.flags.some((f) => f.labelKey === "impersonates" && f.brand === "kaspi.kz")
  );
  assert.ok(impersonated);
});

test("known real brand domain is not flagged as impersonation", () => {
  const result = FraudDetector.analyzeText(
    "ваш чек по оплате доступен по ссылке https://kaspi.kz/receipt"
  );
  const impersonated = result.domainInfo.some((d) =>
    d.flags.some((f) => f.labelKey === "impersonates")
  );
  assert.equal(impersonated, false);
});

test("gracefully handles non-string input instead of throwing", () => {
  assert.doesNotThrow(() => FraudDetector.analyzeText(null));
  assert.doesNotThrow(() => FraudDetector.analyzeText(undefined));
  const result = FraudDetector.analyzeText(undefined);
  assert.equal(result.isFraud, false);
});
