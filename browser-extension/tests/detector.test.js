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

test("text-statistics features (digit/exclamation/uppercase/length/word counts) are populated", () => {
  const { extractFeatures } = require("../detector.js");
  const f = extractFeatures("URGENT!! Send code 1234 NOW!!!");
  assert.equal(f.digitCount, 4);
  assert.equal(f.exclamationCount, 5);
  assert.ok(f.uppercaseCount > 0);
  assert.equal(f.textLength, "URGENT!! Send code 1234 NOW!!!".length);
  assert.equal(f.wordCount, 5);
  assert.ok(f.avgWordLength > 0);
});

test("empty text yields zeroed text-statistics features without crashing", () => {
  const { extractFeatures } = require("../detector.js");
  const f = extractFeatures("");
  assert.equal(f.digitCount, 0);
  assert.equal(f.exclamationCount, 0);
  assert.equal(f.uppercaseCount, 0);
  assert.equal(f.textLength, 0);
  assert.equal(f.wordCount, 0);
  assert.equal(f.avgWordLength, 0);
});

test("many digits/uppercase/exclamation marks push the score up and surface as reasons", () => {
  const result = FraudDetector.analyzeText("URGENT!!! CALL 12345678900 NOW!!!");
  assert.ok(result.reasons.includes("digitCount"));
  assert.ok(result.reasons.includes("exclamationCount"));
  assert.ok(result.reasons.includes("uppercaseCount"));
});

test("wordCount/textLength/avgWordLength never appear as reasons (mirrors fraud_logic's irrelevant set)", () => {
  const result = FraudDetector.analyzeText("this is a perfectly ordinary safe message about the weather today");
  assert.ok(!result.reasons.includes("wordCount"));
  assert.ok(!result.reasons.includes("textLength"));
  assert.ok(!result.reasons.includes("avgWordLength"));
});

test("lrScore() (the ported LogisticRegression member) returns a valid probability", () => {
  const { extractFeatures, lrScore } = require("../detector.js");
  for (const text of ["", "hello there", "URGENT send code now http://bad.xyz"]) {
    const p = lrScore(extractFeatures(text));
    assert.ok(p >= 0 && p <= 1, `lrScore(${JSON.stringify(text)}) = ${p} out of [0,1]`);
  }
});

test("lrScore alone over-scores an ordinary greeting, but the blended analyzeText() result stays safe", () => {
  // Locks in the reason LR_BLEND_WEIGHT is only 0.2 (see detector.js's file
  // header comment): evaluated in isolation, the ported LR member assigns a
  // high fraud probability to this plain, link-free Cyrillic greeting — a
  // dataset-distribution artifact, not a real signal. If this ever stops
  // being true (e.g. the model is retrained on a more representative
  // dataset), LR_BLEND_WEIGHT can safely be revisited/increased.
  const { extractFeatures, lrScore } = require("../detector.js");
  const text = "Привет, как дела? Встретимся завтра в 9:00.";
  const isolatedLrProb = lrScore(extractFeatures(text));
  assert.ok(isolatedLrProb > 0.9, `expected the known over-scoring artifact, got ${isolatedLrProb}`);

  const result = FraudDetector.analyzeText(text);
  assert.equal(result.isFraud, false);
  assert.ok(result.prob < 0.5);
});

// =========================
// Internal helpers exported for testability (mirrors tests/test_fraud_logic.py)
// =========================
test("short brand cores (vtb/dhl) are not fuzzy-matched via edit distance", () => {
  const { brandImpersonation } = require("../detector.js");
  assert.equal(brandImpersonation("vth.com"), null);
  assert.equal(brandImpersonation("dgl-tracking.com"), null);
});

test("single-character typosquat domains are flagged via Levenshtein distance", () => {
  const { brandImpersonation } = require("../detector.js");
  assert.equal(brandImpersonation("kasp1-login.xyz"), "kaspi.kz");
  assert.equal(brandImpersonation("paypa1-secure.com"), "paypal.com");
});

test("brands sharing a core name across TLDs do not flag each other", () => {
  const { brandImpersonation } = require("../detector.js");
  assert.equal(brandImpersonation("sberbank.ru"), null);
  assert.equal(brandImpersonation("sberbank.kz"), null);
});

test("getDomain() does not strip a non-leading www. substring", () => {
  const { getDomain } = require("../detector.js");
  assert.equal(getDomain("http://kaspi-www.secure-login.xyz/path"), "kaspi-www.secure-login.xyz");
});

test("getDomain() strips protocol and path", () => {
  const { getDomain } = require("../detector.js");
  assert.equal(getDomain("https://www.example.com/path?x=1"), "example.com");
  assert.equal(getDomain("http://secure-login.xyz"), "secure-login.xyz");
});

test("extractUrls() finds http and www links", () => {
  const { extractUrls } = require("../detector.js");
  const urls = extractUrls("visit http://a.com or www.b.com today");
  assert.equal(urls.length, 2);
});

test("bare domain is detected without a protocol", () => {
  const { extractBareDomains } = require("../detector.js");
  const domains = extractBareDomains("проверьте на check-tech-base.ru прямо сейчас", []);
  assert.ok(domains.includes("check-tech-base.ru"));
});

test("bare domain is not double-counted when also a full URL", () => {
  const { extractUrls, extractBareDomains } = require("../detector.js");
  const urls = extractUrls("visit http://check-tech-base.ru/verify now");
  const bare = extractBareDomains("visit http://check-tech-base.ru/verify now", urls);
  assert.deepEqual(bare, []);
});

test("domainFlags() flags critical severity for brand impersonation", () => {
  const { domainFlags } = require("../detector.js");
  const flags = domainFlags("kaspi-login.xyz");
  assert.ok(flags.some((f) => f.severity === "critical"));
  assert.ok(flags.some((f) => f.labelKey === "impersonates" && f.brand === "kaspi.kz"));
});

test("domainFlags() flags non-Latin domains as critical even without a known brand match", () => {
  const { domainFlags, brandImpersonation } = require("../detector.js");
  const nonLatinDomain = "асmе-shop.com"; // Cyrillic а/е, not a known brand core
  assert.equal(brandImpersonation(nonLatinDomain), null);
  const flags = domainFlags(nonLatinDomain);
  assert.ok(flags.some((f) => f.labelKey === "nonLatinDomain" && f.severity === "critical"));
});

test("ruleBoost() is zero when no rule combos fire", () => {
  const { extractFeatures, ruleBoost } = require("../detector.js");
  const f = extractFeatures("");
  assert.equal(ruleBoost(f), 0);
});

test("ruleBoost() is capped at 0.30 even when nearly every combo fires", () => {
  const { extractFeatures, ruleBoost } = require("../detector.js");
  const f = extractFeatures(
    "СРОЧНО! Не говорите никому! Ваша карта заблокирована, штраф, отправьте " +
    "код и пароль на безопасный счет http://kaspi-login.xyz " +
    "http://another-link.top прямо сейчас!"
  );
  assert.equal(ruleBoost(f), 0.30);
});

test("riskStyle() boundaries match risk_level()'s threshold-relative 0.3/0.5/0.8 cutoffs", () => {
  const { riskStyle } = require("../detector.js");
  assert.equal(riskStyle(0).levelKey, "low");
  assert.equal(riskStyle(0.29).levelKey, "low");
  assert.equal(riskStyle(0.3).levelKey, "mid");
  assert.equal(riskStyle(0.49).levelKey, "mid");
  assert.equal(riskStyle(0.5).levelKey, "high");
  assert.equal(riskStyle(0.79).levelKey, "high");
  assert.equal(riskStyle(0.8).levelKey, "critical");
  assert.equal(riskStyle(1.0).levelKey, "critical");
});

test("riskStyle() badge never contradicts the isFraud (prob >= 0.5) verdict", () => {
  // Regression test for the bug this fixed: mid used to extend to 0.6, so a
  // prob in [0.5, 0.6) showed a yellow "mid" badge next to an already-FRAUD
  // verdict/advice text (analyzeText()'s isFraud is prob >= 0.5).
  const { riskStyle } = require("../detector.js");
  for (let i = 0; i <= 100; i++) {
    const prob = i / 100;
    const { levelKey } = riskStyle(prob);
    const badgeSaysFraud = levelKey === "high" || levelKey === "critical";
    assert.equal(badgeSaysFraud, prob >= 0.5, `prob=${prob} levelKey=${levelKey}`);
  }
});

test("levenshtein() computes standard edit distance", () => {
  const { levenshtein } = require("../detector.js");
  assert.equal(levenshtein("kitten", "sitting"), 3);
  assert.equal(levenshtein("same", "same"), 0);
  assert.equal(levenshtein("", "abc"), 3);
});

// =========================
// loanCount / voteLinkCount (WhatsApp/Telegram hacked-account scam patterns,
// mirrors tests/test_fraud_logic.py's equivalents)
// =========================
test("loan-request phrasing is detected via loanCount", () => {
  const { extractFeatures } = require("../detector.js");
  const f = extractFeatures("привет, выручи до зарплаты, займи 20000 срочно");
  assert.ok(f.loanCount > 0);
});

test("vote-link phrasing is detected via voteLinkCount, with none of the other scam vocabulary", () => {
  const { extractFeatures } = require("../detector.js");
  const f = extractFeatures(
    "привет проголосуй пожалуйста за мою племянницу в конкурсе вот ссылка http://vote-konkurs-online.ru"
  );
  assert.ok(f.voteLinkCount > 0);
  assert.equal(f.hasLink, 1);
  assert.equal(f.secretCount, 0);
  assert.equal(f.threatCount, 0);
});

test("ruleBoost() adds the loan+urgent and voteLink+hasLink combos", () => {
  const { extractFeatures, ruleBoost } = require("../detector.js");
  const loanUrgent = extractFeatures("срочно выручи, займи денег до зарплаты");
  assert.ok(ruleBoost(loanUrgent) >= 0.12);

  const voteWithLink = extractFeatures("проголосуй за меня http://vote-now.ru");
  assert.ok(ruleBoost(voteWithLink) >= 0.10);
});

// =========================
// countMatches() word-boundary matching, not plain substring (mirrors
// test_fraud_logic.py's test_count_matches_does_not_match_mid_word /
// test_count_matches_still_matches_word_stems)
// =========================
test("word-boundary fix: does not match mid-word ('now' inside 'know', 'pin' inside 'shopping'/'opinion')", () => {
  const { extractFeatures } = require("../detector.js");
  let f = extractFeatures("let me know if 7pm still works for you, no rush");
  assert.equal(f.urgentCount, 0);

  f = extractFeatures("just finished shopping, what's your opinion on the blue one");
  assert.equal(f.secretCount, 0);
});

test("word-boundary fix: still matches left-anchored Cyrillic conjugation stems", () => {
  const { extractFeatures } = require("../detector.js");
  let f = extractFeatures("переведите оплату и займы вернем");
  assert.ok(f.moneyCount > 0); // "оплат" -> "оплату"
  assert.ok(f.loanCount > 0); // "займ" -> "займы"

  f = extractFeatures("нужна проверка документов срочно");
  assert.ok(f.verificationCount > 0); // "провер" -> "проверка"

  f = extractFeatures("urgent: verify your pin now");
  assert.ok(f.urgentCount > 0);
  assert.ok(f.secretCount > 0);
});

// =========================
// explain() MIN_THRESHOLDS (single-occurrence "many X" mislabeling fix,
// mirrors app.py's explain() min_thresholds)
// =========================
test("explain() needs >1 for digit/exclamation/uppercase/url reasons, but >=1 for keyword-list reasons", () => {
  const { analyzeText } = require("../detector.js");
  const single = analyzeText("Check http://a.com, 1 code, OK!");
  assert.ok(!single.reasons.includes("digitCount"));
  assert.ok(!single.reasons.includes("exclamationCount"));
  assert.ok(!single.reasons.includes("uppercaseCount"));
  assert.ok(!single.reasons.includes("urlCount"));

  const many = analyzeText("CODE 12345678 NOW!! ALSO SEE http://a.com AND http://b.com");
  assert.ok(many.reasons.includes("digitCount"));
  assert.ok(many.reasons.includes("exclamationCount"));
  assert.ok(many.reasons.includes("uppercaseCount"));
  assert.ok(many.reasons.includes("urlCount"));
});

test("LR_FEATURE_ORDER exactly matches extractFeatures()'s numeric keys (catches feature-shape drift like the loanCount/voteLinkCount gap this fixed)", () => {
  const { extractFeatures, LR_FEATURE_ORDER } = require("../detector.js");
  const f = extractFeatures("some ordinary text with http://a.com and займ");
  const featureKeys = Object.keys(f).filter((k) => k !== "domains");
  assert.deepEqual(featureKeys.slice().sort(), LR_FEATURE_ORDER.slice().sort());
});
