// The extension must give the SAME result as the website for the same text.
// ensemble_golden.json holds what the site's Python code computes
// (scripts/export_extension_model.py): the 25 rule features, every model's
// probability, the final probability and the percentage text, on ~4,500
// texts - the whole training set, real SMS, disguised scams and texts where
// Python and JavaScript string handling differ.
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { EnsembleScorer, wordTokens } = require("../ensemble.js");
const { analyzeText, extractFeatures, ruleBoost, riskPercent } = require("../detector.js");

const read = (f) => JSON.parse(fs.readFileSync(path.join(__dirname, f), "utf8"));
const model = read("../ensemble_model.json");
const golden = read("ensemble_golden.json");
const scorer = EnsembleScorer.parse(model);
const camel = (s) => s.replace(/_([a-z])/g, (_, c) => c.toUpperCase());

test("golden file is for this model file", () => {
  assert.equal(golden.training_fingerprint, model.training_fingerprint);
  assert.ok(golden.texts.length >= 4000);
});

test("rule features are exactly the site's on every golden text", () => {
  const keys = golden.features.map(camel);
  const bad = [];
  golden.texts.forEach((t, i) => {
    const f = extractFeatures(t);
    keys.forEach((k, j) => { if (f[k] !== golden.x[i][j]) bad.push(`${k} on ${JSON.stringify(t).slice(0, 80)}: ${f[k]} != ${golden.x[i][j]}`); });
  });
  assert.deepEqual(bad.slice(0, 10), []);
});

test("every model and the final probability match scikit-learn (< 1e-9)", () => {
  const worst = {};
  golden.texts.forEach((t, i) => {
    const f = extractFeatures(t);
    const s = scorer.scores(t, f, ruleBoost(f));
    for (const k of Object.keys(golden.scores)) {
      const d = Math.abs(s[k] - golden.scores[k][i]);
      if (!(k in worst) || d > worst[k][0]) worst[k] = [d, t];
    }
  });
  for (const [k, [d, t]] of Object.entries(worst)) {
    assert.ok(d < 1e-9, `${k}: difference ${d} on ${JSON.stringify(t)}`);
  }
});

test("the popup / labels show the same percentage and verdict as the site", () => {
  const bad = [];
  golden.texts.forEach((t, i) => {
    const r = analyzeText(t, scorer);
    if (r.pct !== golden.pct[i] || r.isFraud !== golden.scores.final[i] >= 0.5) {
      bad.push(`${JSON.stringify(t).slice(0, 60)}: ${r.pct} vs ${golden.pct[i]}`);
    }
  });
  assert.deepEqual(bad, []);
  for (const [p, want] of golden.pct_probe) assert.equal(riskPercent(p), want, String(p));
});

test("TF-IDF tokens follow Python's \\w\\w+", () => {
  assert.deepEqual(wordTokens("a ab snake_case 12 é ²³ к0д"), ["ab", "snake_case", "12", "²³", "к0д"]);
});

test("model id is shown from the fingerprint", () => {
  assert.match(scorer.fingerprint, /^[0-9a-f]{16}$/);
});
