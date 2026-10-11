// The extension's on-device model must give the SAME probabilities as the
// scikit-learn model it was exported from (golden.tsv, shared with the
// Android app's test), and must be the very same model file as the app's.
const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const { CharScorer } = require("../scorer.js");

const ROOT = path.join(__dirname, "..", "..");
const MODEL = path.join(__dirname, "..", "char_model.tsv");
const scorer = CharScorer.parse(fs.readFileSync(MODEL, "utf8"));

function unescape(s) {
  return s.replace(/\\(.)/g, (_, c) => ({ t: "\t", n: "\n", r: "\r", "\\": "\\" }[c] ?? "\\" + c));
}

test("matches scikit-learn on the golden texts", () => {
  const lines = fs.readFileSync(path.join(ROOT, "android/app/src/test/resources/golden.tsv"), "utf8")
    .split("\n").filter((l) => l && !l.startsWith("# "));
  assert.ok(lines.length >= 300);
  let worst = 0;
  let worstText = "";
  for (const line of lines) {
    const tab = line.lastIndexOf("\t");
    const text = unescape(line.slice(0, tab));
    const diff = Math.abs(scorer.probability(text) - parseFloat(line.slice(tab + 1)));
    if (diff > worst) { worst = diff; worstText = text; }
  }
  assert.ok(worst < 1e-6, `max difference ${worst} on ${JSON.stringify(worstText)}`);
});

test("is the same model file as the Android app", () => {
  const sha = (p) => crypto.createHash("sha256").update(fs.readFileSync(p)).digest("hex");
  assert.strictEqual(sha(MODEL), sha(path.join(ROOT, "android/app/src/main/assets/char_model.tsv")));
});

test("char_wb n-grams match scikit-learn", () => {
  assert.deepStrictEqual(scorer.ngrams("x"), [" x", "x ", " x "]);
  assert.deepStrictEqual(scorer.ngrams("Ab").slice(0, 6), [" a", "ab", "b ", " ab", "ab ", " ab "]);
});

test("scam is flagged, ordinary message is not, words explain why", () => {
  const scam = scorer.analyze("Срочно! Ваша карта заблокирована. Назовите код из SMS http://kaspi-login.xyz");
  const ok = scorer.analyze("Привет, урок математики перенесли на пятницу, не забудь тетрадь");
  assert.ok(scam.probability > 0.5, String(scam.probability));
  assert.ok(ok.probability < 0.5, String(ok.probability));
  assert.ok(scam.suspiciousWords.length > 0);
});
