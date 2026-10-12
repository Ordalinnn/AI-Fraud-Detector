// normalize.js must turn every text into exactly what text_normalize.py does
// (normalize_golden.tsv, shared with the Android app's test).
const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");
const { normalizeText } = require("../normalize.js");
const { extractFeatures } = require("../detector.js");

const GOLDEN = path.join(__dirname, "..", "..", "android/app/src/test/resources/normalize_golden.tsv");
const unescape = (s) => s.replace(/\\(.)/g, (_, c) => ({ t: "\t", n: "\n", r: "\r", "\\": "\\" }[c] ?? "\\" + c));

test("matches text_normalize.py on every golden text", () => {
  const lines = fs.readFileSync(GOLDEN, "utf8").split("\n").filter((l) => l && !l.startsWith("# "));
  assert.ok(lines.length >= 300, String(lines.length));
  const bad = [];
  for (const line of lines) {
    const [raw, expected] = line.split("\t").map(unescape);
    const got = normalizeText(raw);
    if (got !== expected) bad.push({ raw, expected, got });
  }
  assert.deepStrictEqual(bad.slice(0, 5), []);
});

test("disguised scam words are counted by the rules, look-alike domain still flagged", () => {
  const plain = extractFeatures("Срочно! Ваша карта заблокирована, назовите код");
  const disguised = extractFeatures("Cрoчнo! Ваша кaртa зaблoкирoвaнa, нaзoвитe к0д");
  assert.ok(plain.urgentCount > 0);
  assert.strictEqual(disguised.urgentCount, plain.urgentCount);
  assert.strictEqual(disguised.threatCount, plain.threatCount);
  assert.strictEqual(extractFeatures("Войдите: http://kаspi.kz/login").homoglyphDomain, 1);
});

test("disguised text gets its own reason, ordinary text does not", () => {
  const { analyzeText } = require("../detector.js");
  assert.strictEqual(analyzeText("Ваша к@рта зaблoкирoвана, назовите к0д").reasons[0], "disguised");
  assert.ok(!analyzeText("Привет, урок перенесли на пятницу…").reasons.includes("disguised"));
  const { isDisguised } = require("../normalize.js");
  for (const t of ["площадь 50 м²", "№5", "WhatsAppқа", "Wi-Fi", "Привет ❤️", "семья 👨‍👩‍👧"]) assert.strictEqual(isDisguised(t), false, t);
  for (const t of ["ＳＭＳ", "к о д ы", "к​од", "kаspi"]) assert.strictEqual(isDisguised(t), true, t);
});
