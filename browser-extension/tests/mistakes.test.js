// "This is a mistake" reports and their CSV export (format of
// data/real/team_real_messages_TEMPLATE.csv).
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const M = require("../mistakes.js");

test("report stores the CORRECTED label and the language", () => {
  const a = M.makeEntry({ text: "Привет, урок перенесли", predictedFraud: true, prob: 0.71, source: "popup" });
  assert.equal(a.label, 0);
  assert.equal(a.language, "ru");
  assert.match(a.source, /popup: model said fraud 71%/);
  const b = M.makeEntry({ text: "Сіздің картаңыз бұғатталды", predictedFraud: false, prob: 0.2 });
  assert.equal(b.label, 1);
  assert.equal(b.language, "kk");
  assert.equal(M.makeEntry({ text: "Your parcel", predictedFraud: false }).language, "en");
});

test("newest first, one entry per text, capped", () => {
  let list = [];
  list = M.addTo(list, M.makeEntry({ text: "a message", predictedFraud: true }));
  list = M.addTo(list, M.makeEntry({ text: "b message", predictedFraud: true }));
  list = M.addTo(list, M.makeEntry({ text: "a message", predictedFraud: false }));
  assert.deepEqual(list.map((e) => [e.text, e.label]), [["a message", 1], ["b message", 0]]);
  for (let i = 0; i < 600; i++) list = M.addTo(list, M.makeEntry({ text: "m" + i, predictedFraud: true }));
  assert.equal(list.length, 500);
  assert.equal(M.addTo(list, M.makeEntry({ text: "   " })).length, 500, "empty text ignored");
});

test("CSV has the template's columns, quotes commas/newlines, guards formulas", () => {
  const template = fs.readFileSync(path.join(__dirname, "..", "..", "data/real/team_real_messages_TEMPLATE.csv"), "utf8")
    .split(/\r?\n/)[0].trim();
  assert.equal(M.CSV_COLUMNS.join(","), template);
  const csv = M.toCsv([
    { text: 'Карта, "заблокирована"\nсрочно', label: 1, language: "ru", source: "web.whatsapp.com", date: "2026-10-12" },
    { text: "=HYPERLINK(1)", label: 0, language: "en", source: "popup", date: "2026-10-12" },
  ]);
  const lines = csv.split("\n");
  assert.equal(lines[0], template);
  assert.equal(lines[1], '"Карта, ""заблокирована""');
  assert.equal(lines[2], 'срочно",1,ru,,web.whatsapp.com,2026-10-12');
  assert.equal(lines[3], "'=HYPERLINK(1),0,en,,popup,2026-10-12");
});

test("CSV quotes ';' and tabs (Excel with Russian/Kazakh settings splits on ';')", () => {
  const M = require("../mistakes.js");
  const csv = M.toCsv([{ text: "Привет;=cmd|' /C calc'!A0", label: 1, language: "ru", source: "x", date: "2026-10-15" }]);
  assert.match(csv.split("\n")[1], /^"Привет;=cmd\|' \/C calc'!A0",1,ru,,x,2026-10-15$/);
});
