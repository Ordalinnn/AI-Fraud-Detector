// content.js against fake WhatsApp Web / Telegram Web / Gmail pages (jsdom),
// scored by the real on-device model, the way background.js scores them.
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { JSDOM } = require("jsdom");

const { CharScorer } = require("../scorer.js");
const { analyzeText } = require("../detector.js");
const I18N = require("../i18n.js");
const { siteFor, createScanner, findCandidates } = require("../content.js");

const scorer = CharScorer.parse(fs.readFileSync(path.join(__dirname, "..", "char_model.tsv"), "utf8"));
const SCAM = "Здравствуйте, это служба безопасности банка. Ваша карта заблокирована, назовите код из SMS";
const OK = "Привет, урок перенесли на пятницу, не забудь учебник";

let calls = 0;
function score(texts) {
  calls += 1;
  return Promise.resolve(texts.map((t) => {
    const r = analyzeText(t, scorer);
    return { prob: r.prob, pct: r.pct, words: r.modelWords, reasons: r.reasons, domains: r.domainInfo };
  }));
}

const tick = (ms) => new Promise((r) => setTimeout(r, ms));

function page(html) {
  const dom = new JSDOM(`<!doctype html><body>${html}</body>`);
  return dom.window.document;
}

function waBubble(text, out = false) {
  return `<div class="${out ? "message-out" : "message-in"}"><div class="copyable-text" data-pre-plain-text="[10:00, 11.10.2026] Bank: ">` +
    `<span class="selectable-text"><span>${text}</span></span></div></div>`;
}

test("site table covers the three web apps and nothing else", () => {
  assert.equal(siteFor("web.whatsapp.com").host, "web.whatsapp.com");
  assert.equal(siteFor("web.telegram.org").host, "web.telegram.org");
  assert.equal(siteFor("mail.google.com").host, "mail.google.com");
  assert.equal(siteFor("example.com"), null);
  assert.equal(siteFor("evilweb.whatsapp.com.example"), null);
});

test("WhatsApp: scam messages get a label (incoming and own), normal ones don't", async () => {
  const doc = page(`<div id="chat">${waBubble(SCAM)}${waBubble(OK)}${waBubble(SCAM, true)}${waBubble("ок")}</div>`);
  const sc = createScanner({ doc, site: siteFor("web.whatsapp.com"), score, strings: () => I18N.strings("ru") });
  sc.start();
  await sc.scan();
  const badges = doc.querySelectorAll(".afd-badge");
  assert.equal(badges.length, 2);
  assert.match(badges[0].textContent, /^⚠️ Похоже на мошенничество · \d+% · почему\? ▾$/);
  assert.deepEqual(sc.status(), { host: "web.whatsapp.com", running: true, checked: 3, flagged: 2 });
  assert.ok(badges[0].parentElement.previousElementSibling.textContent.includes("заблокирована"));
  sc.stop();
});

test("WhatsApp: a message arriving later is picked up by the observer", async () => {
  const doc = page(`<div id="chat">${waBubble(OK)}</div>`);
  const sc = createScanner({ doc, site: siteFor("web.whatsapp.com"), score, strings: () => I18N.strings("kz") });
  sc.start();
  await tick(500);
  assert.equal(doc.querySelectorAll(".afd-badge").length, 0);
  doc.getElementById("chat").insertAdjacentHTML("beforeend", waBubble("Сіздің картаңыз бұғатталды. SMS-тегі кодты айтыңыз, әйтпесе ақшаңыз жоғалады"));
  await tick(900);
  const badges = doc.querySelectorAll(".afd-badge");
  assert.equal(badges.length, 1);
  assert.match(badges[0].textContent, /Алаяқтыққа ұқсайды/);
  sc.stop();
});

test("re-rendered messages are scored from cache and never get two labels", async () => {
  const doc = page(`<div id="chat">${waBubble(SCAM)}</div>`);
  const sc = createScanner({ doc, site: siteFor("web.whatsapp.com"), score, strings: () => I18N.strings("en") });
  sc.start();
  await sc.scan();
  const before = calls;
  // WhatsApp throws the node away and draws it again when you scroll
  doc.getElementById("chat").innerHTML = waBubble(SCAM);
  await sc.scan();
  await sc.scan();
  assert.equal(calls, before, "no second request for the same text");
  assert.equal(doc.querySelectorAll(".afd-badge").length, 1);
  sc.stop();
});

test("Telegram (web A and web K) and Gmail markup", async () => {
  const tgA = page(`<div class="Message"><div class="text-content">${SCAM}</div></div>` +
    `<div class="Message own"><div class="text-content">${SCAM}</div></div>`);
  const tgK = page(`<div class="bubble"><div class="message"><span class="translatable-message">${SCAM}</span></div></div>` +
    `<div class="bubble is-out"><div class="message"><span class="translatable-message">${SCAM}</span></div></div>`);
  const gm = page(`<div class="a3s">${OK}<div class="a3s">quoted ${SCAM}</div></div><div class="a3s">${SCAM}</div>`);
  for (const [doc, host, expected] of [[tgA, "web.telegram.org", 2], [tgK, "web.telegram.org", 2], [gm, "mail.google.com", 2]]) {
    const sc = createScanner({ doc, site: siteFor(host), score, strings: () => I18N.strings("ru") });
    sc.start();
    await sc.scan();
    assert.equal(doc.querySelectorAll(".afd-badge").length, expected, host);
    sc.stop();
  }
  // Gmail: label goes above the email body, not after a long email; a quoted
  // email nested inside another is part of it and gets no label of its own
  assert.equal(gm.querySelector(".afd-box").nextElementSibling.className, "a3s");
  assert.equal(gm.querySelectorAll(".a3s .afd-badge").length, 0);
});

test("clicking the label shows what is suspicious, clicking again hides it", async () => {
  const text = "Здравствуйте, служба безопасности Kaspi. Ваша карта заблокирована! Срочно назовите код из SMS или перейдите http://kaspi-bank-secure.xyz/login";
  const dom = new JSDOM(`<!doctype html><body><div id="chat">${waBubble(text)}</div></body>`);
  const doc = dom.window.document;
  let bubbled = 0;
  doc.getElementById("chat").addEventListener("click", () => { bubbled += 1; });
  const sc = createScanner({ doc, site: siteFor("web.whatsapp.com"), score, strings: () => I18N.strings("ru") });
  sc.start();
  await sc.scan();
  const badge = doc.querySelector(".afd-badge");
  const details = doc.querySelector(".afd-details");
  assert.equal(details.hidden, true);

  badge.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true, cancelable: true }));
  assert.equal(details.hidden, false);
  assert.equal(badge.getAttribute("aria-expanded"), "true");
  assert.equal(bubbled, 0, "the click must not reach WhatsApp's own handlers");
  const shown = details.textContent;
  assert.match(shown, /Почему это подозрительно/);
  assert.match(shown, /Подозрительные слова:/);
  assert.ok(details.querySelectorAll(".afd-chip").length >= 1);
  assert.match(shown, /Признаки:/);
  assert.ok(details.querySelectorAll(".afd-list li").length >= 2, shown);
  assert.match(shown, /kaspi-bank-secure\.xyz — /, "the fake link and why it is suspicious");
  assert.match(shown, /Не сообщайте коды/);
  const link = details.querySelector("a.afd-link");
  assert.ok(link.href.startsWith("https://ai-fraud-detector-p6jkftqrssphfcsxxepipo.streamlit.app/?text="));
  assert.equal(decodeURIComponent(link.href.split("?text=")[1]), text);

  badge.dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true }));
  assert.equal(details.hidden, true);
  sc.stop();
});

test("short texts are skipped until they grow; reset removes all labels", async () => {
  const doc = page(`<div class="text-content">ок</div>`);
  const site = siteFor("web.telegram.org");
  assert.equal(findCandidates(site, doc.body).length, 0);
  doc.querySelector(".text-content").textContent = SCAM;
  assert.equal(findCandidates(site, doc.body).length, 1);
  const sc = createScanner({ doc, site, score, strings: () => I18N.strings("ru") });
  sc.start();
  await sc.scan();
  assert.equal(doc.querySelectorAll(".afd-badge").length, 1);
  sc.stop();
  sc.reset();
  assert.equal(doc.querySelectorAll(".afd-badge, [data-afd-checked]").length, 0);
});

test("a dead extension (updated while the tab was open) stops the scanner quietly", async () => {
  const doc = page(`<div class="text-content">${SCAM}</div>`);
  const dead = () => Promise.reject(new Error("Extension context invalidated."));
  const sc = createScanner({ doc, site: siteFor("web.telegram.org"), score: dead, strings: () => I18N.strings("ru") });
  sc.start();
  await tick(600);
  assert.equal(sc.running, false);
  assert.equal(doc.querySelectorAll(".afd-badge").length, 0);
});

test("manifest injects content.js only on the three sites", () => {
  const m = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "manifest.json"), "utf8"));
  assert.deepEqual(m.content_scripts[0].matches.sort(),
    ["https://mail.google.com/*", "https://web.telegram.org/*", "https://web.whatsapp.com/*"]);
  assert.deepEqual(m.content_scripts[0].js, ["i18n.js", "mistakes.js", "content.js"]);
  // host permissions = exactly the same three sites (only so the popup can
  // see that the tab is WhatsApp/Telegram/Gmail and say "press F5")
  assert.deepEqual([...m.host_permissions].sort(), [...m.content_scripts[0].matches].sort());
  for (const f of [...m.content_scripts[0].js, ...m.content_scripts[0].css]) {
    assert.ok(fs.existsSync(path.join(__dirname, "..", f)), f);
  }
});

test("'Not fraud' in the details panel saves a mistake report", async () => {
  const saved = [];
  globalThis.Mistakes = { add: async (e) => { saved.push(e); return saved.length; } };
  try {
    const dom = new JSDOM(`<!doctype html><body><div id="chat">${waBubble(SCAM)}</div></body>`, { url: "https://web.whatsapp.com/" });
    const doc = dom.window.document;
    const sc = createScanner({ doc, site: siteFor("web.whatsapp.com"), score, strings: () => I18N.strings("ru") });
    sc.start();
    await sc.scan();
    doc.querySelector(".afd-badge").dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
    const btn = doc.querySelector(".afd-mistake");
    assert.match(btn.textContent, /Это не мошенничество/);
    btn.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
    await tick(10);
    assert.equal(saved.length, 1);
    assert.equal(saved[0].text, SCAM);
    assert.equal(saved[0].predictedFraud, true);
    assert.equal(saved[0].source, "web.whatsapp.com");
    assert.match(btn.textContent, /Спасибо/);
    assert.equal(btn.disabled, true);
    // the red label is replaced by a quiet note right away
    assert.equal(doc.querySelectorAll(".afd-badge").length, 0);
    assert.match(doc.querySelector(".afd-note").textContent, /не мошенничество/);
    assert.equal(sc.status().flagged, 0);
    sc.stop();
  } finally {
    delete globalThis.Mistakes;
  }
});


test("texts the user marked 'not fraud' are not flagged again, even after a redraw", async () => {
  const M = require("../mistakes.js");
  const marks = M.addTo([], M.makeEntry({ text: SCAM, predictedFraud: true, prob: 0.97 }));
  const dismissed = M.notFraudSet(marks);
  const doc = page(`<div id="chat">${waBubble(SCAM)}${waBubble(SCAM.replace("банка", "Kaspi"))}</div>`);
  const sc = createScanner({ doc, site: siteFor("web.whatsapp.com"), score, strings: () => I18N.strings("ru"),
    isDismissed: (t) => dismissed.has(M.textKey(t)) });
  sc.start();
  await sc.scan();
  // only the other (unmarked) scam gets a label
  assert.equal(doc.querySelectorAll(".afd-badge").length, 1);
  assert.ok(doc.querySelector(".afd-box").previousElementSibling.textContent.includes("Kaspi"));
  doc.getElementById("chat").innerHTML = waBubble(SCAM);
  await sc.scan();
  assert.equal(doc.querySelectorAll(".afd-badge").length, 0);
  sc.stop();
});

test("only 'not fraud' marks are skipped, not 'it IS fraud' ones", () => {
  const M = require("../mistakes.js");
  let list = M.addTo([], M.makeEntry({ text: "missed scam text", predictedFraud: false }));
  list = M.addTo(list, M.makeEntry({ text: "  ordinary text  ", predictedFraud: true }));
  const set = M.notFraudSet(list);
  assert.ok(set.has(M.textKey("ordinary text")));
  assert.ok(!set.has(M.textKey("missed scam text")));
});
