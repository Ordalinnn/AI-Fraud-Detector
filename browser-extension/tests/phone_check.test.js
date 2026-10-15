// The offline checker page for phones (static/check/), run in jsdom the way
// an iPhone Shortcut opens it: the page, its scripts and the model are the
// real files; only the network is replaced by reading them from disk.
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { JSDOM, ResourceLoader } = require("jsdom");

const ROOT = path.join(__dirname, "..", "..");
const CHECK = path.join(ROOT, "static", "check");
const SITE = "https://ai-fraud-detector.streamlit.app";
const PAGE_URL = `${SITE}/~/+/app/static/check/index.html`;
const golden = JSON.parse(fs.readFileSync(path.join(__dirname, "ensemble_golden.json"), "utf8"));
const MODEL = fs.readFileSync(path.join(CHECK, "lib", "ensemble_model.json"), "utf8");

const localFile = (url) => path.join(CHECK, decodeURIComponent(new URL(url).pathname.replace(/^.*\/app\/static\/check\//, "")));

class DiskLoader extends ResourceLoader {
  fetch(url) { return Promise.resolve(fs.readFileSync(localFile(url))); }
}

async function open(url, { storage, session, modelGate } = {}) {
  const dom = await JSDOM.fromFile(path.join(CHECK, "index.html"), {
    url,
    runScripts: "dangerously",
    resources: new DiskLoader(),
    pretendToBeVisual: true,
    beforeParse(window) {
      window.fetch = async (u) => {
        assert.match(String(u), /ensemble_model\.json$/);
        if (modelGate) await modelGate;
        return { ok: true, status: 200, json: async () => JSON.parse(MODEL) };
      };
      if (storage) for (const [k, v] of Object.entries(storage)) window.localStorage.setItem(k, v);
      if (session) for (const [k, v] of Object.entries(session)) window.sessionStorage.setItem(k, v);
    },
  });
  await new Promise((resolve) => dom.window.addEventListener("load", resolve));
  return dom;
}

const $ = (dom, id) => dom.window.document.getElementById(id);
const shown = (dom, id) => !$(dom, id).classList.contains("hidden");

async function waitFor(fn, ms = 20000) {
  const end = Date.now() + ms;
  while (!fn()) {
    if (Date.now() > end) throw new Error("timed out");
    await new Promise((r) => setTimeout(r, 20));
  }
}

test("a Shortcut link (#text=...) is checked at once with the website's percentage", async () => {
  const i = golden.texts.findIndex((t, k) => t.length > 30 && golden.scores.final[k] > 0.9);
  const dom = await open(`${PAGE_URL}?lang=ru#text=${encodeURIComponent(golden.texts[i])}`);
  await waitFor(() => shown(dom, "result"));
  assert.equal($(dom, "input").value, golden.texts[i].trim());
  assert.ok($(dom, "banner").textContent.includes(golden.pct[i] + "%"), $(dom, "banner").textContent);
  // the message is not left in the address bar / history
  assert.equal(dom.window.location.hash, "");
  assert.equal(dom.window.location.search, "?lang=ru");
  // "Full analysis on the website" goes to the site itself (not the /~/+/
  // path) with ?text=, the form the website reads on its host
  assert.ok($(dom, "siteLink").href.startsWith(SITE + "/?text="), $(dom, "siteLink").href);
  assert.match($(dom, "footer").textContent, new RegExp(golden.training_fingerprint.slice(0, 8)));
  dom.window.close();
});

test("typed texts give the same verdict and percentage as the website", async () => {
  const dom = await open(PAGE_URL + "?lang=en");
  await dom.window.AFDCheck.modelReady;
  const picks = [0, 7, 120, 900, 1500, 2600, 3900, golden.texts.length - 1].filter((k) => golden.texts[k].trim());
  for (const k of picks) {
    $(dom, "input").value = golden.texts[k];
    await dom.window.AFDCheck.check();
    assert.ok($(dom, "banner").textContent.includes(golden.pct[k] + "%"), `${k}: ${$(dom, "banner").textContent} vs ${golden.pct[k]}`);
  }
  dom.window.close();
});

test("Shortcut started without text explains what to do", async () => {
  const dom = await open(PAGE_URL + "?lang=ru#text=");
  assert.ok(shown(dom, "notice"));
  assert.match($(dom, "notice").textContent, /без текста/);
  assert.ok(!shown(dom, "result"));
  dom.window.close();
});

test("languages: ?lang=, the picker and the remembered choice", async () => {
  let dom = await open(PAGE_URL + "?lang=kz");
  assert.equal($(dom, "pageTitle").textContent, "Хабарламаны тексеру");
  $(dom, "lang").value = "en";
  $(dom, "lang").dispatchEvent(new dom.window.Event("change"));
  assert.equal($(dom, "pageTitle").textContent, "Check a message");
  assert.equal(dom.window.localStorage.getItem("afd-lang"), "en");
  dom.window.close();
  dom = await open(PAGE_URL, { storage: { "afd-lang": "kz" } });
  assert.equal($(dom, "pageTitle").textContent, "Хабарламаны тексеру");
  dom.window.close();
});

test("all three languages have the same page texts", async () => {
  const dom = await open(PAGE_URL);
  const { PAGE } = dom.window.AFDCheck;
  const keys = Object.keys(PAGE.ru).sort();
  for (const lang of ["kz", "en"]) assert.deepEqual(Object.keys(PAGE[lang]).sort(), keys, lang);
  for (const lang of ["kz", "ru", "en"]) {
    for (const [k, v] of Object.entries(PAGE[lang])) assert.ok(v && v.trim(), `${lang}.${k} empty`);
  }
  dom.window.close();
});

test("'the model is wrong' is kept on the phone and exported in the team's CSV format", async () => {
  const dom = await open(PAGE_URL + "?lang=ru#text=" + encodeURIComponent("Срочно! Ваша карта заблокирована, назовите код из SMS"));
  await waitFor(() => shown(dom, "result"));
  $(dom, "mistakeBtn").click();
  const saved = JSON.parse(dom.window.localStorage.getItem("afd-mistakes"));
  assert.equal(saved.length, 1);
  assert.equal(saved[0].label, 0); // model said fraud, user says it is not
  assert.match(saved[0].source, /^phone page: model said fraud/);
  assert.ok(shown(dom, "mistakes"));
  assert.ok($(dom, "mistakeBtn").disabled);
  assert.match(dom.window.Mistakes.toCsv(saved).split("\n")[0], /^text,label,language,scam_type,source,date_collected$/);
  dom.window.close();
});

test("helpers: shared text parsing and the website address", async () => {
  const dom = await open(PAGE_URL);
  const { sharedText, siteRoot } = dom.window.AFDCheck;
  const L = (u) => new URL(u);
  assert.equal(sharedText(L("https://x/a#text=%D0%BA%D0%BE%D0%B4%20%2B%201")), "код + 1");
  assert.equal(sharedText(L("https://x/a?text=a+b")), "a b");
  assert.equal(sharedText(L("https://x/a#text=")), "");
  assert.equal(sharedText(L("https://x/a#text=%E0%A4%A")), "%E0%A4%A"); // broken encoding: kept as is
  assert.equal(sharedText(L("https://x/a")), null);
  assert.equal(siteRoot(L("https://s.streamlit.app/~/+/app/static/check/index.html")), "https://s.streamlit.app/");
  assert.equal(siteRoot(L("http://localhost:8501/app/static/check/index.html")), "http://localhost:8501/");
  dom.window.close();
});

test("the page's copies of the extension files are up to date", () => {
  for (const f of ["i18n.js", "mistakes.js", "normalize.js", "scorer.js", "ensemble.js", "detector.js", "ensemble_model.json"]) {
    assert.ok(fs.readFileSync(path.join(CHECK, "lib", f)).equals(fs.readFileSync(path.join(__dirname, "..", f))),
      `static/check/lib/${f} differs — run python scripts/build_web_checker.py`);
  }
});

test("no regex lookbehind in what the phone page runs (Safari has it only since iOS 16.4)", () => {
  const files = ["app.js", ...["i18n.js", "mistakes.js", "normalize.js", "scorer.js", "ensemble.js", "detector.js"].map((f) => "lib/" + f)];
  for (const f of files) {
    const code = fs.readFileSync(path.join(CHECK, f), "utf8").split("\n").filter((l) => !/^\s*\/\//.test(l)).join("\n");
    assert.ok(!/\(\?<[!=]/.test(code), `${f} uses a lookbehind`);
  }
});


const detector = require("../detector.js");
const { EnsembleScorer } = require("../ensemble.js");
const extScorer = EnsembleScorer.parse(MODEL);

test("text is scored as typed (no trim), like the website's text box", async () => {
  const raw = "\n😀 Ваша карта заблокирована 😀\n\n";
  const want = detector.analyzeText(raw, extScorer).pct;
  assert.notEqual(want, detector.analyzeText(raw.trim(), extScorer).pct, "example must depend on the line breaks");
  const dom = await open(PAGE_URL + "?lang=en");
  await dom.window.AFDCheck.modelReady;
  $(dom, "input").value = raw;
  await dom.window.AFDCheck.check();
  assert.ok($(dom, "banner").textContent.includes(want + "%"), $(dom, "banner").textContent + " vs " + want);
  dom.window.close();
});

test("long messages: the website's 8000-character cap, never half an emoji", async () => {
  const long = "a" + "😀".repeat(9000) + " срочно код";
  const dom = await open(PAGE_URL + "?lang=en#text=" + encodeURIComponent(long));
  await waitFor(() => shown(dom, "result"));
  const kept = $(dom, "input").value;
  assert.equal(Array.from(kept).length, 8000);
  assert.ok(!/[\uD800-\uDBFF]$/.test(kept), "ends with half a surrogate pair");
  assert.ok($(dom, "siteLink").href.includes("?text="), "site link was not built");
  dom.window.close();
});

test("the Shortcut's own setup link in the clipboard is explained, not checked", async () => {
  const own = PAGE_URL + "#text=";
  const dom = await open(PAGE_URL + "?lang=ru#text=" + encodeURIComponent(own));
  assert.ok(shown(dom, "notice"));
  assert.match($(dom, "notice").textContent, /ссылка на эту страницу/);
  assert.ok(!shown(dom, "result"));
  assert.equal($(dom, "input").value, "");
  // and a language switch re-translates the notice
  $(dom, "lang").value = "en";
  $(dom, "lang").dispatchEvent(new dom.window.Event("change"));
  assert.match($(dom, "notice").textContent, /link to this page/);
  dom.window.close();
});

test("a new #text= in an already open tab is checked", async () => {
  const dom = await open(PAGE_URL + "?lang=en#text=" + encodeURIComponent("Привет, как дела? Увидимся завтра"));
  await waitFor(() => shown(dom, "result"));
  const before = $(dom, "banner").textContent;
  dom.window.location.hash = "#text=" + encodeURIComponent("Срочно! Ваша карта заблокирована, назовите код из SMS http://kaspi-login.xyz");
  await waitFor(() => $(dom, "banner").textContent !== before);
  assert.match($(dom, "input").value, /карта заблокирована/);
  dom.window.close();
});

test("Clear while the model is loading cancels the pending check", async () => {
  let release;
  const gate = new Promise((r) => { release = r; });
  const dom = await open(PAGE_URL + "?lang=en#text=" + encodeURIComponent("Срочно! Назовите код из SMS"), { modelGate: gate });
  assert.ok($(dom, "checkBtn").disabled, "button should wait for the model");
  $(dom, "clearBtn").click();
  assert.ok(!$(dom, "checkBtn").disabled);
  release();
  await dom.window.AFDCheck.modelReady;
  await new Promise((r) => setTimeout(r, 50));
  assert.ok(!shown(dom, "result"), "a cleared message must not show a result");
  dom.window.close();
});

test("a reloaded tab brings back the message checked in it", async () => {
  const dom = await open(PAGE_URL + "?lang=en", { session: { "afd-last-text": "Срочно! Назовите код из SMS" } });
  await waitFor(() => shown(dom, "result"));
  assert.equal($(dom, "input").value, "Срочно! Назовите код из SMS");
  $(dom, "clearBtn").click();
  assert.equal(dom.window.sessionStorage.getItem("afd-last-text"), null);
  dom.window.close();
});

test("'Warn family' never forwards a live phishing link", async () => {
  const dom = await open(PAGE_URL);
  await dom.window.AFDCheck.modelReady;
  const text = "Kaspi: счёт заблокирован, подтвердите по ссылке https://Kaspi-Secure-Login.top/verify?id=8812 или www.kaspi-help.xyz";
  const result = dom.window.FraudDetector.analyzeText(text, null);
  const safe = dom.window.AFDCheck.defang(text, result);
  assert.ok(!/https?:\/\//i.test(safe), safe);
  assert.ok(!/kaspi-secure-login\.top/i.test(safe) && !/kaspi-help\.xyz/i.test(safe), safe);
  assert.match(safe, /hxxps:\/\/Kaspi-Secure-Login\[\.\]top/);
  dom.window.close();
});
