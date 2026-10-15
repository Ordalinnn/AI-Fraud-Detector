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

async function open(url, { storage } = {}) {
  const dom = await JSDOM.fromFile(path.join(CHECK, "index.html"), {
    url,
    runScripts: "dangerously",
    resources: new DiskLoader(),
    pretendToBeVisual: true,
    beforeParse(window) {
      window.fetch = async (u) => {
        assert.match(String(u), /ensemble_model\.json$/);
        return { ok: true, status: 200, json: async () => JSON.parse(MODEL) };
      };
      if (storage) for (const [k, v] of Object.entries(storage)) window.localStorage.setItem(k, v);
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
  // "Full analysis on the website" goes to the site itself, not to the /~/+/ path
  assert.ok($(dom, "siteLink").href.startsWith(SITE + "/#text="), $(dom, "siteLink").href);
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
