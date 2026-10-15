const inputEl = document.getElementById("input");
const resultEl = document.getElementById("result");
const riskBannerEl = document.getElementById("riskBanner");
const adviceEl = document.getElementById("advice");
const reasonsEl = document.getElementById("reasons");
const domainsEl = document.getElementById("domains");
const langSelectEl = document.getElementById("langSelect");
const validationMsgEl = document.getElementById("validationMsg");
const modelWordsEl = document.getElementById("modelWords");
const autoscanEl = document.getElementById("autoscan");
const scanStatusEl = document.getElementById("scanStatus");
const mistakeBtnEl = document.getElementById("mistakeBtn");
const mistakeThanksEl = document.getElementById("mistakeThanks");
const mistakesBoxEl = document.getElementById("mistakesBox");
let lastText = "";
const AUTOSCAN_HOSTS = ["web.whatsapp.com", "web.telegram.org", "mail.google.com"];

// The site's full model, run on the device (ensemble.js). Loaded once; if
// it cannot be loaded, analyzeText falls back to the rules.
const scorerPromise = fetch(chrome.runtime.getURL("ensemble_model.json"))
  .then((r) => r.json())
  .then((m) => self.EnsembleScorer.parse(m))
  // the site shows the same "model ..." id, so anyone can check both run
  // the same model version
  .then((sc) => { modelId = sc.fingerprint.slice(0, 8); applyStaticText(); return sc; })
  .catch(() => null);

let currentLang = self.I18N.DEFAULT_LANG;
let modelId = "";
let lastResult = null;

function applyStaticText() {
  const s = self.I18N.strings(currentLang);
  document.documentElement.lang = currentLang;
  document.getElementById("appTitle").textContent = s.appTitle;
  document.getElementById("appSubtitle").textContent = s.appSubtitle;
  document.getElementById("footerText").textContent = `${s.footer} · v${chrome.runtime.getManifest().version}`
    + (modelId ? ` · model ${modelId}` : "");
  document.getElementById("analyzeBtn").textContent = s.analyzeBtn;
  document.getElementById("autoscanLabel").textContent = s.autoscanLabel;
  document.getElementById("mistakesDownload").textContent = s.mistakesDownload;
  document.getElementById("mistakesClear").textContent = s.mistakesClear;
  document.getElementById("mistakesNote").textContent = s.mistakesNote;
  refreshMistakes();
  inputEl.placeholder = s.placeholder;
  langSelectEl.value = currentLang;
  if (!validationMsgEl.classList.contains("hidden")) {
    validationMsgEl.textContent = s.emptyInputError;
  }
}

function domainFlagText(flag, s) {
  const template = s.domainFlagLabels[flag.labelKey] || flag.labelKey;
  return template.replace("{brand}", flag.brand || "");
}

function clearChildren(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
}

function render(result) {
  const s = self.I18N.strings(currentLang);
  resultEl.classList.remove("hidden");

  riskBannerEl.className = `risk-banner ${result.cssClass}`;
  riskBannerEl.textContent = `${result.emoji} ${s.riskLabels[result.levelKey]} — ${result.pct}%`;

  adviceEl.textContent = result.isFraud ? s.adviceBad : s.adviceGood;

  mistakeBtnEl.textContent = result.isFraud ? s.mistakeFlaggedBtn : s.mistakeMissedBtn;
  mistakeBtnEl.disabled = false;
  mistakeThanksEl.classList.add("hidden");

  const words = result.modelWords || [];
  modelWordsEl.classList.toggle("hidden", !(result.isFraud && words.length));
  modelWordsEl.textContent = `${s.modelWordsLabel} ${words.join(", ")}`;

  clearChildren(reasonsEl);
  if (result.reasons.length) {
    for (const key of result.reasons) {
      const li = document.createElement("li");
      li.textContent = s.reasonLabels[key] || key;
      reasonsEl.appendChild(li);
    }
  } else {
    const li = document.createElement("li");
    li.textContent = s.noReasons;
    reasonsEl.appendChild(li);
  }

  clearChildren(domainsEl);
  for (const d of result.domainInfo) {
    const li = document.createElement("li");
    li.appendChild(document.createTextNode(`🌐 ${d.domain}`));
    li.appendChild(document.createElement("br"));
    const flagText = d.flags.length
      ? d.flags.map((f) => (f.severity === "critical" ? "🔴 " : "⚠️ ") + domainFlagText(f, s)).join(" | ")
      : `✅ ${s.noDomainIssues}`;
    li.appendChild(document.createTextNode(flagText));
    domainsEl.appendChild(li);
  }
}

function reRenderIfResultShown() {
  applyStaticText();
  refreshScanStatus();
  if (lastResult) render(lastResult);
}

function showValidationError() {
  const s = self.I18N.strings(currentLang);
  validationMsgEl.textContent = s.emptyInputError;
  validationMsgEl.classList.remove("hidden");
}

function hideValidationError() {
  validationMsgEl.classList.add("hidden");
}

document.getElementById("analyzeBtn").addEventListener("click", async () => {
  const text = inputEl.value.trim();
  if (!text) {
    showValidationError();
    return;
  }
  hideValidationError();
  lastResult = self.FraudDetector.analyzeText(text, await scorerPromise);
  lastText = text;
  chrome.storage.local.set({ lastResult, lastText: text });
  render(lastResult);
});

inputEl.addEventListener("input", () => {
  if (inputEl.value.trim()) hideValidationError();
});

// "This is a mistake": saved on this computer, exportable as CSV for the team.
async function refreshMistakes() {
  const s = self.I18N.strings(currentLang);
  const list = await self.Mistakes.list();
  mistakesBoxEl.classList.toggle("hidden", list.length === 0);
  document.getElementById("mistakesTitle").textContent = s.mistakesTitle.replace("{n}", list.length);
}

mistakeBtnEl.addEventListener("click", async () => {
  if (!lastResult || !lastText) return;
  await self.Mistakes.add({ text: lastText, predictedFraud: lastResult.isFraud, prob: lastResult.prob, source: "popup" });
  mistakeBtnEl.disabled = true;
  mistakeThanksEl.textContent = self.I18N.strings(currentLang).mistakeThanks;
  mistakeThanksEl.classList.remove("hidden");
  refreshMistakes();
});

document.getElementById("mistakesDownload").addEventListener("click", async () => {
  const csv = self.Mistakes.toCsv(await self.Mistakes.list());
  const url = URL.createObjectURL(new Blob(["\ufeff" + csv], { type: "text/csv;charset=utf-8" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = `ai-fraud-detector-mistakes-${new Date().toISOString().slice(0, 10)}.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

document.getElementById("mistakesClear").addEventListener("click", async () => {
  await self.Mistakes.clear();
  refreshMistakes();
});

// Tells the user whether autoscan is actually running on the current tab,
// and if not, what to do (open the site / press F5).
function refreshScanStatus() {
  const s = self.I18N.strings(currentLang);
  if (!autoscanEl.checked) { scanStatusEl.textContent = s.statusOff; return; }
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const tab = tabs && tabs[0];
    if (!tab || tab.id == null) { scanStatusEl.textContent = s.statusOtherPage; return; }
    let host = "";
    try { host = new URL(tab.url || "").hostname; } catch (e) { host = ""; }
    const known = AUTOSCAN_HOSTS.includes(host);
    chrome.tabs.sendMessage(tab.id, { type: "status" }, (res) => {
      if (chrome.runtime.lastError || !res) {
        scanStatusEl.textContent = known ? s.statusReload : s.statusOtherPage;
      } else {
        scanStatusEl.textContent = s.statusActive
          .replace("{checked}", res.checked).replace("{flagged}", res.flagged);
      }
    });
  });
}

chrome.storage.local.get({ autoscan: true }, (data) => {
  autoscanEl.checked = data.autoscan !== false;
  refreshScanStatus();
});
autoscanEl.addEventListener("change", () => {
  chrome.storage.local.set({ autoscan: autoscanEl.checked }, () => setTimeout(refreshScanStatus, 600));
});

langSelectEl.addEventListener("change", () => {
  currentLang = langSelectEl.value;
  self.I18N.setLang(currentLang);
  reRenderIfResultShown();
});

// Restore saved language, then restore the last analysis (e.g. from a
// right-click "check for fraud" that just happened) if there is one.
self.I18N.getLang((lang) => {
  currentLang = lang;
  applyStaticText();
  refreshScanStatus();
  document.body.classList.remove("loading");

  chrome.storage.local.get(["lastResult", "lastText"], (data) => {
    if (data.lastText) { inputEl.value = data.lastText; lastText = data.lastText; }
    if (data.lastResult) {
      lastResult = data.lastResult;
      render(lastResult);
    }
  });
});
