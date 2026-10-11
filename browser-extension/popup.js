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
const AUTOSCAN_HOSTS = ["web.whatsapp.com", "web.telegram.org", "mail.google.com"];

// The on-device ML model (same as the Android app). Loaded once; if it
// cannot be loaded, analyzeText falls back to the rules.
const scorerPromise = fetch(chrome.runtime.getURL("char_model.tsv"))
  .then((r) => r.text())
  .then((tsv) => self.CharScorer.parse(tsv))
  .catch(() => null);

let currentLang = self.I18N.DEFAULT_LANG;
let lastResult = null;

function applyStaticText() {
  const s = self.I18N.strings(currentLang);
  document.documentElement.lang = currentLang;
  document.getElementById("appTitle").textContent = s.appTitle;
  document.getElementById("appSubtitle").textContent = s.appSubtitle;
  document.getElementById("footerText").textContent = `${s.footer} · v${chrome.runtime.getManifest().version}`;
  document.getElementById("analyzeBtn").textContent = s.analyzeBtn;
  document.getElementById("autoscanLabel").textContent = s.autoscanLabel;
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
  chrome.storage.local.set({ lastResult, lastText: text });
  render(lastResult);
});

inputEl.addEventListener("input", () => {
  if (inputEl.value.trim()) hideValidationError();
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
    if (data.lastText) inputEl.value = data.lastText;
    if (data.lastResult) {
      lastResult = data.lastResult;
      render(lastResult);
    }
  });
});
