const inputEl = document.getElementById("input");
const resultEl = document.getElementById("result");
const riskBannerEl = document.getElementById("riskBanner");
const adviceEl = document.getElementById("advice");
const reasonsEl = document.getElementById("reasons");
const domainsEl = document.getElementById("domains");
const langSelectEl = document.getElementById("langSelect");
const validationMsgEl = document.getElementById("validationMsg");

let currentLang = self.I18N.DEFAULT_LANG;
let lastResult = null;

function applyStaticText() {
  const s = self.I18N.strings(currentLang);
  document.documentElement.lang = currentLang;
  document.getElementById("appTitle").textContent = s.appTitle;
  document.getElementById("appSubtitle").textContent = s.appSubtitle;
  document.getElementById("footerText").textContent = s.footer;
  document.getElementById("analyzeBtn").textContent = s.analyzeBtn;
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

document.getElementById("analyzeBtn").addEventListener("click", () => {
  const text = inputEl.value.trim();
  if (!text) {
    showValidationError();
    return;
  }
  hideValidationError();
  lastResult = self.FraudDetector.analyzeText(text);
  chrome.storage.local.set({ lastResult, lastText: text });
  render(lastResult);
});

inputEl.addEventListener("input", () => {
  if (inputEl.value.trim()) hideValidationError();
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
  document.body.classList.remove("loading");

  chrome.storage.local.get(["lastResult", "lastText"], (data) => {
    if (data.lastText) inputEl.value = data.lastText;
    if (data.lastResult) {
      lastResult = data.lastResult;
      render(lastResult);
    }
  });
});
