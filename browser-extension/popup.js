const inputEl = document.getElementById("input");
const resultEl = document.getElementById("result");
const riskBannerEl = document.getElementById("riskBanner");
const adviceEl = document.getElementById("advice");
const reasonsEl = document.getElementById("reasons");
const domainsEl = document.getElementById("domains");
const langSelectEl = document.getElementById("langSelect");

let currentLang = self.I18N.DEFAULT_LANG;
let lastResult = null;

function applyStaticText() {
  const s = self.I18N.strings(currentLang);
  document.getElementById("appTitle").textContent = s.appTitle;
  document.getElementById("appSubtitle").textContent = s.appSubtitle;
  document.getElementById("footerText").textContent = s.footer;
  document.getElementById("analyzeBtn").textContent = s.analyzeBtn;
  inputEl.placeholder = s.placeholder;
  langSelectEl.value = currentLang;
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
      const span = document.createElement("span");
      span.textContent = s.reasonLabels[key] || key;
      reasonsEl.appendChild(span);
    }
  } else {
    const span = document.createElement("span");
    span.textContent = s.noReasons;
    reasonsEl.appendChild(span);
  }

  clearChildren(domainsEl);
  for (const d of result.domainInfo) {
    const div = document.createElement("div");
    div.appendChild(document.createTextNode(`🌐 ${d.domain}`));
    div.appendChild(document.createElement("br"));
    const flagText = d.flags.length
      ? d.flags.map((f) => (f.severity === "critical" ? "🔴 " : "⚠️ ") + domainFlagText(f, s)).join(" | ")
      : `✅ ${s.noDomainIssues}`;
    div.appendChild(document.createTextNode(flagText));
    domainsEl.appendChild(div);
  }
}

function reRenderIfResultShown() {
  applyStaticText();
  if (lastResult) render(lastResult);
}

document.getElementById("analyzeBtn").addEventListener("click", () => {
  const text = inputEl.value.trim();
  if (!text) return;
  lastResult = self.FraudDetector.analyzeText(text);
  chrome.storage.local.set({ lastResult, lastText: text });
  render(lastResult);
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

  chrome.storage.local.get(["lastResult", "lastText"], (data) => {
    if (data.lastText) inputEl.value = data.lastText;
    if (data.lastResult) {
      lastResult = data.lastResult;
      render(lastResult);
    }
  });
});
