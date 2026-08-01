const inputEl = document.getElementById("input");
const resultEl = document.getElementById("result");
const riskBannerEl = document.getElementById("riskBanner");
const adviceEl = document.getElementById("advice");
const reasonsEl = document.getElementById("reasons");
const domainsEl = document.getElementById("domains");

function render(result, text) {
  resultEl.classList.remove("hidden");

  riskBannerEl.className = `risk-banner ${result.cssClass}`;
  riskBannerEl.textContent = `${result.emoji} ${result.label} — ${result.pct}%`;

  adviceEl.textContent = result.isFraud
    ? "⚠️ Do not share codes, passwords, CVV, or card numbers. Do not click the link. Contact the organization only through its official number."
    : "✅ This message looks safe. If unsure, verify through an official source.";

  reasonsEl.innerHTML = result.reasons.length
    ? result.reasons.map((r) => `<span>${r}</span>`).join("")
    : "";

  domainsEl.innerHTML = result.domainInfo.length
    ? result.domainInfo
        .map((d) => {
          const flagText = d.flags.length
            ? d.flags.map((f) => (f.severity === "critical" ? "🔴 " : "⚠️ ") + f.label).join(" | ")
            : "✅ No issues found";
          return `<div>🌐 ${d.domain}<br>${flagText}</div>`;
        })
        .join("")
    : "";
}

document.getElementById("analyzeBtn").addEventListener("click", () => {
  const text = inputEl.value.trim();
  if (!text) return;
  const result = self.FraudDetector.analyzeText(text);
  chrome.storage.local.set({ lastResult: result, lastText: text });
  render(result, text);
});

// If a right-click "check for fraud" analysis just happened, show it here too.
chrome.storage.local.get(["lastResult", "lastText"], (data) => {
  if (data.lastText) inputEl.value = data.lastText;
  if (data.lastResult) render(data.lastResult, data.lastText);
});
