importScripts("i18n.js", "scorer.js", "detector.js");

// The on-device model (char_model.tsv, ~0.9 MB) is parsed once per service
// worker life. If it can't be loaded, analysis falls back to the rules.
let scorerPromise = null;
function getScorer() {
  if (!scorerPromise) {
    scorerPromise = fetch(chrome.runtime.getURL("char_model.tsv"))
      .then((r) => r.text())
      .then((t) => self.CharScorer.parse(t))
      .catch((e) => { console.warn("AI Fraud Detector: model not loaded", e); scorerPromise = null; return null; });
  }
  return scorerPromise;
}

function contextMenuTitleFor(lang) {
  return self.I18N.strings(lang).contextMenuTitle;
}

chrome.runtime.onInstalled.addListener(() => {
  // onInstalled fires on every extension UPDATE, not just a fresh install,
  // but Chrome persists registered context menu ids across updates. Without
  // removeAll() first, create()'s second call with the same fixed id below
  // fails with a "duplicate id" chrome.runtime.lastError on every update.
  chrome.contextMenus.removeAll(() => {
    self.I18N.getLang((lang) => {
      chrome.contextMenus.create({
        id: "check-fraud-selection",
        title: contextMenuTitleFor(lang),
        contexts: ["selection"]
      });
    });
  });
});

// Keep the context menu label in sync if the language is changed from the popup.
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local" && changes.lang) {
    chrome.contextMenus.update("check-fraud-selection", {
      title: contextMenuTitleFor(changes.lang.newValue)
    });
  }
});

chrome.contextMenus.onClicked.addListener(async (info) => {
  if (info.menuItemId !== "check-fraud-selection" || !info.selectionText) return;
  const scorer = await getScorer();

  self.I18N.getLang((lang) => {
    const s = self.I18N.strings(lang);
    const result = self.FraudDetector.analyzeText(info.selectionText, scorer);

    chrome.storage.local.set({
      lastResult: result,
      lastText: info.selectionText
    });

    const message = `${result.pct}%\n${result.isFraud ? s.adviceBad : s.adviceGood}`;

    // Fixed id: a second right-click check replaces the previous notification
    // in place instead of stacking a new one in the notification tray.
    chrome.notifications.create("fraud-check-result", {
      type: "basic",
      iconUrl: "icons/icon128.png",
      title: `${result.emoji} ${s.riskLabels[result.levelKey]}`,
      message,
      priority: result.isFraud ? 2 : 0
    });
  });
});

// Scoring for content.js (automatic checks in WhatsApp Web / Telegram Web /
// Gmail). Message text is scored here, inside the extension, and only
// numbers go back to the page - nothing is sent over the network.
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (!msg || msg.type !== "score" || !Array.isArray(msg.texts)) return false;
  getScorer().then((scorer) => {
    sendResponse(msg.texts.map((t) => {
      const r = self.FraudDetector.analyzeText(String(t).slice(0, 4000), scorer);
      return { prob: r.prob, pct: r.pct, words: r.modelWords, reasons: r.reasons, domains: r.domainInfo };
    }));
  });
  return true; // async response
});
