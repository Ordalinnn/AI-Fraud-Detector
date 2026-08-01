importScripts("i18n.js", "detector.js");

function contextMenuTitleFor(lang) {
  return self.I18N.strings(lang).contextMenuTitle;
}

chrome.runtime.onInstalled.addListener(() => {
  self.I18N.getLang((lang) => {
    chrome.contextMenus.create({
      id: "check-fraud-selection",
      title: contextMenuTitleFor(lang),
      contexts: ["selection"]
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

chrome.contextMenus.onClicked.addListener((info) => {
  if (info.menuItemId !== "check-fraud-selection" || !info.selectionText) return;

  self.I18N.getLang((lang) => {
    const s = self.I18N.strings(lang);
    const result = self.FraudDetector.analyzeText(info.selectionText);

    chrome.storage.local.set({
      lastResult: result,
      lastText: info.selectionText
    });

    const message = `${result.pct}%\n${result.isFraud ? s.adviceBad : s.adviceGood}`;

    chrome.notifications.create({
      type: "basic",
      iconUrl: "icons/icon128.png",
      title: `${result.emoji} ${s.riskLabels[result.levelKey]}`,
      message,
      priority: result.isFraud ? 2 : 0
    });
  });
});
