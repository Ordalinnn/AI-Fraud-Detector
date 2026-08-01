importScripts("detector.js");

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "check-fraud-selection",
    title: 'Check "%s" for fraud/scam',
    contexts: ["selection"]
  });
});

chrome.contextMenus.onClicked.addListener((info) => {
  if (info.menuItemId !== "check-fraud-selection" || !info.selectionText) return;

  const result = self.FraudDetector.analyzeText(info.selectionText);

  chrome.storage.local.set({
    lastResult: result,
    lastText: info.selectionText
  });

  const bodyLines = [`Risk: ${result.pct}%`];
  if (result.isFraud) {
    bodyLines.push("Do not share codes, passwords, CVV, or card numbers. Do not click the link.");
  } else {
    bodyLines.push("Looks safe, but verify through an official source if unsure.");
  }

  chrome.notifications.create({
    type: "basic",
    iconUrl: "icons/icon128.png",
    title: `${result.emoji} ${result.label}`,
    message: bodyLines.join("\n"),
    priority: result.isFraud ? 2 : 0
  });
});
