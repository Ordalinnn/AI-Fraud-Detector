// Self-contained fraud/scam heuristic scorer for the browser extension.
//
// This mirrors the keyword/domain rule logic from the main AI Fraud Detector
// Streamlit app (app.py), but does NOT use the trained ML ensemble (Logistic
// Regression / Random Forest / Gradient Boosting) — scikit-learn models
// can't run inside a browser extension. Instead, the feature weights below
// are a hand-tuned substitute for what the ensemble would output, combined
// with the same rule-based "boost" logic used in the main app. Treat this
// as a lighter, offline approximation of the full web app, not a 1:1 port
// of its accuracy.
//
// Runs entirely locally: no network request is made, so message text never
// leaves the user's device.

const WORD_LISTS = {
  urgent: [
    "срочно", "шұғыл", "быстро", "немедленно", "қазір", "тез",
    "urgent", "now", "immediately", "asap", "прямо сейчас",
    "сейчас же", "без промедления"
  ],
  secret: [
    "код", "пароль", "cvv", "sms", "құпия", "password",
    "пин", "pin", "код подтверждения", "verification code",
    "one time code", "otp"
  ],
  money: [
    "карта", "счет", "банк", "ақша", "төле", "оплата", "перевод",
    "баланс", "средства", "деньги", "transfer", "payment",
    "wallet", "iban"
  ],
  threat: [
    "заблокирована", "удален", "штраф", "угрозой", "бұғатталды", "жабылады",
    "blocked", "suspended", "terminated", "penalty", "freeze",
    "ограничен", "будет закрыт"
  ],
  identity: [
    "паспорт", "иин", "удостоверение", "личность", "identity",
    "document", "id card", "жсн", "құжат"
  ],
  reward: [
    "выиграли", "приз", "бонус", "подарок", "акция", "компенсация",
    "won", "prize", "gift", "bonus", "reward", "ұтыс", "сыйлық"
  ],
  pressure: [
    "не говорите никому", "никому не сообщайте", "это секретно",
    "только сейчас", "последний шанс", "иначе", "do not tell anyone",
    "last chance", "only now", "қазір ғана"
  ],
  suspiciousDomainWords: [
    "login", "verify", "secure", "bonus", "gift",
    "account", "support", "confirm", "prize", "payment", "wallet",
    "security", "update", "auth", "free"
  ],
  suspiciousZones: [".xyz", ".top", ".click", ".site", ".online", ".live", ".info", ".icu"],
  knownBrands: [
    "kaspi.kz", "halykbank.kz", "sberbank.kz", "sberbank.ru",
    "tinkoff.ru", "vtb.ru", "egov.kz", "kazpost.kz", "dhl.com"
  ]
};

function countMatches(textLower, words) {
  return words.reduce((n, w) => n + (textLower.includes(w) ? 1 : 0), 0);
}

function extractUrls(textLower) {
  const matches = textLower.match(/https?:\/\/[^\s]+|www\.[^\s]+/g);
  return matches || [];
}

function getDomain(url) {
  return url.replace(/^https?:\/\//, "").replace(/^www\./, "").split("/")[0];
}

function brandImpersonation(domain) {
  for (const brand of WORD_LISTS.knownBrands) {
    const brandCore = brand.split(".")[0];
    if (domain.includes(brandCore) && domain !== brand && !domain.endsWith("." + brand)) {
      return brand;
    }
  }
  return null;
}

function domainFlags(domain) {
  const flags = [];
  const impersonated = brandImpersonation(domain);
  if (impersonated) flags.push({ label: `Impersonates ${impersonated}`, severity: "critical" });
  if (WORD_LISTS.suspiciousDomainWords.some((w) => domain.includes(w))) {
    flags.push({ label: "Suspicious keyword", severity: "warn" });
  }
  if (domain.length > 20) flags.push({ label: "Long domain", severity: "warn" });
  if (WORD_LISTS.suspiciousZones.some((z) => domain.endsWith(z))) {
    flags.push({ label: "Suspicious TLD", severity: "critical" });
  }
  if (/\d/.test(domain)) flags.push({ label: "Contains digits", severity: "warn" });
  return flags;
}

function extractFeatures(text) {
  const textLower = text.toLowerCase();
  const urls = extractUrls(textLower);
  const domains = urls.map(getDomain);

  let suspiciousDomain = 0, longDomain = 0, suspiciousZone = 0, digitDomain = 0, brandFlag = 0;
  for (const d of domains) {
    if (WORD_LISTS.suspiciousDomainWords.some((w) => d.includes(w))) suspiciousDomain = 1;
    if (d.length > 20) longDomain = 1;
    if (WORD_LISTS.suspiciousZones.some((z) => d.endsWith(z))) suspiciousZone = 1;
    if (/\d/.test(d)) digitDomain = 1;
    if (brandImpersonation(d)) brandFlag = 1;
  }

  const urgentCount = countMatches(textLower, WORD_LISTS.urgent);
  const threatCount = countMatches(textLower, WORD_LISTS.threat);

  return {
    hasLink: urls.length > 0 ? 1 : 0,
    urgentCount,
    secretCount: countMatches(textLower, WORD_LISTS.secret),
    moneyCount: countMatches(textLower, WORD_LISTS.money),
    threatCount,
    identityCount: countMatches(textLower, WORD_LISTS.identity),
    rewardCount: countMatches(textLower, WORD_LISTS.reward),
    pressureCount: countMatches(textLower, WORD_LISTS.pressure),
    suspiciousDomain,
    longDomain,
    suspiciousZone,
    digitDomain,
    brandFlag,
    urlCount: urls.length,
    hasMultipleWarnings: urgentCount > 0 && threatCount > 0 ? 1 : 0,
    domains
  };
}

function heuristicBaseScore(f) {
  let score = 0.03; // small baseline so neutral text isn't exactly 0%
  score += f.hasLink ? 0.08 : 0;
  score += Math.min(f.urgentCount * 0.06, 0.18);
  score += Math.min(f.secretCount * 0.15, 0.30);
  score += Math.min(f.moneyCount * 0.04, 0.12);
  score += Math.min(f.threatCount * 0.08, 0.16);
  score += Math.min(f.identityCount * 0.10, 0.20);
  score += Math.min(f.rewardCount * 0.07, 0.14);
  score += Math.min(f.pressureCount * 0.10, 0.20);
  score += f.suspiciousDomain ? 0.10 : 0;
  score += f.suspiciousZone ? 0.10 : 0;
  score += f.longDomain ? 0.05 : 0;
  score += f.digitDomain ? 0.05 : 0;
  score += f.brandFlag ? 0.15 : 0;
  return Math.min(score, 0.95);
}

function ruleBoost(f) {
  let boost = 0;
  if (f.hasLink && f.secretCount) boost += 0.15;
  if (f.urgentCount && f.moneyCount) boost += 0.12;
  if (f.secretCount && f.moneyCount) boost += 0.15;
  if (f.threatCount && f.hasLink) boost += 0.10;
  if (f.rewardCount && f.moneyCount) boost += 0.12;
  if (f.pressureCount) boost += 0.10;
  if (f.suspiciousZone || f.suspiciousDomain) boost += 0.10;
  if (f.brandFlag) boost += 0.15;
  if (f.hasMultipleWarnings) boost += 0.08;
  if (f.urlCount > 1) boost += 0.05;
  return Math.min(boost, 0.30);
}

const EXPLAIN_LABELS = {
  hasLink: "A link was detected",
  urgentCount: "Urgent action words were found",
  secretCount: "Possible request for code / password / CVV",
  moneyCount: "Bank, money, or card-related words were found",
  threatCount: "Pressure or threat indicators were found",
  identityCount: "Possible request for personal identity data",
  rewardCount: "Prize, bonus, or gift promise was found",
  pressureCount: "Pressure or secrecy phrase was found",
  suspiciousDomain: "Suspicious words were found in the domain",
  longDomain: "The domain is suspiciously long",
  suspiciousZone: "Suspicious domain zone detected",
  digitDomain: "The domain contains numbers",
  brandFlag: "Domain mimics a known bank/brand but isn't the real one",
  hasMultipleWarnings: "Both urgency and threat were detected simultaneously",
  urlCount: "Multiple links were detected"
};

function explain(f) {
  const reasons = [];
  for (const key of Object.keys(EXPLAIN_LABELS)) {
    const value = key === "urlCount" ? (f.urlCount > 1 ? 1 : 0) : f[key];
    if (value) reasons.push(EXPLAIN_LABELS[key]);
  }
  return reasons;
}

function riskStyle(prob) {
  if (prob < 0.3) return { label: "Low risk", cssClass: "risk-low", emoji: "🟢" };
  if (prob < 0.6) return { label: "Suspicious", cssClass: "risk-mid", emoji: "🟡" };
  if (prob < 0.8) return { label: "High risk", cssClass: "risk-high", emoji: "🟠" };
  return { label: "Critical risk", cssClass: "risk-critical", emoji: "🔴" };
}

function analyzeText(text) {
  const f = extractFeatures(text);
  const base = heuristicBaseScore(f);
  const boost = ruleBoost(f);
  const prob = Math.min(0.99, base + boost);
  const style = riskStyle(prob);
  const reasons = explain(f);
  const domainInfo = f.domains.map((d) => ({ domain: d, flags: domainFlags(d) }));
  return {
    prob,
    pct: Math.round(prob * 100),
    isFraud: prob >= 0.5,
    ...style,
    reasons,
    domainInfo
  };
}

// Exposed for popup.js / background.js (classic scripts, no ES modules,
// for maximum compatibility with Manifest V3 service workers).
if (typeof self !== "undefined") {
  self.FraudDetector = { analyzeText };
}
