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
    "карта", "счет", "банк", "ақша", "төле", "оплат", "перевод",
    "баланс", "средства", "деньги", "transfer", "payment", "pay",
    "wallet", "iban"
  ],
  // "оплат" (not "оплата") is a stem: it also catches "оплатить"/
  // "оплатите"/"оплачивает" since matching is substring-based and doesn't
  // handle Russian verb conjugation otherwise. Same idea for "pay".
  verificationService: [
    "провер", "справк", "сертификат", "реестр", "registry",
    "verif", "certificate"
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
    "security", "update", "auth", "free", "check", "registry"
  ],
  suspiciousZones: [".xyz", ".top", ".click", ".site", ".online", ".live", ".info", ".icu"],
  knownBrands: [
    "kaspi.kz", "halykbank.kz", "sberbank.kz", "sberbank.ru",
    "tinkoff.ru", "vtb.ru", "egov.kz", "kazpost.kz", "dhl.com",
    "paypal.com", "amazon.com", "apple.com", "google.com", "microsoft.com",
    "instagram.com", "whatsapp.com", "facebook.com", "netflix.com",
    "temu.com", "shein.com", "aliexpress.com"
  ]
};

// Cyrillic characters that render visually identical (or near-identical) to
// a Latin letter in most fonts — the basis of IDN homograph phishing, e.g.
// "kаspi.kz" with a Cyrillic а (U+0430) looks exactly like kaspi.kz to a
// human but is a completely different domain that substring matching alone
// would miss.
const CONFUSABLES = {
  "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
  "і": "i", "ѕ": "s", "һ": "h", "ј": "j"
};

function deconfuse(domain) {
  return domain.replace(/./gu, (ch) => CONFUSABLES[ch] || ch);
}

function isAscii(s) {
  return /^[\x00-\x7F]*$/.test(s);
}

function levenshtein(a, b) {
  if (a === b) return 0;
  if (!a) return b.length;
  if (!b) return a.length;
  let prevRow = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const curRow = [i];
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      curRow[j] = Math.min(prevRow[j] + 1, curRow[j - 1] + 1, prevRow[j - 1] + cost);
    }
    prevRow = curRow;
  }
  return prevRow[b.length];
}

function countMatches(textLower, words) {
  return words.reduce((n, w) => n + (textLower.includes(w) ? 1 : 0), 0);
}

function extractUrls(textLower) {
  const matches = textLower.match(/https?:\/\/[^\s]+|www\.[^\s]+/g);
  return matches || [];
}

// Real scams increasingly write a domain as plain text (e.g. "check-tech-
// base.ru") instead of a clickable http(s):// or www. link, specifically
// to dodge filters that only look for literal links. This catches those
// bare mentions so they still register as a domain.
const BARE_DOMAIN_PATTERN = /\b[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9-]+)*\.(?:ru|kz|com|net|org|info|xyz|top|site|online|live|icu|click|by|ua|io|me|co)\b/g;

function extractBareDomains(textLower, urls) {
  let masked = textLower;
  for (const u of urls) {
    masked = masked.split(u).join(" ");
  }
  const matches = masked.match(BARE_DOMAIN_PATTERN);
  return matches || [];
}

function getDomain(url) {
  return url.replace(/^https?:\/\//, "").replace(/^www\./, "").split("/")[0];
}

function brandImpersonation(domain) {
  // Also catches two patterns plain substring matching misses: homograph
  // domains that swap in look-alike Cyrillic characters (deconfused before
  // comparison), and single-character typosquats like "kasp1.kz" or
  // "paypa1.com" (caught via edit distance against each domain token).
  //
  // Checked up front against the RAW (not deconfused) domain, against ALL
  // brands at once: some brand cores repeat across TLDs (sberbank.kz vs
  // sberbank.ru), so a per-brand-loop skip would still flag one as
  // impersonating the other. A homoglyph domain that only equals a brand
  // after deconfusion isn't the real domain — it must NOT be caught here.
  if (WORD_LISTS.knownBrands.some((b) => domain === b || domain.endsWith("." + b))) return null;
  const normalized = deconfuse(domain);
  for (const brand of WORD_LISTS.knownBrands) {
    const brandCore = brand.split(".")[0];
    if (normalized.includes(brandCore)) return brand;
    // Short cores (e.g. "vtb", "dhl") are skipped — a 1-edit match against
    // a 3-letter word is too likely to be a false positive.
    if (brandCore.length >= 5) {
      for (const token of normalized.split(/[^a-z0-9]+/)) {
        if (token.length >= 4 && levenshtein(token, brandCore) === 1) return brand;
      }
    }
  }
  return null;
}

function domainFlags(domain) {
  // Returns language-agnostic flag keys; popup.js resolves labelKey (and
  // {brand} for "impersonates") to the current UI language's text.
  const flags = [];
  const impersonated = brandImpersonation(domain);
  if (impersonated) {
    flags.push({ labelKey: "impersonates", brand: impersonated, severity: "critical" });
  } else if (!isAscii(domain)) {
    // No known brand matched, but non-Latin characters in a domain are
    // themselves a classic IDN homograph phishing signal even when the
    // impersonated brand isn't in knownBrands.
    flags.push({ labelKey: "nonLatinDomain", severity: "critical" });
  }
  if (WORD_LISTS.suspiciousDomainWords.some((w) => domain.includes(w))) {
    flags.push({ labelKey: "suspiciousKeyword", severity: "warn" });
  }
  if (domain.length > 20) flags.push({ labelKey: "longDomain", severity: "warn" });
  if (WORD_LISTS.suspiciousZones.some((z) => domain.endsWith(z))) {
    flags.push({ labelKey: "suspiciousTld", severity: "critical" });
  }
  if (/\d/.test(domain)) flags.push({ labelKey: "containsDigits", severity: "warn" });
  return flags;
}

function extractFeatures(text) {
  const textLower = text.toLowerCase();
  const urls = extractUrls(textLower);
  const bareDomains = extractBareDomains(textLower, urls);
  const domains = urls.map(getDomain).concat(bareDomains);
  const linkCount = urls.length + bareDomains.length;

  let suspiciousDomain = 0, longDomain = 0, suspiciousZone = 0, digitDomain = 0, brandFlag = 0, homoglyphDomain = 0;
  for (const d of domains) {
    if (WORD_LISTS.suspiciousDomainWords.some((w) => d.includes(w))) suspiciousDomain = 1;
    if (d.length > 20) longDomain = 1;
    if (WORD_LISTS.suspiciousZones.some((z) => d.endsWith(z))) suspiciousZone = 1;
    if (/\d/.test(d)) digitDomain = 1;
    if (brandImpersonation(d)) brandFlag = 1;
    if (!isAscii(d)) homoglyphDomain = 1;
  }

  const urgentCount = countMatches(textLower, WORD_LISTS.urgent);
  const threatCount = countMatches(textLower, WORD_LISTS.threat);

  return {
    hasLink: linkCount > 0 ? 1 : 0,
    urgentCount,
    secretCount: countMatches(textLower, WORD_LISTS.secret),
    moneyCount: countMatches(textLower, WORD_LISTS.money),
    threatCount,
    identityCount: countMatches(textLower, WORD_LISTS.identity),
    rewardCount: countMatches(textLower, WORD_LISTS.reward),
    verificationCount: countMatches(textLower, WORD_LISTS.verificationService),
    pressureCount: countMatches(textLower, WORD_LISTS.pressure),
    suspiciousDomain,
    longDomain,
    suspiciousZone,
    digitDomain,
    brandFlag,
    homoglyphDomain,
    urlCount: linkCount,
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
  score += Math.min(f.verificationCount * 0.05, 0.15);
  score += f.suspiciousDomain ? 0.10 : 0;
  score += f.suspiciousZone ? 0.10 : 0;
  score += f.longDomain ? 0.05 : 0;
  score += f.digitDomain ? 0.05 : 0;
  score += f.brandFlag ? 0.15 : 0;
  score += f.homoglyphDomain ? 0.15 : 0;
  return Math.min(score, 0.95);
}

function ruleBoost(f) {
  let boost = 0;
  if (f.hasLink && f.secretCount) boost += 0.15;
  if (f.urgentCount && f.moneyCount) boost += 0.12;
  if (f.secretCount && f.moneyCount) boost += 0.15;
  if (f.threatCount && f.hasLink) boost += 0.10;
  if (f.rewardCount && f.moneyCount) boost += 0.12;
  if (f.verificationCount && (f.moneyCount || f.hasLink)) boost += 0.15;
  if (f.pressureCount) boost += 0.10;
  if (f.suspiciousZone || f.suspiciousDomain) boost += 0.10;
  if (f.brandFlag) boost += 0.15;
  if (f.homoglyphDomain) boost += 0.15;
  if (f.hasMultipleWarnings) boost += 0.08;
  if (f.urlCount > 1) boost += 0.05;
  return Math.min(boost, 0.30);
}

// Reason keys, in display order. popup.js/background.js resolve each to
// the current UI language's text via I18N.strings(lang).reasonLabels.
const EXPLAIN_KEYS = [
  "hasLink", "urgentCount", "secretCount", "moneyCount", "threatCount",
  "identityCount", "rewardCount", "pressureCount", "verificationCount",
  "suspiciousDomain", "longDomain", "suspiciousZone", "digitDomain",
  "brandFlag", "homoglyphDomain", "hasMultipleWarnings", "urlCount"
];

function explain(f) {
  const reasonKeys = [];
  for (const key of EXPLAIN_KEYS) {
    const value = key === "urlCount" ? (f.urlCount > 1 ? 1 : 0) : f[key];
    if (value) reasonKeys.push(key);
  }
  return reasonKeys;
}

function riskStyle(prob) {
  // levelKey is resolved to translated text by the caller (popup.js /
  // background.js), same pattern as fraud_logic.risk_level() in the Python app.
  if (prob < 0.3) return { levelKey: "low", cssClass: "risk-low", emoji: "🟢" };
  if (prob < 0.6) return { levelKey: "mid", cssClass: "risk-mid", emoji: "🟡" };
  if (prob < 0.8) return { levelKey: "high", cssClass: "risk-high", emoji: "🟠" };
  return { levelKey: "critical", cssClass: "risk-critical", emoji: "🔴" };
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
