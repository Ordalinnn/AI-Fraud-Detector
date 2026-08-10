// Self-contained fraud/scam heuristic scorer for the browser extension.
//
// This mirrors the keyword/domain rule logic from the main AI Fraud Detector
// Streamlit app (app.py), but does NOT run the full trained ML ensemble
// (Logistic Regression + Random Forest + Gradient Boosting + TF-IDF/NB) —
// scikit-learn models can't run inside a browser extension. The base score
// is a blend of two things: a hand-tuned rule-weighted sum (heuristicBaseScore),
// and the real LogisticRegression member's coefficients ported as a plain
// dot-product + sigmoid (lrScore — see LR_* constants below, regenerated via
// scripts/extract_lr_coefficients.py). The LR component is deliberately
// given a MINORITY weight (see LR_BLEND_WEIGHT): trained in isolation
// (without the other 3 ensemble members averaging it out), it scores some
// perfectly ordinary short conversational messages surprisingly high — e.g.
// an isolated evaluation found it assigns ~0.98 fraud probability to a
// plain "how are you, see you tomorrow at 9" greeting, a dataset-distribution
// artifact rather than a real signal. Full rule_boost() logic is also
// ported unchanged. Treat this whole thing as a lighter, offline
// approximation of the full web app, not a 1:1 port of its accuracy.
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

function isUpperChar(ch) {
  return ch !== ch.toLowerCase() && ch === ch.toUpperCase();
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
  if (text === null || text === undefined) {
    text = "";
  } else if (typeof text !== "string") {
    text = String(text);
  }
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

  // Basic text statistics — mirrors fraud_logic.extract_features()'s
  // digit_count/exclamation_count/uppercase_count/text_length/word_count/
  // avg_word_length. uppercaseCount is measured against the ORIGINAL
  // (non-lowercased) text, same as the Python version.
  const words = textLower.split(/\s+/).filter((w) => w.length > 0);
  const digitCount = (textLower.match(/\d/g) || []).length;
  const exclamationCount = (textLower.match(/!/g) || []).length;
  const uppercaseCount = Array.from(text).filter(isUpperChar).length;
  const textLength = Array.from(text).length;
  const wordCount = words.length;
  const avgWordLength = wordCount
    ? Math.round((words.reduce((sum, w) => sum + Array.from(w).length, 0) / wordCount) * 100) / 100
    : 0;

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
    digitCount,
    exclamationCount,
    uppercaseCount,
    textLength,
    wordCount,
    avgWordLength,
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
  // Weights below are informed by the coefficients a LogisticRegression
  // trained on the real app.py dataset actually learned for these features
  // (digitCount, uppercaseCount, wordCount, avgWordLength all had
  // meaningfully positive coefficients — several bigger than threatCount's).
  // wordCount and textLength are highly collinear (both just measure
  // message verbosity), so only wordCount is scored to avoid double
  // counting the same signal. exclamationCount is intentionally excluded:
  // the trained model learned an ~0 coefficient for it (co-occurs with
  // urgentCount, which already carries the signal) — still tracked in the
  // feature object for parity/explainability, just not scored here.
  score += Math.min(f.digitCount * 0.01, 0.08);
  score += Math.min(f.uppercaseCount * 0.01, 0.08);
  score += Math.min(f.wordCount * 0.002, 0.08);
  score += Math.min(f.avgWordLength * 0.01, 0.08);
  return Math.min(score, 0.95);
}

// StandardScaler + LogisticRegression(C=1.0, max_iter=1000, random_state=42)
// coefficients, trained on app.py's full training dataset via the exact
// same extractFeatures()/extract_features() vector used below. Regenerate
// with `python scripts/extract_lr_coefficients.py` from the repo root
// whenever fraud_logic.extract_features() changes shape or app.py's
// training dataset changes, and paste the output back in here.
const LR_FEATURE_ORDER = ["hasLink", "urgentCount", "secretCount", "moneyCount", "threatCount", "identityCount", "rewardCount", "pressureCount", "verificationCount", "suspiciousDomain", "longDomain", "suspiciousZone", "digitDomain", "brandFlag", "homoglyphDomain", "digitCount", "exclamationCount", "uppercaseCount", "textLength", "wordCount", "avgWordLength", "urlCount", "hasMultipleWarnings"];
const LR_MEAN = [0.017345, 0.157617, 0.071644, 0.441176, 0.061086, 0.031674, 0.048265, 0.014329, 0.047511, 0.002262, 0.0, 0.001508, 0.0, 0.000754, 0.0, 0.095023, 0.0, 0.00905, 65.760935, 10.09276, 5.818077, 0.017345, 0.010558];
const LR_SCALE = [0.130555, 0.374587, 0.285647, 0.650392, 0.239488, 0.175131, 0.227967, 0.118842, 0.216246, 0.047511, 1.0, 0.038807, 1.0, 0.027451, 1.0, 0.497374, 1.0, 0.164522, 15.038643, 2.92356, 1.163335, 0.130555, 0.102209];
const LR_COEF = [-0.448267, 1.453043, 0.689823, 0.471641, 0.336216, 0.494019, 0.346125, 0.75348, 0.230036, 0.300533, 0.0, 0.156384, 0.0, 0.142334, 0.0, 0.429622, 0.0, 0.427079, 0.825451, 1.148422, 0.801198, -0.448267, 0.21409];
const LR_INTERCEPT = 0.340179;

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function lrScore(f) {
  let z = LR_INTERCEPT;
  for (let i = 0; i < LR_FEATURE_ORDER.length; i++) {
    const x = f[LR_FEATURE_ORDER[i]];
    z += ((x - LR_MEAN[i]) / LR_SCALE[i]) * LR_COEF[i];
  }
  return sigmoid(z);
}

// Minority weight for the ported LR component in the blended base score
// (see the file header comment for why: evaluated on its own, the LR model
// over-scores some plain conversational messages that never appear near
// its training templates). Meaningfully influences the result without
// being able to single-handedly flip a message the heuristic scores as safe.
const LR_BLEND_WEIGHT = 0.2;

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
// wordCount/textLength/avgWordLength are deliberately excluded (mirrors
// fraud_logic.explain()'s `irrelevant` set): they're near-always non-zero
// for any message, so surfacing them as "reasons" would be noise rather
// than a meaningful signal.
const EXPLAIN_KEYS = [
  "hasLink", "urgentCount", "secretCount", "moneyCount", "threatCount",
  "identityCount", "rewardCount", "pressureCount", "verificationCount",
  "suspiciousDomain", "longDomain", "suspiciousZone", "digitDomain",
  "brandFlag", "homoglyphDomain", "digitCount", "exclamationCount",
  "uppercaseCount", "hasMultipleWarnings", "urlCount"
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
  const base = (1 - LR_BLEND_WEIGHT) * heuristicBaseScore(f) + LR_BLEND_WEIGHT * lrScore(f);
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

// Also exposed via CommonJS exports (Node's `require`) so the test suite
// can exercise internal helpers directly instead of only indirectly through
// analyzeText(). Guarded because classic browser/service-worker scripts have
// no `module` global.
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    analyzeText, extractFeatures, lrScore, heuristicBaseScore, ruleBoost,
    brandImpersonation, domainFlags, getDomain, levenshtein, riskStyle,
    extractUrls, extractBareDomains
  };
}
