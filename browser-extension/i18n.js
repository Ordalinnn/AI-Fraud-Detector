// Shared translations for the extension's popup and background notifications.
// Loaded as a classic script by both popup.html and background.js
// (importScripts), so it must not use ES module syntax.

const DEFAULT_LANG = "en";

const TRANSLATIONS = {
  en: {
    statusActive: "✅ Autoscan is working on this tab: {checked} messages checked, {flagged} suspicious.",
    statusReload: "⚠️ Reload this tab (F5): the extension was installed or updated after the page was opened.",
    statusOtherPage: "ℹ️ Autoscan works in WhatsApp Web, Telegram Web and Gmail. Open one of them in this tab (already open? press F5).",
    statusOff: "Autoscan is off.",
    autoscanLabel: "Automatically check new messages in WhatsApp Web, Telegram Web and Gmail",
    badgeLabel: "Looks like fraud",
    badgeWords: "Suspicious words:",
    badgeAdvice: "Don't share codes, passwords or card details and don't open links. Contact the organization only via its official number.",
    modelWordsLabel: "Words that raised the risk:",
    appTitle: "AI Fraud Detector",
    appSubtitle: "Runs locally — nothing leaves your device",
    placeholder: "Paste a suspicious SMS, message, or call transcript…",
    analyzeBtn: "🚀 Analyze",
    footer: "On-device ML model (character n-grams) + rules — not a certified security tool.",
    adviceBad: "⚠️ Do not share codes, passwords, CVV, or card numbers. Do not click the link. Contact the organization only through its official number.",
    adviceGood: "✅ This message looks safe. If unsure, verify through an official source.",
    noReasons: "No strong fraud indicators were found.",
    noDomainIssues: "No issues found",
    emptyInputError: "Please enter text first.",
    contextMenuTitle: 'Check "%s" for fraud/scam',
    langLabel: "Language",
    riskLabels: { low: "Low risk", mid: "Suspicious", high: "High risk", critical: "Critical risk" },
    reasonLabels: {
      hasLink: "A link was detected",
      urgentCount: "Urgent action words were found",
      secretCount: "Possible request for code / password / CVV",
      moneyCount: "Bank, money, or card-related words were found",
      loanCount: "A request to borrow or urgently send money was detected",
      voteLinkCount: "A link asking you to vote for or support someone was detected",
      threatCount: "Pressure or threat indicators were found",
      identityCount: "Possible request for personal identity data",
      rewardCount: "Prize, bonus, or gift promise was found",
      pressureCount: "Pressure or secrecy phrase was found",
      verificationCount: "Mentions a paid \"verification service\" or \"official registry\"",
      suspiciousDomain: "Suspicious words were found in the domain",
      longDomain: "The domain is suspiciously long",
      suspiciousZone: "Suspicious domain zone detected",
      digitDomain: "The domain contains numbers",
      brandFlag: "Domain mimics a known bank/brand but isn't the real one",
      homoglyphDomain: "Domain contains Cyrillic characters disguised as Latin letters",
      digitCount: "The text contains many numbers or codes",
      exclamationCount: "Many exclamation marks were used",
      uppercaseCount: "Many uppercase letters were used",
      hasMultipleWarnings: "Both urgency and threat were detected simultaneously",
      urlCount: "Multiple links were detected"
    },
    domainFlagLabels: {
      impersonates: "Impersonates {brand}",
      suspiciousKeyword: "Suspicious keyword",
      longDomain: "Long domain",
      suspiciousTld: "Suspicious TLD",
      containsDigits: "Contains digits",
      nonLatinDomain: "Non-Latin lookalike characters"
    }
  },
  ru: {
    statusActive: "✅ Автопроверка работает на этой вкладке: проверено сообщений — {checked}, подозрительных — {flagged}.",
    statusReload: "⚠️ Обновите эту вкладку (F5): расширение установили или обновили, когда страница уже была открыта.",
    statusOtherPage: "ℹ️ Автопроверка работает в WhatsApp Web, Telegram Web и Gmail. Откройте один из них в этой вкладке (если уже открыт — нажмите F5).",
    statusOff: "Автопроверка выключена.",
    autoscanLabel: "Автоматически проверять новые сообщения в WhatsApp Web, Telegram Web и Gmail",
    badgeLabel: "Похоже на мошенничество",
    badgeWords: "Подозрительные слова:",
    badgeAdvice: "Не сообщайте коды, пароли и данные карты, не переходите по ссылкам. Связывайтесь с организацией только по официальному номеру.",
    modelWordsLabel: "Слова, повысившие риск:",
    appTitle: "AI Fraud Detector",
    appSubtitle: "Работает локально — ничего не покидает ваше устройство",
    placeholder: "Вставьте подозрительное SMS, сообщение или текст звонка…",
    analyzeBtn: "🚀 Анализ",
    footer: "ML-модель на устройстве (символьные n-граммы) + правила — не сертифицированный инструмент безопасности.",
    adviceBad: "⚠️ Не сообщайте код, пароль, CVV или номер карты. Не переходите по ссылке. Свяжитесь с организацией только по официальному номеру.",
    adviceGood: "✅ Сообщение выглядит безопасным. Если сомневаетесь, проверьте через официальный источник.",
    noReasons: "Сильные признаки мошенничества не найдены.",
    noDomainIssues: "Проблем не найдено",
    emptyInputError: "Сначала введите текст.",
    contextMenuTitle: 'Проверить "%s" на мошенничество',
    langLabel: "Язык",
    riskLabels: { low: "Низкий риск", mid: "Подозрительно", high: "Высокий риск", critical: "Очень высокий риск" },
    reasonLabels: {
      hasLink: "Обнаружена ссылка",
      urgentCount: "Есть срочный призыв к действию",
      secretCount: "Возможен запрос кода / пароля / CVV",
      moneyCount: "Есть слова о банке, деньгах или карте",
      loanCount: "Обнаружена просьба одолжить или срочно перевести деньги",
      voteLinkCount: "Обнаружена ссылка с просьбой проголосовать/поддержать от имени знакомого",
      threatCount: "Есть признаки давления или угрозы",
      identityCount: "Возможен запрос личных документов",
      rewardCount: "Есть обещание выигрыша, бонуса или подарка",
      pressureCount: "Есть давление или просьба держать в секрете",
      verificationCount: "Упоминается платная \"проверка\" или \"официальный реестр\"",
      suspiciousDomain: "В домене есть подозрительные слова",
      longDomain: "Домен подозрительно длинный",
      suspiciousZone: "Обнаружена подозрительная доменная зона",
      digitDomain: "В домене есть цифры",
      brandFlag: "Домен маскируется под известный банк/бренд",
      homoglyphDomain: "В домене есть кириллические символы, похожие на латинские буквы",
      digitCount: "В тексте много чисел или кодов",
      exclamationCount: "Используется много восклицательных знаков",
      uppercaseCount: "Используется много заглавных букв",
      hasMultipleWarnings: "Одновременно присутствуют срочность и угроза",
      urlCount: "Обнаружено несколько ссылок"
    },
    domainFlagLabels: {
      impersonates: "Маскируется под {brand}",
      suspiciousKeyword: "Подозрительное слово",
      longDomain: "Длинный домен",
      suspiciousTld: "Подозрительная зона",
      containsDigits: "Содержит цифры",
      nonLatinDomain: "Похожие на латиницу кириллические символы"
    }
  },
  kz: {
    statusActive: "✅ Бұл бетте автотексеру жұмыс істеп тұр: тексерілген хабарлама — {checked}, күдіктісі — {flagged}.",
    statusReload: "⚠️ Бұл бетті жаңартыңыз (F5): кеңейтім бет ашылғаннан кейін орнатылды немесе жаңартылды.",
    statusOtherPage: "ℹ️ Автотексеру WhatsApp Web, Telegram Web және Gmail-да жұмыс істейді. Солардың бірін осы бетте ашыңыз (ашық болса — F5 басыңыз).",
    statusOff: "Автотексеру өшірулі.",
    autoscanLabel: "WhatsApp Web, Telegram Web және Gmail-дағы жаңа хабарламаларды автоматты тексеру",
    badgeLabel: "Алаяқтыққа ұқсайды",
    badgeWords: "Күдікті сөздер:",
    badgeAdvice: "Кодтарды, құпиясөздерді және карта деректерін айтпаңыз, сілтемелерге өтпеңіз. Ұйыммен тек ресми нөмір арқылы байланысыңыз.",
    modelWordsLabel: "Тәуекелді арттырған сөздер:",
    appTitle: "AI Fraud Detector",
    appSubtitle: "Жергілікті жұмыс істейді — деректер құрылғыдан шықпайды",
    placeholder: "Күмәнді SMS, хабарлама немесе қоңырау мәтінін қойыңыз…",
    analyzeBtn: "🚀 Талдау",
    footer: "Құрылғыдағы ML-модель (символдық n-грамдар) + ережелер — ресми қауіпсіздік құралы емес.",
    adviceBad: "⚠️ Код, пароль, CVV немесе карта нөмірін бермеңіз. Сілтемеге баспаңыз. Ұйыммен тек ресми нөмір арқылы хабарласыңыз.",
    adviceGood: "✅ Хабарлама қауіпсіз көрінеді. Күмәндансаңыз, ресми дереккөз арқылы тексеріңіз.",
    noReasons: "Күшті алаяқтық белгілері табылмады.",
    noDomainIssues: "Мәселе табылмады",
    emptyInputError: "Алдымен мәтін енгізіңіз.",
    contextMenuTitle: '"%s" алаяқтыққа тексеру',
    langLabel: "Тіл",
    riskLabels: { low: "Қауіп төмен", mid: "Күмәнді", high: "Жоғары қауіп", critical: "Өте жоғары қауіп" },
    reasonLabels: {
      hasLink: "Сілтеме анықталды",
      urgentCount: "Шұғыл әрекетке шақыру бар",
      secretCount: "Код / пароль / CVV сұрауы мүмкін",
      moneyCount: "Банк, ақша немесе картаға қатысты сөздер бар",
      loanCount: "Ақша қарызға сұрау белгісі анықталды",
      voteLinkCount: "Таныс адамнан дауыс беру/қолдау сұрайтын сілтеме анықталды",
      threatCount: "Қорқыту немесе қысым белгісі бар",
      identityCount: "Жеке құжат сұралуы мүмкін",
      rewardCount: "Ұтыс, бонус немесе сыйлық уәдесі бар",
      pressureCount: "Қысым немесе құпия ұстау белгісі бар",
      verificationCount: "Ақылы \"тексеру қызметі\" немесе \"ресми реестр\" туралы сөз бар",
      suspiciousDomain: "Доменде күмәнді сөздер бар",
      longDomain: "Домен ұзындығы күмәнді",
      suspiciousZone: "Күмәнді домен зонасы анықталды",
      digitDomain: "Доменде цифрлар бар",
      brandFlag: "Домен белгілі банк/брендке ұқсатылған",
      homoglyphDomain: "Доменде латын әрпіне ұқсас кириллица таңбалары бар",
      digitCount: "Мәтінде көп сан немесе код кездеседі",
      exclamationCount: "Көп леп белгісі қолданылған",
      uppercaseCount: "Үлкен әріптер көп қолданылған",
      hasMultipleWarnings: "Бір уақытта шұғылдық және қорқыту бар",
      urlCount: "Бірнеше сілтеме анықталды"
    },
    domainFlagLabels: {
      impersonates: "{brand} ұқсатады",
      suspiciousKeyword: "Күмәнді сөз",
      longDomain: "Ұзын домен",
      suspiciousTld: "Күмәнді зона",
      containsDigits: "Цифрлар бар",
      nonLatinDomain: "Латынға ұқсас кириллица таңбалары"
    }
  }
};

// Until the user picks a language, follow the browser's: a Kazakh or Russian
// Chrome gets Kazakh or Russian labels instead of English.
function i18nBrowserLang() {
  const nav = (typeof navigator !== "undefined" && navigator.language || "").toLowerCase();
  if (nav.startsWith("kk")) return "kz";
  if (nav.startsWith("ru")) return "ru";
  return DEFAULT_LANG;
}

function i18nGetLang(callback) {
  chrome.storage.local.get(["lang"], (data) => callback(data.lang || i18nBrowserLang()));
}

function i18nSetLang(lang, callback) {
  chrome.storage.local.set({ lang }, callback || function () {});
}

function i18nStrings(lang) {
  return TRANSLATIONS[lang] || TRANSLATIONS[DEFAULT_LANG];
}

// Flattens a (possibly nested, e.g. reasonLabels/riskLabels/domainFlagLabels)
// translation object into dotted key paths, so nested keys are compared for
// parity too, not just the top level.
function collectKeyPaths(obj, prefix) {
  let paths = [];
  for (const key of Object.keys(obj)) {
    const path = prefix ? `${prefix}.${key}` : key;
    const value = obj[key];
    if (value && typeof value === "object" && !Array.isArray(value)) {
      paths = paths.concat(collectKeyPaths(value, path));
    } else {
      paths.push(path);
    }
  }
  return paths;
}

// Mirrors translations.py's validate_translations(): raises listing exactly
// which language is missing which key(s) if the language dicts don't all
// have identical key sets (including nested reasonLabels/riskLabels/
// domainFlagLabels keys). Called at load time below so a missing
// translation is caught immediately, not silently as `undefined` text
// showing up in the popup the first time a user picks that language.
function validateTranslations(translations) {
  const langs = Object.keys(translations);
  const keySets = {};
  for (const lang of langs) {
    keySets[lang] = new Set(collectKeyPaths(translations[lang], ""));
  }
  const allKeys = new Set();
  for (const lang of langs) {
    for (const k of keySets[lang]) allKeys.add(k);
  }
  const problems = [];
  for (const lang of langs) {
    const missing = [...allKeys].filter((k) => !keySets[lang].has(k)).sort();
    if (missing.length) problems.push(`${lang} is missing: ${JSON.stringify(missing)}`);
  }
  if (problems.length) {
    throw new Error("Translation key mismatch between languages:\n" + problems.join("\n"));
  }
}

validateTranslations(TRANSLATIONS);

if (typeof self !== "undefined") {
  self.I18N = { TRANSLATIONS, DEFAULT_LANG, browserLang: i18nBrowserLang, getLang: i18nGetLang, setLang: i18nSetLang, strings: i18nStrings };
}

// Also exposed via CommonJS exports so the test suite can exercise
// validateTranslations()/collectKeyPaths() directly (including against
// deliberately-broken fixtures), not just the load-time throw above.
// Guarded because classic browser/service-worker scripts have no `module`.
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    TRANSLATIONS, DEFAULT_LANG, browserLang: i18nBrowserLang, getLang: i18nGetLang, setLang: i18nSetLang,
    strings: i18nStrings, validateTranslations, collectKeyPaths
  };
}
