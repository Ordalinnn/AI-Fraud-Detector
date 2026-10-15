// Offline message checker for phones (iPhone first).
//
// Runs the website's own model in the browser — the same files as the
// browser extension (lib/, copied by scripts/build_web_checker.py), so the
// percentage is the one the website shows. The message never leaves the
// phone: an iPhone Shortcut opens this page with the text after "#text=",
// and a URL fragment is not sent to any server. sw.js keeps the page and
// the model on the phone, so it also opens with no internet and when the
// website is asleep.

(function () {
  "use strict";

  const PAGE = {
    kz: {
      pageTitle: "Хабарламаны тексеру",
      subtitle: "Телефонның өзінде тексеріледі — мәтін ешқайда жіберілмейді",
      placeholder: "Күдікті SMS-ті, хабарламаны немесе қоңырау мәтінін осында қойыңыз…",
      paste: "📋 Қою",
      clear: "Тазалау",
      check: "🔍 Тексеру",
      loading: "Модель жүктелуде…",
      modelFailed: "Модель жүктелмеді — тексеру тек ережелер бойынша (дәлдігі төменірек).",
      pasteFailed: "Қою мүмкін болмады: өрісті басып тұрып, «Қою» таңдаңыз.",
      emptyShortcut: "Команда мәтінсіз іске қосылды: хабарламаны көшіріңіз немесе «Бөлісу» арқылы жіберіңіз.",
      offlineReady: "✅ Интернетсіз де жұмыс істейді.",
      offlineSaving: "Интернетсіз жұмыс істеу үшін сақталуда…",
      offlineNone: "Бұл браузерде интернетсіз режим жоқ — тексеру бәрібір телефонда жүреді.",
      installHint: "📲 Белгіше ретінде қосыңыз: Safari-де «Бөлісу» → «Басты экранға қосу». Сонда интернетсіз де бір рет басып ашылады.",
      siteLink: "Сайтта толық талдау ↗",
      share: "📤 Жақындарды ескерту",
      shareText: "⚠️ Абай болыңыз: бұл хабарлама алаяқтыққа ұқсайды ({pct}%). Кодты, құпиясөзді, карта деректерін ешкімге айтпаңыз.\n\n«{text}»",
      mistakeThanks: "Рахмет! Белгіленді. Тек осы телефонда сақталады.",
      mistakesNote: "Белгілер тек осы телефонда сақталады. CSV-ді жүктеп, жоба командасына жіберіңіз; алдымен аттарды, телефон мен карта нөмірлерін [NAME], [PHONE], [CARD] деп ауыстырыңыз.",
      modelLine: "Сайттағы модель (5 модель + стекинг) · model {id} · ресми қауіпсіздік құралы емес",
    },
    ru: {
      pageTitle: "Проверка сообщения",
      subtitle: "Проверка идёт на телефоне — текст никуда не отправляется",
      placeholder: "Вставьте сюда подозрительное SMS, сообщение или текст звонка…",
      paste: "📋 Вставить",
      clear: "Очистить",
      check: "🔍 Проверить",
      loading: "Загружаем модель…",
      modelFailed: "Модель не загрузилась — проверка только по правилам (менее точная).",
      pasteFailed: "Не удалось вставить: нажмите и подержите поле, затем «Вставить».",
      emptyShortcut: "Команда запустилась без текста: скопируйте сообщение или поделитесь им.",
      offlineReady: "✅ Работает и без интернета.",
      offlineSaving: "Сохраняем для работы без интернета…",
      offlineNone: "В этом браузере нет режима без интернета — но проверка всё равно идёт на телефоне.",
      installHint: "📲 Добавьте значок: в Safari «Поделиться» → «На экран „Домой“». Тогда проверка открывается в одно касание, даже без интернета.",
      siteLink: "Подробный анализ на сайте ↗",
      share: "📤 Предупредить близких",
      shareText: "⚠️ Осторожно: это сообщение похоже на мошенничество ({pct}%). Никому не сообщайте коды, пароли и данные карты.\n\n«{text}»",
      mistakeThanks: "Спасибо! Отмечено. Хранится только на этом телефоне.",
      mistakesNote: "Отметки хранятся только на этом телефоне. Скачайте CSV и отправьте команде проекта; перед отправкой замените имена, телефоны и номера карт на [NAME], [PHONE], [CARD].",
      modelLine: "Модель сайта (5 моделей + стекинг) · model {id} · не сертифицированный инструмент безопасности",
    },
    en: {
      pageTitle: "Check a message",
      subtitle: "Checked on your phone — the text is not sent anywhere",
      placeholder: "Paste a suspicious SMS, message or call transcript here…",
      paste: "📋 Paste",
      clear: "Clear",
      check: "🔍 Check",
      loading: "Loading the model…",
      modelFailed: "The model didn't load — rules-only check (less accurate).",
      pasteFailed: "Couldn't paste: press and hold the box, then tap “Paste”.",
      emptyShortcut: "The shortcut ran without text: copy the message or share it.",
      offlineReady: "✅ Works offline too.",
      offlineSaving: "Saving for offline use…",
      offlineNone: "This browser has no offline mode — the check still runs on your phone.",
      installHint: "📲 Add it as an icon: in Safari tap Share → “Add to Home Screen”. Then it opens with one tap, even offline.",
      siteLink: "Full analysis on the website ↗",
      share: "📤 Warn family",
      shareText: "⚠️ Careful: this message looks like a scam ({pct}%). Never share codes, passwords or card details.\n\n“{text}”",
      mistakeThanks: "Thanks! Marked. Saved only on this phone.",
      mistakesNote: "Marks are stored only on this phone. Download the CSV and send it to the project team; first replace names, phone and card numbers with [NAME], [PHONE], [CARD].",
      modelLine: "The website's model (5 models + stacking) · model {id} · not a certified security tool",
    },
  };
  const LANGS = ["kz", "ru", "en"];
  const LANG_KEY = "afd-lang";
  const MISTAKES_KEY = "afd-mistakes";
  const MAX_CHARS = 4000;

  const $ = (id) => document.getElementById(id);
  const store = {
    get(key) { try { return window.localStorage.getItem(key); } catch (e) { return null; } },
    set(key, value) { try { window.localStorage.setItem(key, value); } catch (e) { /* private mode */ } },
  };

  // ---------- language ----------
  function pickLang() {
    const fromUrl = new URLSearchParams(location.search).get("lang");
    if (LANGS.includes(fromUrl)) return fromUrl;
    const saved = store.get(LANG_KEY);
    if (LANGS.includes(saved)) return saved;
    const nav = String(navigator.language || "").toLowerCase();
    if (nav.startsWith("kk")) return "kz";
    if (nav.startsWith("en")) return "en";
    return "ru";
  }
  let lang = pickLang();
  const P = () => PAGE[lang];
  const S = () => self.I18N.strings(lang);

  // ---------- text shared by the Shortcut ----------
  // "#text=<URL-encoded>" (preferred: a fragment never reaches a server) or
  // "?text=..." as the website accepts. Returns null when nothing was
  // shared, "" when the Shortcut ran without any text.
  function sharedText(loc) {
    const m = /^#text=(.*)$/s.exec(loc.hash || "");
    let raw = null;
    if (m) {
      raw = m[1].replace(/\+/g, " ");
      try { raw = decodeURIComponent(raw); } catch (e) { /* keep as is */ }
    } else {
      const q = new URLSearchParams(loc.search);
      if (q.has("text")) raw = q.get("text");
    }
    if (raw === null) return null;
    return raw.trim().slice(0, MAX_CHARS);
  }

  // The website's address, for "Full analysis on the website": this page is
  // <site>/app/static/check/index.html (on Streamlit Cloud with a "/~/+/"
  // prefix in front of "app/").
  function siteRoot(loc) {
    return loc.origin + loc.pathname.replace(/(~\/\+\/)?app\/static\/check\/[^/]*$/, "");
  }

  // ---------- model ----------
  let scorer = null;
  let modelId = "";
  let modelState = "loading"; // loading | ready | failed
  const modelReady = fetch("lib/ensemble_model.json")
    .then((r) => { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
    .then((m) => {
      scorer = self.EnsembleScorer.parse(m);
      modelId = scorer.fingerprint.slice(0, 8);
      modelState = "ready";
    })
    .catch(() => { modelState = "failed"; })
    .then(() => applyText());

  // ---------- rendering ----------
  let last = null; // { text, result }

  function setText(id, value) { $(id).textContent = value; }

  function applyText() {
    const p = P();
    document.documentElement.lang = lang === "kz" ? "kk" : lang;
    document.title = p.pageTitle + " · AI Fraud Detector";
    setText("pageTitle", p.pageTitle);
    setText("subtitle", p.subtitle);
    $("input").placeholder = p.placeholder;
    setText("inputLabel", p.pageTitle);
    setText("pasteBtn", p.paste);
    setText("clearBtn", p.clear);
    setText("checkBtn", modelState === "loading" && pendingCheck ? p.loading : p.check);
    setText("siteLink", p.siteLink);
    setText("shareBtn", p.share);
    setText("mistakesNote", p.mistakesNote);
    setText("mistakesDownload", S().mistakesDownload);
    setText("mistakesClear", S().mistakesClear);
    setText("installHint", p.installHint);
    setText("footer", modelState === "ready" ? p.modelLine.replace("{id}", modelId) : "");
    $("lang").value = lang;
    renderStatus();
    renderMistakes();
    if (last) render(last.result);
  }

  function showNotice(msg) {
    const el = $("notice");
    el.textContent = msg;
    el.classList.toggle("hidden", !msg);
  }

  function clearChildren(el) { while (el.firstChild) el.removeChild(el.firstChild); }

  function render(result) {
    const s = S();
    const p = P();
    $("result").classList.remove("hidden");
    const banner = $("banner");
    banner.className = "banner " + result.cssClass;
    banner.textContent = `${result.emoji} ${s.riskLabels[result.levelKey]} — ${result.pct}%`;
    const fill = $("meterFill");
    fill.className = "meter-fill " + result.cssClass;
    fill.style.width = Math.max(2, Math.min(100, result.prob * 100)) + "%";
    setText("advice", result.isFraud ? s.adviceBad : s.adviceGood);

    const words = result.modelWords || [];
    $("words").classList.toggle("hidden", !(result.isFraud && words.length));
    setText("words", `${s.modelWordsLabel} ${words.join(", ")}`);

    const reasons = $("reasons");
    clearChildren(reasons);
    for (const key of (result.reasons.length ? result.reasons : [null])) {
      const li = document.createElement("li");
      li.textContent = key ? (s.reasonLabels[key] || key) : s.noReasons;
      reasons.appendChild(li);
    }

    const domains = $("domains");
    clearChildren(domains);
    for (const d of result.domainInfo) {
      const li = document.createElement("li");
      const flags = d.flags.length
        ? d.flags.map((f) => (f.severity === "critical" ? "🔴 " : "⚠️ ")
          + (s.domainFlagLabels[f.labelKey] || f.labelKey).replace("{brand}", f.brand || "")).join(" · ")
        : "✅ " + s.noDomainIssues;
      li.textContent = `🌐 ${d.domain} — ${flags}`;
      domains.appendChild(li);
    }

    $("shareBtn").classList.toggle("hidden", !(result.isFraud && typeof navigator.share === "function"));
    const mistake = $("mistakeBtn");
    mistake.textContent = result.isFraud ? s.mistakeFlaggedBtn : s.mistakeMissedBtn;
    mistake.disabled = !!(last && last.marked);
    $("mistakeThanks").classList.toggle("hidden", !(last && last.marked));
    setText("mistakeThanks", p.mistakeThanks);
    $("siteLink").href = siteRoot(location) + "#text=" + encodeURIComponent(last ? last.text : "");
  }

  // ---------- checking ----------
  let pendingCheck = false;

  async function check() {
    const text = $("input").value.trim().slice(0, MAX_CHARS);
    if (!text) { showNotice(S().emptyInputError); return; }
    showNotice("");
    if (modelState === "loading") {
      pendingCheck = true;
      $("checkBtn").disabled = true;
      applyText();
      await modelReady;
      pendingCheck = false;
      $("checkBtn").disabled = false;
    }
    if (modelState === "failed") showNotice(P().modelFailed);
    const result = self.FraudDetector.analyzeText(text, scorer);
    last = { text, result, marked: false };
    render(result);
    applyText();
    const box = $("result");
    if (box.scrollIntoView) box.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ---------- "the model is wrong" (kept on this phone) ----------
  function mistakes() {
    try { return JSON.parse(store.get(MISTAKES_KEY) || "[]"); } catch (e) { return []; }
  }
  function renderMistakes() {
    const list = mistakes();
    $("mistakes").classList.toggle("hidden", list.length === 0);
    setText("mistakesTitle", S().mistakesTitle.replace("{n}", list.length));
  }

  // ---------- offline status / install hint ----------
  let offline = "none"; // none | saving | ready
  function renderStatus() {
    const p = P();
    setText("status", offline === "ready" ? p.offlineReady : offline === "saving" ? p.offlineSaving : p.offlineNone);
    const ios = /iphone|ipad|ipod/i.test(navigator.userAgent)
      || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
    const standalone = navigator.standalone === true
      || (window.matchMedia && window.matchMedia("(display-mode: standalone)").matches);
    $("installHint").classList.toggle("hidden", !(ios && !standalone));
  }

  function registerServiceWorker() {
    if (!("serviceWorker" in navigator) || !window.isSecureContext) return;
    offline = "saving";
    navigator.serviceWorker.register("sw.js")
      .then(() => navigator.serviceWorker.ready)
      .then(() => { offline = "ready"; renderStatus(); })
      .catch(() => { offline = "none"; renderStatus(); });
  }

  // ---------- events ----------
  $("checkBtn").addEventListener("click", check);
  $("input").addEventListener("input", () => showNotice(""));
  $("clearBtn").addEventListener("click", () => {
    $("input").value = "";
    last = null;
    $("result").classList.add("hidden");
    showNotice("");
    $("input").focus();
  });
  $("pasteBtn").addEventListener("click", async () => {
    try {
      const text = await navigator.clipboard.readText();
      if (text && text.trim()) { $("input").value = text.trim().slice(0, MAX_CHARS); showNotice(""); check(); }
      else showNotice(P().pasteFailed);
    } catch (e) {
      showNotice(P().pasteFailed);
    }
  });
  $("lang").addEventListener("change", () => {
    lang = $("lang").value;
    store.set(LANG_KEY, lang);
    applyText();
  });
  $("shareBtn").addEventListener("click", () => {
    if (!last) return;
    const text = P().shareText.replace("{pct}", last.result.pct).replace("{text}", last.text.slice(0, 300));
    navigator.share({ text }).catch(() => {});
  });
  $("mistakeBtn").addEventListener("click", () => {
    if (!last || last.marked) return;
    const entry = self.Mistakes.makeEntry({
      text: last.text, predictedFraud: last.result.isFraud, prob: last.result.prob, source: "phone page",
    });
    store.set(MISTAKES_KEY, JSON.stringify(self.Mistakes.addTo(mistakes(), entry)));
    last.marked = true;
    render(last.result);
    renderMistakes();
  });
  $("mistakesDownload").addEventListener("click", () => {
    const csv = "﻿" + self.Mistakes.toCsv(mistakes());
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = "ai-fraud-detector-mistakes.csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  });
  $("mistakesClear").addEventListener("click", () => {
    store.set(MISTAKES_KEY, "[]");
    renderMistakes();
  });

  // ---------- start ----------
  applyText();
  registerServiceWorker();
  const shared = sharedText(location);
  if (shared !== null) {
    // keep the message out of the address bar / history once it is read
    try {
      const clean = new URL(location.href);
      clean.hash = "";
      clean.searchParams.delete("text");
      history.replaceState(null, "", clean.pathname + clean.search);
    } catch (e) { /* ignore */ }
    if (shared) { $("input").value = shared; check(); } else showNotice(P().emptyShortcut);
  }

  window.AFDCheck = { PAGE, sharedText, siteRoot, check, modelReady };
})();
