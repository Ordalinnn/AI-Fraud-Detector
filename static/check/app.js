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
      pasteFailed: "Қою мүмкін болмады: шыққан «Вставить» (Paste) батырмасын басыңыз немесе өрісті басып тұрып, «Вставить» таңдаңыз.",
      emptyShortcut: "Команда мәтінсіз іске қосылды: хабарламаны көшіріңіз немесе «Поделиться» (Share) арқылы жіберіңіз.",
      offlineReady: "✅ Интернетсіз де жұмыс істейді.",
      offlineReadyTab: "✅ Телефонға сақталды: бұл бет соңғы күндері ашылса, интернетсіз де ашылады. Әрқашан интернетсіз — басты экрандағы белгішеден.",
      offlineSaving: "Интернетсіз жұмыс істеу үшін сақталуда…",
      offlineNone: "Бұл браузерде интернетсіз режим жоқ — тексеру бәрібір телефонда жүреді.",
      offlineFailed: "Интернетсіз жұмыс үшін сақталмады — тексеру бәрібір телефонда жүреді.",
      installHint: "📲 Белгіше ретінде қосыңыз: Safari-де «Поделиться» (Share; көрінбесе — алдымен «•••») → «На экран „Домой“» (Add to Home Screen). Белгішені бір рет интернет бар кезде ашыңыз — содан кейін ол интернетсіз де бір басумен ашылады.",
      siteLink: "Сайтта толық талдау (мәтін серверге жіберіледі) ↗",
      copiedLink: "⚠️ Алмасу буферінде хабарлама емес, осы беттің сілтемесі тұр (команданы баптау кезінде көшірілген). Тексергіңіз келетін хабарламаны көшіріп, команданы қайта іске қосыңыз.",
      share: "📤 Жақындарды ескерту",
      shareText: "⚠️ Абай болыңыз: бұл хабарлама алаяқтыққа ұқсайды ({pct}%). Кодты, құпиясөзді, карта деректерін ешкімге айтпаңыз.\n\n«{text}»",
      mistakeThanks: "Рахмет! Белгіленді. Тек осы телефонда сақталады.",
      mistakesNote: "Белгілер тек осы телефонда сақталады, ал Safari бетті 7 күн ашпаса оларды өшіріп тастауы мүмкін — CSV файлын бірден жүктеп, жоба командасына жіберіңіз. Алдымен аттарды, телефон мен карта нөмірлерін [АТЫ], [ТЕЛЕФОН], [КАРТА] деп ауыстырыңыз.",
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
      pasteFailed: "Не удалось вставить: нажмите «Вставить» во всплывающей подсказке или нажмите и подержите поле и выберите «Вставить».",
      emptyShortcut: "Команда запустилась без текста: скопируйте сообщение или поделитесь им.",
      offlineReady: "✅ Работает и без интернета.",
      offlineReadyTab: "✅ Сохранено на телефоне: страница откроется и без интернета, если вы открывали её в последние дни. Всегда без интернета — со значка на экране «Домой».",
      offlineSaving: "Сохраняем для работы без интернета…",
      offlineNone: "В этом браузере нет режима без интернета — но проверка всё равно идёт на телефоне.",
      offlineFailed: "Не удалось сохранить для работы без интернета — проверка всё равно идёт на телефоне.",
      installHint: "📲 Добавьте значок: в Safari «Поделиться» (если её не видно — сначала «•••») → «На экран „Домой“». Откройте значок один раз при интернете — дальше проверка открывается в одно касание даже без интернета.",
      siteLink: "Подробный анализ на сайте (текст уйдёт на сервер) ↗",
      copiedLink: "⚠️ В буфере обмена лежало не сообщение, а ссылка на эту страницу (её копировали, когда настраивали команду). Скопируйте само подозрительное сообщение и запустите команду ещё раз.",
      share: "📤 Предупредить близких",
      shareText: "⚠️ Осторожно: это сообщение похоже на мошенничество ({pct}%). Никому не сообщайте коды, пароли и данные карты.\n\n«{text}»",
      mistakeThanks: "Спасибо! Отмечено. Хранится только на этом телефоне.",
      mistakesNote: "Отметки хранятся только на этом телефоне, а Safari может стереть их, если страницу не открывать 7 дней, — скачайте CSV сразу и отправьте команде проекта. Перед отправкой замените имена, телефоны и номера карт на [ИМЯ], [ТЕЛЕФОН], [КАРТА].",
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
      pasteFailed: "Couldn't paste: tap “Paste” in the bubble, or press and hold the box and choose “Paste”.",
      emptyShortcut: "The shortcut ran without text: copy the message or share it.",
      offlineReady: "✅ Works offline too.",
      offlineReadyTab: "✅ Saved on the phone: the page opens offline too if you used it in the last few days. Always offline — from the Home Screen icon.",
      offlineSaving: "Saving for offline use…",
      offlineNone: "This browser has no offline mode — the check still runs on your phone.",
      offlineFailed: "Couldn't save for offline use — the check still runs on your phone.",
      installHint: "📲 Add it as an icon: in Safari tap Share (if you can't see it, tap “•••” first) → “Add to Home Screen”. Open the icon once while online — after that it opens with one tap, even offline.",
      siteLink: "Full analysis on the website (sends the text to the server) ↗",
      copiedLink: "⚠️ The clipboard held a link to this page (copied while setting up the shortcut), not a message. Copy the suspicious message itself and run the shortcut again.",
      share: "📤 Warn family",
      shareText: "⚠️ Careful: this message looks like a scam ({pct}%). Never share codes, passwords or card details.\n\n“{text}”",
      mistakeThanks: "Thanks! Marked. Saved only on this phone.",
      mistakesNote: "Marks are stored only on this phone, and Safari may erase them if the page isn't opened for 7 days — download the CSV right away and send it to the project team. First replace names, phone and card numbers with [NAME], [PHONE], [CARD].",
      modelLine: "The website's model (5 models + stacking) · model {id} · not a certified security tool",
    },
  };
  const LANGS = ["kz", "ru", "en"];
  const LANG_KEY = "afd-lang";
  const MISTAKES_KEY = "afd-mistakes";
  // Same cap as the website (app.py MAX_INPUT_CHARS), counted in code
  // points like Python, so an emoji is never cut in half.
  const MAX_CHARS = 8000;
  const SITE_LINK_CHARS = 1500; // as the extension's "open on the website" link
  const cut = (s, n) => Array.from(String(s)).slice(0, n).join("");
  const LAST_KEY = "afd-last-text"; // this tab only: survives an iOS tab reload

  const $ = (id) => document.getElementById(id);
  const store = {
    get(key) { try { return window.localStorage.getItem(key); } catch (e) { return null; } },
    set(key, value) { try { window.localStorage.setItem(key, value); } catch (e) { /* private mode */ } },
    // sessionStorage: read with one argument, write with two (null removes)
    session(key, value) {
      try {
        if (arguments.length === 1) return window.sessionStorage.getItem(key);
        if (value === null) window.sessionStorage.removeItem(key);
        else window.sessionStorage.setItem(key, value);
      } catch (e) { /* storage blocked */ }
      return null;
    },
  };

  // ---------- language ----------
  function pickLang() {
    const fromUrl = new URLSearchParams(location.search).get("lang");
    if (LANGS.includes(fromUrl)) { store.set(LANG_KEY, fromUrl); return fromUrl; }
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
    return cut(raw.trim(), MAX_CHARS); // the website strips shared text too
  }

  // The Shortcut's "Get Clipboard" fallback sent the setup link itself
  // (copied from the instructions), not a message: same rule as the site.
  const isOwnLink = (text) => /^https?:\/\/\S+[?&#]text=\S*$/.test(text);

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

  // ---------- state ----------
  let last = null;      // { text, result, marked }
  let notice = null;    // key of the shown notice (re-translated on a language switch)
  let checking = 0;     // number of the latest check: a newer one or Clear wins
  let pending = false;  // a check is waiting for the model
  // none (no service workers) | saving | ready | failed
  let offline = ("serviceWorker" in navigator && window.isSecureContext) ? "saving" : "none";
  // opened by the Shortcut: then the "add an icon" hint is not repeated
  const openedByShortcut = sharedText(location) !== null;

  // ---------- rendering ----------
  function setText(id, value) { $(id).textContent = value; }

  function noticeText(key) { return P()[key] || S()[key] || ""; }

  function showNotice(key) {
    notice = key || null;
    const el = $("notice");
    el.textContent = notice ? noticeText(notice) : "";
    el.classList.toggle("hidden", !notice);
  }

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
    const waiting = pending && modelState === "loading";
    setText("checkBtn", waiting ? p.loading : p.check);
    $("checkBtn").disabled = waiting;
    setText("siteLink", p.siteLink);
    setText("shareBtn", p.share);
    setText("mistakesNote", p.mistakesNote);
    setText("mistakesDownload", S().mistakesDownload);
    setText("mistakesClear", S().mistakesClear);
    setText("installHint", p.installHint);
    setText("footer", modelState === "ready" ? p.modelLine.replace("{id}", modelId) : "");
    $("lang").value = lang;
    if (notice) setText("notice", noticeText(notice));
    renderStatus();
    renderMistakes();
    if (last) render();
  }

  function clearChildren(el) { while (el.firstChild) el.removeChild(el.firstChild); }

  function render() {
    const { result, text, marked } = last;
    const s = S();
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
    mistake.disabled = marked;
    $("mistakeThanks").classList.toggle("hidden", !marked);
    setText("mistakeThanks", P().mistakeThanks);
    // ?text= (not #text=): the website only reads the query string on its host
    $("siteLink").href = siteRoot(location) + "?text=" + encodeURIComponent(cut(text, SITE_LINK_CHARS));
  }

  // ---------- checking ----------
  // The text is scored exactly as typed, like the website's text box
  // (leading/trailing line breaks count for the length features).
  async function check() {
    const text = cut($("input").value, MAX_CHARS);
    if (!text.trim()) { showNotice("emptyInputError"); return; }
    showNotice(null);
    const id = ++checking;
    if (modelState === "loading") {
      pending = true;
      applyText();
      await modelReady;
      if (id !== checking) return; // cleared or replaced while the model loaded
      pending = false;
    }
    if (modelState === "failed") showNotice("modelFailed");
    last = { text, result: self.FraudDetector.analyzeText(text, scorer), marked: false };
    store.session(LAST_KEY, text);
    applyText();
    const box = $("result");
    if (box.scrollIntoView) box.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function clearAll() {
    checking++;
    pending = false;
    $("input").value = "";
    last = null;
    store.session(LAST_KEY, null);
    $("result").classList.add("hidden");
    showNotice(null);
    applyText();
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
  function renderStatus() {
    const p = P();
    const ios = /iphone|ipad|ipod/i.test(navigator.userAgent)
      || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
    const standalone = navigator.standalone === true
      || (window.matchMedia && window.matchMedia("(display-mode: standalone)").matches);
    // Safari may erase a tab's saved copy after 7 days without the page
    // being used; the Home Screen app is exempt (WebKit storage policy)
    const ready = ios && !standalone ? p.offlineReadyTab : p.offlineReady;
    setText("status", { ready, saving: p.offlineSaving, failed: p.offlineFailed }[offline] || p.offlineNone);
    $("installHint").classList.toggle("hidden", !(ios && !standalone && !openedByShortcut));
  }

  function setOffline(state) { offline = state; renderStatus(); }

  // Registered after the model has loaded, so a first visit on mobile data
  // does not download the 3 MB model twice at the same time.
  function registerServiceWorker() {
    if (!("serviceWorker" in navigator) || !window.isSecureContext) return;
    setOffline("saving");
    navigator.serviceWorker.register("sw.js").then((reg) => {
      if (reg.active && !reg.installing) { setOffline("ready"); return; }
      const sw = reg.installing || reg.waiting;
      if (!sw) return;
      sw.addEventListener("statechange", () => {
        if (sw.state === "activated") setOffline("ready");
        if (sw.state === "redundant") setOffline(reg.active ? "ready" : "failed"); // a download failed
      });
    }).catch(() => setOffline("failed"));
    // ask Safari/Chrome not to evict the saved page and marks (best effort)
    if (navigator.storage && navigator.storage.persist) navigator.storage.persist().catch(() => {});
  }

  // A link may not carry a working phishing address to relatives: the
  // shared quote has "hxxp://" and "[.]" instead of live links.
  function defang(text, result) {
    let t = text.replace(/\bhttp(s?):\/\//gi, "hxxp$1://").replace(/\bwww\./gi, "www[.]");
    for (const d of result.domainInfo) {
      if (!d.domain) continue;
      const re = new RegExp(d.domain.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi");
      t = t.replace(re, (m) => m.replace(/\./g, "[.]"));
    }
    return t;
  }

  // ---------- text shared by the Shortcut ----------
  function takeShared() {
    const shared = sharedText(location);
    if (shared === null) return false;
    // keep the message out of the address bar / history once it is read
    try {
      const clean = new URL(location.href);
      clean.hash = "";
      clean.searchParams.delete("text");
      history.replaceState(null, "", clean.pathname + clean.search);
    } catch (e) { /* ignore */ }
    if (!shared) { showNotice("emptyShortcut"); return true; }
    if (isOwnLink(shared)) { showNotice("copiedLink"); return true; }
    $("input").value = shared;
    check();
    return true;
  }

  // ---------- events ----------
  $("checkBtn").addEventListener("click", check);
  $("input").addEventListener("input", () => { if (notice === "emptyInputError") showNotice(null); });
  $("clearBtn").addEventListener("click", () => { clearAll(); $("input").focus(); });
  $("pasteBtn").addEventListener("click", async () => {
    let text = "";
    try { text = await navigator.clipboard.readText(); } catch (e) { text = ""; }
    if (!text || !text.trim()) { showNotice("pasteFailed"); return; }
    if (isOwnLink(text.trim())) { showNotice("copiedLink"); return; }
    $("input").value = cut(text, MAX_CHARS);
    check();
  });
  $("lang").addEventListener("change", () => {
    lang = $("lang").value;
    store.set(LANG_KEY, lang);
    applyText();
  });
  $("shareBtn").addEventListener("click", () => {
    if (!last) return;
    const quote = defang(cut(last.text.trim(), 300), last.result);
    const text = P().shareText.replace("{pct}", last.result.pct).replace("{text}", quote);
    navigator.share({ text }).catch(() => {});
  });
  $("mistakeBtn").addEventListener("click", () => {
    if (!last || last.marked) return;
    const entry = self.Mistakes.makeEntry({
      text: last.text, predictedFraud: last.result.isFraud, prob: last.result.prob, source: "phone page",
    });
    store.set(MISTAKES_KEY, JSON.stringify(self.Mistakes.addTo(mistakes(), entry)));
    last.marked = true;
    render();
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
  // a new #text=... in a tab that is already open (no reload happens)
  window.addEventListener("hashchange", takeShared);

  // ---------- start ----------
  applyText();
  modelReady.then(registerServiceWorker);
  if (!takeShared()) {
    // iOS reloads background tabs: bring back the message checked in this tab
    const again = store.session(LAST_KEY);
    if (again) { $("input").value = again; check(); }
  }

  window.AFDCheck = { PAGE, sharedText, siteRoot, check, modelReady, defang, cut };
})();
