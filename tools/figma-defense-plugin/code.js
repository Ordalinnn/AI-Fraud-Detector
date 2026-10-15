// AI Fraud Detector — defense update (Figma Slides plugin).
//
// One run updates the deck "AI Fraud Detector — НОУ қорғауы" for the
// 7-minute defense (plan: docs/presentation_update_kz.md):
//   1. light colour scheme (the website's palette);
//   2. current numbers: 5 models + stacking, 94.7 %, new examples;
//   3. a 5th model on the "how it works" slide;
//   4. two new slides: the stacking calculation and "real-world checks";
//   5. backup slides moved to an "Қосымша" section and skipped;
//   6. speaker notes with timing (≈ 6 min 50 s).
// Safe to run twice: new slides are not duplicated, finished edits are
// reported as "already done". Undo: Ctrl+Z or File → Version history.

const ID = {
  title: "4:2", problem: "4:46", goal: "4:63", apparatus: "10:2", data: "5:4",
  system: "5:39", lr: "13:2", tfidf: "13:26", oldEnsemble: "10:11", results: "5:76",
  confusion: "5:109", errors: "28:17", xai: "6:6", demo: "6:29", limits: "6:43", end: "6:59",
};
const NEW_STACK = "NEW · Стекинг";
const NEW_REAL = "NEW · Шынайы өмір";

const report = { done: [], missed: [] };
const ok = (m) => report.done.push(m);
const miss = (m) => report.missed.push(m);

// ---------- colours ----------
const hex = (c) => "#" + [c.r, c.g, c.b].map((v) => Math.round(v * 255).toString(16).padStart(2, "0")).join("");
const rgb = (h) => ({ r: parseInt(h.slice(1, 3), 16) / 255, g: parseInt(h.slice(3, 5), 16) / 255, b: parseInt(h.slice(5, 7), 16) / 255 });
const solid = (h) => ({ type: "SOLID", color: rgb(h) });

const C = {
  bgA: "#eef3fc", bgB: "#f8fafd", panel: "#dce7fb", card: "#ffffff", line: "#d6e0f0", chip: "#e2e8f0",
  ink: "#0f172a", body: "#334155", muted: "#475569", faint: "#64748b",
  blue: "#1d4ed8", teal: "#0d9488", red: "#e11d48", amber: "#b45309", amberBg: "#fff4db",
};
const DARK = new Set(["#0b1220", "#151e32", "#404d66", "#4a381a"]);
const SEMANTIC_RED = /Критикалық|өткізіп|FN|алаяқтық\s*$|^\s*28\s*$/;

function firstSolid(node) {
  if (!("fills" in node) || !Array.isArray(node.fills)) return null;
  const p = node.fills.find((f) => f.type === "SOLID" && f.visible !== false && (f.opacity === undefined || f.opacity > 0.5));
  return p ? hex(p.color) : null;
}
function contains(b, x, y) { return b && x >= b.x && x <= b.x + b.width && y >= b.y && y <= b.y + b.height; }

// The colour visually behind a node (original colours): an earlier sibling
// shape under its centre, else the nearest filled ancestor, else the slide.
function backgroundOf(node, slide) {
  const bb = node.absoluteBoundingBox;
  const cx = bb ? bb.x + bb.width / 2 : 0;
  const cy = bb ? bb.y + bb.height / 2 : 0;
  let cur = node;
  while (cur && cur !== slide) {
    const p = cur.parent;
    if (!p) break;
    const idx = p.children.indexOf(cur);
    for (let i = idx - 1; i >= 0; i--) {
      const s = p.children[i];
      if (!s.visible || !(s.type === "RECTANGLE" || s.type === "FRAME")) continue;
      const f = firstSolid(s);
      if (f && contains(s.absoluteBoundingBox, cx, cy)) return f;
    }
    if (p === slide || p.type === "SLIDE") return firstSolid(p);
    const pf = firstSolid(p);
    if (pf) return pf;
    cur = p;
  }
  return firstSolid(slide);
}

function textOf(node) {
  if (node.type === "TEXT") return node.characters;
  if ("findAllWithCriteria" in node) return node.findAllWithCriteria({ types: ["TEXT"] }).map((t) => t.characters).join(" ");
  return "";
}

function redFor(slide, chars) {
  return slide.id === ID.title || SEMANTIC_RED.test(chars) ? C.red : C.blue;
}

function shapeColour(node, slide, slideBg, h) {
  switch (h) {
    case "#0b1220":
      if (node.type === "VECTOR" || node.type === "ELLIPSE") return null; // QR codes, chart dots
      return slideBg === "#0b1220" ? C.bgA : C.panel;
    case "#151e32": return C.card;
    case "#404d66": return node.type === "RECTANGLE" || node.type === "LINE" ? C.line : C.chip;
    case "#d1ccc2": return C.line;
    case "#19c37d": return C.teal;
    case "#4a381a": return C.amberBg;
    case "#ff3b4e": {
      if (slide.id === ID.title) return C.red;
      if (slide.id === ID.data && node.width > 200) return C.red; // "686 алаяқтық" bar
      if (/Критикалық|FN/.test(textOf(node))) return C.red;
      return C.blue;
    }
    default: return null;
  }
}

function textColour(slide, bg, h, chars) {
  const onDark = DARK.has(bg);
  if (h === "#ff3b4e") return bg === "#ff3b4e" ? null : redFor(slide, chars);
  if (h === "#19c37d") return C.teal;
  if (!onDark) return null;
  return { "#ffffff": C.ink, "#b0b9c9": C.muted, "#9ea3ad": C.faint, "#ffb020": C.amber }[h] || null;
}

async function loadFontsOf(t) {
  const fonts = t.fontName === figma.mixed ? t.getStyledTextSegments(["fontName"]).map((s) => s.fontName) : [t.fontName];
  await Promise.all(fonts.map((f) => figma.loadFontAsync(f)));
}

async function recolour(slide) {
  const slideBg = firstSolid(slide);
  const plan = [];
  for (const n of slide.findAll(() => true)) {
    if (n.type === "TEXT") {
      const bg = backgroundOf(n, slide);
      const segs = n.getStyledTextSegments(["fills"]);
      const ranges = [];
      for (const s of segs) {
        const p = s.fills.find((f) => f.type === "SOLID");
        if (!p) continue;
        const to = textColour(slide, bg, hex(p.color), s.characters);
        if (to) ranges.push([s.start, s.end, to]);
      }
      if (ranges.length) plan.push(["text", n, ranges]);
    } else {
      const fill = firstSolid(n);
      const toFill = fill ? shapeColour(n, slide, slideBg, fill) : null;
      let toStroke = null;
      if ("strokes" in n && Array.isArray(n.strokes)) {
        const s = n.strokes.find((p) => p.type === "SOLID");
        if (s && ["#ff3b4e", "#404d66"].includes(hex(s.color))) toStroke = shapeColour(n, slide, slideBg, hex(s.color));
      }
      const addBorder = (fill === "#151e32") || (fill === "#ffffff" && n.type === "FRAME" && n.parent === slide && !n.strokes.length);
      if (toFill || toStroke || addBorder) plan.push(["shape", n, toFill, toStroke, addBorder]);
    }
  }
  for (const [kind, n, a, b, c] of plan) {
    if (kind === "text") {
      await loadFontsOf(n);
      for (const [s, e, to] of a) n.setRangeFills(s, e, [solid(to)]);
    } else {
      if (a) n.fills = n.fills.map((p) => (p.type === "SOLID" && p.visible !== false ? Object.assign({}, p, { color: rgb(a) }) : p));
      if (b) n.strokes = n.strokes.map((p) => (p.type === "SOLID" ? Object.assign({}, p, { color: rgb(b) }) : p));
      if (c && "strokes" in n) { n.strokes = [solid(C.line)]; n.strokeWeight = 1.5; n.strokeAlign = "INSIDE"; }
    }
  }
  const newBg = { "#0b1220": C.bgA, "#f4f1ea": C.bgB, "#ff3b4e": C.blue }[slideBg];
  if (newBg) slide.fills = [solid(newBg)];
  return plan.length;
}

// ---------- text edits (keep each run's style) ----------
function texts(slide) { return slide.findAllWithCriteria({ types: ["TEXT"] }); }

async function sub(node, from, to) {
  const i = node.characters.indexOf(from);
  if (i < 0) return false;
  await loadFontsOf(node);
  if (to.length) node.insertCharacters(i + from.length, to, "BEFORE");
  node.deleteCharacters(i, i + from.length);
  return true;
}

// Replace `from` -> `to` in the first text of the slide containing it
// (or the n-th one, top to bottom, when `nth` is given).
async function edit(slide, from, to, nth) {
  let hits = texts(slide).filter((t) => t.characters.includes(from));
  if (nth !== undefined) hits = hits.sort((a, b) => a.absoluteTransform[1][2] - b.absoluteTransform[1][2]).slice(nth, nth + 1);
  if (!hits.length) {
    if (texts(slide).some((t) => t.characters.includes(to))) return ok(`уже сделано: ${to.slice(0, 40)}`);
    return miss(`слайд ${slide.name}: не найден текст «${from.slice(0, 50)}»`);
  }
  await sub(hits[0], from, to);
  ok(`${slide.name}: ${from.slice(0, 30)} → ${to.slice(0, 30)}`);
}

// Replace a whole text node that equals / starts with `head`.
async function setText(slide, head, to, exact) {
  const t = texts(slide).find((n) => (exact ? n.characters.trim() === head : n.characters.startsWith(head)));
  if (!t) {
    if (texts(slide).some((n) => n.characters.startsWith(to.slice(0, 25)))) return ok(`уже сделано: ${to.slice(0, 40)}`);
    return miss(`слайд ${slide.name}: не найден текст «${head.slice(0, 50)}»`);
  }
  await sub(t, t.characters, to);
  ok(`${slide.name}: ${head.slice(0, 30)} → ${to.slice(0, 30)}`);
}

// ---------- "how it works": add the 5th model ----------
async function addFifthModel(slide) {
  if (texts(slide).some((t) => t.characters.startsWith("Символдық n-грамдар"))) return ok("5-я модель уже есть");
  const names = ["Logistic Regression", "Random Forest", "Gradient Boosting", "TF-IDF + Naive Bayes"];
  const boxes = names.map((nm) => {
    const t = texts(slide).find((n) => n.characters.trim() === nm);
    return t && t.parent && t.parent !== slide && t.parent.type === "FRAME" ? t.parent : null;
  });
  if (boxes.some((b) => !b)) return miss("слайд 6: не нашёл карточки моделей — 5-ю модель добавьте вручную (копия карточки TF-IDF)");
  const step = boxes[1].y - boxes[0].y;
  const copy = boxes[3].clone();
  if (copy.parent !== boxes[3].parent) boxes[3].parent.appendChild(copy);
  for (const b of boxes) b.y -= step / 2;
  copy.x = boxes[3].x;
  copy.y = boxes[3].y + step;
  for (const t of copy.findAllWithCriteria({ types: ["TEXT"] })) {
    if (t.characters.trim() === "TF-IDF + Naive Bayes") await sub(t, t.characters, "Символдық n-грамдар + LR");
    else await sub(t, t.characters, "сөз бөліктерінен: картаңыз / карту");
  }
  ok("слайд 6: добавлена 5-я модель");
}

// ---------- new slides ----------
async function fonts() {
  await Promise.all([
    { family: "Montserrat", style: "ExtraBold" }, { family: "Montserrat", style: "Bold" },
    { family: "Inter", style: "Regular" }, { family: "Inter", style: "Medium" },
    { family: "Inter", style: "Semi Bold" }, { family: "Inter", style: "Bold" },
    { family: "JetBrains Mono", style: "Bold" },
  ].map((f) => figma.loadFontAsync(f)));
}
const F = {
  head: { family: "Montserrat", style: "ExtraBold" }, headB: { family: "Montserrat", style: "Bold" },
  reg: { family: "Inter", style: "Regular" }, med: { family: "Inter", style: "Medium" },
  semi: { family: "Inter", style: "Semi Bold" }, bold: { family: "Inter", style: "Bold" },
  mono: { family: "JetBrains Mono", style: "Bold" },
};

function addFrame(parent, x, y, w, h, fill, radius, border) {
  const f = figma.createFrame();
  parent.appendChild(f);
  f.resize(w, h);
  f.fills = [solid(fill)];
  if (radius) f.cornerRadius = radius;
  if (border) { f.strokes = [solid(C.line)]; f.strokeWeight = 1.5; f.strokeAlign = "INSIDE"; }
  f.clipsContent = false;
  f.x = x; f.y = y;
  return f;
}
function addText(parent, font, size, color, chars, x, y, w, opts) {
  const t = figma.createText();
  parent.appendChild(t);
  t.fontName = font;
  t.fontSize = size;
  t.characters = chars;
  t.fills = [solid(color)];
  if (opts && opts.track) t.letterSpacing = { unit: "PERCENT", value: opts.track };
  if (opts && opts.lh) t.lineHeight = { unit: "PERCENT", value: opts.lh };
  if (w) { t.textAutoResize = "HEIGHT"; t.resize(w, t.height); } else t.textAutoResize = "WIDTH_AND_HEIGHT";
  t.x = x; t.y = y;
  return t;
}
function addRect(parent, x, y, w, h, fill) {
  const r = figma.createRectangle();
  parent.appendChild(r);
  r.resize(w, h);
  r.fills = [solid(fill)];
  r.x = x; r.y = y;
  return r;
}
const label = (s, chars) => addText(s, F.mono, 22, C.blue, chars, 120, 110, 0, { track: 8 });

function buildStackingSlide() {
  const s = figma.createSlide();
  s.name = NEW_STACK;
  s.fills = [solid(C.bgB)];
  label(s, "07 · СТЕКИНГ: ЕСЕПТЕУ МЫСАЛЫ");
  const q = addFrame(s, 120, 168, 1680, 96, C.card, 20, true);
  addText(q, F.med, 30, C.ink, "«Ваша посылка задержана. Оплатите доставку 350 тг по ссылке kazpost-delivery.top»", 40, 27, 1600);

  const cols = [120, 236, 690, 860, 1040];
  const head = ["", "МОДЕЛЬ", "pᵢ", "logit pᵢ", "салмақ wᵢ"];
  head.forEach((h, i) => h && addText(s, F.mono, 18, C.faint, h, cols[i], 304, 0, { track: 6 }));
  const rows = [
    ["M₁", "Логистикалық регрессия", "0,9998", "8,33", "0,10"],
    ["M₂", "Кездейсоқ орман", "0,480", "−0,08", "0,23"],
    ["M₃", "Градиенттік бустинг", "0,555", "0,22", "0,40"],
    ["M₄", "TF-IDF + Байес", "0,811", "1,45", "1,43"],
    ["M₅", "Символдық n-грамдар", "0,974", "3,63", "1,68"],
    ["ΔR", "Ережелер (.top, kazpost)", "0,25", "—", "0,29"],
  ];
  rows.forEach((r, k) => {
    const y = 344 + k * 76;
    addRect(s, 120, y, 1020, 1.5, C.line);
    addText(s, F.headB, 28, C.blue, r[0], cols[0], y + 20);
    addText(s, F.semi, 26, C.ink, r[1], cols[1], y + 22, 440);
    addText(s, F.med, 26, C.body, r[2], cols[2], y + 22);
    addText(s, F.med, 26, C.body, r[3], cols[3], y + 22);
    addText(s, F.bold, 26, k === 4 || k === 3 ? C.blue : C.ink, r[4], cols[4], y + 22);
  });
  addRect(s, 120, 344 + 6 * 76, 1020, 1.5, C.line);

  const card = addFrame(s, 1180, 304, 620, 496, C.card, 24, true);
  addText(card, F.mono, 18, C.blue, "СТЕКИНГ ФОРМУЛАСЫ", 40, 36, 0, { track: 6 });
  addText(card, F.semi, 26, C.ink, "z = b + Σ wᵢ · logit pᵢ + w_R · ΔR", 40, 80, 540);
  addText(card, F.reg, 22, C.faint, "logit p = ln(p / (1 − p)),  b = 0,56", 40, 128, 540);
  addRect(card, 40, 176, 540, 1.5, C.line);
  addText(card, F.med, 26, C.body, "z = 0,56 + 0,83 − 0,02 + 0,09\n    + 2,07 + 6,10 + 0,07 ≈ 9,71", 40, 200, 540, { lh: 140 });
  addText(card, F.head, 38, C.ink, "P = σ(9,71) ≈ 0,9999", 40, 306, 540);
  const chip = addFrame(card, 40, 392, 330, 56, "#ffe4e8", 14);
  addText(chip, F.bold, 26, C.red, "99,9% → Критикалық", 20, 12);

  addText(s, F.reg, 24, C.muted,
    "RF (0,48) мен GB (0,55) екіұдай, бірақ стекинг сенімдірек мәтіндік модельдерге көбірек салмақ береді. " +
    "Ескі формула — қарапайым орташа — осы жерде небәрі 0,76 беретін еді. Пайыз 99,9%-дан аспайды: жүйе ешқашан «100% сенімдімін» демейді.",
    120, 870, 1680, { lh: 140 });
  return s;
}

function buildRealWorldSlide() {
  const s = figma.createSlide();
  s.name = NEW_REAL;
  s.fills = [solid(C.bgA)];
  label(s, "09 · ШЫНАЙЫ ӨМІРДЕ ТЕКСЕРУ");
  addText(s, F.head, 56, C.ink, "Жюри ескертулерінен кейін — жүйені шынайы жағдайда тексердік", 120, 168, 1500, { lh: 115 });
  const cards = [
    ["бұрын 18,2%", "0,3%", "жалған дабыл 2 242 нақты қарапайым SMS-та (оқытуда болмаған)", C.blue],
    ["бұрын 73,6%", "88,4%", "қазақ тіліндегі дәлдік", C.blue],
    ["бұрын 46–83%", "93%", "жасырылған алаяқтық ұсталды: «к0д», латын әріптері, көрінбейтін таңбалар", C.blue],
    ["бұрын 8,2%", "0,7%", "калибрлеу қатесі (ECE): «80% қауіп» шынымен ~80% дегенді білдіреді", C.teal],
  ];
  cards.forEach((c, i) => {
    const f = addFrame(s, 120 + i * 430, 380, 390, 420, C.card, 24, true);
    addText(f, F.med, 24, C.faint, c[0], 36, 36);
    addText(f, F.head, 92, c[3], c[1], 32, 80);
    addText(f, F.med, 24, C.body, c[2], 36, 214, 318, { lh: 140 });
  });
  addText(s, F.med, 26, C.body, "Оқытуда мүлдем болмаған алаяқтық түрлері: 76,0% → 84,1% ұсталды", 120, 860, 1680);
  addText(s, F.reg, 20, C.faint, "Дереккөз: тәуелсіз тексеру, reports/jury_evaluation және reports/obfuscation (GitHub: Ordalinnn/AI-Fraud-Detector)", 120, 912, 1680);
  return s;
}

// ---------- speaker notes ----------
const NOTES = {
  [ID.title]: "**Нұрасыл · 0:20**\n- Құрметті қазылар алқасы, сәлеметсіздер ме! Біз — Тараз НЗМ-нің 12С сынып оқушылары Ордалин Нұрасыл мен Ғани Ақжол.\n- Жобамыз — «AI Fraud Detector»: алаяқтық хабарламаны танып, оның ықтималдығын есептейтін математикалық модель.",
  [ID.problem]: "**Нұрасыл · 0:25**\n- Қазақстанда тіркелген қылмыстың 19,5%-ы — интернет-алаяқтық.\n- Ең жиі үш схема: «картаңыз бұғатталды», «банк қызметкері» қоңырауы, жалған курьер мен ұтыс.\n- Ортақ белгілері — шұғылдық, код сұрау, сілтеме.",
  [ID.goal]: "**Нұрасыл · 0:20**\n- Мақсат — хабарламаның алаяқтық болу ықтималдығын математикалық түрде бағалау.\n- Негізгі міндет — бірнеше статистикалық модельді біріктіріп, нәтижені түсінікті ету.",
  [ID.apparatus]: "**Нұрасыл · 0:20**\n- Гипотеза: бірнеше әдісті стекинг арқылы біріктірсек, бағалау дәлірек әрі түсініктірек болады.\n- Тексеру — тәуелсіз 282 хабарламада, кілт сөз әдісімен салыстыру.",
  [ID.data]: "**Нұрасыл · 0:25**\n- Үш тілде 1 398 хабарлама: 686 алаяқтық, 712 қауіпсіз, 24-тен астам схема.\n- Жюри шынайы хабарламаларды сұрады — тағы 400 нақты SMS қостық.\n- → Сөз Ақжолға.",
  [ID.system]: "**Ақжол · 0:40**\n- Алдымен маскировканы аламыз: «к0д» → «код».\n- 25 белгі есептеледі, бес модель бағалайды: үшеуі белгілер бойынша, екеуі мәтіннен тікелей.\n- Ең маңыздысы — стекинг: соңғы логистикалық регрессия әр модельге қаншалықты сену керегін деректерден өзі үйренеді.\n- Ескі орташалауда үш әлсіз модель күшті модельді басып кететін еді.",
  [ID.lr]: "**Ақжол · 0:35**\n- Әр модельдің негізі — сигмоида: z — белгілердің салмақталған қосындысы, σ оны 0 мен 1 аралығындағы ықтималдыққа айналдырады.\n- Мысалда құпиялылық сөздері, ұзындық пен цифрлар z-ті 3,14-ке көтереді → 96%.\n- «Ертең сабақ болады» — небәрі 5%.",
  [ID.tfidf]: "**Қосымша — сұрақ қойылса**\n- TF: сөздің хабарламадағы жиілігі, IDF = ln(N / df) — сиректігі.\n- Байес: «код» сөзі кездессе, алаяқтық ықтималдығы 49%-дан 83%-ға өседі.",
  [ID.oldEnsemble]: "**Ескі нұсқа (жазбаша жұмыс)** — қарапайым орташа. Жаңа стекинг слайдымен салыстыру үшін ғана.",
  [ID.results]: "**Ақжол · 0:40**\n- Тәуелсіз 282 хабарламада дәлдік 82,3%-дан 94,7%-ға өсті.\n- Recall: кілт сөз әдісі алаяқтықтың 42%-ын ұстайды, біздікі — 93%.\n- Жалған дабыл 152 қауіпсіз хабарламаның 6-ында ғана.\n- 15 түрлі бөліністе орташа — 96,1%.",
  [ID.confusion]: "**Қосымша — сұрақ қойылса**\n- TN 146, FP 6, FN 9, TP 121. Барлық метрика осы төрт саннан есептеледі.",
  [ID.errors]: "**Қосымша — сұрақ қойылса**\n- 6 жалған дабыл, 9 өткізіп алынған алаяқтық.\n- «Мама…» схемасы бұрын 0,42 болатын, енді 0,99.\n- Әлсіз жерлер: романтикалық (41%) және инвестициялық (66%) алаяқтықтың жаңа түрлері.",
  [ID.xai]: "**Нұрасыл · 0:25**\n- Жүйе себебін көрсетеді: жалған домен, триггер сөздер, модельдердің келісімі.\n- Бір модель үш жерде: сайт, браузер кеңейтпесі, Android.\n- Кеңейтпе сайттағы модельдің өзін есептейді, мәтін құрылғыдан шықпайды.",
  [ID.demo]: "**Нұрасыл · 0:40**\n- Қазір тікелей көрсетемін. [Сайтты алдын ала ашып қойыңыз!]\n- Алаяқтық SMS-ті қойып, қауіп деңгейін, белгіленген сөздерді және 5 модельдің бағасын көрсету.\n- Уақыт болса — WhatsApp Web-тегі кеңейтпе белгісі.\n- Қазылар QR-код арқылы өздері де тексере алады.",
  [ID.limits]: "**Нұрасыл · 0:25**\n- Шектеулер: нақты қазақ және орыс хабарламалары әлі аз, романтикалық және инвестициялық схемалар нашар ұсталады, бұл — прототип.\n- Келесі қадам — «Бұл қате» батырмасы арқылы нақты деректер жинау және бүкіл хат алмасуды талдау.",
  [ID.end]: "**Нұрасыл · 0:20**\n- 5 модельдің стекинг жүйесі тәуелсіз сынақта 94,7% дәлдік көрсетті — кілт сөз әдісінен әлдеқайда жоғары.\n- Гипотеза расталды.\n- Назарларыңызға рахмет, сұрақтарыңызға дайынбыз!",
};
const NOTES_STACK = "**Ақжол · 0:40**\n- Әр модельдің ықтималдығын логитке айналдырып, үйренген салмағына көбейтеміз.\n- RF мен GB екіұдай (0,48 және 0,55), бірақ стекинг сенімдірек мәтіндік модельдерге көбірек салмақ береді.\n- z ≈ 9,7 → P = 99,9%, критикалық қауіп. Ескі орташа формула осы жерде небәрі 0,76 беретін еді.";
const NOTES_REAL = "**Ақжол · 0:35**\n- Жюри «шынайы өмірде қалай?» деп сұрады — тексердік.\n- Оқытуда болмаған 2 242 нақты SMS-та жалған дабыл 18%-дан 0,3%-ға түсті.\n- Қазақ тілі — 88%. Алаяқтар «к0д» деп жасырса да — 93%.\n- Пайыз калибрленген: «80%» шынымен 80% дегенді білдіреді.\n- → Сөз Нұрасылға.";

// ---------- main ----------
async function main() {
  await fonts();
  const all = figma.getSlideGrid().flat();
  const byId = Object.fromEntries(all.map((s) => [s.id, s]));
  const S = {};
  for (const [k, id] of Object.entries(ID)) {
    S[k] = byId[id];
    if (!S[k]) throw new Error(`Слайд ${id} (${k}) не найден — это не тот файл или слайды пересоздавались`);
  }

  // 1. colours
  let n = 0;
  for (const s of Object.values(S)) n += await recolour(s);
  ok(`перекрашено элементов: ${n}`);

  // 2. text edits — only on the first run (they look for the old texts)
  if (all.some((s) => s.name === NEW_STACK)) {
    ok("тексты уже обновлены прошлым запуском");
  } else {
  await edit(S.apparatus, "нәтижелерін ережелер негізінде біріктірсек", "нәтижелерін стекинг арқылы біріктірсек");
  await edit(S.data, "Шаблондар әртүрлі банк атауларымен кеңейтілді.",
    "Шаблондар әртүрлі банк атауларымен кеңейтілді. Оған қоса 400 нақты SMS (UCI корпусы) — модель шынайы хабарламалар стилін де көрсін.");

  await edit(S.system, "төрт модельдің", "бес модельдің");
  await setText(S.system, "SMS, чат немесе қоңырау транскрипті", "SMS, чат немесе транскрипт · «к0д» → «код»");
  await setText(S.system, "20 белгі", "25 белгі", true);
  await setText(S.system, "Орташа ықтималдық + ереже бонусы", "Стекинг + ереже бонусы");
  await setText(S.system, "бонус ≤ 30%", "салмақтарды деректерден үйренеді", true);
  await setText(S.system, "Ереже бонусы жеке сөзге",
    "Стекинг — соңғы логистикалық регрессия: әр модельге қанша сенуді деректерден өзі үйренеді (символдық n-грам 1,68 · TF-IDF 1,43 · GB 0,40 · ережелер 0,29 · RF 0,23 · LR 0,10).");
  await addFifthModel(S.system);

  await edit(S.lr, "2,0 → 0,12", "2,97 → 0,05");
  await setText(S.lr, "z = 2,81 → 0,94", "z = 3,14 → 0,96", true);
  await setText(S.lr, "z = 0,26 + 2,14", "z = 0,05 + 1,79 + 1,22 + 1,18 − 1,10 ≈ 3,14");
  await setText(S.lr, "0,26 — бос мүше",
    "0,05 — бос мүше w₀ · 1,79 — құпиялылық сөздері · 1,22 — мәтін ұзындығы · 1,18 — цифрлар саны · −1,10 — қалған белгілер");
  await edit(S.lr, "σ(2,81)", "σ(3,14)");
  await edit(S.lr, "0,060", "0,043");
  await edit(S.lr, "≈ 0,94", "≈ 0,96");
  await edit(S.lr, "ықтималдығы 94%", "ықтималдығы 96%");

  await setText(S.tfidf, "07 · TF-IDF", "ҚОСЫМША · TF-IDF ЖӘНЕ НАИВТІ БАЙЕС");
  await setText(S.oldEnsemble, "08 · АНСАМБЛЬ", "ЕСКІ НҰСҚА · ОРТАША АНСАМБЛЬ (ЖАЗБАША ЖҰМЫС)");

  await setText(S.results, "09 · НӘТИЖЕЛЕР", "08 · НӘТИЖЕЛЕР");
  await setText(S.results, "79%", "93%", true);
  await setText(S.results, "Ансамбль қарапайым кілт сөз", "Жаңа ансамбль алаяқтықтың 93%-ын ұстайды — кілт сөз әдісінен 2 есе көп");
  await setText(S.results, "Ансамбль", "Жаңа ансамбль", true);
  await edit(S.results, "82,3%", "94,7%", 0);
  await edit(S.results, "82,3%", "95,3%", 0);
  await setText(S.results, "78,5%", "93,1%", true);
  await setText(S.results, "80,3%", "94,2%", true);
  await setText(S.results, "93,0%", "99,1%", true);
  await setText(S.results, "Бір сценарийден",
    "Сол 282 хабарлама (group-aware 80/20: 1116 / 282), бір сценарийдің нұсқалары тек бір жаққа түседі. Жазбаша жұмыстағы ансамбль: 82,3% · ең жақсы жалғыз модель (символдық n-грам): 93,6% · 15 бөлініс бойынша орташа: 96,1% ± 1,4.");

  await setText(S.confusion, "10 · ШАТАСУ", "ҚОСЫМША · ШАТАСУ МАТРИЦАСЫ");
  await setText(S.confusion, "130", "146", true);
  await setText(S.confusion, "22", "6", true);
  await setText(S.confusion, "28", "9", true);
  await setText(S.confusion, "102", "121", true);
  await edit(S.confusion, "(102 + 130)", "(121 + 146)");
  await edit(S.confusion, "= 82,3%", "= 94,7%", 0);
  await edit(S.confusion, "102 / (102 + 22)", "121 / (121 + 6)");
  await edit(S.confusion, "82,3%", "95,3%");
  await edit(S.confusion, "102 / (102 + 28)", "121 / (121 + 9)");
  await edit(S.confusion, "78,5%", "93,1%");
  await edit(S.confusion, "0,823 · 0,785 / 1,608", "0,953 · 0,931 / 1,884");
  await edit(S.confusion, "80,3%", "94,2%");

  await setText(S.errors, "11 · ҚАТЕЛЕРДІ", "ҚОСЫМША · ҚАТЕЛЕРДІ ТАЛДАУ");
  await setText(S.errors, "22", "6", true);
  await setText(S.errors, "28", "9", true);
  await edit(S.errors, "0,42", "0,99");
  await edit(S.errors, "Күдікті", "Критикалық");
  await setText(S.errors, "Шын мәнінде алаяқтық",
    "Жазбаша жұмыстағы жүйе 0,42 берген (Күдікті). Символдық n-грам моделі мен стекинг қосылғаннан кейін — 0,99 (Критикалық)");
  await setText(S.errors, "Қорытынды: модель сөз",
    "Енді әлсіз жерлер: оқытуда болмаған романтикалық (41%) және инвестициялық (66%) алаяқтық — бүкіл хат алмасуды талдау керек");

  await setText(S.xai, "12 · ТҮСІНДІРМЕЛІ", "10 · ТҮСІНДІРМЕЛІ AI");
  await edit(S.xai, "4 модель келіспесе", "5 модель келіспесе");
  await setText(S.xai, "Телефон мен браузер", "Сайт · кеңейтпе · Android", true);
  await setText(S.xai, "PWA-қосымша", "Кеңейтпе сайттағы модельдің өзін құрылғыда есептейді; мәтін құрылғыдан шықпайды");

  await setText(S.demo, "13 · ДЕМО", "11 · ДЕМО");

  await setText(S.limits, "14 · ШЕКТЕУЛЕР", "12 · ШЕКТЕУЛЕР МЕН ДАМУ");
  await setText(S.limits, "— Оқыту жиыны шағын", "— Нақты қазақ және орыс хабарламалары әлі аз: тексеру құралы дайын, деректер жиналуда");
  await setText(S.limits, "— Мүлдем жаңа схемаларды", "— Романтикалық және инвестициялық алаяқтықтың жаңа түрлері нашар ұсталады (41% және 66%)");
  await setText(S.limits, "Деректер жиынын нақты", "Нақты хабарламаларды «Бұл қате» батырмасы арқылы жинау");
  await setText(S.limits, "Тілдер санын көбейту", "Бүкіл хат алмасуды талдау (тексердік: әзірге жалған дабылды көбейтеді)");
  await setText(S.limits, "Түсіндірмелерді мобильді", "Android қосымшасына толық ансамбльді енгізу");

  await setText(S.end, "20 белгі жүйесі", "25 белгі және 5 модельдің стекинг жүйесі әзірленді; үш тілде 1 398 хабарлама + 400 нақты SMS");
  await setText(S.end, "Тәуелсіз сынақта", "Тәуелсіз сынақта: 94,7% дәлдік, 94,2% F1, 99,1% ROC-AUC (кілт сөз әдісі: 69,5% / 56,1%)");
  await setText(S.end, "Гипотеза расталды", "Гипотеза расталды: модельдерді стекингпен біріктіру кілт сөзбен сүзуден де, ескі орташалаудан да тиімді");
  await setText(S.end, "Қолдану аясы", "Сайт, браузер кеңейтпесі және Android қосымшасы; мектептерде цифрлық сауаттылық құралы; бастапқы коды ашық");
  }

  // 3. new slides (once)
  let stack = all.find((s) => s.name === NEW_STACK);
  if (!stack) { stack = buildStackingSlide(); ok("создан слайд «Стекинг: есептеу мысалы»"); }
  let real = all.find((s) => s.name === NEW_REAL);
  if (!real) { real = buildRealWorldSlide(); ok("создан слайд «Шынайы өмірде тексеру»"); }
  stack.speakerNotes = NOTES_STACK;
  real.speakerNotes = NOTES_REAL;

  // 4. notes, order, skipped backup slides
  for (const [id, text] of Object.entries(NOTES)) {
    try { byId[id].speakerNotes = text; } catch (e) { miss(`заметки ${id}: ${e.message}`); }
  }
  const main = [S.title, S.problem, S.goal, S.apparatus, S.data, S.system, S.lr, stack, S.results, real, S.xai, S.demo, S.limits, S.end];
  const backup = [S.tfidf, S.confusion, S.errors, S.oldEnsemble];
  try {
    for (const s of main) s.isSkippedSlide = false;
    for (const s of backup) s.isSkippedSlide = true;
  } catch (e) { miss("скрыть запасные слайды не удалось: " + e.message); }
  reorder(main, backup);
}

// Put the slides in the defense order. Figma has changed this API over
// time, so try several ways and check the result after each one.
function currentOrder() { return figma.getSlideGrid().flat().map((s) => s.id); }
function reorder(main, backup) {
  const used = new Set([...main, ...backup].map((s) => s.id));
  const others = figma.getSlideGrid().flat().filter((s) => !used.has(s.id));
  const extra = [...backup, ...others];
  const want = [...main, ...extra].map((s) => s.id);
  const same = () => currentOrder().join() === want.join();
  if (same()) return ok("порядок слайдов уже правильный");
  const rowOf = () => figma.currentPage.children.find((c) => c.type === "SLIDE_GRID");
  const attempts = [
    ["setSlideGrid, 2 раздела", () => figma.setSlideGrid([main, extra]), true],
    ["setCanvasGrid, 2 раздела", () => figma.setCanvasGrid([main, extra]), true],
    ["setSlideGrid, 1 ряд", () => figma.setSlideGrid([[...main, ...extra]]), false],
    ["setCanvasGrid, 1 ряд", () => figma.setCanvasGrid([[...main, ...extra]]), false],
    ["insertChild", () => {
      const row = main[0].parent;
      [...main, ...extra].forEach((s, i) => row.insertChild(i, s));
    }, false],
  ];
  const errors = [];
  for (const [name, fn, twoRows] of attempts) {
    try {
      fn();
      if (!same()) { errors.push(`${name}: порядок не изменился`); continue; }
      if (twoRows) {
        const grid = rowOf();
        if (grid && grid.children[0]) grid.children[0].name = "Қорғау · 7 минут";
        if (grid && grid.children[1]) grid.children[1].name = "Қосымша (сұрақтарға)";
      }
      return ok(`порядок слайдов исправлен (${name})`);
    } catch (e) {
      errors.push(`${name}: ${e && e.message ? e.message : e}`);
    }
  }
  miss("порядок слайдов не изменился — " + errors.join(" | "));
}

main()
  .then(() => {
    const msg = report.missed.length
      ? `Готово, но ${report.missed.length} правок не найдено: ${report.missed.join(" | ")}`
      : `Готово: ${report.done.length} изменений, без ошибок`;
    figma.closePlugin(msg);
  })
  .catch((e) => figma.closePlugin("Ошибка: " + (e && e.message ? e.message : e)));
