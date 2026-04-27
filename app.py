import re
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Fraud Detector",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# LANGUAGE
# =========================
# Language selector is now placed inside the sidebar.
# This block also fixes old saved values like "🇷🇺 Русский".
LANG_OPTIONS = ["🇰🇿 KZ", "🇷🇺 RU", "🇬🇧 EN"]
OLD_LANG_MAP = {
    "🇰🇿 Қазақша": "🇰🇿 KZ",
    "🇷🇺 Русский": "🇷🇺 RU",
    "🇬🇧 English": "🇬🇧 EN",
}

if "lang" not in st.session_state:
    st.session_state.lang = "🇷🇺 RU"

st.session_state.lang = OLD_LANG_MAP.get(st.session_state.lang, st.session_state.lang)

if st.session_state.lang not in LANG_OPTIONS:
    st.session_state.lang = "🇷🇺 RU"

lang = st.session_state.lang

TEXT = {
    "🇰🇿 KZ": {
        "title": "AI Fraud Detector 3,1415",
        "subtitle": "SMS, интернет-хабарлама және қоңырау транскриптіндегі алаяқтықты анықтайтын AI-прототип.",
        "mode": "Тексеру режимі",
        "sms": "SMS / хабарлама",
        "bank": "Банк режимі",
        "call": "Қоңырау транскрипті",
        "file": "Файл анализі",
        "demo": "Демо мысалдар",
        "input_title": "Мәтінді енгізіңіз",
        "input_label": "SMS, хабарлама немесе қоңырау транскрипті:",
        "upload": "TXT файл жүктеу",
        "analyze": "🚀 Анализ жасау",
        "features": "Жүйе мүмкіндіктері",
        "result": "Анализ нәтижесі",
        "risk": "Алаяқтық ықтималдығы",
        "detected": "Табылған белгілер",
        "threshold": "Шешім шегі",
        "model": "Модель",
        "low": "Қауіп төмен",
        "mid": "Күмәнді",
        "high": "Жоғары қауіп",
        "critical": "Өте жоғары қауіп",
        "why": "Неге жүйе осындай шешім шығарды?",
        "domain": "Домен анализі",
        "vector": "Сандық белгілер",
        "report": "Есепті жүктеу",
        "history": "Тексеру тарихы",
        "advice": "Қауіпсіздік кеңесі",
        "bad_advice": "Код, пароль, CVV немесе карта нөмірін бермеңіз. Сілтемеге өтпеңіз. Банкке тек ресми нөмір арқылы хабарласыңыз.",
        "good_advice": "Хабарлама қауіпсіз көрінеді. Бірақ күмән болса, ресми дереккөз арқылы тексеріңіз.",
        "no_text": "Алдымен мәтін енгізіңіз.",
        "no_features": "Күшті алаяқтық белгілері табылмады.",
        "no_domain": "Мәтінде URL немесе домен табылмады.",
        "download": "📥 TXT есепті жүктеу",
        "how": "Бұл қалай жұмыс істейді?",
        "footer": "Қолданбалы математика + машиналық оқыту прототипі",
    },
    "🇷🇺 RU": {
        "title": "AI Fraud Detector",
        "subtitle": "AI-прототип для обнаружения мошенничества в SMS, сообщениях и транскриптах звонков.",
        "mode": "Режим проверки",
        "sms": "SMS / сообщение",
        "bank": "Банковский режим",
        "call": "Транскрипт звонка",
        "file": "Анализ файла",
        "demo": "Демо-примеры",
        "input_title": "Введите текст",
        "input_label": "SMS, сообщение или транскрипт звонка:",
        "upload": "Загрузить TXT файл",
        "analyze": "🚀 Сделать анализ",
        "features": "Возможности системы",
        "result": "Результат анализа",
        "risk": "Вероятность мошенничества",
        "detected": "Найденные признаки",
        "threshold": "Порог решения",
        "model": "Модель",
        "low": "Низкий риск",
        "mid": "Подозрительно",
        "high": "Высокий риск",
        "critical": "Очень высокий риск",
        "why": "Почему система приняла такое решение?",
        "domain": "Анализ домена",
        "vector": "Числовые признаки",
        "report": "Скачать отчет",
        "history": "История проверок",
        "advice": "Совет по безопасности",
        "bad_advice": "Не сообщайте код, пароль, CVV или номер карты. Не переходите по ссылке. Свяжитесь с банком только по официальному номеру.",
        "good_advice": "Сообщение выглядит безопасным. Но если есть сомнения, проверьте через официальный источник.",
        "no_text": "Сначала введите текст.",
        "no_features": "Сильные признаки мошенничества не найдены.",
        "no_domain": "URL или домен в тексте не найден.",
        "download": "📥 Скачать TXT отчет",
        "how": "Как это работает?",
        "footer": "Прототип на основе прикладной математики и машинного обучения",
    },
    "🇬🇧 EN": {
        "title": "AI Fraud Detector",
        "subtitle": "An AI prototype for detecting fraud in SMS, messages, and call transcripts.",
        "mode": "Check mode",
        "sms": "SMS / message",
        "bank": "Banking mode",
        "call": "Call transcript",
        "file": "File analysis",
        "demo": "Demo examples",
        "input_title": "Enter text",
        "input_label": "SMS, message, or call transcript:",
        "upload": "Upload TXT file",
        "analyze": "🚀 Analyze",
        "features": "System features",
        "result": "Analysis result",
        "risk": "Fraud probability",
        "detected": "Detected features",
        "threshold": "Decision threshold",
        "model": "Model",
        "low": "Low risk",
        "mid": "Suspicious",
        "high": "High risk",
        "critical": "Critical risk",
        "why": "Why did the system make this decision?",
        "domain": "Domain analysis",
        "vector": "Numeric features",
        "report": "Download report",
        "history": "Check history",
        "advice": "Security advice",
        "bad_advice": "Do not share codes, passwords, CVV, or card numbers. Do not open suspicious links. Contact the bank only through the official number.",
        "good_advice": "The message looks safe. If unsure, verify it through official sources.",
        "no_text": "Please enter text first.",
        "no_features": "No strong fraud indicators were found.",
        "no_domain": "No URL or domain was found in the text.",
        "download": "📥 Download TXT report",
        "how": "How does it work?",
        "footer": "Applied Mathematics + Machine Learning Prototype",
    },
}

T = TEXT[lang]

# =========================
# DATA
# =========================
data = [
    ["срочно отправьте код из SMS", 1],
    ["ваша карта заблокирована перейдите по ссылке http://secure-login.xyz", 1],
    ["перейдите по ссылке http://kaspi-login.xyz", 1],
    ["назовите пароль чтобы защитить счет", 1],
    ["введите CVV и номер карты", 1],
    ["вы выиграли приз оплатите доставку", 1],
    ["ваш аккаунт будет удален срочно подтвердите вход", 1],
    ["не говорите никому и отправьте код", 1],
    ["сотрудник банка просит код из SMS", 1],
    ["ваш счет под угрозой продиктуйте пароль", 1],
    ["құттықтаймыз сіз ұтыс ұттыңыз карта деректерін енгізіңіз", 1],
    ["сіздің картаңыз бұғатталды SMS кодты жіберіңіз", 1],
    ["шұғыл түрде сілтемеге өтіп аккаунтты растаңыз", 1],
    ["банк қызметкерімін кодты айтыңыз", 1],
    ["привет как дела", 0],
    ["завтра урок математики в 9", 0],
    ["встреча в 15:00", 0],
    ["ваш заказ доставлен", 0],
    ["спасибо за покупку", 0],
    ["добрый день документы готовы", 0],
    ["сегодня тренировка в 18:00", 0],
    ["сәлем қалайсың", 0],
    ["ертең математика сабағы болады", 0],
    ["үй тапсырмасын жібердім", 0],
]

urgent_words = ["срочно", "шұғыл", "быстро", "немедленно", "қазір", "тез", "urgent", "now"]
secret_words = ["код", "пароль", "cvv", "sms", "құпия", "password"]
money_words = ["карта", "счет", "банк", "ақша", "төле", "оплата", "перевод"]
threat_words = ["заблокирована", "удален", "штраф", "угрозой", "бұғатталды", "жабылады"]
suspicious_domain_words = ["login", "verify", "secure", "bonus", "gift", "bank", "kaspi"]
suspicious_zones = [".xyz", ".top", ".click", ".site", ".online"]

# =========================
# FUNCTIONS
# =========================
def extract_urls(text):
    return re.findall(r"https?://[^\s]+|www\.[^\s]+", text.lower())

def get_domain(url):
    url = url.replace("https://", "").replace("http://", "").replace("www.", "")
    return url.split("/")[0]

def count_matches(text, words):
    text = text.lower()
    return sum(1 for w in words if w in text)

def extract_features(text):
    text = text.lower()
    urls = extract_urls(text)
    domains = [get_domain(u) for u in urls]

    suspicious_domain = 0
    long_domain = 0
    suspicious_zone = 0
    digit_domain = 0

    for d in domains:
        if any(w in d for w in suspicious_domain_words):
            suspicious_domain = 1
        if len(d) > 20:
            long_domain = 1
        if any(d.endswith(z) for z in suspicious_zones):
            suspicious_zone = 1
        if any(ch.isdigit() for ch in d):
            digit_domain = 1

    return {
        "has_link": int(len(urls) > 0),
        "urgent_count": count_matches(text, urgent_words),
        "secret_count": count_matches(text, secret_words),
        "money_count": count_matches(text, money_words),
        "threat_count": count_matches(text, threat_words),
        "suspicious_domain": suspicious_domain,
        "long_domain": long_domain,
        "suspicious_zone": suspicious_zone,
        "digit_domain": digit_domain,
        "digit_count": sum(ch.isdigit() for ch in text),
        "exclamation_count": text.count("!"),
    }, domains

def explain(features):
    if lang == "🇰🇿 KZ":
        labels = {
            "has_link": "Сілтеме анықталды",
            "urgent_count": "Шұғыл әрекетке шақыру бар",
            "secret_count": "Код / пароль / CVV сұрауы мүмкін",
            "money_count": "Банк, ақша немесе картаға қатысты сөздер бар",
            "threat_count": "Қорқыту немесе қысым жасау белгісі бар",
            "suspicious_domain": "Доменде күмәнді сөздер бар",
            "long_domain": "Домен ұзындығы күмәнді",
            "suspicious_zone": "Күмәнді домен зонасы анықталды",
            "digit_domain": "Доменде цифрлар бар",
            "digit_count": "Мәтінде көп сан немесе код кездеседі",
            "exclamation_count": "Көп леп белгісі қолданылған",
        }
    elif lang == "🇷🇺 RU":
        labels = {
            "has_link": "Обнаружена ссылка",
            "urgent_count": "Есть срочный призыв к действию",
            "secret_count": "Возможен запрос кода / пароля / CVV",
            "money_count": "Есть слова о банке, деньгах или карте",
            "threat_count": "Есть признаки давления или угрозы",
            "suspicious_domain": "В домене есть подозрительные слова",
            "long_domain": "Домен подозрительно длинный",
            "suspicious_zone": "Обнаружена подозрительная доменная зона",
            "digit_domain": "В домене есть цифры",
            "digit_count": "В тексте много чисел или кодов",
            "exclamation_count": "Используется много восклицательных знаков",
        }
    else:
        labels = {
            "has_link": "A link was detected",
            "urgent_count": "Urgent action words were found",
            "secret_count": "Possible request for code / password / CVV",
            "money_count": "Bank, money, or card-related words were found",
            "threat_count": "Pressure or threat indicators were found",
            "suspicious_domain": "Suspicious words were found in the domain",
            "long_domain": "The domain is suspiciously long",
            "suspicious_zone": "Suspicious domain zone detected",
            "digit_domain": "The domain contains numbers",
            "digit_count": "The text contains many numbers or codes",
            "exclamation_count": "Many exclamation marks were used",
        }
    return [labels[k] for k, v in features.items() if v > 0]

def risk_style(prob):
    if prob < 0.3:
        return T["low"], "risk-low", "🟢"
    if prob < 0.6:
        return T["mid"], "risk-mid", "🟡"
    if prob < 0.8:
        return T["high"], "risk-high", "🟠"
    return T["critical"], "risk-critical", "🔴"

# =========================
# TRAIN MODEL
# =========================
rows, labels = [], []
for text, label in data:
    f, _ = extract_features(text)
    rows.append(f)
    labels.append(label)

X_train = pd.DataFrame(rows)
y_train = np.array(labels)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression())
])
model.fit(X_train, y_train)

# =========================
# PREMIUM UI STYLE
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 15% 15%, rgba(37,99,235,0.17), transparent 28%),
        radial-gradient(circle at 80% 0%, rgba(20,184,166,0.18), transparent 32%),
        linear-gradient(135deg, #f8fafc 0%, #eef2ff 45%, #ecfeff 100%);
}

.block-container {
    padding-top: 1.4rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #111827 60%, #020617 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #e5e7eb !important;
}

/* Fix white text inside select boxes */
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.22) !important;
}

[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span,
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] input,
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] svg {
    color: #0f172a !important;
    fill: #0f172a !important;
}

/* Fix radio language buttons in sidebar */
[data-testid="stSidebar"] .stRadio label {
    color: #e5e7eb !important;
    font-weight: 700 !important;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    padding: 8px 10px;
    border-radius: 14px;
    margin-right: 6px;
}

[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSlider [data-testid="stTickBar"],
[data-testid="stSidebar"] .stSlider div {
    color: #e5e7eb !important;
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 38px;
    border-radius: 34px;
    background:
        linear-gradient(135deg, rgba(15,23,42,0.98), rgba(30,64,175,0.95) 52%, rgba(13,148,136,0.94)),
        url('https://images.unsplash.com/photo-1550751827-4bd374c3f58b?auto=format&fit=crop&w=1600&q=80');
    background-blend-mode: multiply;
    background-size: cover;
    color: white;
    box-shadow: 0 30px 80px rgba(15, 23, 42, 0.28);
    margin-bottom: 26px;
    border: 1px solid rgba(255,255,255,0.18);
}

.hero:before {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    right: -70px;
    top: -80px;
    background: rgba(255,255,255,0.13);
    filter: blur(3px);
    border-radius: 50%;
}

.hero-grid {
    position: relative;
    display: grid;
    grid-template-columns: 1.6fr 0.9fr;
    gap: 24px;
    align-items: center;
}

.hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 13px;
    border-radius: 999px;
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.23);
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.3px;
    margin-bottom: 16px;
}

.hero-title {
    font-size: 54px;
    line-height: 1.02;
    font-weight: 900;
    letter-spacing: -1.7px;
    margin-bottom: 14px;
}

.hero-subtitle {
    font-size: 18px;
    line-height: 1.6;
    opacity: 0.91;
    max-width: 760px;
}

.badge {
    display: inline-block;
    padding: 9px 14px;
    border-radius: 999px;
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.22);
    margin-right: 8px;
    margin-top: 17px;
    font-size: 13px;
    font-weight: 700;
    backdrop-filter: blur(8px);
}

.hero-panel {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 28px;
    padding: 22px;
    backdrop-filter: blur(16px);
}

.hero-panel-title {
    font-size: 14px;
    opacity: 0.82;
    font-weight: 700;
    margin-bottom: 8px;
}

.hero-panel-value {
    font-size: 38px;
    font-weight: 900;
    letter-spacing: -1px;
}

.hero-panel-small {
    font-size: 13px;
    opacity: 0.82;
    line-height: 1.5;
    margin-top: 8px;
}

.glass-card {
    background: rgba(255,255,255,0.84);
    border: 1px solid rgba(226,232,240,0.92);
    border-radius: 30px;
    padding: 25px;
    box-shadow: 0 20px 55px rgba(15, 23, 42, 0.09);
    margin-bottom: 19px;
    backdrop-filter: blur(18px);
}

.section-title {
    font-size: 24px;
    font-weight: 900;
    color: #0f172a;
    letter-spacing: -0.6px;
    margin-bottom: 8px;
}

.section-subtitle {
    color: #64748b;
    font-size: 14px;
    margin-bottom: 15px;
}

.feature-list p {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    padding: 12px 14px;
    border-radius: 16px;
    margin: 8px 0;
    color: #334155;
    font-weight: 650;
}

.metric-card {
    background: rgba(255,255,255,0.92);
    border-radius: 26px;
    padding: 23px 18px;
    box-shadow: 0 16px 42px rgba(15, 23, 42, 0.08);
    border: 1px solid #e5e7eb;
    text-align: center;
    transition: all .2s ease;
}

.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 22px 50px rgba(15, 23, 42, 0.12);
}

.metric-icon {
    width: 42px;
    height: 42px;
    border-radius: 15px;
    background: linear-gradient(135deg, #2563eb, #14b8a6);
    color: white;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 10px;
    font-size: 20px;
}

.metric-label {
    color: #64748b;
    font-size: 13px;
    font-weight: 750;
    text-transform: uppercase;
    letter-spacing: 0.45px;
}

.metric-value {
    color: #0f172a;
    font-size: 34px;
    font-weight: 900;
    margin-top: 5px;
}

.risk-low, .risk-mid, .risk-high, .risk-critical {
    border-radius: 30px;
    padding: 24px 26px;
    font-size: 28px;
    font-weight: 900;
    margin: 20px 0;
    border: 1px solid rgba(255,255,255,0.8);
    box-shadow: 0 18px 45px rgba(15,23,42,0.08);
}
.risk-low {background:linear-gradient(135deg,#dcfce7,#f0fdf4);color:#166534;}
.risk-mid {background:linear-gradient(135deg,#fef9c3,#fffbeb);color:#854d0e;}
.risk-high {background:linear-gradient(135deg,#ffedd5,#fff7ed);color:#9a3412;}
.risk-critical {background:linear-gradient(135deg,#fee2e2,#fff1f2);color:#991b1b;}

.feature-chip {
    display: inline-flex;
    align-items: center;
    padding: 10px 14px;
    border-radius: 999px;
    background: #eef2ff;
    color: #3730a3;
    font-weight: 750;
    margin: 6px;
    border: 1px solid #c7d2fe;
    box-shadow: 0 8px 18px rgba(55,48,163,0.08);
}

.domain-box {
    padding: 15px 16px;
    border-radius: 18px;
    background: #0f172a;
    color: #e2e8f0;
    font-family: monospace;
    margin-bottom: 10px;
    border: 1px solid #334155;
}

.stButton > button {
    border-radius: 18px !important;
    padding: 0.8rem 1rem !important;
    font-weight: 850 !important;
    border: 0 !important;
    background: linear-gradient(135deg, #2563eb, #0f766e) !important;
    color: white !important;
    box-shadow: 0 14px 30px rgba(37,99,235,0.23) !important;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 20px 42px rgba(37,99,235,0.30) !important;
}

textarea {
    border-radius: 22px !important;
    border: 1px solid #cbd5e1 !important;
    box-shadow: inset 0 1px 4px rgba(15,23,42,0.05) !important;
}

[data-testid="stTabs"] button {
    font-weight: 800;
    border-radius: 16px 16px 0 0;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    margin-top: 34px;
    padding: 18px;
}

@media (max-width: 900px) {
    .hero-grid { grid-template-columns: 1fr; }
    .hero-title { font-size: 38px; }
}
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown(f"## 🔐 {T['title']}")
    st.caption("Smart fraud detection prototype")

    selected_lang = st.radio(
        "🌍 Language / Тіл / Язык",
        LANG_OPTIONS,
        index=LANG_OPTIONS.index(st.session_state.lang),
        horizontal=True,
        key="lang_selector"
    )
    if selected_lang != st.session_state.lang:
        st.session_state.lang = selected_lang
        st.rerun()

    st.divider()

    mode = st.selectbox(
        T["mode"],
        [T["sms"], T["bank"], T["call"], T["file"]]
    )

    threshold = st.slider(T["threshold"], 0.1, 0.9, 0.5, 0.05)

    st.divider()
    st.markdown(f"### 🧪 {T['demo']}")
    demo = st.selectbox(
        T["demo"],
        ["Fraud SMS", "Fraud Call", "Safe Message", "Kazakh Fraud"]
    )

    st.divider()
    st.markdown("### 🛡️ Project stack")
    st.markdown("• Logistic Regression")
    st.markdown("• Feature Engineering")
    st.markdown("• Domain Analysis")
    st.markdown("• Explainable AI")

demo_texts = {
    "Fraud SMS": "Срочно! Ваша карта заблокирована. Отправьте код из SMS и перейдите по ссылке http://secure-login.xyz",
    "Fraud Call": "Здравствуйте, я сотрудник банка. Назовите код из SMS, чтобы мы защитили ваш счет.",
    "Safe Message": "Привет, завтра урок математики в 9:00. Не забудь тетрадь.",
    "Kazakh Fraud": "Сіздің картаңыз бұғатталды. Шұғыл түрде SMS кодты жіберіңіз және http://kaspi-login.xyz сайтына кіріңіз."
}

# =========================
# HERO
# =========================
st.markdown(f"""
<div class="hero">
    <div class="hero-grid">
        <div>
            <div class="hero-kicker">⚡ AI-powered safety scanner</div>
            <div class="hero-title">🔐 {T['title']}</div>
            <div class="hero-subtitle">{T['subtitle']}</div>
            <span class="badge">Logistic Regression</span>
            <span class="badge">Domain Analysis</span>
            <span class="badge">Explainable AI</span>
            <span class="badge">Risk Report</span>
        </div>
        <div class="hero-panel">
            <div class="hero-panel-title">Prototype readiness</div>
            <div class="hero-panel-value">Demo-ready</div>
            <div class="hero-panel-small">Analyzes text, links, pressure words, secret-code requests and suspicious domains.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# INPUT
# =========================
left, right = st.columns([2.1, 0.9], gap="large")

with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">✍️ {T["input_title"]}</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Paste a suspicious message or call transcript and check the risk level.</div>', unsafe_allow_html=True)

    uploaded = None
    if mode == T["file"]:
        uploaded = st.file_uploader(T["upload"], type=["txt"])

    if uploaded:
        input_text = uploaded.read().decode("utf-8", errors="ignore")
        st.success("File uploaded successfully")
    else:
        input_text = st.text_area(
            T["input_label"],
            value=demo_texts[demo],
            height=210
        )

    analyze = st.button(T["analyze"], use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown(f"""
    <div class="glass-card feature-list">
        <div class="section-title">⚙️ {T['features']}</div>
        <p>🧾 Text analysis</p>
        <p>🌐 Domain analysis</p>
        <p>🧠 Logistic regression</p>
        <p>🔍 Explainable result</p>
        <p>📥 Downloadable report</p>
    </div>
    """, unsafe_allow_html=True)

# =========================
# HISTORY
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# ANALYSIS
# =========================
if analyze:
    if not input_text.strip():
        st.warning(T["no_text"])
    else:
        features, domains = extract_features(input_text)
        X_input = pd.DataFrame([features])
        prob = model.predict_proba(X_input)[0][1]
        pred = int(prob >= threshold)
        risk_label, risk_class, emoji = risk_style(prob)
        explanations = explain(features)

        st.markdown(f'<div class="section-title">📊 {T["result"]}</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="metric-icon">%</div><div class="metric-label">{T["risk"]}</div><div class="metric-value">{prob*100:.1f}%</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-icon">⚠️</div><div class="metric-label">{T["detected"]}</div><div class="metric-value">{len(explanations)}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-icon">🎚️</div><div class="metric-label">{T["threshold"]}</div><div class="metric-value">{threshold:.2f}</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="metric-icon">🧠</div><div class="metric-label">{T["model"]}</div><div class="metric-value">LR</div></div>', unsafe_allow_html=True)

        st.markdown(f'<div class="{risk_class}">{emoji} {risk_label}</div>', unsafe_allow_html=True)
        st.progress(float(prob))

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Explain AI",
            f"🌐 {T['domain']}",
            f"🧠 {T['vector']}",
            f"📥 {T['report']}",
            f"📜 {T['history']}"
        ])

        with tab1:
            st.subheader(T["why"])
            if explanations:
                chips = "".join([f'<span class="feature-chip">{e}</span>' for e in explanations])
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.success(T["no_features"])

            st.subheader("Feature contribution")
            coef = model.named_steps["clf"].coef_[0]
            contrib = []
            for name, value, w in zip(X_train.columns, X_input.iloc[0], coef):
                contrib.append([name, value, round(w, 3), round(value * w, 3)])

            contrib_df = pd.DataFrame(contrib, columns=["Feature", "Value", "Weight", "Contribution"])
            st.dataframe(contrib_df.sort_values("Contribution", ascending=False), use_container_width=True)

            st.subheader(T["advice"])
            if pred == 1:
                st.error(T["bad_advice"])
            else:
                st.success(T["good_advice"])

        with tab2:
            st.subheader(T["domain"])
            if domains:
                for d in domains:
                    st.markdown(f'<div class="domain-box">🌐 {d}</div>', unsafe_allow_html=True)
            else:
                st.info(T["no_domain"])

        with tab3:
            st.subheader(T["vector"])
            st.dataframe(X_input, use_container_width=True)

        with tab4:
            report = f"""
AI Fraud Detector Report
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Input:
{input_text}

Fraud risk: {prob*100:.1f}%
Risk level: {risk_label}
Threshold: {threshold}

Detected features:
{chr(10).join("- " + e for e in explanations) if explanations else "No strong fraud indicators"}

Domains:
{chr(10).join(domains) if domains else "No domains"}

Advice:
{T["bad_advice"] if pred == 1 else T["good_advice"]}
"""
            st.download_button(
                T["download"],
                report,
                file_name="fraud_analysis_report.txt",
                use_container_width=True
            )

        st.session_state.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Risk %": round(prob * 100, 1),
            "Level": risk_label,
            "Text": input_text[:70]
        })

        with tab5:
            if st.session_state.history:
                st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
            else:
                st.write("No history")

with st.expander(f"📘 {T['how']}"):
    if lang == "🇰🇿 KZ":
        st.write("1. Мәтіннен белгілер алынады. 2. Олар сандық векторға айналады. 3. Логистикалық регрессия ықтималдық есептейді. 4. Сайт нәтиже мен кеңес береді.")
    elif lang == "🇷🇺 RU":
        st.write("1. Из текста извлекаются признаки. 2. Они превращаются в числовой вектор. 3. Логистическая регрессия считает вероятность. 4. Сайт показывает результат и совет.")
    else:
        st.write("1. Features are extracted from the text. 2. They are converted into a numeric vector. 3. Logistic regression calculates probability. 4. The site shows the result and advice.")

st.markdown(f'<div class="footer">{T["footer"]}</div>', unsafe_allow_html=True)
