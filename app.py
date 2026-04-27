import re
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    return [labels[k] for k, v in features.items() if v > 0]

def risk_style(prob):
    if prob < 0.3:
        return "LOW RISK", "Қауіп төмен", "risk-low", "🟢"
    if prob < 0.6:
        return "SUSPICIOUS", "Күмәнді", "risk-mid", "🟡"
    if prob < 0.8:
        return "HIGH RISK", "Жоғары қауіп", "risk-high", "🟠"
    return "CRITICAL", "Өте жоғары қауіп", "risk-critical", "🔴"

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
# PAGE CONFIG + STYLE
# =========================
st.set_page_config(
    page_title="AI Fraud Detector",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #eef2ff 0%, #f8fafc 45%, #ecfeff 100%);
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.hero {
    padding: 34px;
    border-radius: 28px;
    background: linear-gradient(135deg, #111827 0%, #1e3a8a 55%, #0f766e 100%);
    color: white;
    box-shadow: 0 22px 55px rgba(15, 23, 42, 0.22);
    margin-bottom: 24px;
}

.hero-title {
    font-size: 48px;
    font-weight: 800;
    margin-bottom: 8px;
    letter-spacing: -1px;
}

.hero-subtitle {
    font-size: 18px;
    opacity: 0.88;
    max-width: 900px;
}

.badge {
    display: inline-block;
    padding: 8px 14px;
    border-radius: 999px;
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.24);
    margin-right: 8px;
    margin-top: 16px;
    font-size: 14px;
}

.glass-card {
    background: rgba(255,255,255,0.78);
    border: 1px solid rgba(226,232,240,0.9);
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
    backdrop-filter: blur(12px);
    margin-bottom: 18px;
}

.metric-card {
    background: white;
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 14px 35px rgba(15, 23, 42, 0.08);
    border: 1px solid #e5e7eb;
    text-align: center;
}

.metric-label {
    color: #64748b;
    font-size: 14px;
    font-weight: 600;
}

.metric-value {
    color: #0f172a;
    font-size: 34px;
    font-weight: 800;
    margin-top: 6px;
}

.risk-low, .risk-mid, .risk-high, .risk-critical {
    border-radius: 24px;
    padding: 26px;
    font-size: 28px;
    font-weight: 800;
    margin: 18px 0;
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.10);
}

.risk-low {
    background: linear-gradient(135deg, #dcfce7, #bbf7d0);
    color: #166534;
    border: 1px solid #86efac;
}

.risk-mid {
    background: linear-gradient(135deg, #fef9c3, #fde68a);
    color: #854d0e;
    border: 1px solid #facc15;
}

.risk-high {
    background: linear-gradient(135deg, #ffedd5, #fed7aa);
    color: #9a3412;
    border: 1px solid #fb923c;
}

.risk-critical {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    color: #991b1b;
    border: 1px solid #f87171;
}

.feature-chip {
    display: inline-block;
    padding: 10px 14px;
    border-radius: 14px;
    background: #eef2ff;
    color: #3730a3;
    font-weight: 600;
    margin: 5px;
    border: 1px solid #c7d2fe;
}

.domain-box {
    padding: 14px;
    border-radius: 16px;
    background: #0f172a;
    color: #e2e8f0;
    font-family: monospace;
    margin-bottom: 8px;
}

.advice-box {
    padding: 22px;
    border-radius: 22px;
    background: white;
    border-left: 7px solid #ef4444;
    box-shadow: 0 14px 35px rgba(15, 23, 42, 0.08);
}

.small-muted {
    color: #64748b;
    font-size: 14px;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## 🔐 AI Fraud Detector")
    st.write("Жоба режимдері")

    mode = st.selectbox(
        "Тексеру режимі",
        ["SMS / хабарлама", "Банковский режим", "Қоңырау транскрипті", "Файл анализі"]
    )

    threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)

    st.divider()
    st.markdown("### 🧪 Демо мысалдар")
    demo = st.selectbox(
        "Мысал таңдаңыз",
        [
            "Fraud SMS",
            "Fraud Call",
            "Safe Message",
            "Kazakh Fraud",
        ]
    )

    demo_texts = {
        "Fraud SMS": "Срочно! Ваша карта заблокирована. Отправьте код из SMS и перейдите по ссылке http://secure-login.xyz",
        "Fraud Call": "Здравствуйте, я сотрудник банка. Назовите код из SMS, чтобы мы защитили ваш счет.",
        "Safe Message": "Привет, завтра урок математики в 9:00. Не забудь тетрадь.",
        "Kazakh Fraud": "Сіздің картаңыз бұғатталды. Шұғыл түрде SMS кодты жіберіңіз және http://kaspi-login.xyz сайтына кіріңіз."
    }

    st.divider()
    st.markdown("### 📌 Модель")
    st.write("Algorithm: Logistic Regression")
    st.write("Input: text features")
    st.write("Output: fraud probability")

# =========================
# HERO
# =========================
st.markdown("""
<div class="hero">
    <div class="hero-title">AI Fraud Detector</div>
    <div class="hero-subtitle">
        SMS, интернет-хабарлама және қоңырау транскриптіндегі алаяқтық белгілерін анықтайтын
        логистикалық регрессияға негізделген AI-прототип.
    </div>
    <span class="badge">Logistic Regression</span>
    <span class="badge">Domain Analysis</span>
    <span class="badge">Explainable AI</span>
    <span class="badge">Risk Report</span>
</div>
""", unsafe_allow_html=True)

# =========================
# INPUT SECTION
# =========================
left, right = st.columns([2.1, 1])

with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("✍️ Мәтінді енгізіңіз")

    default_text = demo_texts[demo]

    uploaded = None
    if mode == "Файл анализі":
        uploaded = st.file_uploader("TXT файл жүктеу", type=["txt"])

    if uploaded:
        input_text = uploaded.read().decode("utf-8", errors="ignore")
        st.success("Файл сәтті жүктелді.")
    else:
        input_text = st.text_area(
            "SMS, хабарлама немесе қоңырау транскрипті:",
            value=default_text,
            height=190
        )

    analyze = st.button("🚀 Анализ жасау", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown("""
    <div class="glass-card">
        <h3>⚙️ Жүйе мүмкіндіктері</h3>
        <p>✅ мәтіннен белгілерді шығарады</p>
        <p>✅ доменді тексереді</p>
        <p>✅ fraud risk пайызын есептейді</p>
        <p>✅ шешімді түсіндіреді</p>
        <p>✅ есепті жүктеуге мүмкіндік береді</p>
    </div>
    """, unsafe_allow_html=True)

# =========================
# ANALYSIS
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

if analyze:
    if not input_text.strip():
        st.warning("Алдымен мәтін енгізіңіз.")
    else:
        features, domains = extract_features(input_text)
        X_input = pd.DataFrame([features])
        prob = model.predict_proba(X_input)[0][1]
        pred = int(prob >= threshold)
        risk_en, risk_kz, risk_class, emoji = risk_style(prob)
        explanations = explain(features)

        st.markdown("## 📊 Анализ нәтижесі")

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Fraud Risk</div>
            <div class="metric-value">{prob*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        m2.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Detected Features</div>
            <div class="metric-value">{len(explanations)}</div>
        </div>
        """, unsafe_allow_html=True)

        m3.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Threshold</div>
            <div class="metric-value">{threshold:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        m4.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Model</div>
            <div class="metric-value">LR</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="{risk_class}">
            {emoji} {risk_en} — {risk_kz}
        </div>
        """, unsafe_allow_html=True)

        st.progress(float(prob))

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Explain AI",
            "🌐 Domain",
            "🧠 Feature Vector",
            "📥 Report",
            "📜 History"
        ])

        with tab1:
            st.subheader("Неге жүйе осындай шешім шығарды?")
            if explanations:
                chips = "".join([f'<span class="feature-chip">{e}</span>' for e in explanations])
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.success("Күшті алаяқтық белгілері табылмады.")

            st.divider()
            st.subheader("Белгілердің модельге әсері")
            coef = model.named_steps["clf"].coef_[0]
            contrib = []
            for name, value, w in zip(X_train.columns, X_input.iloc[0], coef):
                contrib.append([name, value, round(w, 3), round(value * w, 3)])

            contrib_df = pd.DataFrame(contrib, columns=["Feature", "Value", "Weight", "Contribution"])
            st.dataframe(contrib_df.sort_values("Contribution", ascending=False), use_container_width=True)

            st.subheader("Қауіпсіздік кеңесі")
            if pred == 1:
                st.markdown("""
                <div class="advice-box">
                    <b>Қауіп жоғары.</b><br>
                    Код, пароль, CVV немесе карта нөмірін бермеңіз.
                    Сілтемеге өтпеңіз. Банкке тек ресми нөмір арқылы хабарласыңыз.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success("Хабарлама қауіпсіз көрінеді. Бірақ күмән болса, ресми дереккөз арқылы тексеріңіз.")

        with tab2:
            st.subheader("Домен анализі")
            if domains:
                for d in domains:
                    st.markdown(f'<div class="domain-box">{d}</div>', unsafe_allow_html=True)

                    domain_warnings = []
                    if len(d) > 20:
                        domain_warnings.append("Домен ұзын")
                    if any(w in d for w in suspicious_domain_words):
                        domain_warnings.append("Күмәнді сөздер бар")
                    if any(d.endswith(z) for z in suspicious_zones):
                        domain_warnings.append("Күмәнді зона")
                    if any(ch.isdigit() for ch in d):
                        domain_warnings.append("Цифрлар бар")

                    if domain_warnings:
                        for w in domain_warnings:
                            st.warning(w)
                    else:
                        st.success("Доменде айқын қауіп белгісі табылмады.")
            else:
                st.info("Мәтінде URL немесе домен табылмады.")

        with tab3:
            st.subheader("Модельге берілген сандық белгілер")
            st.dataframe(X_input, use_container_width=True)

            st.subheader("Оқыту деректерінің қысқаша көрінісі")
            st.dataframe(pd.DataFrame(data, columns=["Text", "Label"]).head(10), use_container_width=True)

        with tab4:
            report = f"""
AI Fraud Detector Report
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Input:
{input_text}

Fraud risk: {prob*100:.1f}%
Risk level: {risk_kz}
Threshold: {threshold}

Detected features:
{chr(10).join("- " + e for e in explanations) if explanations else "No strong fraud indicators"}

Domains:
{chr(10).join(domains) if domains else "No domains"}

Advice:
{"Do not share SMS codes, passwords, CVV or card data. Do not open suspicious links." if pred == 1 else "Looks relatively safe, but verify through official sources if unsure."}
"""
            st.download_button(
                "📥 TXT есепті жүктеу",
                report,
                file_name="fraud_analysis_report.txt",
                use_container_width=True
            )

        st.session_state.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Risk %": round(prob * 100, 1),
            "Level": risk_kz,
            "Text": input_text[:70]
        })

        with tab5:
            if st.session_state.history:
                st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
            else:
                st.write("Әзірге тарих жоқ.")

st.markdown("""
<div class="footer">
    AI Fraud Detector • Applied Mathematics + Machine Learning Prototype
</div>
""", unsafe_allow_html=True)
