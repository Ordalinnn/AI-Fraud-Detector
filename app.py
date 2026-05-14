```python
import re
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="AI Fraud Detector",
    page_icon="🔐",
    layout="wide"
)

# =========================================
# LANGUAGE
# =========================================
LANG_OPTIONS = ["🇷🇺 RU", "🇬🇧 EN"]

if "lang" not in st.session_state:
    st.session_state.lang = "🇷🇺 RU"

lang = st.session_state.lang

TEXT = {
    "🇷🇺 RU": {
        "title": "AI Fraud Detector",
        "subtitle": "AI-система для обнаружения мошенничества в SMS, сообщениях и звонках.",
        "input": "Введите сообщение",
        "analyze": "Проверить",
        "risk": "Вероятность мошенничества",
        "features": "Обнаруженные признаки",
        "safe": "Низкий риск",
        "mid": "Подозрительно",
        "high": "Высокий риск",
        "critical": "Критический риск",
        "history": "История",
        "domains": "Домены",
        "vector": "Признаки",
        "no_domain": "Ссылок не найдено",
        "advice": "Совет безопасности",
        "bad_advice": "Не сообщайте коды, CVV и пароли. Не переходите по подозрительным ссылкам.",
        "good_advice": "Сообщение выглядит безопасным.",
    },

    "🇬🇧 EN": {
        "title": "AI Fraud Detector",
        "subtitle": "AI system for detecting fraud in SMS, messages and call transcripts.",
        "input": "Enter message",
        "analyze": "Analyze",
        "risk": "Fraud probability",
        "features": "Detected features",
        "safe": "Low risk",
        "mid": "Suspicious",
        "high": "High risk",
        "critical": "Critical risk",
        "history": "History",
        "domains": "Domains",
        "vector": "Features",
        "no_domain": "No links found",
        "advice": "Security advice",
        "bad_advice": "Do not share codes, CVV or passwords. Avoid suspicious links.",
        "good_advice": "The message looks safe.",
    }
}

T = TEXT[lang]

# =========================================
# TRAINING DATA
# =========================================
data = [
    ["срочно отправьте код из sms", 1],
    ["ваша карта заблокирована перейдите по ссылке", 1],
    ["введите cvv и номер карты", 1],
    ["подтвердите пароль", 1],
    ["your account is blocked verify now", 1],
    ["send otp code immediately", 1],
    ["transfer money to safe account", 1],

    ["привет как дела", 0],
    ["встреча завтра в 9", 0],
    ["спасибо за покупку", 0],
    ["your order has been delivered", 0],
    ["meeting tomorrow at school", 0],
]

urgent_words = [
    "срочно",
    "urgent",
    "immediately",
    "быстро"
]

secret_words = [
    "код",
    "пароль",
    "cvv",
    "otp",
    "password"
]

money_words = [
    "карта",
    "банк",
    "деньги",
    "money",
    "payment",
    "account"
]

threat_words = [
    "заблокирована",
    "blocked",
    "suspended",
    "удален"
]

suspicious_zones = [
    ".xyz",
    ".top",
    ".site",
    ".click"
]

# =========================================
# FUNCTIONS
# =========================================
def extract_urls(text):
    return re.findall(r"https?://[^\s]+|www\.[^\s]+", text.lower())

def get_domain(url):
    url = url.replace("https://", "").replace("http://", "").replace("www.", "")
    return url.split("/")[0]

def count_matches(text, words):
    return sum(1 for w in words if w in text)

def extract_features(text):
    text = text.lower()

    urls = extract_urls(text)
    domains = [get_domain(u) for u in urls]

    suspicious_zone = 0

    for d in domains:
        if any(d.endswith(z) for z in suspicious_zones):
            suspicious_zone = 1

    features = {
        "has_link": int(len(urls) > 0),
        "urgent_count": count_matches(text, urgent_words),
        "secret_count": count_matches(text, secret_words),
        "money_count": count_matches(text, money_words),
        "threat_count": count_matches(text, threat_words),
        "suspicious_zone": suspicious_zone,
        "digit_count": sum(ch.isdigit() for ch in text),
        "text_length": len(text),
        "word_count": len(text.split())
    }

    return features, domains

def explain(features):
    labels = {
        "has_link": "Suspicious link",
        "urgent_count": "Urgency detected",
        "secret_count": "Password/code request",
        "money_count": "Bank or money words",
        "threat_count": "Threat detected",
        "suspicious_zone": "Suspicious domain zone",
        "digit_count": "Many numbers detected",
    }

    return [
        labels[k]
        for k, v in features.items()
        if v > 0 and k in labels
    ]

def risk_style(prob):
    if prob < 0.3:
        return T["safe"], "risk-low"

    if prob < 0.6:
        return T["mid"], "risk-mid"

    if prob < 0.8:
        return T["high"], "risk-high"

    return T["critical"], "risk-critical"

# =========================================
# TRAIN MODEL
# =========================================
rows = []
labels = []

for text, label in data:
    f, _ = extract_features(text)
    rows.append(f)
    labels.append(label)

X = pd.DataFrame(rows)
y = np.array(labels)

lr_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression())
])

rf_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier())
])

gb_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", GradientBoostingClassifier())
])

lr_model.fit(X, y)
rf_model.fit(X, y)
gb_model.fit(X, y)

# =========================================
# MINIMAL CSS
# =========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f5f7fb;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: white;
    border-right: 1px solid #e5e7eb;
}

.hero {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 24px;
    padding: 34px;
    margin-bottom: 24px;
}

.hero-title {
    font-size: 42px;
    font-weight: 700;
    color: #111827;
    margin-bottom: 10px;
}

.hero-subtitle {
    color: #6b7280;
    font-size: 16px;
    line-height: 1.6;
    max-width: 700px;
}

.card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 20px;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 18px;
    color: #111827;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(4,1fr);
    gap: 14px;
    margin-bottom: 20px;
}

.metric-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 20px;
}

.metric-label {
    color: #6b7280;
    font-size: 13px;
}

.metric-value {
    font-size: 30px;
    font-weight: 700;
    margin-top: 8px;
    color: #111827;
}

.stButton > button {
    background: #111827 !important;
    color: white !important;
    border-radius: 14px !important;
    border: none !important;
    height: 50px;
    font-weight: 600 !important;
}

textarea {
    border-radius: 16px !important;
    border: 1px solid #d1d5db !important;
}

.risk-low,
.risk-mid,
.risk-high,
.risk-critical {
    border-radius: 18px;
    padding: 18px 22px;
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 18px;
}

.risk-low {
    background: #ecfdf5;
    color: #065f46;
}

.risk-mid {
    background: #fefce8;
    color: #854d0e;
}

.risk-high {
    background: #fff7ed;
    color: #c2410c;
}

.risk-critical {
    background: #fef2f2;
    color: #991b1b;
}

.feature-chip {
    display: inline-block;
    padding: 8px 12px;
    border-radius: 999px;
    background: #f3f4f6;
    color: #374151;
    margin: 4px;
    font-size: 13px;
}

.domain-box {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 14px;
    margin-bottom: 10px;
    font-family: monospace;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# =========================================
# SIDEBAR
# =========================================
with st.sidebar:
    st.markdown("## 🔐 AI Fraud Detector")

    selected_lang = st.radio(
        "Language",
        LANG_OPTIONS,
        horizontal=True
    )

    if selected_lang != st.session_state.lang:
        st.session_state.lang = selected_lang
        st.rerun()

    st.divider()

    threshold = st.slider(
        "Threshold",
        0.1,
        0.9,
        0.5,
        0.05
    )

# =========================================
# HERO
# =========================================
st.markdown(f"""
<div class="hero">
    <div class="hero-title">🔐 AI Fraud Detector</div>
    <div class="hero-subtitle">
        {T["subtitle"]}
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================
# INPUT
# =========================================
st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown(
    f'<div class="section-title">{T["input"]}</div>',
    unsafe_allow_html=True
)

input_text = st.text_area(
    "",
    height=180,
    placeholder="Paste suspicious message here..."
)

analyze = st.button(
    T["analyze"],
    use_container_width=True
)

st.markdown('</div>', unsafe_allow_html=True)

# =========================================
# HISTORY
# =========================================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================================
# ANALYSIS
# =========================================
if analyze and input_text.strip():

    features, domains = extract_features(input_text)

    X_input = pd.DataFrame([features])

    lr_prob = lr_model.predict_proba(X_input)[0][1]
    rf_prob = rf_model.predict_proba(X_input)[0][1]
    gb_prob = gb_model.predict_proba(X_input)[0][1]

    prob = (lr_prob + rf_prob + gb_prob) / 3

    risk_label, risk_class = risk_style(prob)

    explanations = explain(features)

    # ======================
    # METRICS
    # ======================
    st.markdown("""
    <div class="metric-grid">
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    c1.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{T["risk"]}</div>
        <div class="metric-value">{prob*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    c2.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{T["features"]}</div>
        <div class="metric-value">{len(explanations)}</div>
    </div>
    """, unsafe_allow_html=True)

    c3.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Threshold</div>
        <div class="metric-value">{threshold}</div>
    </div>
    """, unsafe_allow_html=True)

    c4.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Models</div>
        <div class="metric-value">3</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="{risk_class}">
        ⚠️ {risk_label}
    </div>
    """, unsafe_allow_html=True)

    st.progress(float(prob))

    # ======================
    # TABS
    # ======================
    tab1, tab2, tab3, tab4 = st.tabs([
        "Analysis",
        "Domains",
        "Features",
        T["history"]
    ])

    # ======================
    # TAB 1
    # ======================
    with tab1:

        st.subheader(T["features"])

        if explanations:
            chips = "".join([
                f'<span class="feature-chip">{e}</span>'
                for e in explanations
            ])

            st.markdown(chips, unsafe_allow_html=True)

        else:
            st.success("No strong fraud indicators")

        st.subheader(T["advice"])

        if prob >= threshold:
            st.error(T["bad_advice"])
        else:
            st.success(T["good_advice"])

    # ======================
    # TAB 2
    # ======================
    with tab2:

        if domains:

            for d in domains:

                st.markdown(f"""
                <div class="domain-box">
                    🌐 {d}
                </div>
                """, unsafe_allow_html=True)

        else:
            st.info(T["no_domain"])

    # ======================
    # TAB 3
    # ======================
    with tab3:

        feature_df = pd.DataFrame({
            "Feature": list(features.keys()),
            "Value": list(features.values())
        })

        st.dataframe(
            feature_df,
            use_container_width=True,
            hide_index=True
        )

    # ======================
    # SAVE HISTORY
    # ======================
    st.session_state.history.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Risk": f"{prob*100:.1f}%",
        "Result": risk_label,
        "Text": input_text[:50]
    })

    # ======================
    # TAB 4
    # ======================
    with tab4:

        if st.session_state.history:

            st.dataframe(
                pd.DataFrame(st.session_state.history),
                use_container_width=True,
                hide_index=True
            )

# =========================================
# FOOTER
# =========================================
st.markdown("""
<div style="
text-align:center;
margin-top:40px;
color:#6b7280;
font-size:13px;
">
AI Fraud Detector · Minimal UI
</div>
""", unsafe_allow_html=True)
```
