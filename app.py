# Premium Fraud Detector — Full Improved Streamlit Code

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

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Premium Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# LANGUAGE
# =====================================================
LANG = {
    "EN": {
        "title": "Premium Fraud Detector",
        "subtitle": "AI-powered scam analysis for SMS, links, calls, and suspicious messages.",
        "analyze": "Analyze Message",
        "input": "Enter suspicious message",
        "risk": "Fraud Risk",
        "safe": "Low Risk",
        "mid": "Suspicious",
        "high": "High Risk",
        "critical": "Critical",
        "features": "Detected Indicators",
        "domains": "Domain Analysis",
        "history": "History",
        "clear": "Clear History",
    }
}

T = LANG["EN"]

# =====================================================
# TRAINING DATA
# =====================================================
data = [
    ["your account is blocked send OTP immediately", 1],
    ["verify your card using secure-login.xyz", 1],
    ["urgent transfer money to safe account", 1],
    ["claim your prize enter card details", 1],
    ["hello see you tomorrow", 0],
    ["meeting moved to friday", 0],
    ["your order has arrived", 0],
    ["thank you for your payment", 0],
]

urgent_words = [
    "urgent", "immediately", "now", "asap", "срочно"
]

secret_words = [
    "otp", "password", "pin", "cvv", "code"
]

money_words = [
    "bank", "money", "card", "payment", "account"
]

threat_words = [
    "blocked", "suspended", "deleted"
]

suspicious_zones = [".xyz", ".top", ".site", ".click"]

suspicious_domain_words = [
    "secure", "verify", "login", "bank", "wallet"
]

# =====================================================
# HELPERS
# =====================================================
def extract_urls(text):
    return re.findall(r"https?://[^\s]+|www\.[^\s]+", text.lower())


def get_domain(url):
    url = url.replace("https://", "").replace("http://", "")
    return url.split("/")[0]


def count_matches(text, words):
    return sum(1 for w in words if w in text)


def extract_features(text):
    text = text.lower()

    urls = extract_urls(text)
    domains = [get_domain(u) for u in urls]

    suspicious_domain = 0
    suspicious_zone = 0

    for d in domains:
        if any(x in d for x in suspicious_domain_words):
            suspicious_domain = 1

        if any(d.endswith(z) for z in suspicious_zones):
            suspicious_zone = 1

    features = {
        "has_link": int(len(urls) > 0),
        "urgent_count": count_matches(text, urgent_words),
        "secret_count": count_matches(text, secret_words),
        "money_count": count_matches(text, money_words),
        "threat_count": count_matches(text, threat_words),
        "suspicious_domain": suspicious_domain,
        "suspicious_zone": suspicious_zone,
        "digit_count": sum(c.isdigit() for c in text),
        "length": len(text),
    }

    return features, domains


# =====================================================
# TRAIN MODEL
# =====================================================
@st.cache_resource

def train_model():
    rows = []
    labels = []

    for text, label in data:
        f, _ = extract_features(text)
        rows.append(f)
        labels.append(label)

    X = pd.DataFrame(rows)
    y = np.array(labels)

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression())
    ])

    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100))
    ])

    gb = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier())
    ])

    lr.fit(X, y)
    rf.fit(X, y)
    gb.fit(X, y)

    return lr, rf, gb


lr_model, rf_model, gb_model = train_model()

# =====================================================
# PREMIUM CSS
# =====================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(59,130,246,0.12), transparent 30%),
        radial-gradient(circle at bottom right, rgba(168,85,247,0.12), transparent 30%),
        linear-gradient(135deg,#0b0f14 0%,#111827 100%);

    color: white;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg,#05070a 0%,#0f172a 100%);

    border-right: 1px solid rgba(255,255,255,0.05);
}

.card {
    background: rgba(17,24,39,0.72);

    border: 1px solid rgba(255,255,255,0.06);

    border-radius: 24px;

    padding: 26px;

    backdrop-filter: blur(18px);

    box-shadow: 0 12px 40px rgba(0,0,0,0.25);

    margin-bottom: 18px;

    transition: all .2s ease;
}

.card:hover {
    transform: translateY(-2px);
    border-color: rgba(96,165,250,0.25);
}

.page-title {
    font-size: 60px;
    font-weight: 700;
    color: white;
    letter-spacing: -3px;
}

.page-subtitle {
    color: #94a3b8;
    font-size: 16px;
    margin-top: 10px;
    line-height: 1.7;
}

.top-banner {
    background:
        linear-gradient(
            135deg,
            rgba(59,130,246,0.18),
            rgba(168,85,247,0.14)
        );

    border: 1px solid rgba(255,255,255,0.08);

    border-radius: 22px;

    padding: 22px 28px;

    margin: 24px 0;

    backdrop-filter: blur(20px);
}

.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 20px;
}

.stat-card {
    background: rgba(255,255,255,0.04);

    border: 1px solid rgba(255,255,255,0.06);

    border-radius: 18px;

    padding: 20px;
}

.stat-label {
    color: #94a3b8;
    font-size: 12px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.stat-value {
    font-size: 36px;
    color: white;
    font-weight: 700;
}

.risk-track {
    height: 10px;
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    overflow: hidden;
    margin-top: 16px;
}

.risk-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg,#ef4444,#dc2626);
}

textarea {
    background: rgba(255,255,255,0.04) !important;

    border: 1px solid rgba(255,255,255,0.08) !important;

    border-radius: 18px !important;

    color: white !important;

    font-size: 15px !important;
}

.stButton > button {
    background:
        linear-gradient(135deg,#2563eb,#7c3aed) !important;

    color: white !important;

    border: none !important;

    border-radius: 14px !important;

    padding: .9rem 1.4rem !important;

    font-weight: 600 !important;

    width: 100%;
}

.chip {
    display: inline-block;

    padding: 8px 14px;

    margin: 4px;

    border-radius: 999px;

    background: rgba(96,165,250,0.12);

    border: 1px solid rgba(96,165,250,0.18);

    color: #bfdbfe;

    font-size: 12px;
}

.footer {
    text-align: center;
    margin-top: 50px;
    color: #64748b;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:

    st.markdown("""
    <div class="card">
        <h2 style="margin:0;color:white;">🛡️ Fraud Detector</h2>
        <p style="color:#94a3b8;font-size:13px;margin-top:10px;">
            AI-powered cybersecurity prototype
        </p>
    </div>
    """, unsafe_allow_html=True)

    threshold = st.slider(
        "Detection Threshold",
        0.1,
        0.9,
        0.5,
        0.05
    )

    st.markdown("### Features")

    features = [
        "🔍 Text analysis",
        "🌐 Domain inspection",
        "🧠 Ensemble ML",
        "📊 Explainable AI",
        "⚠ Scam simulation",
        "📈 Live metrics"
    ]

    for f in features:
        st.markdown(f"- {f}")

# =====================================================
# HEADER
# =====================================================
st.markdown(
    f"""
    <div class="page-title">
        Premium Fraud Detector
    </div>

    <div class="page-subtitle">
        {T['subtitle']}
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<div class="top-banner">
    <div style="font-size:12px;color:#93c5fd;text-transform:uppercase;letter-spacing:1px;">
        Live AI Threat Analysis
    </div>

    <div style="font-size:24px;font-weight:700;color:white;margin-top:8px;">
        Real-time fraud detection powered by ensemble machine learning
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# INPUT
# =====================================================
left, right = st.columns([2.2, 1], gap="large")

with left:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    input_text = st.text_area(
        T["input"],
        height=200,
        value="Urgent! Your account is blocked. Verify now using secure-login.xyz"
    )

    analyze = st.button(T["analyze"])

    st.markdown('</div>', unsafe_allow_html=True)

with right:

    st.markdown("""
    <div class="card">

    <div style="font-size:12px;color:#60a5fa;text-transform:uppercase;margin-bottom:14px;">
        Security Stack
    </div>

    <div style="padding:10px 0;color:#cbd5e1;">🔍 NLP & Pattern Analysis</div>
    <div style="padding:10px 0;color:#cbd5e1;">🌐 Domain Intelligence</div>
    <div style="padding:10px 0;color:#cbd5e1;">🧠 Logistic Regression</div>
    <div style="padding:10px 0;color:#cbd5e1;">🌲 Random Forest</div>
    <div style="padding:10px 0;color:#cbd5e1;">⚡ Gradient Boosting</div>
    <div style="padding:10px 0;color:#cbd5e1;">📊 Explainable AI Output</div>

    </div>
    """, unsafe_allow_html=True)

# =====================================================
# HISTORY
# =====================================================
if "history" not in st.session_state:
    st.session_state.history = []

# =====================================================
# ANALYSIS
# =====================================================
if analyze:

    features, domains = extract_features(input_text)

    X_input = pd.DataFrame([features])

    lr_prob = lr_model.predict_proba(X_input)[0][1]
    rf_prob = rf_model.predict_proba(X_input)[0][1]
    gb_prob = gb_model.predict_proba(X_input)[0][1]

    prob = (lr_prob + rf_prob + gb_prob) / 3

    if features["has_link"] and features["secret_count"]:
        prob += 0.12

    prob = min(prob, 0.99)

    risk_pct = prob * 100

    if prob < 0.3:
        level = T["safe"]
    elif prob < 0.6:
        level = T["mid"]
    elif prob < 0.8:
        level = T["high"]
    else:
        level = T["critical"]

    # =====================
    # STATS
    # =====================

    st.markdown(f"""
    <div class="stat-grid">

        <div class="stat-card">
            <div class="stat-label">Fraud Risk</div>
            <div class="stat-value">{risk_pct:.1f}%</div>
        </div>

        <div class="stat-card">
            <div class="stat-label">Domains</div>
            <div class="stat-value">{len(domains)}</div>
        </div>

        <div class="stat-card">
            <div class="stat-label">Verdict</div>
            <div class="stat-value" style="font-size:26px;">{level}</div>
        </div>

    </div>
    """, unsafe_allow_html=True)

    # =====================
    # RISK CARD
    # =====================

    st.markdown(f"""
    <div class="card">

        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div style="color:#cbd5e1;font-size:14px;">
                Fraud Probability
            </div>

            <div style="padding:8px 14px;border-radius:999px;background:rgba(239,68,68,0.15);color:#fca5a5;font-size:12px;font-weight:600;">
                {level}
            </div>
        </div>

        <div style="font-size:72px;font-weight:700;color:white;margin-top:18px;letter-spacing:-4px;">
            {risk_pct:.1f}%
        </div>

        <div class="risk-track">
            <div class="risk-fill" style="width:{risk_pct:.1f}%"></div>
        </div>

    </div>
    """, unsafe_allow_html=True)

    # =====================
    # EXPLANATION
    # =====================

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader(T["features"])

    explanations = []

    if features["has_link"]:
        explanations.append("Contains suspicious link")

    if features["urgent_count"]:
        explanations.append("Urgency language detected")

    if features["secret_count"]:
        explanations.append("Requests sensitive information")

    if features["money_count"]:
        explanations.append("Mentions banking/payment")

    if features["threat_count"]:
        explanations.append("Threat or pressure detected")

    if features["suspicious_zone"]:
        explanations.append("Suspicious domain zone")

    for e in explanations:
        st.markdown(f'<span class="chip">{e}</span>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # =====================
    # DOMAIN ANALYSIS
    # =====================

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader(T["domains"])

    if domains:
        for d in domains:
            st.markdown(f"""
            <div style="
                padding:16px;
                border-radius:16px;
                background:rgba(255,255,255,0.04);
                border:1px solid rgba(255,255,255,0.06);
                margin-bottom:12px;
            ">
                <div style="font-size:15px;color:white;font-weight:600;">
                    {d}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No suspicious domains detected")

    st.markdown('</div>', unsafe_allow_html=True)

    # =====================
    # MODEL BREAKDOWN
    # =====================

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Model Breakdown")

    model_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "Gradient Boosting"
        ],
        "Probability": [
            round(lr_prob * 100, 1),
            round(rf_prob * 100, 1),
            round(gb_prob * 100, 1)
        ]
    })

    st.dataframe(model_df, use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # =====================
    # HISTORY
    # =====================

    st.session_state.history.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Risk": f"{risk_pct:.1f}%",
        "Verdict": level,
        "Preview": input_text[:60]
    })

# =====================================================
# HISTORY PANEL
# =====================================================

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader(T["history"])

if st.session_state.history:

    st.dataframe(
        pd.DataFrame(st.session_state.history),
        use_container_width=True,
        hide_index=True
    )

    if st.button(T["clear"]):
        st.session_state.history = []
        st.rerun()

else:
    st.info("No analysis history yet")

st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================

st.markdown(
    f"""
    <div class="footer">
        Premium Fraud Detector · AI Security Prototype · {datetime.now().year}
    </div>
    """,
    unsafe_allow_html=True
)
```

## Installation

```bash
pip install streamlit scikit-learn pandas numpy
```

## Run

```bash
streamlit run app.py
```
