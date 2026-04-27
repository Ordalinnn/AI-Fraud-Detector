import re
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="AI Fraud Detector",
    page_icon="🔐",
    layout="wide"
)

# =========================
# Dataset
# =========================

data = [
    # FRAUD
    ["срочно отправьте код из SMS", 1],
    ["ваша карта заблокирована перейдите по ссылке http://secure-login.xyz", 1],
    ["перейдите по ссылке http://kaspi-login.xyz", 1],
    ["назовите пароль чтобы защитить счет", 1],
    ["введите CVV и номер карты", 1],
    ["вы выиграли приз оплатите доставку", 1],
    ["ваш аккаунт будет удален срочно подтвердите вход", 1],
    ["не говорите никому и отправьте код", 1],
    ["служба безопасности банка просит назвать SMS код", 1],
    ["ваш счет под угрозой продиктуйте пароль", 1],
    ["переведите деньги сейчас иначе карта заблокируется", 1],
    ["подтвердите личность через ссылку http://verify-bank.top", 1],
    ["вам начислен бонус перейдите по ссылке", 1],
    ["для получения выплаты введите данные карты", 1],
    ["ваш Kaspi аккаунт заблокирован срочно войдите", 1],
    ["оплатите комиссию чтобы получить выигрыш", 1],
    ["ваш номер выбран победителем отправьте данные", 1],
    ["срочно оплатите штраф по ссылке", 1],
    ["құттықтаймыз сіз ұтыс ұттыңыз карта деректерін енгізіңіз", 1],
    ["сіздің картаңыз бұғатталды SMS кодты жіберіңіз", 1],
    ["шұғыл түрде сілтемеге өтіп аккаунтты растаңыз", 1],
    ["банк қызметкерімін кодты айтыңыз", 1],
    ["құпиясөзді жіберіңіз әйтпесе аккаунт жабылады", 1],
    ["urgent verify your bank account using this link", 1],
    ["your account is locked enter your card data", 1],
    ["send the verification code now", 1],

    # SAFE
    ["привет как дела", 0],
    ["завтра урок математики в 9", 0],
    ["встреча в 15:00", 0],
    ["ваш заказ доставлен", 0],
    ["спасибо за покупку", 0],
    ["добрый день документы готовы", 0],
    ["сегодня тренировка в 18:00", 0],
    ["мама я пришел домой", 0],
    ["напоминаем о родительском собрании", 0],
    ["домашнее задание отправлено", 0],
    ["завтра контрольная по физике", 0],
    ["ваш баланс пополнен", 0],
    ["спасибо за регистрацию на мероприятие", 0],
    ["ссылка на урок будет отправлена позже", 0],
    ["жду тебя возле школы", 0],
    ["сәлем қалайсың", 0],
    ["ертең математика сабағы болады", 0],
    ["үй тапсырмасын жібердім", 0],
    ["кездесу сағат 15:00-де", 0],
    ["құжаттар дайын болды", 0],
    ["hello see you tomorrow", 0],
    ["your meeting starts at 10 am", 0],
]

urgent_words = ["срочно", "шұғыл", "urgent", "now", "немедленно", "қазір", "тез", "быстро"]
money_words = ["карта", "счет", "банк", "cvv", "ақша", "төлем", "перевод", "оплата", "account", "bank", "card"]
secret_words = ["код", "пароль", "sms", "cvv", "құпия", "құпиясөз", "password", "verification code"]
threat_words = ["заблокирован", "заблокирована", "удален", "штраф", "под угрозой", "бұғатталды", "жабылады", "locked"]
reward_words = ["выиграли", "приз", "бонус", "ұтыс", "сыйлық", "winner", "bonus", "gift"]
domain_words = ["login", "verify", "secure", "bank", "kaspi", "bonus", "gift"]
bad_zones = [".xyz", ".top", ".click", ".site", ".online"]


# =========================
# Functions
# =========================

def detect_language(text):
    t = text.lower()
    kazakh_letters = "әғқңөұүіһ"
    russian_letters = "ыэёъщ"
    english_words = ["your", "account", "bank", "urgent", "password", "verify"]

    if any(c in t for c in kazakh_letters):
        return "🇰🇿 Қазақша"
    if any(c in t for c in russian_letters):
        return "🇷🇺 Русский"
    if any(w in t for w in english_words):
        return "🇬🇧 English"
    return "🌐 Mixed / Unknown"


def extract_urls(text):
    return re.findall(r"https?://[^\s]+|www\.[^\s]+", text.lower())


def get_domain(url):
    url = url.replace("https://", "").replace("http://", "").replace("www.", "")
    return url.split("/")[0]


def count_words(text, words):
    t = text.lower()
    return sum(1 for w in words if w in t)


def extract_features(text):
    t = text.lower()
    urls = extract_urls(t)
    domains = [get_domain(u) for u in urls]

    has_link = int(len(urls) > 0)
    urgent_count = count_words(t, urgent_words)
    money_count = count_words(t, money_words)
    secret_count = count_words(t, secret_words)
    threat_count = count_words(t, threat_words)
    reward_count = count_words(t, reward_words)

    suspicious_domain = 0
    long_domain = 0
    suspicious_zone = 0

    for d in domains:
        if any(w in d for w in domain_words):
            suspicious_domain = 1
        if len(d) > 20:
            long_domain = 1
        if any(d.endswith(z) for z in bad_zones):
            suspicious_zone = 1

    digit_count = sum(ch.isdigit() for ch in t)
    exclamation_count = t.count("!")

    features = {
        "has_link": has_link,
        "urgent_count": urgent_count,
        "money_count": money_count,
        "secret_count": secret_count,
        "threat_count": threat_count,
        "reward_count": reward_count,
        "suspicious_domain": suspicious_domain,
        "long_domain": long_domain,
        "suspicious_zone": suspicious_zone,
        "digit_count": digit_count,
        "exclamation_count": exclamation_count,
    }

    flags = []
    if has_link:
        flags.append("🔗 Link")
    if urgent_count:
        flags.append("⚡ Urgency")
    if money_count:
        flags.append("💳 Money/Bank")
    if secret_count:
        flags.append("🔐 Code/Password")
    if threat_count:
        flags.append("🚨 Threat")
    if reward_count:
        flags.append("🎁 Fake reward")
    if suspicious_domain:
        flags.append("🌐 Suspicious domain")
    if long_domain:
        flags.append("📏 Long domain")
    if suspicious_zone:
        flags.append("⚠️ Risky domain zone")
    if digit_count >= 4:
        flags.append("🔢 Many digits")
    if exclamation_count >= 2:
        flags.append("❗ Emotional pressure")

    return features, flags, domains


# =========================
# Model
# =========================

X = pd.DataFrame([extract_features(text)[0] for text, label in data])
y = np.array([label for text, label in data])

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])
model.fit(X, y)


# =========================
# Session
# =========================

if "history" not in st.session_state:
    st.session_state.history = []


# =========================
# UI CSS
# =========================

st.markdown("""
<style>
.big-title {
    font-size: 44px;
    font-weight: 900;
}
.sub {
    color: #6b7280;
    font-size: 18px;
}
.card {
    padding: 20px;
    border-radius: 18px;
    background: #f9fafb;
    border: 1px solid #e5e7eb;
}
.flag {
    display: inline-block;
    padding: 8px 14px;
    border-radius: 999px;
    background: #eef2ff;
    font-weight: 700;
}
.feature {
    padding: 10px 14px;
    margin: 6px 5px 6px 0;
    border-radius: 999px;
    background: #f3f4f6;
    display: inline-block;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# =========================
# Page
# =========================

st.markdown('<div class="big-title">🔐 AI Fraud Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Logistic Regression + Domain Analysis + Call Transcript Detection</div>', unsafe_allow_html=True)
st.divider()

left, right = st.columns([2, 1])

with left:
    mode = st.radio("Input type", ["SMS / Message", "Call transcript", "Upload TXT file"], horizontal=True)

    default_sms = "Срочно! Ваша карта заблокирована. Отправьте код из SMS и перейдите по ссылке http://secure-login.xyz"
    default_call = "Здравствуйте, я сотрудник банка. Назовите код из SMS, чтобы мы защитили ваш счет."

    if mode == "SMS / Message":
        text = st.text_area("Enter message:", value=default_sms, height=170)
    elif mode == "Call transcript":
        text = st.text_area("Enter call transcript:", value=default_call, height=170)
    else:
        uploaded = st.file_uploader("Upload .txt file", type=["txt"])
        text = ""
        if uploaded:
            text = uploaded.read().decode("utf-8", errors="ignore")
            st.text_area("Loaded text:", value=text, height=170)

    threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)
    analyze = st.button("🚀 Analyze", use_container_width=True)

with right:
    st.markdown("### What this prototype does")
    st.markdown("""
    <div class="card">
    ✅ Detects language<br>
    ✅ Finds fraud flags<br>
    ✅ Checks suspicious domains<br>
    ✅ Uses logistic regression<br>
    ✅ Gives risk probability<br>
    ✅ Saves check history
    </div>
    """, unsafe_allow_html=True)

if analyze:
    if not text.strip():
        st.warning("Please enter text first.")
    else:
        features, flags, domains = extract_features(text)
        X_input = pd.DataFrame([features])
        prob = model.predict_proba(X_input)[0][1]
        lang = detect_language(text)

        st.divider()
        st.markdown("## 📊 Result")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fraud risk", f"{prob * 100:.1f}%")
        c2.metric("Flags", len(flags))
        c3.metric("Threshold", threshold)
        c4.markdown(f'<div class="flag">{lang}</div>', unsafe_allow_html=True)

        st.progress(float(prob))

        if prob < 0.3:
            st.success("✅ Low risk")
            level = "Low risk"
        elif prob < 0.6:
            st.warning("🟡 Suspicious")
            level = "Suspicious"
        elif prob < 0.8:
            st.warning("🟠 High risk")
            level = "High risk"
        else:
            st.error("🔴 Very high fraud probability")
            level = "Very high fraud probability"

        st.markdown("## 🚩 Detected flags")
        if flags:
            st.markdown(" ".join([f'<span class="feature">{f}</span>' for f in flags]), unsafe_allow_html=True)
        else:
            st.info("No strong fraud flags detected.")

        if domains:
            st.markdown("## 🌐 Domains")
            for d in domains:
                st.code(d)

        st.markdown("## 💡 Recommendation")
        if prob >= threshold:
            st.error("Do not share SMS code, password, CVV, card number, or personal data. Do not open suspicious links.")
        else:
            st.success("Looks relatively safe, but verify through official sources if unsure.")

        st.markdown("## 🧠 Numerical features")
        st.dataframe(X_input, use_container_width=True)

        st.session_state.history.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "language": lang,
            "risk": round(prob * 100, 1),
            "level": level,
            "preview": text[:70]
        })

        with st.expander("📌 Logistic regression coefficients"):
            coef = model.named_steps["clf"].coef_[0]
            coef_df = pd.DataFrame({
                "Feature": X.columns,
                "Coefficient": coef
            }).sort_values("Coefficient", ascending=False)
            st.dataframe(coef_df, use_container_width=True)

        with st.expander("📘 How it works"):
            st.write("""
            The system extracts numerical features from the text: links, urgent words, bank-related words,
            secret code requests, suspicious domains, digits and emotional pressure.
            Then logistic regression calculates the probability of fraud.
            """)

st.divider()
st.markdown("## 📜 Check history")

if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
else:
    st.info("No checks yet.")
