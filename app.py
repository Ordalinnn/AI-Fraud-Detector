import re
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Dataset
# -----------------------------
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

# -----------------------------
# Functions
# -----------------------------
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
        "has_link": "Сілтеме бар",
        "urgent_count": "Шұғыл сөздер бар",
        "secret_count": "Код/пароль/CVV сұрауы мүмкін",
        "money_count": "Банк/ақша/карта туралы сөздер бар",
        "threat_count": "Қорқыту немесе қысым жасау бар",
        "suspicious_domain": "Доменде күмәнді сөздер бар",
        "long_domain": "Домен тым ұзын",
        "suspicious_zone": "Күмәнді домен зонасы",
        "digit_domain": "Доменде цифр бар",
        "digit_count": "Мәтінде көп сан/код кездеседі",
        "exclamation_count": "Леп белгілері көп",
    }
    return [labels[k] for k, v in features.items() if v > 0]

# -----------------------------
# Train model
# -----------------------------
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

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="AI Fraud Detector", page_icon="🔐", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []

st.title("🔐 AI Fraud Detector Ultimate")
st.write("SMS, интернет-хабарлама және қоңырау транскриптіндегі алаяқтықты логистикалық регрессия арқылы анықтайтын прототип.")

left, right = st.columns([2, 1])

with left:
    mode = st.selectbox("Режим", ["Обычный", "Банковский", "Қоңырау транскрипті"])

    samples = {
        "Обычный": "Вы выиграли приз! Срочно перейдите по ссылке http://bonus-gift.xyz",
        "Банковский": "Ваша карта заблокирована. Отправьте код из SMS и перейдите на http://secure-login.xyz",
        "Қоңырау транскрипті": "Здравствуйте, я сотрудник банка. Назовите код из SMS, чтобы мы защитили ваш счет."
    }

    text = st.text_area("Мәтінді енгізіңіз:", value=samples[mode], height=170)

    uploaded = st.file_uploader("TXT файл жүктеу", type=["txt"])
    if uploaded:
        text = uploaded.read().decode("utf-8", errors="ignore")
        st.info("Файл жүктелді.")

    threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)

    if mode == "Банковский":
        threshold = max(threshold, 0.45)

    analyze = st.button("🚀 Анализ жасау", use_container_width=True)

with right:
    st.subheader("Жүйе мүмкіндіктері")
    st.write("✅ мәтін талдау")
    st.write("✅ домен анализі")
    st.write("✅ логистикалық регрессия")
    st.write("✅ Explain AI")
    st.write("✅ тарих")
    st.write("✅ есеп жүктеу")

if analyze:
    features, domains = extract_features(text)
    X_input = pd.DataFrame([features])
    prob = model.predict_proba(X_input)[0][1]
    pred = int(prob >= threshold)
    explanations = explain(features)

    st.divider()
    st.subheader("📊 Нәтиже")

    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud risk", f"{prob*100:.1f}%")
    c2.metric("Табылған белгілер", len(explanations))
    c3.metric("Threshold", threshold)

    if prob < 0.3:
        st.success("🟢 Қауіп төмен")
        level = "Қауіп төмен"
    elif prob < 0.6:
        st.warning("🟡 Күмәнді")
        level = "Күмәнді"
    elif prob < 0.8:
        st.warning("🟠 Жоғары қауіп")
        level = "Жоғары қауіп"
    else:
        st.error("🔴 Алаяқтық ықтималдығы өте жоғары")
        level = "Өте жоғары қауіп"

    st.progress(float(prob))

    st.subheader("🔍 Неге бұлай шешті?")
    if explanations:
        for e in explanations:
            st.write(f"- {e}")
    else:
        st.write("Күшті алаяқтық белгілері табылмады.")

    if domains:
        st.subheader("🌐 Домен анализі")
        for d in domains:
            st.code(d)

    st.subheader("🧠 Explain AI: белгілердің үлесі")
    coef = model.named_steps["clf"].coef_[0]
    contrib = []
    for name, value, w in zip(X_train.columns, X_input.iloc[0], coef):
        contrib.append([name, value, w, value * w])

    contrib_df = pd.DataFrame(contrib, columns=["Feature", "Value", "Weight", "Contribution"])
    st.dataframe(contrib_df.sort_values("Contribution", ascending=False), use_container_width=True)

    st.subheader("📌 Модельге берілген сандық вектор")
    st.dataframe(X_input, use_container_width=True)

    st.subheader("💡 Кеңес")
    if pred == 1:
        st.error("Код, пароль, CVV, карта нөмірін бермеңіз. Сілтемеге өтпеңіз. Ресми банк нөміріне өзіңіз хабарласыңыз.")
    else:
        st.success("Хабарлама қауіпсіз көрінеді, бірақ күмән болса ресми дереккөз арқылы тексеріңіз.")

    report = f"""
AI Fraud Detector Report
Date: {datetime.now()}

Input text:
{text}

Fraud risk: {prob*100:.1f}%
Risk level: {level}

Detected features:
{chr(10).join("- " + e for e in explanations)}

Domains:
{chr(10).join(domains) if domains else "No domains"}

Advice:
{"Do not share codes, passwords, CVV or card data." if pred == 1 else "Looks safe, but verify from official sources if unsure."}
"""

    st.download_button("📥 Есепті жүктеу", report, file_name="fraud_report.txt")

    st.session_state.history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "text": text[:60],
        "risk": round(prob * 100, 1),
        "level": level
    })

st.divider()
st.subheader("📜 Тексеру тарихы")
if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
else:
    st.write("Әзірге тарих жоқ.")

with st.expander("📘 Бұл қалай жұмыс істейді?"):
    st.write("""
1. Мәтіннен белгілер алынады: сілтеме, шұғыл сөздер, код, карта, домен.
2. Белгілер сандық векторға айналады.
3. Логистикалық регрессия алаяқтық ықтималдығын есептейді.
4. Сайт нәтиже, түсіндірме және қауіпсіздік кеңесін береді.
""")

with st.expander("🧪 Тест мысалдары"):
    st.code("Срочно! Ваша карта заблокирована. Отправьте код из SMS и перейдите по ссылке http://secure-login.xyz")
    st.code("Привет, завтра урок математики в 9:00")
    st.code("Здравствуйте, я сотрудник банка. Назовите код из SMS.")
