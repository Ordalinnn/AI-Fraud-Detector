import re
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# -----------------------------
# 1. Training data
# -----------------------------
train_data = [
    ["Срочно отправьте код из SMS", 1],
    ["Ваша карта заблокирована перейдите по ссылке http://secure-login.xyz", 1],
    ["Назовите пароль чтобы защитить счет", 1],
    ["Вы выиграли приз перейдите по ссылке", 1],
    ["Введите CVV и номер карты", 1],
    ["Не говорите никому и отправьте код", 1],
    ["Ваш аккаунт будет удален срочно подтвердите вход", 1],
    ["Перейдите на http://kaspi-login.xyz", 1],
    ["Здравствуйте, завтра будет урок математики", 0],
    ["Мама, я пришел домой", 0],
    ["Ваш заказ доставлен", 0],
    ["Напоминаем о встрече в 15:00", 0],
    ["Спасибо за покупку в официальном магазине", 0],
    ["Добрый день, документы готовы", 0],
    ["Сегодня тренировка в 18:00", 0],
    ["Ваш баланс пополнен", 0],
]

# -----------------------------
# 2. Dictionaries
# -----------------------------
urgent_words = [
    "срочно", "шұғыл", "быстро", "немедленно", "қазір",
    "тез", "urgent", "now", "сразу"
]

money_words = [
    "карта", "счет", "ақша", "банк", "cvv", "пароль",
    "код", "sms", "перевод", "оплата", "төле"
]

threat_words = [
    "заблокирована", "удален", "штраф", "проблема",
    "бұғатталды", "жабылады", "тоқтатылады"
]

secret_words = [
    "код", "пароль", "cvv", "номер карты", "жеке мәлімет",
    "құпия", "sms код"
]

suspicious_domain_words = [
    "login", "verify", "secure", "bonus", "gift", "bank", "kaspi"
]

suspicious_zones = [".xyz", ".top", ".click", ".site", ".online"]


# -----------------------------
# 3. Feature extraction
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
    text_lower = text.lower()
    urls = extract_urls(text_lower)
    domains = [get_domain(u) for u in urls]

    has_link = int(len(urls) > 0)
    urgent_count = count_matches(text_lower, urgent_words)
    money_count = count_matches(text_lower, money_words)
    threat_count = count_matches(text_lower, threat_words)
    secret_count = count_matches(text_lower, secret_words)

    suspicious_domain = 0
    long_domain = 0
    suspicious_zone = 0

    for d in domains:
        if any(w in d for w in suspicious_domain_words):
            suspicious_domain = 1
        if len(d) > 20:
            long_domain = 1
        if any(d.endswith(z) for z in suspicious_zones):
            suspicious_zone = 1

    digit_count = sum(ch.isdigit() for ch in text_lower)
    exclamation_count = text_lower.count("!")

    features = {
        "has_link": has_link,
        "urgent_count": urgent_count,
        "money_count": money_count,
        "threat_count": threat_count,
        "secret_count": secret_count,
        "suspicious_domain": suspicious_domain,
        "long_domain": long_domain,
        "suspicious_zone": suspicious_zone,
        "digit_count": digit_count,
        "exclamation_count": exclamation_count,
    }

    explanations = []

    if has_link:
        explanations.append("Хабарламада сілтеме бар")
    if urgent_count > 0:
        explanations.append("Шұғыл әрекетке итермелейтін сөздер бар")
    if money_count > 0:
        explanations.append("Банк/ақша/картаға қатысты сөздер бар")
    if secret_count > 0:
        explanations.append("Құпия код немесе жеке мәлімет сұрауы мүмкін")
    if threat_count > 0:
        explanations.append("Қорқыту немесе қысым жасау белгілері бар")
    if suspicious_domain:
        explanations.append("Доменде күмәнді сөздер бар")
    if long_domain:
        explanations.append("Домен тым ұзын")
    if suspicious_zone:
        explanations.append("Күмәнді домен зонасы анықталды")
    if digit_count >= 4:
        explanations.append("Көп сан/код кездеседі")
    if exclamation_count >= 2:
        explanations.append("Көп леп белгісі қолданылған")

    return features, explanations, domains


# -----------------------------
# 4. Prepare model
# -----------------------------
rows = []
labels = []

for text, label in train_data:
    features, _, _ = extract_features(text)
    rows.append(features)
    labels.append(label)

X_train = pd.DataFrame(rows)
y_train = np.array(labels)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression())
])

model.fit(X_train, y_train)


# -----------------------------
# 5. Streamlit interface
# -----------------------------
# -----------------------------
# 5. Improved Streamlit interface
# -----------------------------
st.set_page_config(
    page_title="AI Fraud Detector",
    page_icon="🔐",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #1f2937;
}
.subtitle {
    font-size: 18px;
    color: #6b7280;
}
.card {
    padding: 22px;
    border-radius: 18px;
    background-color: #f9fafb;
    border: 1px solid #e5e7eb;
}
.safe {
    padding: 20px;
    border-radius: 16px;
    background-color: #dcfce7;
    color: #166534;
    font-size: 24px;
    font-weight: 700;
}
.warning {
    padding: 20px;
    border-radius: 16px;
    background-color: #fef9c3;
    color: #854d0e;
    font-size: 24px;
    font-weight: 700;
}
.danger {
    padding: 20px;
    border-radius: 16px;
    background-color: #fee2e2;
    color: #991b1b;
    font-size: 24px;
    font-weight: 700;
}
.feature-box {
    padding: 12px;
    margin: 6px 0;
    border-radius: 12px;
    background-color: white;
    border-left: 5px solid #6366f1;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔐 AI Fraud Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">SMS, интернет-хабарлама және қоңырау транскриптіндегі алаяқтық белгілерін анықтайтын AI-прототип</div>',
    unsafe_allow_html=True
)

st.divider()

left, right = st.columns([2, 1])

with left:
    st.markdown("### ✍️ Мәтінді енгізіңіз")
    input_type = st.radio(
        "Тексеру түрі:",
        ["SMS / хабарлама", "Қоңырау транскрипті"],
        horizontal=True
    )

    sample_sms = "Срочно! Ваша карта заблокирована. Отправьте код из SMS и перейдите по ссылке http://secure-login.xyz"
    sample_call = "Здравствуйте, я сотрудник банка. Назовите код из SMS, чтобы мы защитили ваш счет."

    if input_type == "SMS / хабарлама":
        default_text = sample_sms
    else:
        default_text = sample_call

    text = st.text_area(
        "Мәтін:",
        value=default_text,
        height=180
    )

    threshold = st.slider(
        "Шешім шегі / Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05
    )

    analyze_button = st.button("🚀 Анализ жасау", use_container_width=True)

with right:
    st.markdown("### ℹ️ Жүйе не істейді?")
    st.markdown("""
    <div class="card">
    ✅ мәтінді талдайды<br>
    ✅ күмәнді белгілерді табады<br>
    ✅ доменді тексереді<br>
    ✅ логистикалық регрессия арқылы ықтималдық есептейді<br>
    ✅ қорытынды кеңес береді
    </div>
    """, unsafe_allow_html=True)

if analyze_button:
    if not text.strip():
        st.warning("Алдымен мәтін енгізіңіз.")
    else:
        features, explanations, domains = extract_features(text)
        X_input = pd.DataFrame([features])

        probability = model.predict_proba(X_input)[0][1]
        prediction = int(probability >= threshold)

        st.divider()
        st.markdown("## 📊 Анализ нәтижесі")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Алаяқтық ықтималдығы", f"{probability * 100:.1f}%")

        with col2:
            st.metric("Табылған белгілер саны", len(explanations))

        with col3:
            st.metric("Threshold", threshold)

        if probability >= 0.8:
            st.markdown(
                f'<div class="danger">⚠️ Жоғары қауіп: {probability * 100:.1f}%</div>',
                unsafe_allow_html=True
            )
        elif probability >= threshold:
            st.markdown(
                f'<div class="warning">⚠️ Күмәнді: {probability * 100:.1f}%</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="safe">✅ Қауіп төмен: {probability * 100:.1f}%</div>',
                unsafe_allow_html=True
            )

        st.progress(float(probability))

        st.markdown("## 🔍 Табылған белгілер")

        if explanations:
            for item in explanations:
                st.markdown(
                    f'<div class="feature-box">• {item}</div>',
                    unsafe_allow_html=True
                )
        else:
            st.success("Күшті алаяқтық белгілері табылмады.")

        if domains:
            st.markdown("## 🌐 Анықталған домендер")
            for d in domains:
                st.code(d)

        st.markdown("## 🧠 Модельге берілген сандық белгілер")
        st.dataframe(X_input, use_container_width=True)

        st.markdown("## 💡 Кеңес")
        if prediction == 1:
            st.error("Жеке мәлімет, SMS код, карта нөмірі немесе пароль бермеңіз. Сілтемеге өтпеңіз.")
        else:
            st.success("Хабарлама қауіпсіз көрінеді, бірақ күмән болса ресми дереккөз арқылы тексеріңіз.")

        with st.expander("📌 Логистикалық регрессия коэффициенттері"):
            coef = model.named_steps["clf"].coef_[0]
            coef_df = pd.DataFrame({
                "Feature": X_train.columns,
                "Coefficient": coef
            }).sort_values(by="Coefficient", ascending=False)
            st.dataframe(coef_df, use_container_width=True)