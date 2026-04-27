import streamlit as st
import numpy as np
import pandas as pd
import re
from sklearn.linear_model import LogisticRegression

# --- СЕССИЯ (история) ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- ДАННЫЕ ---
data = [
    # FRAUD = 1
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
    ["переведите деньги сейчас иначе карта заблокируется", 1],
    ["подтвердите личность через ссылку http://verify-bank.top", 1],
    ["вам начислен бонус перейдите по ссылке", 1],
    ["для получения выплаты введите данные карты", 1],
    ["ваш Kaspi аккаунт заблокирован срочно войдите", 1],
    ["служба безопасности банка просит назвать SMS код", 1],
    ["оплатите комиссию чтобы получить выигрыш", 1],
    ["ваш номер выбран победителем отправьте данные", 1],
    ["перейдите на сайт и подтвердите пароль", 1],
    ["срочно оплатите штраф по ссылке", 1],
    ["құттықтаймыз сіз ұтыс ұттыңыз карта деректерін енгізіңіз", 1],
    ["сіздің картаңыз бұғатталды SMS кодты жіберіңіз", 1],
    ["шұғыл түрде сілтемеге өтіп аккаунтты растаңыз", 1],
    ["банк қызметкерімін кодты айтыңыз", 1],
    ["құпиясөзді жіберіңіз әйтпесе аккаунт жабылады", 1],

    # SAFE = 0
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
]

# --- ПРИЗНАКИ ---
def extract(text):
    text = text.lower()

    link = int("http" in text)
    urgent = int("срочно" in text or "шұғыл" in text)
    code = int("код" in text)
    card = int("карта" in text or "cvv" in text)
    domain = int(".xyz" in text or "login" in text)

    return [link, urgent, code, card, domain]

X = np.array([extract(x[0]) for x in data])
y = np.array([x[1] for x in data])

model = LogisticRegression()
model.fit(X, y)

# --- АНАЛИЗ ---
def analyze(text):
    x = np.array([extract(text)])
    prob = model.predict_proba(x)[0][1]

    explanations = []

    if "http" in text:
        explanations.append("Сілтеме табылды")
    if "срочно" in text:
        explanations.append("Шұғыл сөз бар")
    if "код" in text:
        explanations.append("SMS код сұралады")
    if "карта" in text:
        explanations.append("Карта мәліметі сұралады")
    if ".xyz" in text:
        explanations.append("Күмәнді домен")

    return prob, explanations

# --- UI ---
st.title("🔐 AI Fraud Detector")

st.markdown("### Мәтінді енгізіңіз немесе файл жүктеңіз")

text = st.text_area("Хабарлама немесе транскрипт:")

file = st.file_uploader("TXT файл жүктеу", type=["txt"])

if file is not None:
    text = file.read().decode("utf-8")

if st.button("🚀 Анализ"):
    if not text.strip():
        st.warning("Мәтін енгізіңіз")
    else:
        prob, explanations = analyze(text)

        # --- УРОВНИ ---
        if prob < 0.3:
            level = "Қауіп төмен"
            st.success(f"{level} ({prob:.2f})")
        elif prob < 0.6:
            level = "Күмәнді"
            st.warning(f"{level} ({prob:.2f})")
        elif prob < 0.8:
            level = "Жоғары қауіп"
            st.warning(f"{level} ({prob:.2f})")
        else:
            level = "Алаяқтық ықтималдығы өте жоғары"
            st.error(f"{level} ({prob:.2f})")

        # --- ПРОГРЕСС ---
        st.progress(prob)

        # --- ОБЪЯСНЕНИЕ ---
        st.markdown("### 🧠 Неге бұл нәтиже?")
        if explanations:
            for e in explanations:
                st.write(f"- {e}")
        else:
            st.write("Айқын алаяқтық белгілері табылмады")

        # --- СОВЕТ ---
        st.markdown("### 💡 Кеңес")
        if prob > 0.5:
            st.error("Код, пароль, карта мәліметін бермеңіз!")
        else:
            st.success("Хабарлама қауіпсіз сияқты")

        # --- СОХРАНЕНИЕ В ИСТОРИЮ ---
        st.session_state.history.append({
            "text": text[:50],
            "risk": prob,
            "level": level
        })

# --- ИСТОРИЯ ---
st.markdown("## 📜 Тексеру тарихы")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)
else:
    st.write("Тарих жоқ")