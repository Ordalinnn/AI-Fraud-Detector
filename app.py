import re
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from pathlib import Path
import base64
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Fraud Detector",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# LOGO HELPER
# =========================
def image_to_base64(path: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return ""
    return base64.b64encode(file_path.read_bytes()).decode()

LOGO_B64 = image_to_base64("logo.png")
LOGO_HTML = f"data:image/png;base64,{LOGO_B64}" if LOGO_B64 else ""

# =========================
# LANGUAGE
# =========================
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
        "title": "Fraud Detector",
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
        "analyze": "Анализ жасау →",
        "features": "Мүмкіндіктер",
        "result": "Нәтиже",
        "risk": "Алаяқтық ықтималдығы",
        "detected": "Белгілер",
        "threshold": "Шешім шегі",
        "model": "Модель",
        "low": "Қауіп төмен",
        "mid": "Күмәнді",
        "high": "Жоғары қауіп",
        "critical": "Өте жоғары қауіп",
        "why": "Неге осындай шешім?",
        "domain": "Домен",
        "vector": "Белгілер",
        "report": "Есеп жүктеу",
        "history": "Тарих",
        "advice": "Кеңес",
        "bad_advice": "Код, пароль, CVV немесе карта нөмірін бермеңіз. Сілтемеге өтпеңіз. Банкке тек ресми нөмір арқылы хабарласыңыз.",
        "good_advice": "Хабарлама қауіпсіз көрінеді. Бірақ күмән болса, ресми дереккөз арқылы тексеріңіз.",
        "no_text": "Алдымен мәтін енгізіңіз.",
        "no_features": "Күшті алаяқтық белгілері табылмады.",
        "no_domain": "Мәтінде URL немесе домен табылмады.",
        "download": "TXT есепті жүктеу",
        "how": "Жүйе қалай жұмыс істейді?",
        "footer": "AI Fraud Detector · Applied ML Prototype",
        "model_metrics": "Модель метрикалары",
        "accuracy": "Дәлдік",
        "clear_history": "Тарихты тазалау",
        "char_count": "Таңба",
        "word_count": "Сөз",
        "text_stats": "Мәтін статистикасы",
        "ensemble": "Ансамбль болжамы",
        "no_history": "Тарих жоқ",
    },
    "🇷🇺 RU": {
        "title": "Fraud Detector",
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
        "analyze": "Анализировать →",
        "features": "Возможности",
        "result": "Результат анализа",
        "risk": "Вероятность мошенничества",
        "detected": "Признаки",
        "threshold": "Порог решения",
        "model": "Модель",
        "low": "Низкий риск",
        "mid": "Подозрительно",
        "high": "Высокий риск",
        "critical": "Критический риск",
        "why": "Почему такой результат?",
        "domain": "Домен",
        "vector": "Признаки",
        "report": "Скачать отчёт",
        "history": "История",
        "advice": "Совет",
        "bad_advice": "Не сообщайте код, пароль, CVV или номер карты. Не переходите по ссылке. Свяжитесь с банком только по официальному номеру.",
        "good_advice": "Сообщение выглядит безопасным. Если есть сомнения — проверьте через официальный источник.",
        "no_text": "Сначала введите текст.",
        "no_features": "Сильные признаки мошенничества не найдены.",
        "no_domain": "URL или домен в тексте не найден.",
        "download": "Скачать TXT отчёт",
        "how": "Как работает система?",
        "footer": "AI Fraud Detector · Прикладной ML-прототип",
        "model_metrics": "Метрики моделей",
        "accuracy": "Точность",
        "clear_history": "Очистить историю",
        "char_count": "Символов",
        "word_count": "Слов",
        "text_stats": "Статистика текста",
        "ensemble": "Ансамблевый прогноз",
        "no_history": "История пуста",
    },
    "🇬🇧 EN": {
        "title": "Fraud Detector",
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
        "analyze": "Analyze →",
        "features": "Features",
        "result": "Analysis result",
        "risk": "Fraud probability",
        "detected": "Indicators",
        "threshold": "Decision threshold",
        "model": "Model",
        "low": "Low risk",
        "mid": "Suspicious",
        "high": "High risk",
        "critical": "Critical risk",
        "why": "Why this decision?",
        "domain": "Domain",
        "vector": "Features",
        "report": "Download report",
        "history": "History",
        "advice": "Safety advice",
        "bad_advice": "Do not share codes, passwords, CVV, or card numbers. Do not open links. Contact your bank via the official number only.",
        "good_advice": "This message looks safe. When in doubt, verify through official sources.",
        "no_text": "Please enter text first.",
        "no_features": "No strong fraud indicators were found.",
        "no_domain": "No URL or domain was found in the text.",
        "download": "Download TXT report",
        "how": "How does it work?",
        "footer": "AI Fraud Detector · Applied ML Prototype",
        "model_metrics": "Model Metrics",
        "accuracy": "Accuracy",
        "clear_history": "Clear history",
        "char_count": "Characters",
        "word_count": "Words",
        "text_stats": "Text statistics",
        "ensemble": "Ensemble prediction",
        "no_history": "No history yet",
    },
}

T = TEXT[lang]

# =========================
# TRAINING DATA (unchanged)
# =========================
data = [
    ["срочно отправьте код из SMS", 1],
    ["ваша карта заблокирована перейдите по ссылке http://secure-login.xyz", 1],
    ["перейдите по ссылке http://kaspi-login.xyz", 1],
    ["назовите пароль чтобы защитить счет", 1],
    ["введите CVV и номер карты", 1],
    ["ваш аккаунт будет удален срочно подтвердите вход", 1],
    ["не говорите никому и отправьте код", 1],
    ["сотрудник банка просит код из SMS", 1],
    ["ваш счет под угрозой продиктуйте пароль", 1],
    ["мы обнаружили подозрительную активность подтвердите личность", 1],
    ["подтвердите вход иначе аккаунт будет удален", 1],
    ["переведите деньги на безопасный счет", 1],
    ["это служба безопасности банка срочно назовите код", 1],
    ["вам одобрен кредит отправьте код подтверждения", 1],
    ["ваша карта временно ограничена подтвердите данные", 1],
    ["банк предупреждает о списании подтвердите операцию", 1],
    ["ваш личный кабинет заблокирован войдите по ссылке", 1],
    ["вы выиграли приз оплатите доставку", 1],
    ["вы получили бонус перейдите по ссылке и введите данные", 1],
    ["ваша посылка задержана оплатите пошлину по ссылке", 1],
    ["подтвердите оплату иначе штраф", 1],
    ["ваш номер выиграл в акции отправьте данные карты", 1],
    ["мы из полиции ваш счет используется мошенниками срочно действуйте", 1],
    ["ваш родственник попал в аварию срочно переведите деньги", 1],
    ["на ваше имя оформлен кредит срочно свяжитесь с оператором", 1],
    ["это госуслуги подтвердите учетную запись иначе доступ будет закрыт", 1],
    ["вам начислена компенсация введите номер карты для получения", 1],
    ["urgent your card is blocked verify your account now", 1],
    ["your account will be suspended enter your password", 1],
    ["security department needs your verification code", 1],
    ["you won a prize pay delivery fee by link", 1],
    ["transfer money to a safe account immediately", 1],
    ["your parcel is on hold pay customs fee", 1],
    ["confirm your identity using this secure login link", 1],
    ["click here to verify your bank account details now", 1],
    ["your otp code is required to protect your account", 1],
    ["do not tell anyone call us back with your pin", 1],
    ["last chance to claim your reward enter card details", 1],
    ["your account has been compromised login immediately", 1],
    ["құттықтаймыз сіз ұтыс ұттыңыз карта деректерін енгізіңіз", 1],
    ["сіздің картаңыз бұғатталды SMS кодты жіберіңіз", 1],
    ["шұғыл түрде сілтемеге өтіп аккаунтты растаңыз", 1],
    ["банк қызметкерімін кодты айтыңыз", 1],
    ["қауіпсіз шотқа ақша аударыңыз", 1],
    ["жеке кабинетіңіз жабылады құпия кодты енгізіңіз", 1],
    ["сіздің шотыңыз бұғатталды ресми нөмірге хабарласыңыз", 1],
    ["жеделдетілген несие алыңыз кодты растаңыз", 1],
    ["guaranteed 30% weekly returns send money to activate account", 1],
    ["work from home earn 500 a day send registration fee", 1],
    ["your investment is ready withdraw by entering card details", 1],
    ["гарантированный доход 30 процентов переведите деньги для активации", 1],
    ["удаленная работа заработок 500 в день отправьте регистрационный взнос", 1],
    ["привет как дела", 0],
    ["завтра урок математики в 9", 0],
    ["встреча в 15:00", 0],
    ["ваш заказ доставлен", 0],
    ["спасибо за покупку", 0],
    ["добрый день документы готовы", 0],
    ["сегодня тренировка в 18:00", 0],
    ["ваш чек доступен в приложении", 0],
    ["напоминаем о записи к врачу завтра", 0],
    ["ваш заказ готов к выдаче", 0],
    ["посылка доставлена в пункт выдачи", 0],
    ["оплата прошла успешно спасибо", 0],
    ["сәлем қалайсың", 0],
    ["ертең математика сабағы болады", 0],
    ["үй тапсырмасын жібердім", 0],
    ["hello see you tomorrow at school", 0],
    ["your appointment is confirmed", 0],
    ["your order has been shipped and will arrive in 3 days", 0],
    ["meeting rescheduled to friday at 10am", 0],
    ["thank you for your payment receipt attached", 0],
    ["reminder your prescription is ready for pickup", 0],
    ["your monthly statement is now available in the app", 0],
    ["restaurant reservation confirmed for saturday 7pm", 0],
    ["your package was delivered to your front door", 0],
    ["happy birthday have a great day", 0],
    ["кездесу жоспарланды сейсенбіде сағат 14-те", 0],
    ["тапсырысыңыз дайын алып кетуге болады", 0],
    ["your salary has been credited to your account", 0],
]

urgent_words = ["срочно", "шұғыл", "быстро", "немедленно", "қазір", "тез", "urgent", "now", "immediately", "asap", "прямо сейчас", "сейчас же", "без промедления"]
secret_words = ["код", "пароль", "cvv", "sms", "құпия", "password", "пин", "pin", "код подтверждения", "verification code", "one time code", "otp"]
money_words = ["карта", "счет", "банк", "ақша", "төле", "оплата", "перевод", "баланс", "средства", "деньги", "transfer", "payment", "wallet", "iban"]
threat_words = ["заблокирована", "удален", "штраф", "угрозой", "бұғатталды", "жабылады", "blocked", "suspended", "terminated", "penalty", "freeze", "ограничен", "будет закрыт"]
suspicious_domain_words = ["login", "verify", "secure", "bonus", "gift", "bank", "kaspi", "account", "support", "confirm", "prize", "payment", "wallet", "security", "update", "auth", "free"]
suspicious_zones = [".xyz", ".top", ".click", ".site", ".online", ".live", ".info", ".icu"]
identity_words = ["паспорт", "иин", "удостоверение", "личность", "identity", "document", "id card", "жсн", "құжат"]
reward_words = ["выиграли", "приз", "бонус", "подарок", "акция", "компенсация", "won", "prize", "gift", "bonus", "reward", "ұтыс", "сыйлық"]
pressure_phrases = ["не говорите никому", "никому не сообщайте", "это секретно", "только сейчас", "последний шанс", "иначе", "do not tell anyone", "last chance", "only now", "қазір ғана"]

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
    suspicious_domain = 0
    long_domain = 0
    suspicious_zone = 0
    digit_domain = 0
    for d in domains:
        if any(w in d for w in suspicious_domain_words): suspicious_domain = 1
        if len(d) > 20: long_domain = 1
        if any(d.endswith(z) for z in suspicious_zones): suspicious_zone = 1
        if any(ch.isdigit() for ch in d): digit_domain = 1
    words = text_lower.split()
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    return {
        "has_link": int(len(urls) > 0),
        "urgent_count": count_matches(text_lower, urgent_words),
        "secret_count": count_matches(text_lower, secret_words),
        "money_count": count_matches(text_lower, money_words),
        "threat_count": count_matches(text_lower, threat_words),
        "identity_count": count_matches(text_lower, identity_words),
        "reward_count": count_matches(text_lower, reward_words),
        "pressure_count": count_matches(text_lower, pressure_phrases),
        "suspicious_domain": suspicious_domain,
        "long_domain": long_domain,
        "suspicious_zone": suspicious_zone,
        "digit_domain": digit_domain,
        "digit_count": sum(ch.isdigit() for ch in text_lower),
        "exclamation_count": text_lower.count("!"),
        "uppercase_count": sum(1 for ch in text if ch.isupper()),
        "text_length": len(text),
        "word_count": len(words),
        "avg_word_length": round(avg_word_len, 2),
        "url_count": len(urls),
        "has_multiple_warnings": int(count_matches(text_lower, urgent_words) > 0 and count_matches(text_lower, threat_words) > 0),
    }, domains

def explain(features):
    if lang == "🇰🇿 KZ":
        labels = {
            "has_link": "Сілтеме анықталды", "urgent_count": "Шұғыл әрекетке шақыру бар",
            "secret_count": "Код / пароль / CVV сұрауы", "money_count": "Банк, ақша немесе картаға сөздер",
            "threat_count": "Қорқыту немесе қысым", "identity_count": "Жеке ақпарат сұралуы",
            "reward_count": "Ұтыс немесе бонус уәдесі", "pressure_count": "Қысым белгісі",
            "suspicious_domain": "Доменде күмәнді сөздер", "long_domain": "Ұзын домен",
            "suspicious_zone": "Күмәнді домен зонасы", "digit_domain": "Доменде цифрлар",
            "digit_count": "Мәтінде көп сан", "exclamation_count": "Көп леп белгісі",
            "uppercase_count": "Үлкен әріптер", "has_multiple_warnings": "Шұғылдық + қорқыту",
            "url_count": "Бірнеше сілтеме",
        }
    elif lang == "🇷🇺 RU":
        labels = {
            "has_link": "Обнаружена ссылка", "urgent_count": "Срочный призыв к действию",
            "secret_count": "Запрос кода / пароля / CVV", "money_count": "Слова о банке, деньгах, карте",
            "threat_count": "Угроза или давление", "identity_count": "Запрос личных данных",
            "reward_count": "Обещание выигрыша или бонуса", "pressure_count": "Давление или просьба о секрете",
            "suspicious_domain": "Подозрительные слова в домене", "long_domain": "Подозрительно длинный домен",
            "suspicious_zone": "Подозрительная доменная зона", "digit_domain": "Цифры в домене",
            "digit_count": "Много чисел в тексте", "exclamation_count": "Много восклицательных знаков",
            "uppercase_count": "Много заглавных букв", "has_multiple_warnings": "Срочность + угроза одновременно",
            "url_count": "Несколько ссылок",
        }
    else:
        labels = {
            "has_link": "A link was detected", "urgent_count": "Urgent action words found",
            "secret_count": "Request for code / password / CVV", "money_count": "Bank, money, or card words found",
            "threat_count": "Pressure or threat indicators", "identity_count": "Request for personal identity data",
            "reward_count": "Prize, bonus, or gift promise", "pressure_count": "Pressure or secrecy phrase",
            "suspicious_domain": "Suspicious words in domain", "long_domain": "Suspiciously long domain",
            "suspicious_zone": "Suspicious domain zone", "digit_domain": "Domain contains numbers",
            "digit_count": "Many numbers in text", "exclamation_count": "Many exclamation marks",
            "uppercase_count": "Many uppercase letters", "has_multiple_warnings": "Urgency + threat detected",
            "url_count": "Multiple links detected",
        }
    irrelevant = {"text_length", "word_count", "avg_word_length"}
    return [labels[k] for k, v in features.items() if v > 0 and k in labels and k not in irrelevant]

def risk_style(prob):
    if prob < 0.3: return T["low"], "low", "●"
    if prob < 0.6: return T["mid"], "mid", "●"
    if prob < 0.8: return T["high"], "high", "●"
    return T["critical"], "critical", "●"

def rule_boost(features):
    boost = 0.0
    if features["has_link"] and features["secret_count"]: boost += 0.15
    if features["urgent_count"] and features["money_count"]: boost += 0.12
    if features["secret_count"] and features["money_count"]: boost += 0.15
    if features["threat_count"] and features["has_link"]: boost += 0.10
    if features["reward_count"] and features["money_count"]: boost += 0.12
    if features["pressure_count"]: boost += 0.10
    if features["suspicious_zone"] or features["suspicious_domain"]: boost += 0.10
    if features["has_multiple_warnings"]: boost += 0.08
    if features["url_count"] > 1: boost += 0.05
    return min(boost, 0.30)

@st.cache_resource(show_spinner=False)
def train_models():
    rows, labels = [], []
    for text, label in data:
        f, _ = extract_features(text)
        rows.append(f)
        labels.append(label)
    X_train = pd.DataFrame(rows)
    y_train = np.array(labels)
    lr_model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42))])
    rf_model = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
    gb_model = Pipeline([("scaler", StandardScaler()), ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42))])
    lr_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    lr_cv = cross_val_score(lr_model, X_train, y_train, cv=5, scoring="accuracy").mean()
    rf_cv = cross_val_score(rf_model, X_train, y_train, cv=5, scoring="accuracy").mean()
    gb_cv = cross_val_score(gb_model, X_train, y_train, cv=5, scoring="accuracy").mean()
    metrics = {"Logistic Regression": round(lr_cv * 100, 1), "Random Forest": round(rf_cv * 100, 1), "Gradient Boosting": round(gb_cv * 100, 1)}
    return lr_model, rf_model, gb_model, metrics, X_train, y_train

lr_model, rf_model, gb_model, model_metrics, X_train, y_train = train_models()

# =========================
# MINIMAL CSS
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
}

.stApp {
    background: #f5f5f3;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1200px;
}

/* ---- SIDEBAR ---- */
[data-testid="stSidebar"] {
    background: #0e0e0e;
    border-right: 1px solid #1e1e1e;
}

[data-testid="stSidebar"] * {
    color: #c8c8c8 !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-weight: 500 !important;
    letter-spacing: -0.3px;
}

[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: #888 !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 500 !important;
}

[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
    background: #1a1a1a !important;
    border: 1px solid #333 !important;
    border-radius: 8px !important;
    color: #e0e0e0 !important;
}

[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span,
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] input {
    color: #e0e0e0 !important;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background: #1a1a1a;
    border: 1px solid #333;
    padding: 6px 10px;
    border-radius: 6px;
    margin-right: 4px;
    font-size: 12px !important;
}

[data-testid="stSidebar"] hr {
    border-color: #222 !important;
}

/* ---- HEADER ---- */
.page-header {
    padding: 0 0 2rem 0;
    border-bottom: 1px solid #ddd;
    margin-bottom: 2rem;
}

.page-eyebrow {
    font-size: 11px;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #888;
    font-weight: 500;
    margin-bottom: 10px;
}

.page-title {
    font-size: 36px;
    font-weight: 300;
    color: #111;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 10px;
}

.page-title strong {
    font-weight: 600;
}

.page-subtitle {
    font-size: 14px;
    color: #777;
    font-weight: 400;
    line-height: 1.6;
    max-width: 640px;
}

/* ---- CARDS ---- */
.card {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
}

.card-label {
    font-size: 11px;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #999;
    font-weight: 500;
    margin-bottom: 14px;
}

/* ---- STAT CELLS ---- */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}

.stat-cell {
    background: #fff;
    border: 1px solid #e8e8e8;
    border-radius: 10px;
    padding: 16px 14px;
}

.stat-cell-label {
    font-size: 11px;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #aaa;
    font-weight: 500;
    margin-bottom: 6px;
}

.stat-cell-value {
    font-size: 26px;
    font-weight: 300;
    color: #111;
    letter-spacing: -0.5px;
    line-height: 1;
}

/* ---- RISK DISPLAY ---- */
.risk-bar-wrap {
    background: #fff;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 22px 24px;
    margin: 16px 0;
}

.risk-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
}

.risk-label-text {
    font-size: 13px;
    font-weight: 500;
    color: #555;
    letter-spacing: 0.2px;
}

.risk-pct {
    font-size: 32px;
    font-weight: 300;
    letter-spacing: -1px;
    line-height: 1;
}

.risk-verdict {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 5px 12px;
    border-radius: 99px;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.3px;
}

.verdict-low { background: #f0faf0; color: #2d7a2d; border: 1px solid #c2e8c2; }
.verdict-mid { background: #fffbeb; color: #8a6000; border: 1px solid #edd97a; }
.verdict-high { background: #fff4ec; color: #b84a00; border: 1px solid #f5c49d; }
.verdict-critical { background: #fff0f0; color: #aa1f1f; border: 1px solid #f5aaaa; }

.dot-low { color: #3aaa3a; }
.dot-mid { color: #d4a017; }
.dot-high { color: #e05c00; }
.dot-critical { color: #cc1a1a; }

.risk-track {
    height: 3px;
    background: #f0f0f0;
    border-radius: 99px;
    overflow: hidden;
    margin-top: 6px;
}

.risk-fill-low    { height: 100%; background: #3aaa3a; border-radius: 99px; }
.risk-fill-mid    { height: 100%; background: #d4a017; border-radius: 99px; }
.risk-fill-high   { height: 100%; background: #e05c00; border-radius: 99px; }
.risk-fill-critical { height: 100%; background: #cc1a1a; border-radius: 99px; }

/* ---- CHIPS ---- */
.chip-wrap { display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0; }

.chip {
    display: inline-flex;
    align-items: center;
    padding: 6px 12px;
    border-radius: 99px;
    background: #f4f4f2;
    color: #444;
    font-size: 12px;
    font-weight: 500;
    border: 1px solid #e4e4e0;
    letter-spacing: 0.1px;
}

/* ---- DOMAIN BOX ---- */
.domain-row {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 14px 0;
    border-bottom: 1px solid #f0f0f0;
}

.domain-name {
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    color: #111;
    font-weight: 500;
    word-break: break-all;
    flex: 1;
}

.domain-flags { display: flex; flex-wrap: wrap; gap: 6px; }

.domain-flag {
    font-size: 11px;
    padding: 3px 9px;
    border-radius: 99px;
    font-weight: 500;
}

.flag-warn { background: #fff8e6; color: #8a6000; border: 1px solid #edcc82; }
.flag-danger { background: #fff0f0; color: #aa1f1f; border: 1px solid #f5aaaa; }
.flag-ok { background: #f0faf0; color: #2d7a2d; border: 1px solid #c2e8c2; }

/* ---- BUTTON ---- */
.stButton > button {
    border-radius: 8px !important;
    padding: 0.65rem 1.4rem !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    letter-spacing: 0.2px !important;
    border: 1px solid #111 !important;
    background: #111 !important;
    color: #fff !important;
    box-shadow: none !important;
    transition: opacity 0.15s !important;
}

.stButton > button:hover {
    opacity: 0.85 !important;
    transform: none !important;
}

/* ---- TEXTAREA / INPUTS ---- */
textarea {
    border-radius: 8px !important;
    border: 1px solid #d8d8d8 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    background: #fff !important;
    box-shadow: none !important;
}

textarea:focus {
    border-color: #111 !important;
    box-shadow: none !important;
}

/* ---- TABS ---- */
[data-testid="stTabs"] button {
    font-weight: 500 !important;
    font-size: 13px !important;
    letter-spacing: 0.1px !important;
    border-radius: 6px 6px 0 0 !important;
    color: #777 !important;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: #111 !important;
    font-weight: 600 !important;
}

/* ---- SIDEBAR METRICS ---- */
.sb-metric-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #1e1e1e;
}

.sb-metric-name { font-size: 11px; color: #666; }
.sb-metric-val { font-size: 13px; font-weight: 600; }

/* ---- ADVICE ---- */
.advice-box {
    border-radius: 8px;
    padding: 14px 16px;
    font-size: 13px;
    line-height: 1.6;
    font-weight: 400;
    margin-top: 16px;
}

.advice-safe { background: #f0faf0; color: #2d7a2d; border: 1px solid #c2e8c2; }
.advice-danger { background: #fff0f0; color: #aa1f1f; border: 1px solid #f5aaaa; }

/* ---- FOOTER ---- */
.footer {
    text-align: center;
    color: #bbb;
    font-size: 11px;
    letter-spacing: 0.5px;
    margin-top: 48px;
    padding-top: 20px;
    border-top: 1px solid #e8e8e8;
    text-transform: uppercase;
}

/* Feature list */
.feature-item {
    padding: 9px 0;
    border-bottom: 1px solid #f2f2f2;
    font-size: 13px;
    color: #555;
    display: flex;
    align-items: center;
    gap: 10px;
}

.feature-item:last-child { border-bottom: none; }
.feature-dot { width: 5px; height: 5px; background: #ccc; border-radius: 50%; flex-shrink: 0; }

/* Model breakdown mini cells */
.model-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin: 14px 0;
}

.model-cell {
    background: #f9f9f7;
    border: 1px solid #e8e8e4;
    border-radius: 8px;
    padding: 12px 10px;
    text-align: center;
}

.model-cell-name { font-size: 10px; color: #aaa; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 5px; }
.model-cell-val { font-size: 18px; font-weight: 300; color: #111; }
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## ◈ Fraud Detector")
    st.caption("AI-powered prototype")

    selected_lang = st.radio(
        "Language",
        LANG_OPTIONS,
        index=LANG_OPTIONS.index(st.session_state.lang),
        horizontal=True,
        key="lang_selector"
    )
    if selected_lang != st.session_state.lang:
        st.session_state.lang = selected_lang
        st.rerun()

    st.divider()

    mode = st.selectbox(T["mode"], [T["sms"], T["call"], T["file"]])
    threshold = st.slider(T["threshold"], 0.1, 0.9, 0.5, 0.05)

    st.divider()
    st.markdown(f"**{T['demo']}**")

    demo_texts = {
        "Fraud SMS": "Срочно! Ваша карта заблокирована. Отправьте код из SMS и перейдите по ссылке http://secure-login.xyz",
        "Fraud Call": "Здравствуйте, я сотрудник службы безопасности банка. По вашему счету подозрительная операция. Назовите код из SMS, чтобы мы отменили перевод.",
        "Safe Message": "Привет, завтра урок математики в 9:00. Не забудь тетрадь.",
        "Fake Delivery": "Ваша посылка задержана. Срочно оплатите таможенную пошлину по ссылке http://delivery-pay-online.xyz",
        "Fake Prize": "Поздравляем! Вы выиграли приз. Для получения подарка введите номер карты и CVV.",
        "Relative Scam": "Ваш родственник попал в аварию. Срочно переведите деньги, никому не говорите.",
        "Fake Job Offer": "Поздравляем, вы приняты на удаленную работу. Для оформления выплат отправьте фото удостоверения, номер карты и OTP-код из SMS.",
        "Marketplace Scam": "Здравствуйте, я покупатель с маркетплейса. Подтвердите получение оплаты: перейдите по ссылке https://safe-deal-confirm.top и введите данные карты.",
        "Fake Utility Debt": "Уведомление ЖКХ: у вас долг за коммунальные услуги. Во избежание отключения света оплатите сегодня по ссылке http://pay-service-24.site.",
        "Investment Scam": "Гарантированный доход 30% в неделю! Переведите деньги на инвестиционный счет и сообщите код подтверждения для активации."
    }

    demo = st.selectbox("", list(demo_texts.keys()))

    st.divider()
    st.markdown(f"**{T['model_metrics']}**")
    for model_name, acc in model_metrics.items():
        color = "#4ade80" if acc >= 90 else "#facc15"
        short = model_name.replace("Logistic Regression", "Log. Reg.").replace("Gradient Boosting", "Grad. Boost")
        st.markdown(
            f'<div class="sb-metric-row"><span class="sb-metric-name">{short}</span>'
            f'<span class="sb-metric-val" style="color:{color}">{acc}%</span></div>',
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown("**Stack**")
    for item in ["Ensemble LR + RF + GB", "Feature engineering", "Domain analysis", "Explainable AI"]:
        st.markdown(f'<div style="font-size:12px;color:#555;padding:3px 0">· {item}</div>', unsafe_allow_html=True)

# =========================
# PAGE HEADER
# =========================
st.markdown(f"""
<div class="page-header">
    <div class="page-eyebrow">AI · Machine Learning · Security</div>
    <div class="page-title"><strong>Fraud</strong> Detector</div>
    <div class="page-subtitle">{T['subtitle']}</div>
</div>
""", unsafe_allow_html=True)

# =========================
# INPUT SECTION
# =========================
left, right = st.columns([2.2, 0.8], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="card-label">{T["input_title"]}</div>', unsafe_allow_html=True)

    with st.form("analysis_form", clear_on_submit=False):
        uploaded = None
        if mode == T["file"]:
            uploaded = st.file_uploader(T["upload"], type=["txt"])

        if uploaded:
            input_text = uploaded.read().decode("utf-8", errors="ignore")
            st.success("File loaded.")
        else:
            input_text = st.text_area(
                T["input_label"],
                value=demo_texts[demo],
                height=180,
                label_visibility="collapsed"
            )

        char_count = len(input_text)
        word_count_val = len(input_text.split()) if input_text.strip() else 0
        st.caption(f"{T['char_count']}: {char_count}  ·  {T['word_count']}: {word_count_val}")
        analyze = st.form_submit_button(T["analyze"], use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="card-label">{T["features"]}</div>', unsafe_allow_html=True)
    features_list = [
        "Text & pattern analysis", "Domain inspection", "Ensemble ML (LR+RF+GB)",
        "Explainable output", "Downloadable report", "Real scam scenarios",
        "Rule-based risk boost", "Model accuracy display"
    ]
    for f in features_list:
        st.markdown(f'<div class="feature-item"><div class="feature-dot"></div>{f}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

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

        lr_prob = lr_model.predict_proba(X_input)[0][1]
        rf_prob = rf_model.predict_proba(X_input)[0][1]
        gb_prob = gb_model.predict_proba(X_input)[0][1]
        raw_prob = (lr_prob + rf_prob + gb_prob) / 3.0
        boost = rule_boost(features)
        prob = min(0.99, raw_prob + boost)
        pred = int(prob >= threshold)
        risk_label, risk_class, emoji = risk_style(prob)
        explanations = explain(features)

        st.markdown(f'<div class="card-label" style="margin-top:8px">{T["result"]}</div>', unsafe_allow_html=True)

        # Stats row
        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-cell">
                <div class="stat-cell-label">{T["risk"]}</div>
                <div class="stat-cell-value">{prob*100:.1f}%</div>
            </div>
            <div class="stat-cell">
                <div class="stat-cell-label">{T["detected"]}</div>
                <div class="stat-cell-value">{len(explanations)}</div>
            </div>
            <div class="stat-cell">
                <div class="stat-cell-label">{T["threshold"]}</div>
                <div class="stat-cell-value">{threshold:.2f}</div>
            </div>
            <div class="stat-cell">
                <div class="stat-cell-label">URLs</div>
                <div class="stat-cell-value">{len(domains)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Risk bar
        pct = prob * 100
        st.markdown(f"""
        <div class="risk-bar-wrap">
            <div class="risk-row">
                <span class="risk-label-text">{T["risk"]}</span>
                <span class="verdict-{risk_class} risk-verdict"><span class="dot-{risk_class}">{emoji}</span> {risk_label}</span>
            </div>
            <div class="risk-pct">{pct:.1f}<span style="font-size:18px;color:#aaa;font-weight:300">%</span></div>
            <div class="risk-track"><div class="risk-fill-{risk_class}" style="width:{pct:.1f}%"></div></div>
        </div>
        """, unsafe_allow_html=True)

        # Model breakdown
        st.markdown(f"""
        <div style="margin-bottom:6px;font-size:11px;letter-spacing:0.8px;text-transform:uppercase;color:#aaa;font-weight:500">{T["ensemble"]}</div>
        <div class="model-row">
            <div class="model-cell">
                <div class="model-cell-name">Log. Reg.</div>
                <div class="model-cell-val">{lr_prob*100:.1f}%</div>
            </div>
            <div class="model-cell">
                <div class="model-cell-name">Rand. Forest</div>
                <div class="model-cell-val">{rf_prob*100:.1f}%</div>
            </div>
            <div class="model-cell">
                <div class="model-cell-name">Grad. Boost</div>
                <div class="model-cell-val">{gb_prob*100:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Explain", f"{T['domain']}", f"{T['vector']}", f"{T['report']}", f"{T['history']}"
        ])

        with tab1:
            st.markdown(f'<div class="card-label">{T["why"]}</div>', unsafe_allow_html=True)
            if explanations:
                chips = "".join([f'<span class="chip">{e}</span>' for e in explanations])
                st.markdown(f'<div class="chip-wrap">{chips}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="advice-box advice-safe">✓ {T["no_features"]}</div>', unsafe_allow_html=True)

            st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
            st.markdown('<div class="card-label">Feature contribution · Logistic Regression</div>', unsafe_allow_html=True)
            coef = lr_model.named_steps["clf"].coef_[0]
            feature_names = list(X_input.columns)
            contrib = [[name, round(float(val), 3), round(float(w), 3), round(float(val) * round(float(w), 3), 3)]
                       for name, val, w in zip(feature_names, X_input.iloc[0], coef)]
            contrib_df = pd.DataFrame(contrib, columns=["Feature", "Value", "Weight", "Contribution"])
            st.dataframe(contrib_df.sort_values("Contribution", ascending=False), use_container_width=True, hide_index=True)

            st.markdown('<div class="card-label" style="margin-top:16px">Feature importance · Random Forest</div>', unsafe_allow_html=True)
            rf_importances = rf_model.named_steps["clf"].feature_importances_
            imp_df = pd.DataFrame({"Feature": feature_names, "Importance": [round(float(i), 4) for i in rf_importances]}).sort_values("Importance", ascending=False)
            st.dataframe(imp_df, use_container_width=True, hide_index=True)

            st.markdown(f'<div class="card-label" style="margin-top:16px">{T["advice"]}</div>', unsafe_allow_html=True)
            if pred == 1:
                st.markdown(f'<div class="advice-box advice-danger">⚠ {T["bad_advice"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="advice-box advice-safe">✓ {T["good_advice"]}</div>', unsafe_allow_html=True)

        with tab2:
            st.markdown(f'<div class="card-label">{T["domain"]}</div>', unsafe_allow_html=True)
            if domains:
                for d in domains:
                    flags = []
                    if any(w in d for w in suspicious_domain_words):
                        flags.append('<span class="domain-flag flag-warn">suspicious keyword</span>')
                    if len(d) > 20:
                        flags.append('<span class="domain-flag flag-warn">long domain</span>')
                    if any(d.endswith(z) for z in suspicious_zones):
                        flags.append('<span class="domain-flag flag-danger">suspicious TLD</span>')
                    if any(ch.isdigit() for ch in d):
                        flags.append('<span class="domain-flag flag-warn">contains digits</span>')
                    if not flags:
                        flags.append('<span class="domain-flag flag-ok">no issues</span>')
                    flag_html = "".join(flags)
                    st.markdown(f"""
                    <div class="domain-row">
                        <div class="domain-name">{d}</div>
                        <div class="domain-flags">{flag_html}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="color:#aaa;font-size:13px;padding:12px 0">{T["no_domain"]}</div>', unsafe_allow_html=True)

        with tab3:
            st.markdown(f'<div class="card-label">{T["vector"]}</div>', unsafe_allow_html=True)
            display_df = X_input.T.reset_index()
            display_df.columns = ["Feature", "Value"]
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            st.markdown(f'<div class="card-label" style="margin-top:16px">{T["text_stats"]}</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric(T["char_count"], char_count)
            c2.metric(T["word_count"], word_count_val)
            c3.metric("URLs", len(domains))

        with tab4:
            report = f"""AI Fraud Detector Report
========================
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Language: {lang}

INPUT TEXT:
{input_text}

ANALYSIS RESULTS:
-----------------
Logistic Regression:  {lr_prob*100:.1f}%
Random Forest:        {rf_prob*100:.1f}%
Gradient Boosting:    {gb_prob*100:.1f}%
Ensemble average:     {raw_prob*100:.1f}%
Rule boost:           +{boost*100:.1f}%
Final fraud risk:     {prob*100:.1f}%
Risk level:           {risk_label}
Decision threshold:   {threshold}
Prediction:           {"FRAUD" if pred == 1 else "SAFE"}

DETECTED INDICATORS ({len(explanations)}):
{chr(10).join("- " + e for e in explanations) if explanations else "None"}

DETECTED DOMAINS ({len(domains)}):
{chr(10).join("- " + d for d in domains) if domains else "None"}

TEXT STATS:
Characters: {char_count}  |  Words: {word_count_val}

SECURITY ADVICE:
{T["bad_advice"] if pred == 1 else T["good_advice"]}
"""
            st.markdown(f'<div class="card-label">{T["report"]}</div>', unsafe_allow_html=True)
            st.download_button(
                T["download"],
                report,
                file_name=f"fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                use_container_width=True
            )

        st.session_state.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Risk %": round(prob * 100, 1),
            "Level": risk_label,
            "Verdict": "FRAUD" if pred == 1 else "SAFE",
            "Indicators": len(explanations),
            "Preview": input_text[:60] + ("…" if len(input_text) > 60 else ""),
        })

        with tab5:
            if st.session_state.history:
                st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
                if st.button(T["clear_history"]):
                    st.session_state.history = []
                    st.rerun()
            else:
                st.markdown(f'<div style="color:#aaa;font-size:13px">{T["no_history"]}</div>', unsafe_allow_html=True)

# =========================
# HOW IT WORKS
# =========================
with st.expander(f"↓ {T['how']}"):
    if lang == "🇰🇿 KZ":
        st.write("1. Мәтіннен 18 белгі алынады. 2. Олар сандық векторға айналады. 3. 3 ML модель ықтималдық есептейді. 4. Ереже күшейткіш қолданылады. 5. Нәтиже мен кеңес беріледі.")
    elif lang == "🇷🇺 RU":
        st.write("1. Из текста извлекаются 18 признаков. 2. Они преобразуются в числовой вектор. 3. Три ML-модели (LR, RF, GB) независимо вычисляют вероятность мошенничества. 4. Вероятности усредняются и применяется rule-based boost. 5. Система выводит результат и совет по безопасности.")
    else:
        st.write("1. 18 features are extracted from the text. 2. Converted into a numeric vector. 3. Three ML models (LR, RF, GB) independently predict fraud probability. 4. Probabilities are averaged and a rule-based boost is applied. 5. The app displays the result, explanation, and safety advice.")

st.markdown(f'<div class="footer">{T["footer"]} · {datetime.now().year}</div>', unsafe_allow_html=True)
