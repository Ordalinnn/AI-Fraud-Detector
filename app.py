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
    page_title="AI Fraud Detector",
    page_icon="🔐",
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
        "title": "AI Fraud Detector",
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
        "model_metrics": "Модель метрикалары",
        "accuracy": "Дәлдік",
        "clear_history": "Тарихты тазалау",
        "char_count": "Таңба саны",
        "word_count": "Сөз саны",
        "text_stats": "Мәтін статистикасы",
        "ensemble": "Ансамбль болжамы",
        "no_history": "Тарих жоқ",
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
        "model_metrics": "Метрики модели",
        "accuracy": "Точность",
        "clear_history": "Очистить историю",
        "char_count": "Символов",
        "word_count": "Слов",
        "text_stats": "Статистика текста",
        "ensemble": "Ансамблевый прогноз",
        "no_history": "История пуста",
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
# EXPANDED TRAINING DATA
# =========================
data = [
    # FRAUD: bank / card / account
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

    # FRAUD: delivery / prize / fake government / relatives
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

    # FRAUD: English examples
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

    # FRAUD: Kazakh examples
    ["құттықтаймыз сіз ұтыс ұттыңыз карта деректерін енгізіңіз", 1],
    ["сіздің картаңыз бұғатталды SMS кодты жіберіңіз", 1],
    ["шұғыл түрде сілтемеге өтіп аккаунтты растаңыз", 1],
    ["банк қызметкерімін кодты айтыңыз", 1],
    ["қауіпсіз шотқа ақша аударыңыз", 1],
    ["жеке кабинетіңіз жабылады құпия кодты енгізіңіз", 1],
    ["сіздің шотыңыз бұғатталды ресми нөмірге хабарласыңыз", 1],
    ["жеделдетілген несие алыңыз кодты растаңыз", 1],

    # FRAUD: Investment / job scams
    ["guaranteed 30% weekly returns send money to activate account", 1],
    ["work from home earn 500 a day send registration fee", 1],
    ["your investment is ready withdraw by entering card details", 1],
    ["гарантированный доход 30 процентов переведите деньги для активации", 1],
    ["удаленная работа заработок 500 в день отправьте регистрационный взнос", 1],

    # SAFE examples — expanded and more diverse
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

urgent_words = [
    "срочно", "шұғыл", "быстро", "немедленно", "қазір", "тез",
    "urgent", "now", "immediately", "asap", "прямо сейчас",
    "сейчас же", "без промедления"
]
secret_words = [
    "код", "пароль", "cvv", "sms", "құпия", "password",
    "пин", "pin", "код подтверждения", "verification code",
    "one time code", "otp"
]
money_words = [
    "карта", "счет", "банк", "ақша", "төле", "оплата", "перевод",
    "баланс", "средства", "деньги", "transfer", "payment",
    "wallet", "iban"
]
threat_words = [
    "заблокирована", "удален", "штраф", "угрозой", "бұғатталды", "жабылады",
    "blocked", "suspended", "terminated", "penalty", "freeze",
    "ограничен", "будет закрыт"
]
suspicious_domain_words = [
    "login", "verify", "secure", "bonus", "gift", "bank", "kaspi",
    "account", "support", "confirm", "prize", "payment", "wallet",
    "security", "update", "auth", "free"
]
suspicious_zones = [".xyz", ".top", ".click", ".site", ".online", ".live", ".info", ".icu"]

identity_words = [
    "паспорт", "иин", "удостоверение", "личность", "identity",
    "document", "id card", "жсн", "құжат"
]
reward_words = [
    "выиграли", "приз", "бонус", "подарок", "акция", "компенсация",
    "won", "prize", "gift", "bonus", "reward", "ұтыс", "сыйлық"
]
pressure_phrases = [
    "не говорите никому", "никому не сообщайте", "это секретно",
    "только сейчас", "последний шанс", "иначе", "do not tell anyone",
    "last chance", "only now", "қазір ғана"
]

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
    text_lower = text.lower()
    urls = extract_urls(text_lower)
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
        # FIX: new meaningful features added
        "text_length": len(text),
        "word_count": len(words),
        "avg_word_length": round(avg_word_len, 2),
        "url_count": len(urls),
        "has_multiple_warnings": int(
            count_matches(text_lower, urgent_words) > 0
            and count_matches(text_lower, threat_words) > 0
        ),
    }, domains

def explain(features):
    if lang == "🇰🇿 KZ":
        labels = {
            "has_link": "Сілтеме анықталды",
            "urgent_count": "Шұғыл әрекетке шақыру бар",
            "secret_count": "Код / пароль / CVV сұрауы мүмкін",
            "money_count": "Банк, ақша немесе картаға қатысты сөздер бар",
            "threat_count": "Қорқыту немесе қысым жасау белгісі бар",
            "identity_count": "Жеке құжат немесе жеке ақпарат сұралуы мүмкін",
            "reward_count": "Ұтыс, бонус немесе сыйлық уәдесі бар",
            "pressure_count": "Қысым жасау немесе құпия ұстау белгісі бар",
            "suspicious_domain": "Доменде күмәнді сөздер бар",
            "long_domain": "Домен ұзындығы күмәнді",
            "suspicious_zone": "Күмәнді домен зонасы анықталды",
            "digit_domain": "Доменде цифрлар бар",
            "digit_count": "Мәтінде көп сан немесе код кездеседі",
            "exclamation_count": "Көп леп белгісі қолданылған",
            "uppercase_count": "Үлкен әріптер көп қолданылған",
            "has_multiple_warnings": "Бір уақытта шұғылдық және қорқыту бар",
            "url_count": "Бірнеше сілтеме анықталды",
        }
    elif lang == "🇷🇺 RU":
        labels = {
            "has_link": "Обнаружена ссылка",
            "urgent_count": "Есть срочный призыв к действию",
            "secret_count": "Возможен запрос кода / пароля / CVV",
            "money_count": "Есть слова о банке, деньгах или карте",
            "threat_count": "Есть признаки давления или угрозы",
            "identity_count": "Возможен запрос личных документов или данных",
            "reward_count": "Есть обещание выигрыша, бонуса или подарка",
            "pressure_count": "Есть давление или просьба держать всё в секрете",
            "suspicious_domain": "В домене есть подозрительные слова",
            "long_domain": "Домен подозрительно длинный",
            "suspicious_zone": "Обнаружена подозрительная доменная зона",
            "digit_domain": "В домене есть цифры",
            "digit_count": "В тексте много чисел или кодов",
            "exclamation_count": "Используется много восклицательных знаков",
            "uppercase_count": "Используется много заглавных букв",
            "has_multiple_warnings": "Одновременно присутствуют срочность и угроза",
            "url_count": "Обнаружено несколько ссылок",
        }
    else:
        labels = {
            "has_link": "A link was detected",
            "urgent_count": "Urgent action words were found",
            "secret_count": "Possible request for code / password / CVV",
            "money_count": "Bank, money, or card-related words were found",
            "threat_count": "Pressure or threat indicators were found",
            "identity_count": "Possible request for personal identity data",
            "reward_count": "Prize, bonus, or gift promise was found",
            "pressure_count": "Pressure or secrecy phrase was found",
            "suspicious_domain": "Suspicious words were found in the domain",
            "long_domain": "The domain is suspiciously long",
            "suspicious_zone": "Suspicious domain zone detected",
            "digit_domain": "The domain contains numbers",
            "digit_count": "The text contains many numbers or codes",
            "exclamation_count": "Many exclamation marks were used",
            "uppercase_count": "Many uppercase letters were used",
            "has_multiple_warnings": "Both urgency and threat were detected simultaneously",
            "url_count": "Multiple links were detected",
        }

    # Only explain features that are relevant (non-zero and meaningful)
    irrelevant = {"text_length", "word_count", "avg_word_length"}
    return [labels[k] for k, v in features.items() if v > 0 and k in labels and k not in irrelevant]

def risk_style(prob):
    if prob < 0.3:
        return T["low"], "risk-low", "🟢"
    if prob < 0.6:
        return T["mid"], "risk-mid", "🟡"
    if prob < 0.8:
        return T["high"], "risk-high", "🟠"
    return T["critical"], "risk-critical", "🔴"

def rule_boost(features):
    """
    Rule-based boost for realistic scam patterns.
    Capped at 0.30 to prevent runaway scores on safe messages.
    """
    boost = 0.0
    if features["has_link"] and features["secret_count"]:
        boost += 0.15
    if features["urgent_count"] and features["money_count"]:
        boost += 0.12
    if features["secret_count"] and features["money_count"]:
        boost += 0.15
    if features["threat_count"] and features["has_link"]:
        boost += 0.10
    if features["reward_count"] and features["money_count"]:
        boost += 0.12
    if features["pressure_count"]:
        boost += 0.10
    if features["suspicious_zone"] or features["suspicious_domain"]:
        boost += 0.10
    if features["has_multiple_warnings"]:
        boost += 0.08
    if features["url_count"] > 1:
        boost += 0.05
    return min(boost, 0.30)  # FIX: cap boost so safe messages aren't over-flagged

# =========================
# TRAIN MODELS (ENSEMBLE)
# =========================
@st.cache_resource(show_spinner=False)
def train_models():
    rows, labels = [], []
    for text, label in data:
        f, _ = extract_features(text)
        rows.append(f)
        labels.append(label)

    X_train = pd.DataFrame(rows)
    y_train = np.array(labels)

    # FIX: Use ensemble of 3 models for more reliable predictions
    lr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])
    rf_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    gb_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42))
    ])

    lr_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)

    # Cross-validation accuracy for each model
    lr_cv = cross_val_score(lr_model, X_train, y_train, cv=5, scoring="accuracy").mean()
    rf_cv = cross_val_score(rf_model, X_train, y_train, cv=5, scoring="accuracy").mean()
    gb_cv = cross_val_score(gb_model, X_train, y_train, cv=5, scoring="accuracy").mean()

    metrics = {
        "Logistic Regression": round(lr_cv * 100, 1),
        "Random Forest": round(rf_cv * 100, 1),
        "Gradient Boosting": round(gb_cv * 100, 1),
    }

    return lr_model, rf_model, gb_model, metrics, X_train, y_train


lr_model, rf_model, gb_model, model_metrics, X_train, y_train = train_models()

# =========================
# UI STYLE
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
        linear-gradient(135deg, rgba(15,23,42,0.98), rgba(30,64,175,0.95) 52%, rgba(13,148,136,0.94));
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

.site-logo {
    width: 86px;
    height: 86px;
    object-fit: contain;
    filter: drop-shadow(0 0 18px rgba(34,211,238,0.45));
}

.sidebar-logo {
    width: 72px;
    height: 72px;
    object-fit: contain;
    filter: drop-shadow(0 0 12px rgba(34,211,238,0.45));
    margin-bottom: 8px;
}

.logo-row {
    display: flex;
    align-items: center;
    gap: 18px;
    margin-bottom: 10px;
}

.logo-title {
    font-size: 42px;
    line-height: 1.05;
    font-weight: 900;
    letter-spacing: -1.2px;
}

.logo-title span {
    color: #22d3ee;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    margin-top: 34px;
    padding: 18px;
}

.metrics-bar {
    background: #f1f5f9;
    border-radius: 20px;
    padding: 18px 22px;
    margin: 12px 0;
    border: 1px solid #e2e8f0;
}

.metrics-row {
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
}

.metrics-item {
    background: white;
    border-radius: 14px;
    padding: 12px 18px;
    border: 1px solid #e2e8f0;
    flex: 1;
    min-width: 120px;
    text-align: center;
}

.metrics-item-label {
    font-size: 12px;
    color: #64748b;
    font-weight: 700;
    text-transform: uppercase;
}

.metrics-item-val {
    font-size: 22px;
    font-weight: 900;
    color: #0f172a;
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
    if LOGO_HTML:
        st.markdown(f'<img src="{LOGO_HTML}" class="sidebar-logo">', unsafe_allow_html=True)
    st.markdown(f"## {T['title']}")
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
        [T["sms"], T["call"], T["file"]]
    )

    threshold = st.slider(T["threshold"], 0.1, 0.9, 0.5, 0.05)

    st.divider()
    st.markdown(f"### 🧪 {T['demo']}")

    demo_texts = {
        "Fraud SMS": "Срочно! Ваша карта заблокирована. Отправьте код из SMS и перейдите по ссылке http://secure-login.xyz",
        "Fraud Call": "Здравствуйте, я сотрудник службы безопасности банка. По вашему счету подозрительная операция. Назовите код из SMS, чтобы мы отменили перевод.",
        "Safe Message": "Привет, завтра урок математики в 9:00. Не забудь тетрадь.",
        "Fake Delivery": "Ваша посылка задержана. Срочно оплатите таможенную пошлину по ссылке http://delivery-pay-online.xyz",
        "Fake Prize": "Поздравляем! Вы выиграли приз. Для получения подарка введите номер карты и CVV.",
        "Relative Scam": "Ваш родственник попал в аварию. Срочно переведите деньги, никому не говорите.",
        "Fake Job Offer": "Поздравляем, вы приняты на удаленную работу. Для оформления выплат отправьте фото удостоверения, номер карты и OTP-код из SMS.",
        "Marketplace Prepayment Scam": "Здравствуйте, я покупатель с маркетплейса. Подтвердите получение оплаты: перейдите по ссылке https://safe-deal-confirm.top и введите данные карты.",
        "Fake Utility Debt": "Уведомление ЖКХ: у вас долг за коммунальные услуги. Во избежание отключения света оплатите сегодня по ссылке http://pay-service-24.site.",
        "Investment Scam": "Гарантированный доход 30% в неделю! Переведите деньги на инвестиционный счет и сообщите код подтверждения для активации."
    }

    demo = st.selectbox(
        T["demo"],
        list(demo_texts.keys())
    )

    st.divider()
    st.markdown(f"### 📊 {T['model_metrics']}")
    for model_name, acc in model_metrics.items():
        color = "#16a34a" if acc >= 90 else "#ca8a04"
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.1)">'
            f'<span style="color:#94a3b8;font-size:12px">{model_name}</span>'
            f'<span style="color:{color};font-weight:800;font-size:13px">{acc}%</span></div>',
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown("### 🛡️ Project stack")
    for item in ["Ensemble (LR + RF + GB)", "Feature Engineering", "Domain Analysis",
                 "Explainable AI", "Rule-based Risk Boost", "Real-life Scam Scenarios"]:
        st.markdown(f"• {item}")

# =========================
# HERO
# =========================
logo_block = f'<img src="{LOGO_HTML}" class="site-logo">' if LOGO_HTML else '<div style="font-size:58px">🔐</div>'

st.markdown(f"""
<div class="hero">
    <div class="hero-grid">
        <div>
            <div class="logo-row">
                {logo_block}
                <div>
                    <div class="hero-kicker">⚡ AI-powered safety scanner</div>
                    <div class="logo-title"><span>AI</span> FRAUD<br>DETECTOR</div>
                </div>
            </div>
            <div class="hero-subtitle">{T['subtitle']}</div>
            <span class="badge">Ensemble ML</span>
            <span class="badge">Domain Analysis</span>
            <span class="badge">Explainable AI</span>
            <span class="badge">Risk Report</span>
        </div>
        <div class="hero-panel">
            <div class="hero-panel-title">Prototype readiness</div>
            <div class="hero-panel-value">Demo-ready</div>
            <div class="hero-panel-small">Analyzes text, links, pressure words, secret-code requests and suspicious domains using an ensemble of 3 ML models.</div>
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

    with st.form("analysis_form", clear_on_submit=False):
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

        # FIX: Show live character/word count below textarea
        char_count = len(input_text)
        word_count_val = len(input_text.split()) if input_text.strip() else 0
        st.caption(f"📝 {T['char_count']}: {char_count} | {T['word_count']}: {word_count_val}")

        analyze = st.form_submit_button(T["analyze"], use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown(f"""
    <div class="glass-card feature-list">
        <div class="section-title">⚙️ {T['features']}</div>
        <p>🧾 Text analysis</p>
        <p>🌐 Domain analysis</p>
        <p>🧠 Ensemble ML (LR+RF+GB)</p>
        <p>🔍 Explainable result</p>
        <p>📥 Downloadable report</p>
        <p>🚨 Real scam scenarios</p>
        <p>⚡ Rule-based risk boost</p>
        <p>📊 Model accuracy shown</p>
    </div>
    """, unsafe_allow_html=True)

# =========================
# HISTORY INIT
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

        # FIX: Ensemble prediction — average probabilities from all 3 models
        lr_prob = lr_model.predict_proba(X_input)[0][1]
        rf_prob = rf_model.predict_proba(X_input)[0][1]
        gb_prob = gb_model.predict_proba(X_input)[0][1]
        raw_prob = (lr_prob + rf_prob + gb_prob) / 3.0

        boost = rule_boost(features)
        prob = min(0.99, raw_prob + boost)
        pred = int(prob >= threshold)
        risk_label, risk_class, emoji = risk_style(prob)
        explanations = explain(features)

        st.markdown(f'<div class="section-title">📊 {T["result"]}</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="metric-icon">%</div><div class="metric-label">{T["risk"]}</div><div class="metric-value">{prob*100:.1f}%</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-icon">⚠️</div><div class="metric-label">{T["detected"]}</div><div class="metric-value">{len(explanations)}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-icon">🎚️</div><div class="metric-label">{T["threshold"]}</div><div class="metric-value">{threshold:.2f}</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="metric-icon">🧠</div><div class="metric-label">{T["model"]}</div><div class="metric-value">3-way</div></div>', unsafe_allow_html=True)

        st.markdown(f'<div class="{risk_class}">{emoji} {risk_label}</div>', unsafe_allow_html=True)
        st.progress(float(prob))

        # FIX: Show per-model breakdown
        st.markdown(f"**{T['ensemble']}:**")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Logistic Regression", f"{lr_prob*100:.1f}%")
        mc2.metric("Random Forest", f"{rf_prob*100:.1f}%")
        mc3.metric("Gradient Boosting", f"{gb_prob*100:.1f}%")

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

            st.subheader("Feature contribution (Logistic Regression)")
            coef = lr_model.named_steps["clf"].coef_[0]
            feature_names = list(X_input.columns)
            contrib = []
            for name, value, w in zip(feature_names, X_input.iloc[0], coef):
                contrib.append([name, round(float(value), 3), round(float(w), 3), round(float(value) * round(float(w), 3), 3)])

            contrib_df = pd.DataFrame(contrib, columns=["Feature", "Value", "Weight", "Contribution"])
            # FIX: Highlight most impactful rows in dataframe
            st.dataframe(
                contrib_df.sort_values("Contribution", ascending=False),
                use_container_width=True,
                hide_index=True
            )

            # FIX: Also show Random Forest importances
            st.subheader("Feature importance (Random Forest)")
            rf_importances = rf_model.named_steps["clf"].feature_importances_
            imp_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": [round(float(i), 4) for i in rf_importances]
            }).sort_values("Importance", ascending=False)
            st.dataframe(imp_df, use_container_width=True, hide_index=True)

            st.subheader(T["advice"])
            if pred == 1:
                st.error(T["bad_advice"])
            else:
                st.success(T["good_advice"])

        with tab2:
            st.subheader(T["domain"])
            if domains:
                for d in domains:
                    flags = []
                    if any(w in d for w in suspicious_domain_words):
                        flags.append("⚠️ Suspicious keyword")
                    if len(d) > 20:
                        flags.append("⚠️ Long domain")
                    if any(d.endswith(z) for z in suspicious_zones):
                        flags.append("🔴 Suspicious TLD")
                    if any(ch.isdigit() for ch in d):
                        flags.append("⚠️ Contains digits")
                    flag_str = " | ".join(flags) if flags else "✅ No issues found"
                    st.markdown(
                        f'<div class="domain-box">🌐 {d}<br><small style="color:#94a3b8">{flag_str}</small></div>',
                        unsafe_allow_html=True
                    )
            else:
                st.info(T["no_domain"])

        with tab3:
            st.subheader(T["vector"])
            # FIX: Transpose for better readability
            display_df = X_input.T.reset_index()
            display_df.columns = ["Feature", "Value"]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # FIX: Show text statistics
            st.subheader(T["text_stats"])
            ts1, ts2, ts3 = st.columns(3)
            ts1.metric(T["char_count"], char_count)
            ts2.metric(T["word_count"], word_count_val)
            ts3.metric("URLs", len(domains))

        with tab4:
            report = f"""
AI Fraud Detector Report
========================
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Language: {lang}

INPUT TEXT:
{input_text}

ANALYSIS RESULTS:
-----------------
Logistic Regression probability: {lr_prob*100:.1f}%
Random Forest probability:       {rf_prob*100:.1f}%
Gradient Boosting probability:   {gb_prob*100:.1f}%
Ensemble average (raw):          {raw_prob*100:.1f}%
Rule boost applied:              +{boost*100:.1f}%
Final fraud risk:                {prob*100:.1f}%
Risk level:                      {risk_label}
Decision threshold:              {threshold}
Prediction:                      {"FRAUD" if pred == 1 else "SAFE"}

DETECTED FEATURES ({len(explanations)}):
{chr(10).join("- " + e for e in explanations) if explanations else "No strong fraud indicators"}

DETECTED DOMAINS ({len(domains)}):
{chr(10).join("- " + d for d in domains) if domains else "No domains"}

TEXT STATS:
Characters: {char_count}
Words: {word_count_val}

SECURITY ADVICE:
{T["bad_advice"] if pred == 1 else T["good_advice"]}
"""
            st.download_button(
                T["download"],
                report,
                file_name=f"fraud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                use_container_width=True
            )

        # FIX: Save to history with richer info
        st.session_state.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Risk %": round(prob * 100, 1),
            "Level": risk_label,
            "Verdict": "🚨 FRAUD" if pred == 1 else "✅ SAFE",
            "Features": len(explanations),
            "Text": input_text[:70] + ("..." if len(input_text) > 70 else ""),
        })

        with tab5:
            if st.session_state.history:
                st.dataframe(
                    pd.DataFrame(st.session_state.history),
                    use_container_width=True,
                    hide_index=True
                )
                # FIX: Button to clear history
                if st.button(T["clear_history"]):
                    st.session_state.history = []
                    st.rerun()
            else:
                st.info(T["no_history"])

# =========================
# HOW IT WORKS
# =========================
with st.expander(f"📘 {T['how']}"):
    if lang == "🇰🇿 KZ":
        st.write("1. Мәтіннен 18 белгі алынады. 2. Олар сандық векторға айналады. 3. 3 ML модель ықтималдық есептейді. 4. Ереже күшейткіш қолданылады. 5. Сайт нәтиже мен кеңес береді.")
    elif lang == "🇷🇺 RU":
        st.write("1. Из текста извлекаются 18 признаков. 2. Они превращаются в числовой вектор. 3. 3 модели ML считают вероятность мошенничества. 4. Применяется ансамблирование и правило-буст. 5. Сайт показывает результат и совет по безопасности.")
    else:
        st.write("1. 18 features are extracted from the text. 2. They are converted into a numeric vector. 3. Three ML models (LR, RF, GB) independently predict fraud probability. 4. Probabilities are averaged and a rule-based boost is applied. 5. The app shows the result, explanation, and safety advice.")

st.markdown(f'<div class="footer">{T["footer"]} · Ensemble ML · {datetime.now().year}</div>', unsafe_allow_html=True)
