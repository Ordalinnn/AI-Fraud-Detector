import json
import os
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
from pathlib import Path
import base64
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from fraud_logic import (
    domain_flags, extract_features, rule_boost, risk_level, highlight_text,
)
from translations import LANG_OPTIONS, OLD_LANG_MAP, DEFAULT_LANG, get_translations

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
# PWA SUPPORT (installable "Add to Home Screen" on mobile)
# =========================
# Streamlit renders st.markdown into the page body, not the document <head>,
# so a manifest link / theme-color meta / service worker registration can
# only be added by reaching into the real top-level document from inside a
# components.html iframe.
#
# IMPORTANT: use window.top, not window.parent. Streamlit Community Cloud
# wraps the actual app in its OWN nested "streamlitApp" iframe inside the
# real top-level page, so this component lives two frames deep. window.parent
# only reaches that middle iframe (confirmed by inspecting a live deployment:
# our tags landed in the streamlitApp iframe's <head>, while the real outer
# page still showed Streamlit Cloud's own manifest/icon). window.top always
# jumps straight to the true outermost document regardless of nesting depth.
#
# IMPORTANT #2: don't hardcode "/app/static/..." as an absolute, root-relative
# path. Streamlit Community Cloud proxies the actual app (including static
# files) only under an internal "/~/+/" path prefix — a request to the bare
# "/app/static/icon.png" at the domain root gets swallowed by Streamlit
# Cloud's own routing and returns its wrapper HTML instead of our file
# (confirmed with a direct fetch against a live deployment: 200 OK but
# text/html, not image/png). That prefix is Cloud-specific and doesn't exist
# on Codespaces/self-hosted, so hardcoding it would break those instead.
# Fix: resolve the static path against window.parent.location — the actual
# app iframe's own URL — which already contains whatever prefix the current
# host needs, then use that fully-resolved absolute URL on window.top's tags.
#
# Hosts like Streamlit Community Cloud inject their OWN <link rel="manifest">
# / favicon tags for their platform branding, so this must forcibly replace
# any existing tags rather than skip-if-present, or our icon never wins.
#
# Bump this whenever the icon/manifest content changes. Browsers (especially
# iOS Safari) cache favicons/manifests very aggressively, sometimes
# independent of a normal page reload — appending a version query string
# forces a fresh fetch instead of relying on the user to clear their cache.
# If you change static/manifest.json or static/icon.png, bump this AND the
# matching "?v=" inside static/manifest.json's own icon entries.
PWA_ASSET_VERSION = "3"

components.html("""
<script>
(function () {
    const topDoc = window.top.document;
    const v = "__PWA_VERSION__";
    const baseHref = window.parent.location.href;

    function resolvedUrl(path) {
        return new URL(path, baseHref).href + '?v=' + v;
    }

    function setLink(rel, path, type) {
        topDoc.querySelectorAll('link[rel="' + rel + '"]').forEach(function (el) {
            el.remove();
        });
        const link = topDoc.createElement('link');
        link.rel = rel;
        link.href = resolvedUrl(path);
        if (type) link.type = type;
        topDoc.head.appendChild(link);
    }

    setLink('manifest', 'app/static/manifest.json');
    setLink('icon', 'app/static/icon.png', 'image/png');
    setLink('shortcut icon', 'app/static/icon.png', 'image/png');
    setLink('alternate icon', 'app/static/icon.png', 'image/png');
    setLink('apple-touch-icon', 'app/static/icon.png');

    topDoc.querySelectorAll('meta[name="theme-color"]').forEach(function (el) {
        el.remove();
    });
    const meta = topDoc.createElement('meta');
    meta.name = 'theme-color';
    meta.content = '#2563eb';
    topDoc.head.appendChild(meta);

    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register(resolvedUrl('app/static/sw.js')).catch(function () {});
    }
})();
</script>
""".replace("__PWA_VERSION__", PWA_ASSET_VERSION), height=0, width=0)

# =========================
# LOGO HELPER
# =========================
@st.cache_data(show_spinner=False)
def image_to_base64(path: str) -> str:
    """Reads an image file and returns it as a base64 string, so it can be
    embedded directly in HTML (e.g. <img src="data:image/png;base64,...">)
    without a separate network request. Cached so the file is only read once
    per session instead of on every Streamlit rerun."""
    file_path = Path(path)
    if not file_path.exists():
        return ""
    return base64.b64encode(file_path.read_bytes()).decode()

LOGO_B64 = image_to_base64("logo.png")
LOGO_HTML = f"data:image/png;base64,{LOGO_B64}" if LOGO_B64 else ""

# =========================
# HISTORY PERSISTENCE
# =========================
HISTORY_FILE = Path("history.json")

def load_history():
    """Loads past analysis results from history.json (created on first
    analysis) so the History tab survives an app restart, not just a
    Streamlit rerun. Returns an empty list if the file doesn't exist or
    is corrupted."""
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []

def save_history(history):
    """Writes the full history list back to history.json. Called after
    every analysis and after clearing history. Never stored in git
    (see .gitignore) since it can contain pasted message content."""
    try:
        HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass

FEEDBACK_FILE = Path("feedback.json")

def load_feedback():
    """Loads user-submitted "was this correct?" feedback from feedback.json,
    used only to track potential mislabeled predictions for future dataset
    improvements — not read back into the UI anywhere."""
    if FEEDBACK_FILE.exists():
        try:
            return json.loads(FEEDBACK_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []

def save_feedback(feedback):
    """Writes the full feedback list back to feedback.json. Also never
    committed to git — see .gitignore."""
    try:
        FEEDBACK_FILE.write_text(json.dumps(feedback, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass

# =========================
# DEEP ANALYSIS (optional, external — sends text to the Anthropic API)
# =========================
# Unlike the rest of the app, this feature sends the message text to a
# third-party API and therefore breaks the "nothing leaves your device"
# guarantee. It's opt-in per analysis (a button, not an automatic call) and
# clearly labeled as such in the UI. Configure via Streamlit secrets:
#   .streamlit/secrets.toml -> ANTHROPIC_API_KEY = "sk-ant-..."
# (never commit that file — it's already in .gitignore) or the
# ANTHROPIC_API_KEY environment variable for non-Streamlit-Cloud hosts.
DEEP_ANALYSIS_MODEL = "claude-haiku-4-5-20251001"

def get_anthropic_api_key():
    """Reads the API key from Streamlit secrets first, then the
    environment. Returns None (not an error) if unconfigured, so the
    feature can degrade gracefully instead of crashing the app."""
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")
        if key:
            return key
    except FileNotFoundError:
        # No secrets.toml present — expected when the key is configured via
        # environment variable instead. StreamlitSecretNotFoundError
        # subclasses FileNotFoundError.
        pass
    return os.environ.get("ANTHROPIC_API_KEY")

def run_deep_analysis(text, lang_name):
    """Sends the message to Claude for a genuine reasoning-based second
    opinion (not keyword matching): what manipulation tactics it uses, if
    any, and whether it looks like a real scam pattern. Returns the
    response text. Raises on any failure — the caller is responsible for
    catching and showing a friendly error, since network/API issues are
    expected here (rate limits, missing key, no internet)."""
    import anthropic

    api_key = get_anthropic_api_key()
    if not api_key:
        raise RuntimeError("no_api_key")

    client = anthropic.Anthropic(api_key=api_key)
    prompt = (
        f"You are a fraud-detection assistant. Analyze the following message for signs "
        f"of a scam. Respond in {lang_name}, in 3-5 concise sentences: name the specific "
        f"manipulation tactics you notice (if any), say whether this matches a real-world "
        f"scam pattern, and what the recipient should do. If it looks safe, say so plainly "
        f"and explain why.\n\nMessage to analyze:\n\"\"\"\n{text}\n\"\"\""
    )
    response = client.messages.create(
        model=DEEP_ANALYSIS_MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text

# =========================
# LANGUAGE
# =========================
# LANG_OPTIONS, OLD_LANG_MAP, and all translated strings live in
# translations.py, which validates at import time that every language has
# the same set of keys (see validate_translations() there).
if "lang" not in st.session_state:
    st.session_state.lang = DEFAULT_LANG

st.session_state.lang = OLD_LANG_MAP.get(st.session_state.lang, st.session_state.lang)

if st.session_state.lang not in LANG_OPTIONS:
    st.session_state.lang = DEFAULT_LANG

def apply_lang_change():
    """Runs before the rerun triggered by the radio click, so the new
    language is already in session_state when the script body executes
    (avoids needing a second, explicit st.rerun())."""
    st.session_state.lang = st.session_state.lang_selector

lang = st.session_state.lang
T = get_translations(lang)

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

    # FRAUD: Tech support scams
    ["внимание ваш компьютер заражен вирусом позвоните в службу поддержки немедленно", 1],
    ["your computer has been infected call microsoft support immediately to fix it", 1],
    ["компьютеріңізде вирус табылды қолдау қызметіне дереу хабарласыңыз", 1],

    # FRAUD: Romance scams
    ["любимая мне нужны деньги на билет чтобы прилететь к тебе переведи на карту", 1],
    ["i love you but i am stuck at customs please send money to release my package", 1],

    # FRAUD: Crypto / investment platform scams
    ["ваш криптокошелек скомпрометирован подтвердите seed фразу немедленно", 1],
    ["confirm your wallet seed phrase now or you will lose access to your crypto", 1],
    ["инвестируйте в криптовалюту и удвойте деньги за 24 часа переведите биткоин", 1],

    # FRAUD: QR code scams
    ["отсканируйте qr код для оплаты парковки штраф удвоится если не оплатить сейчас", 1],
    ["scan this qr code to pay the parking fine immediately or it will double", 1],

    # FRAUD: Charity scams
    ["помогите пострадавшим от наводнения переведите пожертвование на карту срочно", 1],
    ["donate now to help earthquake victims click this link to send money", 1],

    # FRAUD: Subscription / free trial scams
    ["ваша бесплатная подписка заканчивается сегодня введите данные карты чтобы не потерять доступ", 1],
    ["your free trial ends today enter your card details now to keep access", 1],

    # FRAUD: SIM swap / number deactivation scams
    ["ваш номер будет отключен через час подтвердите данные по ссылке", 1],
    ["your sim card will be deactivated confirm your identity now by clicking here", 1],

    # FRAUD: Social media account recovery scams
    ["ваш инстаграм аккаунт будет удален через 24 часа подтвердите личность по ссылке", 1],
    ["your whatsapp account will be suspended verify now or lose all your chats", 1],
    ["ватсап аккаунтыңыз жойылады растау үшін сілтемеге өтіңіз", 1],

    # FRAUD: Fake tax refund scams
    ["вам одобрен налоговый возврат введите реквизиты карты для получения", 1],
    ["you are eligible for a tax refund enter your bank details to claim it", 1],

    # FRAUD: Lottery / inheritance from abroad
    ["вам полагается наследство от дальнего родственника за границей вышлите данные для перевода", 1],
    ["you have inherited money from a relative abroad send your bank details to claim", 1],

    # FRAUD: More bank/card variants
    ["ваша карта приостановлена нажмите здесь чтобы восстановить доступ", 1],
    ["уважаемый клиент ваш счет заморожен до подтверждения личности", 1],
    ["we detected unusual login activity confirm your identity within 24 hours", 1],
    ["your debit card has been flagged please verify to avoid permanent block", 1],
    ["банкіңіз сізден шұғыл құпия кодты растауды сұрайды", 1],
    ["please update your banking information immediately to avoid service interruption", 1],

    # FRAUD: More delivery / government / relative-in-trouble variants
    ["курьер не смог доставить посылку оплатите повторную доставку по ссылке", 1],
    ["налоговая служба уведомляет о задолженности оплатите штраф немедленно", 1],
    ["your package requires additional customs clearance fee click to pay now", 1],
    ["this is the tax office you owe a fine pay immediately to avoid arrest", 1],
    ["сіздің отбасы мүшесі ауруханада жедел ақша керек", 1],
    ["mom i lost my phone this is my new number please send money urgently", 1],

    # FRAUD: More investment / job variants
    ["earn passive income 1000 dollars daily just sign up and deposit now", 1],
    ["exclusive investment opportunity limited slots deposit today to secure your spot", 1],
    ["стабильный доход без вложений просто зарегистрируйтесь и подтвердите карту", 1],
    ["job offer confirmed send your bank details to receive your first paycheck advance", 1],
    ["сізге жоғары жалақылы жұмыс табылды алдын ала төлем жасаңыз", 1],

    # FRAUD: More tech support variants
    ["apple support your icloud account has been locked verify now to prevent data loss", 1],
    ["предупреждение системы обнаружены критические ошибки звоните в техподдержку", 1],
    ["your license key has expired call this number immediately to renew", 1],

    # FRAUD: More romance scam variants
    ["darling the customs officer needs a fee to release the gift i sent you", 1],
    ["я нахожусь за границей и мне нужна твоя помощь с деньгами на билет", 1],
    ["i cannot video call right now but i need money for an emergency surgery", 1],

    # FRAUD: More crypto variants
    ["binance support team requires your private key to resolve account issue", 1],
    ["удвоение биткоина гарантировано отправьте на этот кошелек и получите х2", 1],
    ["your exchange account is suspended verify your recovery phrase to restore access", 1],
    ["эксклюзивный airdrop подключите кошелек и подпишите транзакцию для получения токенов", 1],

    # FRAUD: More QR code variants
    ["сканируйте код чтобы получить компенсацию за задержку рейса", 1],
    ["scan to claim your free gift card before it expires today", 1],

    # FRAUD: More charity variants
    ["сбор средств для больных детей переведите любую сумму прямо сейчас", 1],
    ["urgent disaster relief fund click here to donate before it is too late", 1],

    # FRAUD: More subscription variants
    ["netflix оплата не прошла обновите данные карты чтобы не потерять доступ", 1],
    ["your streaming subscription payment failed update card details within 24 hours", 1],

    # FRAUD: More SIM swap / social media variants
    ["код восстановления telegram отправлен никому не сообщайте его кроме нас", 1],
    ["your facebook account was reported confirm identity or it will be permanently deleted", 1],
    ["сіздің telegram аккаунтыңызға кіру талабы расталмаса жабылады", 1],
    ["someone tried to log into your account verify now by sharing the code we sent", 1],

    # FRAUD: More tax / inheritance variants
    ["irs notice you have unclaimed funds provide account details to receive payment", 1],
    ["адвокат за границей нашел вам наследство пришлите документы и оплатите сбор", 1],
    ["government stimulus payment pending verify your bank account to receive it", 1],

    # FRAUD: Global brand impersonation (PayPal / Amazon / Apple / Google)
    ["your paypal account has been limited click here to restore full access", 1],
    ["amazon order could not be processed update your payment method immediately", 1],
    ["apple id disabled tap to verify before your account is permanently closed", 1],
    ["google security alert someone tried to access your account verify now", 1],

    # FRAUD: Business email compromise — impersonating a boss or vendor to
    # rush an urgent payment (FBI IC3 2025: among the top scam categories by
    # dollar losses)
    ["срочно оплатите счет поставщику иначе сорвется важная сделка перевод сегодня", 1],
    ["this is the ceo i need you to process an urgent wire transfer today confidentially", 1],
    ["please update the vendor bank account details before today's payment run", 1],
    ["бухгалтерия смените реквизиты поставщика и переведите оплату срочно сегодня", 1],

    # FRAUD: AI-generated deepfake investment videos (FBI IC3 2025 added
    # "AI-related" as a new formal crime category; also reported in
    # Kazakhstan as deepfakes of public figures promoting fake investments)
    ["посмотрите видео известного бизнесмена он лично приглашает вложить деньги и утроить их", 1],
    ["watch this exclusive video where a famous investor personally invites you to triple your money", 1],
    ["белгілі бизнесмен бейнеролда ақшаңызды үш еседе өсіруге шақырады сілтеме арқылы тіркеліңіз", 1],

    # FRAUD: Fake courier demanding the SMS code at the door (reported in
    # Kazakhstan as fake couriers claiming to be from Kaspi.kz)
    ["курьер kaspi доставляет посылку назовите код из смс чтобы подтвердить получение", 1],
    ["delivery courier at your door needs the sms code to confirm you received the package", 1],

    # FRAUD: Expiring rewards points (FTC-reported pattern)
    ["ваши бонусные баллы сгорают сегодня войдите по ссылке чтобы успеть их использовать", 1],
    ["your reward points expire tonight click here before you lose them forever", 1],

    # FRAUD: Traffic violation QR code scam (FTC-reported pattern: fake
    # official notice with a QR code demanding immediate payment)
    ["штраф за нарушение пдд отсканируйте qr код и оплатите сегодня во избежание суда", 1],
    ["unpaid traffic ticket scan this qr code now to avoid court action", 1],

    # FRAUD: Fake "verify you're human" instructions (FTC-reported pattern
    # mimicking CAPTCHA checks to trick users into running malicious steps)
    ["подтвердите что вы не робот и следуйте инструкциям чтобы получить доступ к файлу", 1],
    ["verify you are human by following these steps to unlock your download", 1],

    # FRAUD: QR code swapped at a physical location (table, parking meter)
    ["scan the qr code on the table to pay your bill and get a discount today only", 1],

    # FRAUD: Bank bonus-exchange phishing (reported in Kazakhstan
    # impersonating Halyk Bank)
    ["обменяйте бонусы halyk bank на деньги перейдите по ссылке и подтвердите карту", 1],

    # FRAUD: Unpaid toll SMS scam (FCC/FBI IC3: 60,000+ complaints by 2025,
    # impersonating E-ZPass, FasTrak, and similar toll payment systems)
    ["you have an unpaid toll balance pay now to avoid additional fees and license suspension", 1],
    ["missed toll payment detected settle your balance today via the link to avoid a fine", 1],
    ["у вас есть неоплаченный проезд по платной дороге оплатите сейчас чтобы избежать штрафа", 1],

    # FRAUD: AI voice-cloning "family emergency" scam (2026 reporting: a
    # cloned voice of a grandchild/relative begging for urgent money,
    # generated from a few seconds of public audio)
    ["бабушка это я я попал в аварию мне очень больно пожалуйста переведи деньги никому не говори", 1],
    ["grandma it's me i've been in a car accident please send money right away don't tell anyone", 1],
    ["немен телефон арқылы сөйлесе алмаймын дауысым өзгерген шұғыл ақша керек", 1],

    # FRAUD: "Pig butchering" — romance scam that pivots into a fake crypto
    # investment platform
    ["любимый я нашла платформу для инвестиций в криптовалюту давай вместе вложим и разбогатеем", 1],
    ["darling i've been making great profits on this crypto platform let me help you invest too", 1],

    # FRAUD: Bank "callback" scheme (2026 Russian banking fraud reporting:
    # scammer asks the victim to call back a number instead of clicking a link)
    ["это служба безопасности сбербанка перезвоните на этот номер срочно по поводу вашего счета", 1],
    ["sberbank security department call us back immediately regarding suspicious activity on your account", 1],

    # FRAUD: NFC "tap your phone" scam (2026 reporting: caller talks the
    # victim into tapping their phone on a terminal under a fake pretext,
    # linking it to the scammer's payment device)
    ["поднесите телефон к терминалу чтобы отменить платеж следуйте инструкциям оператора", 1],
    ["hold your phone near the terminal and follow the operator's instructions to cancel the charge", 1],

    # FRAUD: Landlord impersonation / rent redirection scam
    ["это арендодатель переведите оплату за квартиру на новые реквизиты счета", 1],
    ["hi it's your landlord please send this month's rent to my new bank account details below", 1],
    ["арендодатель сменил банк переведите оплату по новым реквизитам во вложении", 1],
    ["your landlord changed banks please transfer rent to the new account attached", 1],
    ["риэлтор сообщает что депозит нужно перевести на другой счет срочно", 1],
    ["the realtor says the deposit must be transferred to a different account urgently", 1],
    ["хозяин дома просит оплатить аренду заранее на карту его помощника", 1],
    ["the homeowner is asking you to prepay rent to his assistant's card instead", 1],
    ["арендодатель пишет что банк заблокировал его счет переведите на этот номер", 1],
    ["the landlord says his bank account is frozen please send rent to this number instead", 1],
    ["үй иесі жаңа шотқа ақы төлеуді сұрайды міне деректемелер", 1],
    ["жалдаушы төлемді басқа шотқа аударуды сұрайды шұғыл", 1],

    # FRAUD: Marketplace fake-verification scam (Avito/OLX/Telegram) — the
    # scammer poses as a BUYER, not the usual seller-side scam, and talks
    # the seller into paying a small fee on a fake "stolen item registry"
    # site before a meetup, promising reimbursement, then disappears once
    # the seller pays and sends the PDF "certificate"
    ["покупатель просит проверить ноутбук на воровство через сайт check-tech-base.ru и оплатить справку", 1],
    ["buyer asks you to verify the laptop isn't stolen through a paid verification website before meeting", 1],
    ["покупатель обещает вернуть деньги за проверку при встрече но пропадает после оплаты", 1],
    ["the buyer promises to reimburse the verification fee at pickup then goes silent after you pay", 1],
    ["покупатель на авито просит оплатить официальный реестр техники прежде чем приедет смотреть товар", 1],
    ["the buyer on the marketplace insists you pay for an official equipment registry check before viewing", 1],
    ["сатып алушы ноутбукты тексеру үшін ақылы сайтқа сілтеме жіберіп ақша төлеуді сұрайды", 1],
    ["сатып алушы кездесуден бұрын заттың тазалығын растайтын ақылы қызметке төлеуді талап етеді", 1],

    # =========================
    # SYNTHETIC AUGMENTATION: the batch below systematically varies real bank
    # and service names, and rephrases the categories above, so the model
    # sees each pattern across many concrete institutions rather than one
    # fixed name. These are written examples, not independently sourced
    # incidents — the underlying patterns are the ones researched above.
    # =========================

    # FRAUD: Bank/card, more bank names
    ["ваша карта kaspi заблокирована перейдите по ссылке для разблокировки", 1],
    ["служба безопасности kaspi обнаружила подозрительную операцию сообщите код из смс", 1],
    ["kaspi ваш счет временно ограничен подтвердите личность по ссылке", 1],
    ["внимание клиент kaspi лимит по карте превышен отправьте код для снятия ограничения", 1],
    ["kaspi онлайн банкинг требует обновления пароля перейдите по ссылке иначе доступ закроется", 1],
    ["уважаемый клиент kaspi на вашей карте обнаружена подозрительная активность звоните немедленно", 1],
    ["ваша карта halyk bank заблокирована перейдите по ссылке для разблокировки", 1],
    ["служба безопасности halyk bank обнаружила подозрительную операцию сообщите код из смс", 1],
    ["halyk bank ваш счет временно ограничен подтвердите личность по ссылке", 1],
    ["внимание клиент halyk bank лимит по карте превышен отправьте код для снятия ограничения", 1],
    ["halyk bank онлайн банкинг требует обновления пароля перейдите по ссылке иначе доступ закроется", 1],
    ["уважаемый клиент halyk bank на вашей карте обнаружена подозрительная активность звоните немедленно", 1],
    ["ваша карта jusan bank заблокирована перейдите по ссылке для разблокировки", 1],
    ["служба безопасности jusan bank обнаружила подозрительную операцию сообщите код из смс", 1],
    ["jusan bank ваш счет временно ограничен подтвердите личность по ссылке", 1],
    ["внимание клиент jusan bank лимит по карте превышен отправьте код для снятия ограничения", 1],
    ["jusan bank онлайн банкинг требует обновления пароля перейдите по ссылке иначе доступ закроется", 1],
    ["уважаемый клиент jusan bank на вашей карте обнаружена подозрительная активность звоните немедленно", 1],
    ["ваша карта sberbank заблокирована перейдите по ссылке для разблокировки", 1],
    ["служба безопасности sberbank обнаружила подозрительную операцию сообщите код из смс", 1],
    ["sberbank ваш счет временно ограничен подтвердите личность по ссылке", 1],
    ["внимание клиент sberbank лимит по карте превышен отправьте код для снятия ограничения", 1],
    ["sberbank онлайн банкинг требует обновления пароля перейдите по ссылке иначе доступ закроется", 1],
    ["уважаемый клиент sberbank на вашей карте обнаружена подозрительная активность звоните немедленно", 1],
    ["ваша карта втб заблокирована перейдите по ссылке для разблокировки", 1],
    ["служба безопасности втб обнаружила подозрительную операцию сообщите код из смс", 1],
    ["втб ваш счет временно ограничен подтвердите личность по ссылке", 1],
    ["внимание клиент втб лимит по карте превышен отправьте код для снятия ограничения", 1],
    ["втб онлайн банкинг требует обновления пароля перейдите по ссылке иначе доступ закроется", 1],
    ["уважаемый клиент втб на вашей карте обнаружена подозрительная активность звоните немедленно", 1],
    ["ваша карта tinkoff заблокирована перейдите по ссылке для разблокировки", 1],
    ["служба безопасности tinkoff обнаружила подозрительную операцию сообщите код из смс", 1],
    ["tinkoff ваш счет временно ограничен подтвердите личность по ссылке", 1],
    ["внимание клиент tinkoff лимит по карте превышен отправьте код для снятия ограничения", 1],
    ["tinkoff онлайн банкинг требует обновления пароля перейдите по ссылке иначе доступ закроется", 1],
    ["уважаемый клиент tinkoff на вашей карте обнаружена подозрительная активность звоните немедленно", 1],

    # FRAUD: Delivery/government/relative, more services
    ["ваша посылка от казпочты задержана на таможне оплатите пошлину по ссылке", 1],
    ["казпочта не может доставить ваш заказ подтвердите адрес и оплатите повторную доставку", 1],
    ["ваша посылка от сдэк задержана на таможне оплатите пошлину по ссылке", 1],
    ["сдэк не может доставить ваш заказ подтвердите адрес и оплатите повторную доставку", 1],
    ["ваша посылка от kaspi post задержана на таможне оплатите пошлину по ссылке", 1],
    ["kaspi post не может доставить ваш заказ подтвердите адрес и оплатите повторную доставку", 1],
    ["ваша посылка от boxberry задержана на таможне оплатите пошлину по ссылке", 1],
    ["boxberry не может доставить ваш заказ подтвердите адрес и оплатите повторную доставку", 1],
    ["your parcel from dhl is held at customs pay the clearance fee to release it", 1],
    ["dhl delivery failed confirm your address and pay a redelivery fee online", 1],
    ["your parcel from fedex is held at customs pay the clearance fee to release it", 1],
    ["fedex delivery failed confirm your address and pay a redelivery fee online", 1],
    ["your parcel from ups is held at customs pay the clearance fee to release it", 1],
    ["ups delivery failed confirm your address and pay a redelivery fee online", 1],
    ["your parcel from usps is held at customs pay the clearance fee to release it", 1],
    ["usps delivery failed confirm your address and pay a redelivery fee online", 1],
    ["это акимат вам назначен штраф за нарушение оплатите немедленно по ссылке", 1],
    ["государственная служба сообщает о неоплаченном сборе оплатите сегодня во избежание пени", 1],
    ["this is the department of motor vehicles you have an unpaid fine pay today online", 1],
    ["government notice you owe an outstanding fee settle it now to avoid penalties", 1],
    ["ваш брат попал в полицию нужны деньги на залог срочно переведи и никому не говори", 1],
    ["your cousin was arrested and needs bail money urgently please transfer now and stay quiet", 1],
    ["сын мама я разбил чужую машину нужны деньги чтобы уладить дело срочно", 1],
    ["son here i crashed someone's car and need money to settle it quickly don't tell dad", 1],

    # FRAUD: Investment/job, more variants
    ["гарантированная доходность 40 процентов в месяц вложите сейчас и получите бонус", 1],
    ["guaranteed 40 percent monthly returns invest now and get a bonus", 1],
    ["эксклюзивный инвестиционный клуб примем только сегодня внесите депозит для входа", 1],
    ["exclusive investment club accepting new members today only deposit now to join", 1],
    ["удаленная работа модератором зарплата 300000 тенге отправьте паспортные данные", 1],
    ["remote moderator job pays 3000 dollars a month send your id documents to apply", 1],
    ["вакансия менеджера без опыта высокая зарплата оплатите оформление документов", 1],
    ["no experience manager position high salary pay a processing fee to get hired", 1],
    ["инвестируйте в акции с гарантией без риска переведите средства на партнерский счет", 1],
    ["invest in guaranteed risk free stocks transfer funds to our partner account today", 1],
    ["форекс трейдинг обучение бесплатно только сегодня внесите депозит для старта", 1],
    ["free forex trading course today only deposit funds to start trading now", 1],
    ["работа из дома печать конвертов оплата после первого взноса за материалы", 1],
    ["work from home stuffing envelopes payment after your first materials deposit", 1],
    ["быстрый заработок в интернете переведите активационный взнос чтобы начать", 1],
    ["quick online income transfer an activation fee to get started today", 1],
    ["хайп проект удваивает вклады каждую неделю успейте вложить сейчас", 1],
    ["this hyip project doubles deposits every week invest now before it closes", 1],
    ["приглашаем в mlm бизнес купите стартовый пакет чтобы начать зарабатывать", 1],
    ["join our mlm business buy the starter kit to begin earning today", 1],
    ["тіркеліңіз біздің инвестициялық платформамызға ақша салыңыз және пайда табыңыз", 1],
    ["жұмысқа алу растамасы үшін құжаттарыңызды және ақы төлеңіз", 1],
    ["удаленная вакансия без собеседования зарплата ежедневно отправьте данные карты", 1],
    ["no interview remote job daily payouts send your card details to get paid", 1],

    # FRAUD: Tech support, more brands
    ["microsoft support your windows license has expired call now to avoid data loss", 1],
    ["предупреждение windows обнаружен опасный вирус звоните в поддержку немедленно", 1],
    ["norton antivirus subscription failed renew now or your device stays unprotected", 1],
    ["mcafee alert critical threats detected call this number to remove them now", 1],
    ["apple support your device storage is full and corrupted call now to fix it", 1],
    ["google chrome security warning your browser is infected call support immediately", 1],
    ["техподдержка windows обнаружила ошибку системы звоните по указанному номеру", 1],
    ["антивирус нортон истек срок действия продлите сейчас по ссылке", 1],
    ["внимание вирус заблокировал ваш компьютер звоните в службу поддержки microsoft", 1],
    ["your computer has been hacked call apple support immediately to secure your data", 1],
    ["предупреждение системы безопасности google обнаружена угроза звоните сейчас", 1],
    ["critical alert your system has multiple infections call tech support now to fix", 1],
    ["лицензия windows истекла продлите немедленно чтобы избежать блокировки", 1],
    ["your antivirus expired today renew now by calling this number to stay protected", 1],
    ["внимание ваш ip адрес скомпрометирован звоните в службу безопасности сейчас", 1],
    ["warning your ip address has been compromised call security support right away", 1],
    ["техническая поддержка apple обнаружила проблему с вашим icloud звоните немедленно", 1],
    ["microsoft detected unusual activity on your account call support now to secure it", 1],

    # FRAUD: Romance, more variants
    ["любимый мне нужна виза чтобы приехать к тебе пришли деньги на оформление документов", 1],
    ["sweetheart i need money for a visa to finally come see you please help me", 1],
    ["дорогая у меня заблокирована карта на границе переведи денег чтобы я смог вылететь", 1],
    ["my love my card is frozen at the airport please send money so i can fly to you", 1],
    ["я на буровой платформе интернет плохой но я тебя люблю пришли денег на связь", 1],
    ["i am stationed overseas and need money for a phone card to keep talking to you", 1],
    ["любимая врач сказал что мне нужна срочная операция помоги с деньгами", 1],
    ["darling the doctor says i need urgent surgery please help me with the cost", 1],
    ["сердце мое посылка с подарками застряла на таможне пришли денег на растаможку", 1],
    ["my heart the gift package i sent is stuck in customs please send the clearance fee", 1],
    ["махаббатым саған жақын арада келемін алдымен билетке ақша керек", 1],
    ["жаным қолым сынды операция үшін ақша жіберші өтінемін", 1],

    # FRAUD: Crypto, more variants
    ["срочно переведите биткоин иначе ваш кошелек будет заблокирован навсегда", 1],
    ["urgent transfer your bitcoin now or your wallet will be permanently locked", 1],
    ["binance поддержка просит подтвердить seed фразу для восстановления доступа", 1],
    ["coinbase support needs your recovery phrase to verify your account now", 1],
    ["инвестируйте в новый токен сейчас цена вырастет в 10 раз завтра", 1],
    ["invest in this new token now the price will 10x by tomorrow guaranteed", 1],
    ["эксклюзивный airdrop подключите кошелек metamask чтобы получить бесплатные токены", 1],
    ["exclusive airdrop connect your metamask wallet to claim free tokens now", 1],
    ["ваш криптосчет binance заморожен подтвердите личность перейдя по ссылке", 1],
    ["your binance account has been frozen verify your identity by clicking this link", 1],
    ["удвойте свой биткоин за 24 часа отправьте на этот адрес и получите х2", 1],
    ["double your bitcoin in 24 hours send to this address and receive double back", 1],
    ["новая криптобиржа дает бонус 500 долларов за регистрацию и депозит", 1],
    ["new crypto exchange gives a 500 dollar bonus for signing up and depositing", 1],
    ["технический сбой на бирже переведите средства на резервный кошелек срочно", 1],
    ["exchange technical error transfer your funds to this backup wallet immediately", 1],
    ["крипто-гуру делится секретным сигналом присоединяйтесь к платному каналу сейчас", 1],
    ["crypto guru sharing a secret signal join the paid channel now before it closes", 1],
    ["ваш nft будет удален подтвердите кошелек по ссылке в течение часа", 1],
    ["your nft will be deleted verify your wallet through this link within the hour", 1],
    ["криптобот гарантированно приносит прибыль каждый день внесите депозит сейчас", 1],
    ["this crypto trading bot guarantees daily profit deposit now to activate it", 1],
    ["тіркеліңіз крипто платформасына бонус алу үшін әмиянды қосыңыз", 1],
    ["әмияныңызды растаңыз әйтпесе токендеріңіз жоғалады", 1],

    # FRAUD: QR code, more variants
    ["отсканируйте qr код чтобы получить кэшбэк за последнюю покупку", 1],
    ["scan this qr code to claim your cashback from your last purchase", 1],
    ["qr код на чеке дает скидку 50 процентов отсканируйте прямо сейчас", 1],
    ["scan the qr code on your receipt for a 50 percent discount today only", 1],
    ["отсканируйте код для участия в розыгрыше автомобиля сегодня последний день", 1],
    ["scan this code to enter today's car giveaway last day to participate", 1],
    ["qr код для бесплатного wifi введите данные карты для подтверждения возраста", 1],
    ["scan this qr code for free wifi enter your card details to verify your age", 1],
    ["отсканируйте qr на объявлении чтобы получить полную информацию о квартире", 1],
    ["scan the qr code on this flyer to view the full apartment listing and pay a deposit", 1],
    ["qr код оплаты штрафа гибдд отсканируйте и оплатите со скидкой сегодня", 1],
    ["scan this qr code to pay your traffic fine with a discount available today only", 1],

    # FRAUD: Charity, more variants
    ["сбор денег для пострадавших от пожара переведите любую сумму на этот номер", 1],
    ["collecting funds for wildfire victims send any amount to this number now", 1],
    ["помогите больному ребенку срочно нужна операция переведите деньги сейчас", 1],
    ["help this sick child urgent surgery needed please transfer money right now", 1],
    ["фонд помощи животным собирает пожертвования переходите по ссылке и переводите", 1],
    ["animal rescue fund collecting donations click the link and contribute now", 1],
    ["сбор на восстановление после наводнения каждый рубль важен переведите сейчас", 1],
    ["flood relief fundraiser every dollar helps please donate through this link now", 1],
    ["благотворительная акция для ветеранов войны переведите средства сегодня", 1],
    ["charity drive for war veterans please donate today through this link", 1],
    ["мешіт қайырымдылық жинауда қазір көмек беріңіз сілтеме арқылы", 1],
    ["балаларға көмек керек шұғыл ақша аударыңыз осы нөмірге", 1],

    # FRAUD: Subscription/trial, more services
    ["netflix payment declined update your card details now to avoid losing access", 1],
    ["netflix оплата не прошла обновите карту сейчас чтобы не потерять доступ", 1],
    ["spotify premium subscription failed update your payment method today", 1],
    ["spotify премиум подписка не оплачена обновите способ оплаты сегодня", 1],
    ["youtube premium trial ending update payment info now to continue without ads", 1],
    ["youtube premium пробный период заканчивается обновите оплату сейчас", 1],
    ["icloud storage full upgrade now by entering your payment details to avoid data loss", 1],
    ["icloud хранилище заполнено обновите тариф введя данные карты", 1],
    ["google one payment failed renew your subscription now to keep your files safe", 1],
    ["google one оплата не прошла продлите подписку введя данные карты", 1],
    ["amazon prime membership expiring renew now with your card to keep free shipping", 1],
    ["amazon prime подписка истекает продлите сейчас картой чтобы сохранить доставку", 1],
    ["kaspi pay подписка не оплачена обновите карту чтобы не потерять сервис", 1],
    ["your streaming trial is ending enter payment details now to avoid interruption", 1],
    ["disney plus payment issue update your card now to keep watching", 1],
    ["disney plus проблема с оплатой обновите карту сейчас чтобы продолжить просмотр", 1],
    ["жазылымыңыз аяқталады картаны жаңартыңыз әйтпесе қызмет тоқтайды", 1],
    ["тегін сынақ мерзімі бітеді картаны растап жазылымды жалғастырыңыз", 1],

    # FRAUD: SIM swap / social media, more platforms
    ["ваш whatsapp аккаунт будет заблокирован подтвердите код который мы отправили", 1],
    ["your whatsapp account will be banned confirm the code we just sent you", 1],
    ["telegram обнаружил вход с нового устройства подтвердите код немедленно", 1],
    ["telegram detected a login from a new device confirm the code immediately", 1],
    ["ваш instagram аккаунт нарушил правила подтвердите личность по ссылке или он будет удален", 1],
    ["your instagram account violated our policy verify your identity or it will be deleted", 1],
    ["facebook security alert someone logged in from another country verify now", 1],
    ["facebook безопасность кто то вошел в аккаунт из другой страны подтвердите сейчас", 1],
    ["tiktok обнаружил подозрительную активность подтвердите аккаунт чтобы не потерять его", 1],
    ["tiktok detected suspicious activity verify your account now or lose access", 1],
    ["gmail аккаунт будет удален через 24 часа подтвердите личность по ссылке", 1],
    ["your gmail account will be deleted in 24 hours verify your identity via this link", 1],
    ["ваш номер будет отключен оператором подтвердите данные чтобы сохранить сим карту", 1],
    ["your number will be disconnected by the carrier confirm your details to keep your sim", 1],
    ["сим карта будет заблокирована из за неактивности подтвердите личность сейчас", 1],
    ["your sim card will be blocked due to inactivity verify your identity now", 1],
    ["код восстановления вашего аккаунта отправлен никому его не сообщайте кроме нас", 1],
    ["your account recovery code was sent do not share it with anyone except us", 1],
    ["ваш linkedin аккаунт будет закрыт подтвердите профиль по ссылке немедленно", 1],
    ["your linkedin account will be closed verify your profile through this link now", 1],
    ["twitter x аккаунт нарушил правила сообщества подтвердите личность или потеряете доступ", 1],
    ["your x account violated community guidelines verify your identity or lose access", 1],
    ["whatsapp құпия кодыңызды ешкімге айтпаңыз тек осы жерге растаңыз", 1],
    ["telegram аккаунтыңыз жаңа құрылғыдан кірді кодты растаңыз", 1],

    # FRAUD: Tax/inheritance, more variants
    ["налоговая служба одобрила возврат введите реквизиты карты чтобы получить деньги", 1],
    ["the tax authority approved your refund enter your card details to receive it", 1],
    ["вам полагается компенсация от государства подтвердите счет для перевода", 1],
    ["you are owed a government compensation payment confirm your account to receive it", 1],
    ["нотариус сообщает о наследстве от родственника за границей вышлите документы и оплатите сбор", 1],
    ["a notary reports an inheritance from a relative abroad send documents and pay a small fee", 1],
    ["нераспределенные средства ожидают вас введите банковские реквизиты для получения", 1],
    ["unclaimed funds are waiting for you enter your bank details to claim them today", 1],
    ["налоговый инспектор требует немедленной оплаты штрафа по ссылке", 1],
    ["a tax inspector demands immediate payment of a fine through this link", 1],
    ["салық қайтарымын алу үшін карта деректерін енгізіңіз қазір", 1],
    ["мұрагерлік құжаттарын растау үшін ақы төлеңіз", 1],

    # FRAUD: Global brand impersonation, more brands
    ["microsoft account suspended verify now to restore access to your files", 1],
    ["netflix account on hold update your payment info to keep streaming", 1],
    ["ebay account limited confirm your identity to continue buying and selling", 1],
    ["your dhl shipment requires payment confirmation click here now", 1],
    ["fedex tracking issue update your delivery address to receive your package", 1],
    ["linkedin account flagged verify your profile or lose access permanently", 1],
    ["microsoft аккаунт приостановлен подтвердите сейчас чтобы восстановить доступ", 1],
    ["netflix аккаунт заморожен обновите платежные данные чтобы продолжить просмотр", 1],
    ["ebay аккаунт ограничен подтвердите личность чтобы продолжить покупки", 1],
    ["ваша посылка dhl требует подтверждения оплаты нажмите здесь сейчас", 1],
    ["проблема с отслеживанием fedex обновите адрес доставки для получения посылки", 1],
    ["linkedin аккаунт помечен подтвердите профиль иначе потеряете доступ навсегда", 1],
    ["your microsoft 365 subscription payment failed update now to avoid losing access", 1],
    ["steam account suspended verify your identity to restore access to your games", 1],
    ["steam аккаунт заблокирован подтвердите личность чтобы восстановить доступ к играм", 1],
    ["your ebay payment could not be processed update your card details now", 1],
    ["amazon security alert unusual sign in attempt verify your account immediately", 1],
    ["amazon аккаунт безопасность необычная попытка входа подтвердите аккаунт немедленно", 1],

    # FRAUD: Business email compromise, more variants
    ["это финансовый директор переведите оплату новому поставщику сегодня конфиденциально", 1],
    ["this is the cfo please process payment to the new vendor today confidentially", 1],
    ["срочно нужна ваша подпись на счете переведите оплату до конца дня", 1],
    ["urgent i need your approval on this invoice please process payment by end of day", 1],
    ["бухгалтерия обновите реквизиты для перевода зарплаты в этом месяце", 1],
    ["accounting please update the payroll account details for this month's transfer", 1],
    ["поставщик сменил банк переведите оплату по новым реквизитам во вложении", 1],
    ["our vendor changed banks please send payment to the new account details attached", 1],
    ["директор просит срочно оплатить счет через личный кабинет без обычной проверки", 1],
    ["the director is asking for an urgent payment to be processed without the usual review", 1],
    ["менеджер сіз шотты дереу төлеуіңіз керек компания атынан", 1],
    ["бухгалтер жаңа деректемелерге ақша аударыңыз шұғыл", 1],

    # FRAUD: AI-generated deepfake investment videos, more variants
    ["белгілі әнші сізді жаңа инвестициялық платформаға шақырады бейнені қараңыз", 1],
    ["известный актер лично приглашает вас в свой инвестиционный проект посмотрите видео", 1],
    ["a famous celebrity personally invites you to join this investment app watch the video", 1],
    ["финансовый блогер показывает как утроить деньги за неделю смотрите эксклюзивное видео", 1],
    ["watch this exclusive interview where a tech billionaire reveals his investment secret", 1],
    ["новости показывают известного предпринимателя рекламирующего эту платформу вложите сейчас", 1],
    ["breaking news a well known entrepreneur is backing this platform invest before it closes", 1],
    ["видеообращение известного банкира приглашает инвестировать через новое приложение", 1],
    ["watch this video message from a famous banker inviting you to invest through this app", 1],
    ["искусственный интеллект гарантирует доход 20 процентов в день присоединяйтесь сейчас", 1],
    ["this ai trading bot guarantees 20 percent daily returns join now before spots run out", 1],
    ["тегін вебинар белгілі кәсіпкерден инвестиция туралы қазір тіркеліңіз", 1],

    # FRAUD: Fake courier demanding the code, more variants
    ["курьер fedex у вашей двери назовите код из смс чтобы подтвердить личность", 1],
    ["fedex courier at your door needs the sms code to confirm your identity", 1],
    ["курьер сдэк доставляет посылку продиктуйте код подтверждения оператору", 1],
    ["cdek courier delivering your package please read the confirmation code to the operator", 1],
    ["курьер kaspi просит код из смс для завершения доставки заказа", 1],
    ["kaspi courier is asking for the sms code to complete your order delivery", 1],
    ["водитель яндекс доставки просит код подтверждения перед передачей заказа", 1],
    ["the delivery driver is asking for your confirmation code before handing over the order", 1],
    ["курьер боксберри у подъезда назовите код чтобы получить посылку", 1],
    ["boxberry courier is outside your building read the code to receive your package", 1],
    ["жеткізуші кодты сұрайды растау үшін қазір айтыңыз", 1],
    ["курьер сізден смс кодты сұрайды тапсырысты алу үшін", 1],

    # FRAUD: Expiring rewards points, more variants
    ["ваши бонусы magnum сгорают завтра войдите по ссылке чтобы успеть их потратить", 1],
    ["your magnum loyalty points expire tomorrow log in now through this link to use them", 1],
    ["бонусные мили аэрофлот истекают сегодня подтвердите аккаунт чтобы сохранить их", 1],
    ["your airline miles expire today verify your account now to keep them", 1],
    ["баллы небольшая карта сгорают через час перейдите по ссылке немедленно", 1],
    ["your loyalty card points expire in one hour click this link immediately to save them", 1],
    ["кэшбэк баллы будут аннулированы сегодня войдите в приложение по ссылке", 1],
    ["your cashback points will be cancelled today log in through this link now", 1],
    ["подарочная карта истекает завтра активируйте ее прямо сейчас по ссылке", 1],
    ["your gift card balance expires tomorrow activate it right now through this link", 1],
    ["бонустарыңыз ертең жойылады сілтеме арқылы кіріп пайдаланыңыз", 1],
    ["ұпайларыңыз сағат ішінде жойылады қазір растаңыз", 1],

    # FRAUD: Traffic violation QR code, more variants
    ["штраф за превышение скорости отсканируйте qr код и оплатите сегодня со скидкой", 1],
    ["speeding fine scan this qr code and pay today with a discount before it doubles", 1],
    ["нарушение парковки зафиксировано камерой оплатите штраф по qr коду немедленно", 1],
    ["parking violation recorded by camera pay the fine via this qr code right away", 1],
    ["штраф за проезд на красный свет отсканируйте код и оплатите в течение часа", 1],
    ["red light violation fine scan the code and pay within one hour to avoid court", 1],
    ["нарушение пдд зафиксировано фотофиксацией оплатите штраф по коду сегодня", 1],
    ["traffic camera recorded a violation pay the fine using this code today only", 1],
    ["штраф гибдд можно оплатить со скидкой 50 процентов через qr код сейчас", 1],
    ["pay your traffic fine with a 50 percent discount by scanning this qr code now", 1],
    ["жол ережесін бұзу тіркелді qr кодты сканерлеп бүгін төлеңіз", 1],
    ["айыппұлды жеңілдікпен төлеу үшін qr кодты қазір сканерлеңіз", 1],

    # FRAUD: Fake "verify you're human" scam, more variants
    ["чтобы продолжить подтвердите что вы не робот следуя этим инструкциям", 1],
    ["to continue please verify you are not a robot by following these steps", 1],
    ["система безопасности требует пройти проверку человек ли вы выполните действия ниже", 1],
    ["our security system requires a human verification check complete the steps below", 1],
    ["нажмите здесь и следуйте инструкциям чтобы подтвердить что вы не бот", 1],
    ["click here and follow the instructions to prove you are not a bot", 1],
    ["для доступа к файлу подтвердите что вы человек выполнив следующие шаги", 1],
    ["to access this file verify you are human by completing the following steps", 1],
    ["проверка безопасности cloudflare выполните действия чтобы продолжить загрузку", 1],
    ["cloudflare security check complete these actions to continue your download", 1],
    ["робот еместігіңізді растаңыз әйтпесе файлға қол жеткізе алмайсыз", 1],
    ["адам екеніңізді тексеру үшін төмендегі қадамдарды орындаңыз", 1],

    # FRAUD: QR code swapped at a physical location, more variants
    ["отсканируйте qr код у входа в парковку чтобы оплатить и получить скидку", 1],
    ["scan the qr code at the parking entrance to pay and get a discount today", 1],
    ["qr код на автомате для оплаты проезда в автобусе отсканируйте сейчас", 1],
    ["scan this qr code on the bus payment terminal to pay your fare now", 1],
    ["новый qr код для оплаты в этом кафе старый больше не работает", 1],
    ["this cafe has a new qr code for payment the old one no longer works", 1],
    ["отсканируйте qr на билборде чтобы получить купон на скидку сегодня", 1],
    ["scan the qr code on this billboard to get a discount coupon today only", 1],
    ["qr код в лифте для оплаты консьерж услуг отсканируйте и введите данные карты", 1],
    ["scan the qr code in the elevator to pay for concierge services enter your card", 1],
    ["тұрақ ақысын төлеу үшін кіреберістегі qr кодты сканерлеңіз", 1],
    ["асхана үстеліндегі жаңа qr код арқылы төлеңіз ескісі жұмыс істемейді", 1],

    # FRAUD: Bank bonus-exchange phishing, more banks
    ["обменяйте бонусы sberbank на деньги перейдите по ссылке и подтвердите карту", 1],
    ["exchange your sberbank bonus points for cash click here and confirm your card", 1],
    ["обменяйте баллы tinkoff на рубли перейдите по ссылке сейчас", 1],
    ["exchange your tinkoff points for cash click this link now to claim them", 1],
    ["переведите бонусы kaspi в тенге подтвердив карту по ссылке", 1],
    ["convert your kaspi bonus points to cash by confirming your card through this link", 1],
    ["обменяйте мили halyk bank на деньги подтвердите данные карты сейчас", 1],
    ["exchange your halyk bank miles for cash confirm your card details now", 1],
    ["втб бонусная программа истекает обменяйте баллы на деньги сегодня по ссылке", 1],
    ["vtb rewards program ending exchange your points for cash today via this link", 1],
    ["банк бонустарын ақшаға айырбастаңыз картаны растап сілтемеге өтіңіз", 1],
    ["бонустарыңызды бүгін айырбастаңыз әйтпесе олар жойылады", 1],

    # FRAUD: Unpaid toll SMS scam, more variants
    ["you have an outstanding toll fee of 6.99 pay now to avoid a late penalty", 1],
    ["final notice unpaid toll balance pay immediately to avoid dmv reporting", 1],
    ["your ezpass account shows an unpaid toll pay today to avoid additional fees", 1],
    ["fastrak toll violation notice pay now before your license is suspended", 1],
    ["ipass unpaid toll detected settle your balance today through this link", 1],
    ["toll authority final reminder pay your outstanding balance now to avoid court", 1],
    ["у вас есть неоплаченный проезд по платной дороге m1 оплатите сегодня", 1],
    ["проезд по трассе не оплачен штраф увеличится если не оплатить сейчас", 1],
    ["финальное уведомление о неоплаченном проезде оплатите немедленно по ссылке", 1],
    ["камера зафиксировала проезд без оплаты пошлины оплатите штраф сейчас", 1],
    ["your toll invoice is overdue pay today to avoid collections and a credit report mark", 1],
    ["unpaid highway toll notice settle now before the fine doubles tomorrow", 1],
    ["жол ақысы төленбеген бүгін төлеңіз әйтпесе айыппұл көбейеді", 1],
    ["ақылы жолмен өту ақысы төленбеді сілтеме арқылы қазір төлеңіз", 1],
    ["toll payment overdue click here to settle your balance and avoid legal action", 1],
    ["your recent toll trip was not paid pay now to prevent account suspension", 1],
    ["state toll authority notice unpaid balance detected pay immediately online", 1],
    ["camera detected an unpaid toll crossing pay the fine today via this link", 1],

    # FRAUD: AI voice-cloning family emergency scam, more variants
    ["мама это я меня задержала полиция нужны деньги на залог не говори папе", 1],
    ["dad it's me i got arrested and need bail money please don't tell mom", 1],
    ["папа я попал в аварию и мне нужно оплатить ремонт машины сейчас", 1],
    ["mom i was in a car crash and need money for repairs right now please hurry", 1],
    ["бабуля я в больнице мне нужна операция переведи деньги быстрее пожалуйста", 1],
    ["grandpa i'm in the hospital and need surgery money please send it quickly", 1],
    ["это твой внук у меня украли телефон и кошелек пришли денег на такси домой", 1],
    ["it's your grandson someone stole my phone and wallet please send taxi money home", 1],
    ["мама не удивляйся моему голосу я простыл мне очень нужны деньги сейчас", 1],
    ["mom don't worry about my voice i have a cold i really need money right now", 1],
    ["папа полиция задержала меня по ошибке нужен залог срочно не говори маме", 1],
    ["dad police arrested me by mistake i need bail immediately don't tell mom", 1],
    ["әжем мен апатқа ұшырадым ақша керек ешкімге айтпа", 1],
    ["атам мен ауруханадамын операцияға ақша керек тез арада", 1],
    ["sister it's me my card got frozen abroad please wire money to this account now", 1],
    ["сестра у меня заблокировали карту за границей переведи деньги на этот счет", 1],
    ["brother i'm stuck at the border and need cash immediately to cross please help", 1],
    ["братан я застрял на границе срочно нужны деньги чтобы проехать помоги", 1],

    # FRAUD: Pig butchering, more variants
    ["милый я заработала 5000 долларов за неделю на этой платформе присоединяйся ко мне", 1],
    ["babe i made 5000 dollars this week on this platform join me and invest too", 1],
    ["любимая моя инвестиционная группа удвоила деньги за месяц вступай скорее", 1],
    ["sweetheart my investment group doubled our money this month join quickly", 1],
    ["дорогой наставник учит меня трейдингу криптовалюты давай учиться вместе", 1],
    ["honey my mentor is teaching me crypto trading let's learn together and invest", 1],
    ["я нашла надежную платформу для роста капитала переведи туда немного денег", 1],
    ["i found a reliable platform to grow our savings send a little money there", 1],
    ["жанашырым мен криптоға ақша салдым пайда көп келші бірге инвестициялайық", 1],
    ["сен де қосыл бұл платформаға бірге байимыз", 1],
    ["милая эта площадка платит каждый день без риска давай попробуем вместе", 1],
    ["this platform pays out daily with zero risk darling let's try it together", 1],

    # FRAUD: Bank "call us back" scheme, more banks
    ["это служба безопасности halyk bank перезвоните срочно по поводу вашего счета", 1],
    ["halyk bank security department call us back immediately about your account", 1],
    ["это служба безопасности kaspi перезвоните на этот номер по поводу операции", 1],
    ["kaspi security team please call us back regarding a recent transaction", 1],
    ["это служба безопасности tinkoff перезвоните срочно операция требует подтверждения", 1],
    ["tinkoff security department call back now this transaction needs confirmation", 1],
    ["это служба безопасности втб перезвоните немедленно на этот номер", 1],
    ["vtb security team call this number back immediately regarding your card", 1],
    ["банк альфа банк служба безопасности просит перезвонить по счету срочно", 1],
    ["alfa bank security is asking you to call back regarding your account urgently", 1],
    ["жусан банк қауіпсіздік қызметі осы нөмірге дереу қоңырау шалыңыз", 1],
    ["forte bank сізден шотыңыз бойынша қайта қоңырау шалуды сұрайды", 1],
    ["это банк отбасы служба безопасности перезвоните срочно по счету", 1],
    ["otbasy bank security asks you to call back immediately about your account", 1],
    ["почта банк служба безопасности сообщает подозрительную операцию перезвоните", 1],
    ["post bank security detected a suspicious transaction please call us back now", 1],
    ["совкомбанк служба безопасности просит подтвердить операцию по телефону", 1],
    ["sovcombank security is asking you to confirm a transaction over the phone now", 1],

    # FRAUD: NFC "tap your phone" scam, more variants
    ["приложите телефон к терминалу пока я отменяю платеж следуйте моим инструкциям", 1],
    ["hold your phone to the terminal while i cancel the payment follow my instructions", 1],
    ["для отмены операции поднесите телефон к считывающему устройству сейчас", 1],
    ["to cancel the transaction hold your phone near the reader device right now", 1],
    ["оператор банка просит поднести телефон к терминалу для защиты счета", 1],
    ["the bank operator is asking you to tap your phone on the terminal to protect your account", 1],
    ["чтобы вернуть деньги приложите телефон к pos терминалу как я скажу", 1],
    ["to get your refund hold your phone against the pos terminal as i instruct", 1],
    ["служба безопасности просит выполнить бесконтактную оплату для проверки карты", 1],
    ["security is asking you to perform a contactless tap to verify your card works", 1],
    ["картаны қорғау үшін телефонды терминалға апарыңыз нұсқауды орындаңыз", 1],
    ["операцияны болдырмау үшін телефонды pos терминалға тигізіңіз", 1],

    # =========================
    # MORE REAL-WORLD-GROUNDED CATEGORIES: additional reported scam patterns
    # (FTC, FBI IC3, CFPB, SSA-OIG, US Marshals, FCC, Action Fraud UK,
    # Airbnb/Booking trust & safety, BBB Scam Tracker) not yet covered above
    # =========================

    # FRAUD: Grandparent bail-money scam — classic elder fraud reported by
    # the FTC where a caller pretends to be a grandchild arrested and in
    # need of bail money, pressuring secrecy from other family members
    ["бабушка это я меня арестовали пришли деньги на залог никому не говори", 1],
    ["grandma it's me i got arrested please wire bail money and don't tell mom and dad", 1],
    ["внучок попал в беду нужны деньги на адвоката срочно и тихо", 1],

    # FRAUD: Fake mystery-shopper / check-cashing scam — FTC-reported pattern
    # where the victim is mailed a check to deposit, then asked to wire back
    # part of it before the check is discovered to be fake
    ["поздравляем вы приняты тайным покупателем обналичьте чек и переведите часть суммы обратно", 1],
    ["congratulations you're hired as a mystery shopper deposit this check and wire back the difference", 1],

    # FRAUD: Puppy / pet adoption scam — Better Business Bureau's most
    # commonly reported online purchase scam, demanding extra fees before
    # the pet is ever shipped
    ["щенок ждет вас оплатите доставку и страховку перед отправкой", 1],
    ["your puppy is ready for shipping pay the crate and insurance fee first", 1],

    # FRAUD: Fake vacation rental listing scam — FTC-reported pattern of a
    # too-good listing that only accepts wire transfer or gift cards
    ["отличная квартира на побережье переведите предоплату сегодня чтобы забронировать", 1],
    ["amazing beachfront rental wire the deposit today by bank transfer only to secure it", 1],

    # FRAUD: IRS/tax authority gift-card scam — IRS.gov consumer alert: a
    # caller claims back taxes are owed and demands payment in gift cards
    ["налоговая служба вы задолжали оплатите штраф подарочными картами немедленно", 1],
    ["this is the irs you owe back taxes pay immediately with gift cards or be arrested", 1],

    # FRAUD: Utility disconnection scam — FTC/utility-industry alert: caller
    # threatens same-day shutoff unless paid instantly with a prepaid card
    ["ваше электричество будет отключено сегодня оплатите долг предоплаченной картой немедленно", 1],
    ["your power will be shut off within the hour pay now with a prepaid card to stop it", 1],

    # FRAUD: Social Security number suspension scam — SSA Office of
    # Inspector General alert: caller claims the victim's SSN is linked to
    # crime and will be suspended unless verified
    ["ваш номер социального страхования приостановлен подтвердите личные данные немедленно", 1],
    ["your social security number has been suspended due to suspicious activity verify your identity now", 1],

    # FRAUD: Jury duty / arrest warrant scam — US Marshals Service public
    # alert: caller impersonates law enforcement claiming a missed jury
    # summons and demands immediate payment to avoid arrest
    ["вы пропустили вызов в суд присяжных оплатите штраф сейчас чтобы избежать ареста", 1],
    ["you missed jury duty pay the fine right now over the phone or officers will arrest you", 1],

    # FRAUD: Ransomware-style "device locked" popup scam — FBI IC3-reported
    # pattern where a full-screen warning claims illegal content was found
    # and demands payment to unlock the device
    ["ваше устройство заблокировано полицией оплатите штраф чтобы восстановить доступ", 1],
    ["your device has been locked by federal authorities pay the fine now to restore access", 1],

    # FRAUD: Wangiri "one ring" callback scam — FCC consumer alert: a single
    # ring from an unknown international number baits the victim into
    # calling back a premium-rate line
    ["у вас пропущенный звонок с международного номера перезвоните немедленно", 1],
    ["you have a missed call from an international number call back right away", 1],

    # FRAUD: Romance scam, soldier deployed overseas — FTC/DOD Inspector
    # General warning: a fake military profile asks for money to pay for
    # leave, communication fees, or equipment
    ["я солдат на службе за границей мне нужны деньги на отпуск домой переведи пожалуйста", 1],
    ["i'm a soldier stationed overseas i need money for leave papers please send it to me", 1],

    # FRAUD: Task-based "like and earn" job scam — FBI IC3 2024 reported
    # surge: victims are paid small amounts for simple tasks, then asked to
    # deposit their own money to "unlock" larger earnings
    ["выполните простое задание лайкните видео и заработайте внесите депозит чтобы разблокировать доход", 1],
    ["complete simple like tasks and earn money deposit funds first to unlock higher paying tasks", 1],

    # FRAUD: Job scam reimbursement check for home-office equipment — FTC
    # pattern: a fake employer sends an overpayment check for equipment and
    # asks the new hire to wire back the remainder
    ["для настройки рабочего места мы вышлем чек обналичьте его и переведите остаток поставщику", 1],
    ["we're sending a check to buy your home office equipment deposit it and wire back the balance", 1],

    # FRAUD: Payroll-diversion BEC — FBI IC3-reported business email
    # compromise subtype where "HR" asks an employee to confirm new bank
    # details for the next payroll run
    ["отдел кадров просит срочно подтвердить новые банковские реквизиты для зарплаты", 1],
    ["hr here please confirm your updated bank details today so payroll isn't delayed", 1],

    # FRAUD: Accidental-overpayment "send it back" scam — FTC/bank-transfer
    # warning: scammer sends a fraudulent payment then urgently asks the
    # victim to return part of it before the original payment is reversed
    ["я по ошибке перевел вам лишнюю сумму пожалуйста верните разницу на другой счет сегодня", 1],
    ["i accidentally sent you extra money please send the difference back to this account today", 1],

    # FRAUD: Sextortion email — FBI IC3-reported pattern: an email claims to
    # have compromising recordings and demands payment in cryptocurrency
    ["у меня есть запись с вашей камеры оплатите биткоином иначе отправлю всем контактам", 1],
    ["i recorded you through your webcam pay in bitcoin now or i will send it to all your contacts", 1],

    # FRAUD: Fake subscription renewal call — BBB/FTC-reported pattern: a
    # bogus invoice for an expensive renewal leads the victim to call a
    # number where they're talked into remote-access "refund" software
    ["ваша подписка на антивирус продлена на 499 позвоните для отмены и возврата средств", 1],
    ["your antivirus subscription renewed for $499 call this number now for a refund", 1],

    # FRAUD: Debt collector demanding gift cards — CFPB warning: a fake debt
    # collector threatens legal action unless paid immediately in
    # untraceable gift cards or wire transfer
    ["это коллекторское агентство оплатите долг подарочными картами сегодня иначе подадим в суд", 1],
    ["this is a debt collector pay immediately with gift cards today or we will sue you", 1],

    # FRAUD: Car warranty expiring robocall — FTC's most-complained-about
    # robocall category for several consecutive years
    ["гарантия на ваш автомобиль истекает продлите сейчас позвонив по этому номеру", 1],
    ["your car's warranty is about to expire press 1 now to renew before coverage ends", 1],

    # FRAUD: Fake tech-support number from a search ad — FTC-reported
    # pattern: a sponsored search result leads to a fake support line that
    # requests remote access to "fix" a nonexistent problem
    ["служба поддержки обнаружила проблему на вашем компьютере разрешите удаленный доступ для решения", 1],
    ["our support team detected an issue on your computer grant remote access so we can fix it", 1],

    # FRAUD: Money-mule recruitment — FBI warning: victims are offered a cut
    # of funds for letting their bank account be used to move money for
    # someone else
    ["нужен человек с банковским счетом для перевода средств хорошая оплата за пару минут", 1],
    ["easy money just let funds pass through your bank account and keep a percentage for yourself", 1],

    # FRAUD: "Pay outside the platform" rental scam — Airbnb/Booking trust &
    # safety warning: the host asks to move payment off the platform where
    # there's no buyer protection
    ["хозяин просит оплатить напрямую переводом чтобы получить скидку вне платформы", 1],
    ["the host is asking you to pay directly by bank transfer outside the platform for a discount", 1],

    # FRAUD: "I lost my phone, this is my new number" family impersonation —
    # UK Action Fraud-reported messaging-app pattern impersonating a child
    # or parent from an unknown number
    ["мама это я потеряла телефон пишу с нового номера срочно нужны деньги", 1],
    ["hi mum it's me i lost my phone this is my new number i urgently need some money", 1],

    # FRAUD: LinkedIn recruiter payroll-setup scam — FTC-reported pattern
    # where a "recruiter" asks for bank details before a start date to
    # supposedly set up direct deposit
    ["рекрутер просит номер карты и банковские реквизиты для оформления зарплаты до выхода на работу", 1],
    ["the recruiter is asking for your bank account number to set up payroll before your start date", 1],

    # FRAUD: Fake jury-duty/court summons email attachment — public alerts
    # from US courts warning of phishing emails with malicious attachments
    # disguised as failure-to-appear notices
    ["вам пришла повестка в суд откройте вложение и подтвердите явку немедленно", 1],
    ["you have a court summons attached open the file now and confirm your appearance immediately", 1],

    # =========================
    # MORE REAL-WORLD-GROUNDED CATEGORIES, ROUND 2: further documented scam
    # patterns (FTC, FBI IC3, CFPB, SSA-OIG, FCC, US Dept of Labor, BBB)
    # =========================

    # FRAUD: Timeshare resale/exit scam — FTC-documented pattern where a
    # caller claims to have a buyer lined up for the victim's timeshare and
    # demands an upfront "closing fee"
    ["у нас есть покупатель на ваш таймшер оплатите комиссию агентству чтобы закрыть сделку", 1],
    ["we have a buyer lined up for your timeshare pay the closing fee upfront to finalize the sale", 1],

    # FRAUD: Advance-fee loan scam — FTC-reported pattern: guaranteed loan
    # approval regardless of credit history, but an upfront "insurance" or
    # "processing" fee must be paid first
    ["кредит одобрен без проверки кредитной истории оплатите страховой взнос для получения средств", 1],
    ["your loan is pre-approved with no credit check pay the processing fee first to release the funds", 1],

    # FRAUD: Sweepstakes / Publishers Clearing House impersonation — classic
    # FTC-tracked scam: victim is told they won a major sweepstakes and must
    # pay taxes or fees before the prize is delivered
    ["вы выиграли крупный приз всероссийской лотереи оплатите налог чтобы получить выигрыш", 1],
    ["you've won the publishers clearing house sweepstakes pay the delivery tax to claim your prize", 1],

    # FRAUD: Credit repair / debt relief advance-fee scam — CFPB/FTC
    # warning: a company promises to erase debt or fix credit for an
    # upfront fee, before any service is performed
    ["мы полностью очистим вашу кредитную историю оплатите услугу заранее результат гарантирован", 1],
    ["we can wipe your debt clean pay our fee upfront and your credit score is guaranteed to improve", 1],

    # FRAUD: Long-term apartment rental application scam — FTC-reported
    # pattern distinct from vacation rentals: a fake listing collects an
    # application or holding fee by wire before any in-person viewing
    ["квартира сдается срочно переведите залог до просмотра чтобы забронировать за вами", 1],
    ["the apartment is available now wire the holding deposit before the viewing to secure it", 1],

    # FRAUD: Dating app "verification badge" scam — FTC 2024 online-dating
    # scam reporting: a match asks the victim to pay for a safety or
    # verification badge through a fake link before meeting in person
    ["для безопасности переведи небольшую сумму за верификационный значок на сайте знакомств", 1],
    ["before we meet just pay a small fee for the dating safety verification badge through this link", 1],

    # FRAUD: Fake online storefront / never-ships ad scam — FTC-reported
    # pattern: a heavily discounted ad on social media leads to a fake store
    # that takes payment and never ships anything
    ["огромная скидка только сегодня оплатите заказ картой на сайте пока распродажа не закончилась", 1],
    ["huge discount today only pay for your order now on the site before the sale ends", 1],

    # FRAUD: Scam-recovery scam — FTC specifically warns this targets people
    # who already lost money to a scam, promising to recover it for an
    # upfront fee
    ["мы можем вернуть деньги которые вы потеряли мошенникам оплатите нашу комиссию заранее", 1],
    ["we specialize in recovering money lost to scammers just pay our recovery fee upfront first", 1],

    # FRAUD: Fake law firm debt-lawsuit threat — CFPB/FTC warning: a caller
    # poses as a law firm or process server threatening an imminent lawsuit
    # unless a debt is paid immediately
    ["это юридическая фирма против вас подан иск оплатите долг сейчас чтобы избежать суда", 1],
    ["this is a law firm a lawsuit has been filed against you pay the debt now to avoid court", 1],

    # FRAUD: Fake unemployment benefits identity theft — US Department of
    # Labor warning: scammers phish for SSN/bank info claiming it's needed
    # to process an unemployment claim
    ["для оформления пособия по безработице подтвердите номер социального страхования и счет", 1],
    ["to process your unemployment claim confirm your social security number and bank account now", 1],

    # FRAUD: Romance scam, customs duty for a gift/inheritance package —
    # FTC-documented romance-scam variant: the online partner claims to be
    # sending an expensive gift or inheritance that's stuck at customs
    ["дорогой я отправила тебе посылку с драгоценностями оплати таможенную пошлину чтобы забрать", 1],
    ["my love i sent you a package with jewelry just pay the customs fee to release it", 1],

    # FRAUD: School-emergency impersonation scam — reported variant of
    # family-emergency scams: caller poses as a school administrator or
    # nurse claiming the victim's child was in an accident
    ["это школа с вашим ребенком произошел несчастный случай срочно переведите деньги на лечение", 1],
    ["this is the school your child has been in an accident please send money for treatment now", 1],

    # FRAUD: Mandatory certification/training fee job scam — FTC job-scam
    # pattern: a new hire is told they must pay for required certification
    # or training materials before starting
    ["для трудоустройства необходимо оплатить обязательный сертификат курса заранее", 1],
    ["before you can start the job you must pay for the required certification course upfront", 1],

    # FRAUD: KYC "verify to keep your account open" scam — bank-
    # impersonation pattern distinct in framing from a blocked-card alert:
    # claims a mandatory identity-verification update is needed
    ["банк проводит обновление данных клиентов подтвердите личность по ссылке иначе счет закроют", 1],
    ["your bank requires a mandatory kyc update verify your identity by this link or your account will close", 1],

    # FRAUD: Global brand impersonation, more brands (Temu/Shein) —
    # FTC-reported pattern impersonating popular shopping platforms about a
    # failed delivery or account issue
    ["temu не может доставить ваш заказ обновите данные оплаты по ссылке", 1],
    ["shein could not deliver your order update your payment information using this link", 1],

    # FRAUD: Fake vendor invoice scam (BEC) — FBI IC3-reported subtype
    # distinct from payroll diversion: a fake vendor emails accounts
    # payable an overdue invoice for services never rendered
    ["поставщик прислал просроченный счет оплатите немедленно чтобы избежать пени", 1],
    ["accounts payable this invoice from our vendor is overdue please pay immediately to avoid late fees", 1],

    # FRAUD: Wage garnishment court scam — reported pattern impersonating a
    # court or sheriff's office claiming wages will be garnished unless a
    # debt is settled by phone today
    ["суд уведомляет об удержании из зарплаты оплатите долг сегодня чтобы остановить процесс", 1],
    ["the court is garnishing your wages starting today pay the debt now by phone to stop it", 1],

    # FRAUD: Google/Yelp review extortion scam — FTC/BBB-reported pattern
    # targeting small business owners: caller threatens fake negative
    # reviews unless paid, or offers to remove them for a fee
    ["мы разместим негативные отзывы о вашем бизнесе если не оплатите нашу услугу удаления", 1],
    ["we'll flood your business with fake negative reviews unless you pay for our removal service", 1],

    # FRAUD: Premium-rate "reply STOP" SMS scam — FCC-reported pattern
    # where replying to an unknown text confirms the number is active and
    # can trigger premium-rate charges
    ["ответьте стоп чтобы отписаться от смс рассылки", 1],
    ["reply stop to this text to unsubscribe from all future messages", 1],

    # FRAUD: Fake solar panel / government energy grant scam — FTC solar-
    # scam alert: a caller claims the victim qualifies for a government
    # grant but must pay an upfront "registration" fee
    ["вам одобрен государственный грант на установку солнечных панелей оплатите регистрационный взнос", 1],
    ["you qualify for a government solar grant pay the registration fee now to lock in your panels", 1],

    # FRAUD: SSA voicemail robocall "case number, press 1" scam — SSA
    # Office of Inspector General: an automated voicemail claims the
    # victim's SSN is linked to a case and instructs them to press 1
    ["ваш номер социального страхования связан с уголовным делом нажмите 1 чтобы поговорить со специалистом", 1],
    ["your social security number is linked to a criminal case press 1 now to speak with an officer", 1],

    # FRAUD: Widow/widower inheritance romance scam — FTC romance-scam
    # sub-pattern: an online partner claims to be a recently widowed
    # foreigner with a large inheritance who needs help transferring funds
    ["я недавно овдовела и получила крупное наследство помоги мне перевести деньги за границу", 1],
    ["i'm a recent widow with a large inheritance i need your help transferring the funds abroad", 1],

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

    # SAFE: matching everyday counterparts to the new scam categories above,
    # including safe messages that mention money/bank/subscriptions to help
    # the model learn those words alone aren't a fraud signal
    ["спасибо перевод получен вовремя", 0],
    ["ваша подписка продлена автоматически как обычно", 0],
    ["your subscription has been renewed successfully thank you", 0],
    ["напоминание оплатите коммунальные услуги до конца месяца в приложении банка", 0],
    ["your tax return has been processed and refund issued to your account on file", 0],
    ["добро пожаловать в клуб лояльности бонусы уже на вашем счету", 0],
    ["ваш билет на самолет подтвержден вылет завтра в 10 утра", 0],
    ["антивирус успешно обновлен угроз не найдено", 0],
    ["your antivirus scan completed no threats found", 0],
    ["сәлем ертең кездесеміз сағат 5-те", 0],
    ["қаражат картаңызға сәтті аударылды", 0],
    ["your instagram password was changed successfully", 0],
    ["your whatsapp backup completed successfully", 0],
    ["конференция начнется через 10 минут ссылка в календаре", 0],
    ["your package qr code is ready show it at the pickup point", 0],
    ["спасибо за пожертвование фонду мы очень ценим вашу помощь", 0],
    ["thank you for your donation to the foundation we truly appreciate it", 0],
    ["ваш кредит одобрен банком менеджер свяжется с вами в течение дня", 0],
    ["зарплата поступила на карту как обычно", 0],
    ["сіздің тапсырысыңыз жолға шықты", 0],

    # SAFE: more everyday counterparts matching the newest scam variants above
    ["ваша карта успешно перевыпущена заберите в отделении банка", 0],
    ["уважаемый клиент ваш вклад пополнен спасибо", 0],
    ["we noticed a new login from your usual device no action needed", 0],
    ["your debit card renewal is complete no action required", 0],
    ["банкіңіз сізге жаңа өнім туралы хабарлайды", 0],
    ["your banking app has a new feature update check it out", 0],
    ["курьер доставил посылку по адресу спасибо за заказ", 0],
    ["налоговая декларация принята возврат будет зачислен через 10 дней", 0],
    ["your package cleared customs and is on its way", 0],
    ["добрый день напоминаем о плановом визите к стоматологу", 0],
    ["мама я долетел все хорошо позвоню вечером", 0],
    ["hi dad landed safely will call you tonight", 0],
    ["ваш ежемесячный отчет по вкладу доступен в приложении", 0],
    ["your job application has been received we will contact you soon", 0],
    ["собеседование назначено на понедельник в 11 утра", 0],
    ["apple id successfully signed in from your new iphone", 0],
    ["ваша система обновлена автоматически проблем не обнаружено", 0],
    ["your license renewal was successful no further action needed", 0],
    ["привет как прошел день", 0],
    ["сегодня хорошая погода идем гулять", 0],
    ["ваш биткоин перевод подтвержден в блокчейне", 0],
    ["your exchange withdrawal has been completed successfully", 0],
    ["airdrop distribution complete tokens are now in your connected wallet", 0],
    ["спасибо за участие в благотворительном забеге", 0],
    ["your donation receipt is attached for tax purposes", 0],
    ["ваша подписка на netflix успешно оплачена", 0],
    ["your streaming subscription payment was successful enjoy", 0],
    ["сіздің telegram құпия кодыңыз сәтті пайдаланылды", 0],
    ["your facebook login was successful from a recognized device", 0],
    ["налоговый возврат зачислен на ваш счет спасибо", 0],
    ["your inheritance paperwork has been filed with the local notary office", 0],
    ["стимулирующая выплата уже зачислена на ваш счет банком", 0],
    ["собрание акционеров состоится в следующую среду", 0],
    ["ваш заказ на маркетплейсе подтвержден продавцом", 0],
    ["thanks for shopping with us your receipt is attached", 0],
    ["сіздің кездесуіңіз сағат 10-да басталады", 0],

    # SAFE: matching counterparts to the real-world-grounded categories above
    ["счет поставщика оплачен по обычному графику согласно договору", 0],
    ["the vendor invoice was approved and will be paid on the usual schedule", 0],
    ["записал новые банковские реквизиты поставщика после звонка с бухгалтерией для сверки", 0],
    ["записала видео с советами по инвестициям для канала выйдет завтра", 0],
    ["thanks for watching my finance channel new episode drops next week", 0],
    ["курьер оставил посылку на ресепшене можно забрать в любое время", 0],
    ["the courier delivered your package to the front desk no code needed", 0],
    ["ваши бонусные баллы за этот месяц уже начислены на счет", 0],
    ["your reward points balance was updated after your last purchase", 0],
    ["штраф оплачен через приложение банка чек сохранен", 0],
    ["your traffic ticket payment was confirmed by the city portal", 0],
    ["антивирус попросил подтвердить что вы не робот перед обновлением базы", 0],
    ["please complete the captcha to continue browsing our website", 0],
    ["меню кафе доступно по qr коду на столе оплата на кассе", 0],
    ["scan the table qr code to view today's menu no payment needed here", 0],
    ["бонусы halyk bank автоматически зачисляются на карту раз в месяц", 0],

    # SAFE: matching counterparts to the newest real-world-grounded categories
    ["your toll payment for this month was processed automatically as usual", 0],
    ["ваш проезд по платной дороге оплачен автоматически с привязанной карты", 0],
    ["бабушка привет это я как ты себя чувствуешь", 0],
    ["grandma hi it's me just checking in how are you feeling", 0],
    ["любимый посмотри какие фото я сделала сегодня на прогулке", 0],
    ["darling check out these photos from my walk today", 0],
    ["служба безопасности банка подтвердила что операция была совершена вами", 0],
    ["the bank's security team confirmed the transaction was made by you", 0],
    ["оплата картой прошла успешно через терминал спасибо за покупку", 0],
    ["thank you for your contactless payment your receipt is in the app", 0],
    ["арендодатель подтвердил получение оплаты за этот месяц спасибо", 0],
    ["landlord confirmed this month's rent payment was received thank you", 0],

    # =========================
    # SYNTHETIC AUGMENTATION (safe side): everyday counterparts mirroring
    # the same bank/service names used in the fraud batch above, so the
    # model learns those institutions' names alone aren't a fraud signal.
    # =========================

    # SAFE: Bank/card, matching bank names
    ["ваша карта kaspi успешно перевыпущена заберите в ближайшем отделении", 0],
    ["kaspi подтверждает что операция была совершена вами беспокоиться не о чем", 0],
    ["kaspi информирует об изменении тарифов с следующего месяца", 0],
    ["ваш лимит по карте kaspi увеличен согласно вашему запросу", 0],
    ["kaspi онлайн банкинг обновлен добавлены новые функции", 0],
    ["спасибо что выбираете kaspi ваш ежемесячный отчет готов", 0],
    ["ваша карта halyk bank успешно перевыпущена заберите в ближайшем отделении", 0],
    ["halyk bank подтверждает что операция была совершена вами беспокоиться не о чем", 0],
    ["halyk bank информирует об изменении тарифов с следующего месяца", 0],
    ["ваш лимит по карте halyk bank увеличен согласно вашему запросу", 0],
    ["halyk bank онлайн банкинг обновлен добавлены новые функции", 0],
    ["спасибо что выбираете halyk bank ваш ежемесячный отчет готов", 0],
    ["ваша карта jusan bank успешно перевыпущена заберите в ближайшем отделении", 0],
    ["jusan bank подтверждает что операция была совершена вами беспокоиться не о чем", 0],
    ["jusan bank информирует об изменении тарифов с следующего месяца", 0],
    ["ваш лимит по карте jusan bank увеличен согласно вашему запросу", 0],
    ["jusan bank онлайн банкинг обновлен добавлены новые функции", 0],
    ["спасибо что выбираете jusan bank ваш ежемесячный отчет готов", 0],
    ["ваша карта sberbank успешно перевыпущена заберите в ближайшем отделении", 0],
    ["sberbank подтверждает что операция была совершена вами беспокоиться не о чем", 0],
    ["sberbank информирует об изменении тарифов с следующего месяца", 0],
    ["ваш лимит по карте sberbank увеличен согласно вашему запросу", 0],
    ["sberbank онлайн банкинг обновлен добавлены новые функции", 0],
    ["спасибо что выбираете sberbank ваш ежемесячный отчет готов", 0],
    ["ваша карта втб успешно перевыпущена заберите в ближайшем отделении", 0],
    ["втб подтверждает что операция была совершена вами беспокоиться не о чем", 0],
    ["втб информирует об изменении тарифов с следующего месяца", 0],
    ["ваш лимит по карте втб увеличен согласно вашему запросу", 0],
    ["втб онлайн банкинг обновлен добавлены новые функции", 0],
    ["спасибо что выбираете втб ваш ежемесячный отчет готов", 0],
    ["ваша карта tinkoff успешно перевыпущена заберите в ближайшем отделении", 0],
    ["tinkoff подтверждает что операция была совершена вами беспокоиться не о чем", 0],
    ["tinkoff информирует об изменении тарифов с следующего месяца", 0],
    ["ваш лимит по карте tinkoff увеличен согласно вашему запросу", 0],
    ["tinkoff онлайн банкинг обновлен добавлены новые функции", 0],
    ["спасибо что выбираете tinkoff ваш ежемесячный отчет готов", 0],

    # SAFE: Delivery/government/relative, matching services
    ["ваша посылка от казпочты доставлена спасибо за ожидание", 0],
    ["казпочта уведомляет что заказ готов к выдаче в отделении", 0],
    ["ваша посылка от сдэк доставлена в пункт выдачи", 0],
    ["сдэк подтверждает успешную доставку вашего заказа", 0],
    ["ваша посылка от kaspi post доставлена по адресу", 0],
    ["kaspi post подтверждает доставку заказа получателю", 0],
    ["ваша посылка от boxberry готова к получению", 0],
    ["boxberry уведомляет что заказ прибыл в пункт выдачи", 0],
    ["your parcel from dhl was delivered successfully thank you", 0],
    ["dhl confirms your package arrived at the destination", 0],
    ["your parcel from fedex has been delivered to your address", 0],
    ["fedex confirms successful delivery of your order", 0],
    ["your parcel from ups was delivered this morning", 0],
    ["ups confirms your package is now at your doorstep", 0],
    ["your parcel from usps arrived safely today", 0],
    ["usps confirms delivery was completed successfully", 0],
    ["акимат сообщает о плановых работах в вашем районе на следующей неделе", 0],
    ["государственная служба подтверждает что ваше заявление принято к рассмотрению", 0],
    ["the department of motor vehicles confirms your renewal was processed", 0],
    ["government notice your application has been received and is being processed", 0],
    ["брат добрался нормально дома все хорошо не переживай", 0],
    ["cousin made it home safe don't worry everything is fine", 0],
    ["сын дома уже уроки сделал ждем тебя к ужину", 0],
    ["son is home already finished homework see you at dinner", 0],

    # SAFE: Investment/job, matching variants
    ["спасибо за интерес к нашей компании собеседование назначено на вторник", 0],
    ["thank you for your interest in our company your interview is set for tuesday", 0],
    ["ваша заявка на вакансию менеджера принята мы свяжемся с вами на этой неделе", 0],
    ["your application for the manager position was received we'll contact you this week", 0],
    ["поздравляем вы приняты на удаленную работу первый день в понедельник", 0],
    ["congratulations you've been hired for the remote position starting monday", 0],
    ["ваш брокерский счет обновлен согласно ежемесячному отчету", 0],
    ["your brokerage statement has been updated as part of your monthly report", 0],
    ["курс по финансовой грамотности начинается на следующей неделе регистрация открыта", 0],
    ["the financial literacy course starts next week registration is open", 0],
    ["ваш инвестиционный портфель вырос на 3 процента за квартал", 0],
    ["your investment portfolio grew by 3 percent this quarter as expected", 0],
    ["hr отдел подтверждает получение ваших документов для трудоустройства", 0],
    ["hr confirms receipt of your employment documents thank you", 0],
    ["оплата за фриланс проект поступила на счет спасибо за работу", 0],
    ["payment for the freelance project has been received thank you for your work", 0],
    ["жұмысқа өтінішіңіз қабылданды сұхбат сейсенбіде болады", 0],
    ["құжаттарыңыз қабылданды апта ішінде хабарласамыз", 0],
    ["your stock brokerage account statement is now available online", 0],
    ["welcome aboard your first day orientation starts at 9am monday", 0],
    ["конференция по инвестициям пройдет в вашем городе в следующем месяце", 0],
    ["the investment conference will be held in your city next month", 0],
    ["зарплата за этот месяц поступила на карту как обычно", 0],
    ["your paycheck for this month was deposited as usual", 0],

    # SAFE: Tech support, matching variants
    ["windows update completed successfully no action needed", 0],
    ["обновление windows прошло успешно действий не требуется", 0],
    ["norton antivirus scan completed no threats found on your device", 0],
    ["антивирус нортон завершил проверку угроз не обнаружено", 0],
    ["mcafee subscription renewed automatically thank you for staying protected", 0],
    ["mcafee подписка продлена автоматически спасибо что остаетесь защищены", 0],
    ["apple confirms your device backup completed successfully last night", 0],
    ["apple подтверждает резервное копирование устройства прошло успешно", 0],
    ["google chrome updated to the latest version automatically", 0],
    ["google chrome автоматически обновлен до последней версии", 0],
    ["техподдержка windows подтверждает что ваша заявка решена", 0],
    ["microsoft support confirms your support ticket has been resolved", 0],
    ["лицензия windows продлена автоматически согласно подписке", 0],
    ["your antivirus subscription renewed automatically as scheduled", 0],
    ["ваш ip адрес не показывает признаков компрометации все в порядке", 0],
    ["no signs of compromise detected on your account everything looks fine", 0],
    ["техническая поддержка apple закрыла ваше обращение решение найдено", 0],
    ["microsoft confirms unusual activity check found nothing to worry about", 0],

    # SAFE: Romance, matching variants
    ["любимый как прошел твой день расскажи мне все", 0],
    ["sweetheart how was your day tell me everything", 0],
    ["дорогая скучаю по тебе увидимся в эти выходные", 0],
    ["my love missing you can't wait to see you this weekend", 0],
    ["я приземлился нормально буду через час дома", 0],
    ["i landed safely will be home in an hour", 0],
    ["любимая врач сказал все в порядке волноваться не о чем", 0],
    ["darling the doctor said everything is fine nothing to worry about", 0],
    ["посылка с подарком дошла спасибо тебе большое любимый", 0],
    ["the gift package arrived thank you so much my love", 0],
    ["махаббатым қалайсың сағындым", 0],
    ["жаным дәрігер жақсы деді уайымдама", 0],

    # SAFE: Crypto, matching variants
    ["ваш перевод биткоина подтвержден в блокчейне успешно", 0],
    ["your bitcoin transfer was confirmed on the blockchain successfully", 0],
    ["binance подтверждает что вход в аккаунт выполнен с вашего устройства", 0],
    ["coinbase confirms the login was from your recognized device", 0],
    ["ваш новый токен добавлен в портфель согласно инвестиционному плану", 0],
    ["your new token was added to your portfolio as planned", 0],
    ["airdrop токены зачислены на ваш кошелек metamask", 0],
    ["airdrop tokens have been credited to your metamask wallet", 0],
    ["binance аккаунт активен никаких проблем не обнаружено", 0],
    ["your binance account is active no issues detected", 0],
    ["ваш биткоин кошелек синхронизирован баланс обновлен", 0],
    ["your bitcoin wallet synced successfully balance updated", 0],
    ["новая биржа подтвердила регистрацию проверьте почту для активации", 0],
    ["the new exchange confirmed your signup check your email to activate", 0],
    ["технический апдейт биржи завершен сервис работает в обычном режиме", 0],
    ["exchange maintenance completed successfully service is back to normal", 0],
    ["спасибо за подписку на крипто-канал первый сигнал придет завтра", 0],
    ["thanks for subscribing to the crypto channel first signal arrives tomorrow", 0],
    ["ваш nft успешно передан новому владельцу транзакция завершена", 0],
    ["your nft was successfully transferred to the new owner transaction complete", 0],
    ["торговый бот показал стабильную но скромную доходность в этом месяце", 0],
    ["the trading bot showed steady but modest returns this month as expected", 0],
    ["тіркелу сәтті аяқталды бонус шотыңызға түсті", 0],
    ["әмиян сәтті расталды барлығы дұрыс", 0],

    # SAFE: QR code, matching variants
    ["кэшбэк за последнюю покупку уже начислен проверьте в приложении", 0],
    ["your cashback from the last purchase was already credited check the app", 0],
    ["скидка по чеку уже применена спасибо за покупку", 0],
    ["the discount from your receipt was already applied thank you for shopping", 0],
    ["розыгрыш автомобиля завершен победитель будет объявлен завтра", 0],
    ["the car giveaway has ended the winner will be announced tomorrow", 0],
    ["wifi в кафе бесплатный пароль указан на чеке", 0],
    ["the cafe wifi is free the password is printed on your receipt", 0],
    ["объявление о квартире полностью доступно на сайте без регистрации", 0],
    ["the apartment listing is fully available on the website with no signup needed", 0],
    ["штраф гибдд уже оплачен через приложение банка чек сохранен", 0],
    ["the traffic fine was already paid through the banking app receipt saved", 0],

    # SAFE: Charity, matching variants
    ["спасибо за пожертвование пострадавшим от пожара ваша помощь очень ценна", 0],
    ["thank you for your donation to wildfire victims your help means a lot", 0],
    ["операция ребенку прошла успешно спасибо всем кто помог", 0],
    ["the child's surgery was successful thank you to everyone who helped", 0],
    ["фонд помощи животным благодарит вас за постоянную поддержку", 0],
    ["the animal rescue fund thanks you for your continued support", 0],
    ["сбор на восстановление после наводнения завершен спасибо всем донорам", 0],
    ["the flood relief fundraiser has concluded thank you to all donors", 0],
    ["благотворительный концерт для ветеранов прошел успешно спасибо за участие", 0],
    ["the charity concert for veterans was a success thank you for attending", 0],
    ["қайырымдылық жиналды рахмет көмегіңіз үшін", 0],
    ["балаға көмек берілді рахмет барлығына", 0],

    # SAFE: Subscription, matching services
    ["netflix payment was successful enjoy your shows", 0],
    ["netflix оплата прошла успешно приятного просмотра", 0],
    ["spotify premium renewed automatically thank you for listening", 0],
    ["spotify премиум продлен автоматически спасибо что слушаете", 0],
    ["youtube premium subscription confirmed enjoy ad free viewing", 0],
    ["youtube premium подписка подтверждена смотрите без рекламы", 0],
    ["icloud storage upgrade was successful your files are backed up", 0],
    ["icloud хранилище успешно обновлено файлы сохранены", 0],
    ["google one subscription renewed your files remain protected", 0],
    ["google one подписка продлена ваши файлы под защитой", 0],
    ["amazon prime membership renewed enjoy free shipping this year", 0],
    ["amazon prime подписка продлена наслаждайтесь бесплатной доставкой", 0],
    ["kaspi pay подписка успешно оплачена спасибо", 0],
    ["your streaming subscription renewed successfully thank you", 0],
    ["disney plus payment was successful enjoy the new season", 0],
    ["disney plus оплата прошла успешно приятного просмотра", 0],
    ["жазылымыңыз сәтті ұзартылды рахмет", 0],
    ["тегін сынақ мерзімі аяқталды жазылым әдеттегідей жалғасады", 0],

    # SAFE: SIM/social media, matching platforms
    ["ваш whatsapp вход подтвержден с известного устройства", 0],
    ["your whatsapp login was confirmed from a recognized device", 0],
    ["telegram подтверждает вход с вашего обычного устройства", 0],
    ["telegram confirms the login was from your usual device", 0],
    ["ваш instagram аккаунт в порядке нарушений не обнаружено", 0],
    ["your instagram account is in good standing no violations found", 0],
    ["facebook подтверждает что это был ваш обычный вход", 0],
    ["facebook confirms this was your regular login as expected", 0],
    ["tiktok аккаунт активен подозрительной активности не обнаружено", 0],
    ["your tiktok account is active no suspicious activity detected", 0],
    ["gmail подтверждает что настройки аккаунта не менялись", 0],
    ["gmail confirms your account settings have not been changed", 0],
    ["ваш номер активен сим карта работает без проблем", 0],
    ["your number is active your sim card is working with no issues", 0],
    ["сим карта продлена автоматически согласно тарифу", 0],
    ["your sim plan was renewed automatically as scheduled", 0],
    ["код восстановления был использован вами успешно", 0],
    ["your recovery code was used successfully by you", 0],
    ["linkedin профиль подтвержден все в порядке", 0],
    ["your linkedin profile is verified everything looks good", 0],
    ["аккаунт x в порядке нарушений сообщества не обнаружено", 0],
    ["your x account is fine no community guideline violations found", 0],
    ["whatsapp сіздің құрылғыңыздан кірілді барлығы дұрыс", 0],
    ["telegram жаңа кіру расталды сіз болдыңыз", 0],

    # SAFE: Tax/inheritance, matching variants
    ["налоговый возврат зачислен на ваш счет согласно декларации", 0],
    ["your tax refund was deposited to your account per your filing", 0],
    ["компенсация от государства зачислена спасибо за терпение", 0],
    ["the government compensation payment was deposited thank you for your patience", 0],
    ["нотариус подтвердил оформление наследства согласно закону", 0],
    ["the notary confirmed the inheritance was processed according to the law", 0],
    ["неоплаченных штрафов на вашем счету не найдено все чисто", 0],
    ["no unpaid fines were found on your record everything is clear", 0],
    ["налоговая инспекция подтвердила получение вашей декларации", 0],
    ["the tax office confirmed receipt of your annual filing", 0],
    ["салық қайтарымы шотыңызға түсті рахмет", 0],
    ["мұрагерлік құжаттар заң бойынша ресімделді", 0],

    # SAFE: Global brand impersonation, matching brands
    ["microsoft account is active and secure no action needed", 0],
    ["netflix account is in good standing enjoy your subscription", 0],
    ["ebay confirms your recent purchase was completed successfully", 0],
    ["your dhl shipment was delivered on schedule thank you", 0],
    ["fedex confirms your package tracking is up to date", 0],
    ["linkedin profile verification completed successfully", 0],
    ["microsoft аккаунт активен и защищен действий не требуется", 0],
    ["netflix аккаунт в порядке приятного просмотра", 0],
    ["ebay подтверждает что ваша покупка завершена успешно", 0],
    ["ваша посылка dhl доставлена по расписанию спасибо", 0],
    ["fedex подтверждает актуальность отслеживания вашей посылки", 0],
    ["linkedin подтвердил профиль успешно", 0],
    ["your microsoft 365 subscription renewed successfully", 0],
    ["steam account login was from your recognized device", 0],
    ["steam аккаунт вход выполнен с вашего обычного устройства", 0],
    ["your ebay payment was processed successfully thank you", 0],
    ["amazon confirms your recent sign in was from your usual location", 0],
    ["amazon подтверждает что вход был выполнен из привычного места", 0],

    # SAFE: Business email compromise, matching variants
    ["финансовый директор одобрил счет оплата пройдет по графику", 0],
    ["the cfo approved the invoice payment will follow the usual schedule", 0],
    ["ваша подпись на счете больше не требуется решение принято", 0],
    ["your approval on the invoice is no longer needed decision was made", 0],
    ["бухгалтерия подтверждает зарплата будет переведена как обычно", 0],
    ["accounting confirms payroll will be transferred as usual this month", 0],
    ["поставщик подтвердил старые реквизиты остаются без изменений", 0],
    ["the vendor confirmed the existing account details remain unchanged", 0],
    ["директор одобрил счет обычная проверка пройдена", 0],
    ["the director approved the invoice after the standard review", 0],
    ["менеджер шотты кәдімгі тәртіппен төледі", 0],
    ["бухгалтер жалақыны әдеттегідей аударады", 0],

    # SAFE: AI-generated deepfake investment videos, matching variants
    ["видео с известным актером было удалено платформой как поддельное", 0],
    ["the video featuring the celebrity was removed by the platform as fake", 0],
    ["финансовый блогер опубликовал новый честный обзор рынка", 0],
    ["the finance blogger posted a new honest market overview today", 0],
    ["платформа предупредила о поддельных видео с ее руководством", 0],
    ["the platform warned users about fake videos impersonating its leadership", 0],
    ["интервью с предпринимателем вышло на официальном канале компании", 0],
    ["the entrepreneur's interview was published on the company's official channel", 0],
    ["вебинар о финансовой грамотности прошел успешно спасибо за участие", 0],
    ["the financial literacy webinar was a success thank you for attending", 0],
    ["белгілі кәсіпкер алаяқтарды ескертті жалған бейнелерден сақ болыңыз", 0],
    ["жарнамалық бейне ресми арнада ғана жарияланады", 0],

    # SAFE: Fake courier demanding a code, matching variants
    ["курьер fedex оставил посылку у двери подпись не требовалась", 0],
    ["the fedex courier left the package at the door no signature needed", 0],
    ["курьер сдэк передал посылку соседке как договаривались", 0],
    ["the cdek courier handed the package to the neighbor as agreed", 0],
    ["курьер kaspi доставил заказ вовремя спасибо", 0],
    ["the kaspi courier delivered the order on time thank you", 0],
    ["водитель яндекс доставки оставил заказ на ресепшене", 0],
    ["the delivery driver left the order at the front desk", 0],
    ["курьер боксберри позвонил в дверь и передал посылку лично", 0],
    ["the boxberry courier rang the bell and handed over the package personally", 0],
    ["жеткізуші тапсырысты есік алдына қойды", 0],
    ["курьер тапсырысты өз қолыма берді рахмет", 0],

    # SAFE: Expiring rewards points, matching variants
    ["ваши бонусы magnum успешно начислены за последнюю покупку", 0],
    ["your magnum loyalty points were credited for your last purchase", 0],
    ["бонусные мили аэрофлот добавлены на счет после перелета", 0],
    ["your airline miles were added to your account after the flight", 0],
    ["баллы небольшая карта обновлены согласно программе лояльности", 0],
    ["your loyalty card points were updated per the rewards program", 0],
    ["кэшбэк баллы зачислены на счет спасибо за покупку", 0],
    ["your cashback points were credited to your account thank you for shopping", 0],
    ["подарочная карта активирована и готова к использованию", 0],
    ["your gift card has been activated and is ready to use", 0],
    ["бонустарыңыз сәтті есептелді рахмет", 0],
    ["ұпайларыңыз жаңартылды бағдарлама бойынша", 0],

    # SAFE: Traffic violation QR code, matching variants
    ["штраф за превышение скорости оплачен через приложение банка", 0],
    ["the speeding fine was paid through the banking app successfully", 0],
    ["нарушение парковки не найдено на вашем счету все чисто", 0],
    ["no parking violations were found on your account everything is clear", 0],
    ["штраф за проезд на красный свет оплачен вовремя чек сохранен", 0],
    ["the red light fine was paid on time receipt saved", 0],
    ["нарушений пдд на вашем счету не зафиксировано", 0],
    ["no traffic violations have been recorded on your account", 0],
    ["штраф гибдд оплачен со скидкой через официальное приложение", 0],
    ["the traffic fine was paid with a discount through the official app", 0],
    ["жол ережесін бұзу тіркелмеді есебіңізде таза", 0],
    ["айыппұл жеңілдікпен ресми қосымша арқылы төленді", 0],

    # SAFE: Fake "verify you're human" scam, matching variants
    ["проверка безопасности пройдена доступ к сайту открыт", 0],
    ["the security check passed access to the website is now open", 0],
    ["система подтвердила что вы не робот доступ разрешен", 0],
    ["the system confirmed you are not a robot access granted", 0],
    ["файл доступен для загрузки проверка человека не требуется", 0],
    ["the file is available for download no human verification needed", 0],
    ["проверка cloudflare завершена страница загружается", 0],
    ["cloudflare check completed the page is now loading", 0],
    ["капча пройдена успешно можно продолжать", 0],
    ["the captcha was completed successfully you may continue", 0],
    ["тексеру сәтті аяқталды сайтқа қол жеткізе аласыз", 0],
    ["адам екеніңіз расталды жүктеу басталды", 0],

    # SAFE: QR code at a physical location, matching variants
    ["оплата парковки через qr код прошла успешно чек в приложении", 0],
    ["the parking payment via qr code was successful receipt in the app", 0],
    ["оплата проезда в автобусе через qr код подтверждена", 0],
    ["the bus fare payment via qr code was confirmed successfully", 0],
    ["меню кафе по qr коду обновлено добавлены новые блюда", 0],
    ["the cafe's qr code menu was updated with new dishes", 0],
    ["купон со скидкой с билборда активирован в приложении", 0],
    ["the discount coupon from the billboard was activated in the app", 0],
    ["оплата консьерж услуг через qr код прошла успешно", 0],
    ["the concierge service payment via qr code went through successfully", 0],
    ["тұрақ ақысы qr код арқылы сәтті төленді", 0],
    ["асхана мәзірі qr код арқылы қолжетімді жаңартылды", 0],

    # SAFE: Bank bonus-exchange, matching banks
    ["бонусы sberbank зачислены на карту согласно программе лояльности", 0],
    ["your sberbank bonus points were credited per the loyalty program", 0],
    ["баллы tinkoff обновлены после последней покупки", 0],
    ["your tinkoff points were updated after your last purchase", 0],
    ["бонусы kaspi зачислены автоматически в конце месяца", 0],
    ["your kaspi bonus points are credited automatically at month's end", 0],
    ["мили halyk bank добавлены после оплаты картой", 0],
    ["your halyk bank miles were added after your card payment", 0],
    ["бонусная программа втб продлена без изменений условий", 0],
    ["the vtb rewards program was extended with no changes to terms", 0],
    ["банк бонустары картаға сәтті есептелді", 0],
    ["бонустар осы айдың соңында автоматты түрде есептеледі", 0],

    # SAFE: Unpaid toll SMS scam, matching variants
    ["ваш проезд по трассе m1 оплачен автоматически с привязанной карты", 0],
    ["your m1 highway toll was paid automatically from your linked card", 0],
    ["ezpass account balance is sufficient no action needed this month", 0],
    ["fastrak toll payment was processed automatically as usual", 0],
    ["ipass confirms your toll balance is fully paid and up to date", 0],
    ["toll authority confirms your account is in good standing", 0],
    ["проезд по платной дороге оплачен без проблем чек сохранен", 0],
    ["штраф за проезд не зафиксирован ваш счет чист", 0],
    ["уведомление о неоплаченном проезде не найдено все в порядке", 0],
    ["камера подтвердила что оплата пошлины прошла успешно", 0],
    ["your toll invoice was settled automatically thank you", 0],
    ["highway toll payment confirmed no further action needed", 0],
    ["жол ақысы автоматты түрде төленді карта арқылы", 0],
    ["ақылы жол ақысы бойынша қарыз жоқ", 0],
    ["toll payment history shows all trips are paid in full", 0],
    ["your recent toll trip was paid successfully via autopay", 0],
    ["state toll authority confirms your account balance is zero", 0],
    ["camera confirms toll was paid the transaction is complete", 0],

    # SAFE: AI voice-cloning family emergency scam, matching variants
    ["мама я дома все хорошо не переживай", 0],
    ["dad i'm home safe everything's fine don't worry", 0],
    ["папа машина в порядке ремонт не нужен", 0],
    ["mom the car is fine no repairs needed", 0],
    ["бабуля я здоров просто заходил проведать тебя", 0],
    ["grandpa i'm healthy just called to check on you", 0],
    ["это твой внук хотел спросить как твое здоровье", 0],
    ["it's your grandson just wanted to ask how you're feeling", 0],
    ["мама голос немного другой потому что я простыл ничего страшного", 0],
    ["mom my voice sounds different i just have a cold nothing serious", 0],
    ["папа полиция ничего не сообщала я дома весь день", 0],
    ["dad the police haven't called i've been home all day", 0],
    ["әжем мен дұрыс жақсымын уайымдама", 0],
    ["атам мен сау-саламатпын тек сәлемдесейін дедім", 0],
    ["sister my card works fine no issues at the border", 0],
    ["сестра карта работает нормально проблем на границе не было", 0],
    ["brother i crossed the border with no issues thanks for asking", 0],
    ["братан все нормально границу прошел без проблем", 0],

    # SAFE: Pig butchering, matching variants
    ["милый посмотри какие фото я сделала на выходных", 0],
    ["babe check out these photos from my weekend trip", 0],
    ["любимая расскажи как прошел твой рабочий день", 0],
    ["sweetheart tell me how your work day went", 0],
    ["дорогой мой наставник по йоге дал новое расписание занятий", 0],
    ["honey my yoga instructor gave me a new class schedule", 0],
    ["я подписалась на курс по кулинарии начинается завтра", 0],
    ["i signed up for a cooking class it starts tomorrow", 0],
    ["жанашырым қалайсың сағындым сені", 0],
    ["сен қалайсың бүгін не істедің", 0],
    ["милая давай в эти выходные сходим в кино", 0],
    ["darling let's go to the movies this weekend", 0],

    # SAFE: Bank "call us back" scheme, matching banks
    ["halyk bank подтверждает что звонок из вашего отделения был легитимным", 0],
    ["halyk bank confirms the call from your branch was legitimate", 0],
    ["kaspi подтверждает что операция была одобрена вами лично", 0],
    ["kaspi confirms the transaction was approved by you personally", 0],
    ["tinkoff служба поддержки закрыла ваше обращение решение найдено", 0],
    ["tinkoff support closed your ticket with a resolution found", 0],
    ["втб подтверждает что звонок был плановой проверкой без проблем", 0],
    ["vtb confirms the call was a routine check nothing to worry about", 0],
    ["альфа банк подтверждает что ваш счет в полном порядке", 0],
    ["alfa bank confirms your account is in perfect standing", 0],
    ["жусан банк қоңырауыңызды растады барлығы дұрыс", 0],
    ["forte bank шотыңызда мәселе жоқ екенін растады", 0],
    ["отбасы банк сіздің өтінішіңізді қабылдады рахмет", 0],
    ["otbasy bank confirms your request was received thank you", 0],
    ["почта банк подтверждает что операция была стандартной", 0],
    ["post bank confirms the transaction was a routine one", 0],
    ["совкомбанк закрыл обращение по счету проблем не найдено", 0],
    ["sovcombank closed the account inquiry no issues were found", 0],

    # SAFE: NFC "tap your phone" scam, matching variants
    ["оплата через nfc прошла успешно чек в приложении", 0],
    ["the nfc payment went through successfully receipt in the app", 0],
    ["бесконтактная оплата подтверждена банком", 0],
    ["the contactless payment was confirmed by the bank", 0],
    ["терминал подтвердил успешную оплату картой", 0],
    ["the terminal confirmed the card payment was successful", 0],
    ["оператор банка подтвердил что операция была совершена вами", 0],
    ["the bank operator confirmed the transaction was made by you", 0],
    ["возврат средств зачислен на карту автоматически", 0],
    ["the refund was credited back to your card automatically", 0],
    ["телефон арқылы төлем сәтті өтті", 0],
    ["терминал төлемді растады рахмет", 0],

    # SAFE: Landlord impersonation, matching variants
    ["хозяин квартиры подтвердил получение оплаты за этот месяц", 0],
    ["your landlord confirmed receipt of this month's rent payment", 0],
    ["арендодатель не менял банк оплата идет как обычно", 0],
    ["the landlord hasn't changed banks payment goes as usual", 0],
    ["риэлтор подтвердил что депозит был получен на основной счет", 0],
    ["the realtor confirmed the deposit was received on the main account", 0],
    ["хозяин дома поблагодарил за своевременную оплату аренды", 0],
    ["the homeowner thanked you for paying rent on time", 0],
    ["арендодатель сообщил что банковский счет не менялся", 0],
    ["the landlord said the bank account has not changed", 0],
    ["үй иесі осы айдың ақысын алды рахмет", 0],
    ["жалдаушы төлемді уақытында алды", 0],

    # SAFE: matching legitimate marketplace exchanges (no verification-fee
    # request, no fake registry site, just a normal in-person deal)
    ["покупатель согласен приехать сегодня вечером посмотреть ноутбук и купить при встрече", 0],
    ["buyer agreed to come check out the laptop in person tonight and pay on pickup", 0],
    ["покупатель уточнил комплектацию и попросил скинуть еще фото при встрече", 0],
    ["the buyer asked about the accessories included and wants a few more photos before coming", 0],
    ["покупатель предложил встретиться в людном месте для безопасности сделки", 0],
    ["the buyer suggested meeting in a public place for a safer in-person deal", 0],
    ["сатып алушы бүгін кешке келіп затты көруге келісті", 0],
    ["сатып алушы қосымша фото сұрап кездесуде төлеуге келісті", 0],

    # SAFE: matching everyday counterparts to the newest real-world-grounded
    # categories above
    ["внук позвонил просто узнать как дела все хорошо", 0],
    ["my grandson just called to check in everything is fine", 0],
    ["адвокат подтвердил что дело закрыто без залога", 0],

    ["магазин подтвердил что программа тайного покупателя не требует обналичивания чеков", 0],
    ["the store confirmed their mystery shopper program never asks you to cash a check", 0],

    ["заводчик прислал видео щенка и согласился на встречу без предоплаты", 0],
    ["the breeder sent a video of the puppy and agreed to a meet-up with no upfront payment", 0],

    ["агентство недвижимости подтвердило бронь после осмотра квартиры", 0],
    ["the rental agency confirmed the booking after we viewed the apartment in person", 0],

    ["налоговая прислала официальное письмо по почте без требования оплаты картами", 0],
    ["the tax office sent an official letter by mail with no demand for gift card payment", 0],

    ["коммунальная компания напомнила об оплате до конца месяца без угроз", 0],
    ["the utility company sent a normal reminder that the bill is due by the end of the month", 0],

    ["ведомство соцстрахования подтвердило что с моим номером все в порядке", 0],
    ["the social security office confirmed my number is fine and nothing is suspended", 0],

    ["в суде подтвердили что моя явка перенесена без штрафа", 0],
    ["the court confirmed my appearance was simply rescheduled with no fine", 0],

    ["антивирус нашел и удалил файл в обычном режиме без блокировки экрана", 0],
    ["the antivirus found and removed the file normally with no screen lock", 0],

    ["пропущенный звонок оказался от коллеги перезвонила на местный номер", 0],
    ["the missed call was just from a colleague i called back the regular local number", 0],

    ["брат в армии написал что все хорошо и служба идет по плану", 0],
    ["my brother in the army wrote that everything is fine and his service is going as planned", 0],

    ["приложение с заданиями выплатило небольшую сумму без каких либо депозитов", 0],
    ["the task app paid out a small amount with no deposit required at any point", 0],

    ["работодатель прислал оборудование напрямую от поставщика без чеков для обналичивания", 0],
    ["my employer shipped the equipment directly from the vendor with no check to cash", 0],

    ["отдел кадров подтвердил реквизиты зарплаты через защищенный внутренний портал", 0],
    ["hr confirmed my payroll details through the secure internal portal as usual", 0],

    ["друг подтвердил перевод был правильным ничего возвращать не нужно", 0],
    ["my friend confirmed the transfer amount was correct nothing needs to be sent back", 0],

    ["получил обычное рекламное письмо без угроз и вложений", 0],
    ["i just got a regular marketing email with no threats or attachments", 0],

    ["служба поддержки подтвердила что подписка не продлевалась и платеж не проходил", 0],
    ["support confirmed the subscription was never renewed and no charge went through", 0],

    ["банк подтвердил что долга по кредиту нет и звонок был мошенническим", 0],
    ["the bank confirmed there is no outstanding debt and the call was fraudulent", 0],

    ["дилер прислал официальное письмо о продлении гарантии без давления", 0],
    ["the dealership sent an official letter about the warranty renewal with no pressure", 0],

    ["перезвонила по номеру с сайта производителя а не из рекламы и решила вопрос без удаленного доступа", 0],
    ["i called the number from the manufacturer's official site and fixed it without remote access", 0],

    ["друг попросил помочь с переездом а не с переводом чужих денег", 0],
    ["a friend asked for help moving apartments not for moving anyone else's money", 0],

    ["хозяин жилья принял оплату через платформу как обычно", 0],
    ["the host accepted payment through the platform as usual with full protection", 0],

    ["сестра написала с обычного номера и все совпало с предыдущими разговорами", 0],
    ["my sister texted from her usual number and everything matched our previous conversations", 0],

    ["рекрутер объяснил что реквизиты зарплаты вносятся после официального оформления в отделе кадров", 0],
    ["the recruiter explained that payroll details are entered through hr after you're officially hired", 0],

    ["получила уведомление из суда по обычной почте без вложений и без угроз", 0],
    ["i received a normal court notice by mail with no attachment and no threats", 0],

    # SAFE: matching everyday counterparts to the round-2 real-world-grounded
    # categories above
    ["агентство по недвижимости подтвердило что таймшер продан без предоплаты с моей стороны", 0],
    ["the timeshare agency confirmed the sale went through with no upfront fee from me", 0],

    ["банк одобрил кредит после стандартной проверки без предоплаты", 0],
    ["the bank approved my loan after the standard credit check with no upfront fee", 0],

    ["получила обычное письмо о розыгрыше без требования оплаты налога", 0],
    ["i got a regular sweepstakes email with no tax payment required to claim anything", 0],

    ["кредитное агентство объяснило что бесплатно помогает разобраться с историей", 0],
    ["the credit counseling service explained their help is free with no upfront fee", 0],

    ["агент показал квартиру лично и взял залог только после осмотра", 0],
    ["the agent showed the apartment in person and only took a deposit after the viewing", 0],

    ["приложение для знакомств не просит платить за верификацию", 0],
    ["the dating app never asked me to pay for any verification badge", 0],

    ["интернет магазин подтвердил заказ и прислал обычный чек", 0],
    ["the online store confirmed my order and sent a normal receipt", 0],

    ["друг предупредил что компании обещающие вернуть деньги от мошенников сами мошенники", 0],
    ["a friend warned me that companies promising to recover scammed money are scams themselves", 0],

    ["банк подтвердил что долга нет и звонок был мошенническим", 0],
    ["the bank confirmed there is no debt and the call was fraudulent", 0],

    ["служба занятости подтвердила заявку через официальный портал без звонков", 0],
    ["the unemployment office confirmed my claim through the official portal with no phone calls", 0],

    ["друг прислал фото подарка без всяких таможенных пошлин", 0],
    ["a friend sent photos of the gift with no customs fee involved at all", 0],

    ["школа позвонила чтобы сообщить об обычном мероприятии никаких происшествий", 0],
    ["the school called about a regular event nothing happened to my child", 0],

    ["работодатель оплатил обучение сам никаких взносов с меня не требовалось", 0],
    ["my employer covered the training cost themselves no payment was required from me", 0],

    ["банк обновил данные через приложение без ссылок и угроз закрытия счета", 0],
    ["the bank updated my details through the app with no link and no threat to close it", 0],

    ["temu подтвердил доставку заказа без запроса обновить оплату", 0],
    ["shein confirmed the delivery of my order with no request to update payment", 0],

    ["бухгалтерия подтвердила что счет поставщика уже оплачен по графику", 0],
    ["accounts payable confirmed the vendor invoice was already paid on the normal schedule", 0],

    ["получила официальное письмо из суда без угроз и с обычными сроками", 0],
    ["i received an official court letter with no threats and the normal timeline", 0],

    ["клиент оставил настоящий положительный отзыв о нашем бизнесе", 0],
    ["a customer left a genuine positive review about our business", 0],

    ["оператор подтвердил что рассылка отключена без каких либо платежей", 0],
    ["the carrier confirmed the texts were unsubscribed with no charges at all", 0],

    ["компания по установке солнечных панелей прислала обычную смету без предоплаты", 0],
    ["the solar company sent a normal quote with no upfront payment required", 0],

    ["получила официальное письмо от соцстрахования по почте без угроз ареста", 0],
    ["i received an official social security letter by mail with no threat of arrest", 0],

    ["познакомились в приложении и просто мило переписываемся без просьб о деньгах", 0],
    ["we matched on the app and are just chatting nicely with no money requests at all", 0],

    # SAFE: messages that legitimately contain a real link — every SAFE
    # example above this point has zero URLs, so has_link risked being
    # learned as an almost pure fraud signal even though ordinary receipts,
    # delivery tracking, and confirmations legitimately contain a link to
    # the real brand's own domain
    ["ваш чек по оплате доступен по ссылке https://kaspi.kz/receipt", 0],
    ["your payment receipt is available at https://kaspi.kz/receipt", 0],
    ["отследить посылку можно по ссылке https://kazpost.kz/tracking", 0],
    ["track your parcel here https://dhl.com/tracking", 0],
    ["ваша справка готова скачайте по ссылке https://egov.kz/documents", 0],
    ["your certificate is ready download it at https://egov.kz/documents", 0],
    ["детали перевода смотрите в приложении https://halykbank.kz/history", 0],
    ["see the transfer details in the app at https://halykbank.kz/history", 0],
    ["ваш заказ подтвержден детали по ссылке https://amazon.com/orders", 0],
    ["your order is confirmed details at https://amazon.com/orders", 0],
    ["счет за подписку доступен по ссылке https://apple.com/billing", 0],
    ["your subscription invoice is available at https://apple.com/billing", 0],
    ["напоминание о встрече подробности по ссылке https://google.com/calendar", 0],
    ["meeting reminder details at https://google.com/calendar", 0],
    ["ваш аккаунт подтвержден пройдите по ссылке https://instagram.com/settings чтобы изменить профиль", 0],
    ["your account settings can be updated at https://instagram.com/settings anytime", 0],
    ["ваша выписка по счету доступна по ссылке https://sberbank.ru/statements", 0],
    ["your account statement is available at https://sberbank.ru/statements", 0],
    ["жеткізу мәртебесін көру үшін сілтемеге өтіңіз https://kazpost.kz/tracking", 0],
    ["чек қолжетімді https://kaspi.kz/receipt сілтемесі арқылы", 0],
]

def explain(features):
    # Only explain features that are relevant (non-zero and meaningful)
    irrelevant = {"text_length", "word_count", "avg_word_length"}
    return [
        T[f"feat_{k}"]
        for k, v in features.items()
        if v > 0 and k not in irrelevant and f"feat_{k}" in T
    ]

def risk_style(prob):
    """Thin wrapper over fraud_logic.risk_level() that resolves the level
    key to the current UI language's translated label."""
    level_key, css_class, emoji = risk_level(prob)
    return T[level_key], css_class, emoji

def render_risk_meter(prob, threshold, title):
    """Pure CSS/HTML risk meter (no charting library needed): a gradient
    track from green to red, a pointer at the current score, and a dashed
    line marking the decision threshold."""
    pct = max(0.0, min(1.0, prob)) * 100
    threshold_pct = max(0.0, min(1.0, threshold)) * 100
    return f"""
    <div class="meter-wrap">
        <div class="meter-title">{title}</div>
        <div class="meter-track">
            <div class="meter-value-label" style="left:{pct:.1f}%;">{pct:.0f}%</div>
            <div class="meter-pointer" style="left:{pct:.1f}%;"></div>
            <div class="meter-threshold-line" style="left:{threshold_pct:.1f}%;"></div>
        </div>
        <div class="meter-scale"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>
    </div>
    """

# =========================
# TRAIN MODELS (ENSEMBLE)
# =========================
@st.cache_resource(show_spinner=False)
def train_models(feature_schema):
    """`feature_schema` isn't used in the body — it exists purely so the
    cache key changes whenever extract_features()'s output shape changes.
    st.cache_resource only invalidates on this function's OWN source
    changing, not on a helper it calls; without this, adding/removing a
    feature (like brand_flag) silently leaves a stale cached model with
    the wrong number of columns, causing a ValueError at predict time."""
    rows, labels, texts = [], [], []
    for text, label in data:
        f, _ = extract_features(text)
        rows.append(f)
        labels.append(label)
        texts.append(text)

    X_train = pd.DataFrame(rows)
    y_train = np.array(labels)

    # Ensemble of 4 models for more reliable predictions
    lr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])
    rf_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=50, random_state=42))
    ])
    gb_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=50, random_state=42))
    ])

    lr_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)

    # 4th ensemble member: learns fraud-indicative language directly from the
    # training text (word/bigram frequencies) instead of only scoring against
    # the hand-curated keyword lists above — this is what lets the system
    # catch scam phrasing that isn't in any of those lists.
    tfidf_model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True)),
        ("clf", MultinomialNB())
    ])
    tfidf_model.fit(texts, y_train)

    # Cross-validation accuracy for each model (3-fold: dataset is small, keeps startup fast)
    lr_cv = cross_val_score(lr_model, X_train, y_train, cv=3, scoring="accuracy").mean()
    rf_cv = cross_val_score(rf_model, X_train, y_train, cv=3, scoring="accuracy").mean()
    gb_cv = cross_val_score(gb_model, X_train, y_train, cv=3, scoring="accuracy").mean()
    tfidf_cv = cross_val_score(tfidf_model, texts, y_train, cv=3, scoring="accuracy").mean()

    metrics = {
        "Logistic Regression": round(lr_cv * 100, 1),
        "Random Forest": round(rf_cv * 100, 1),
        "Gradient Boosting": round(gb_cv * 100, 1),
        "Text Patterns (TF-IDF)": round(tfidf_cv * 100, 1),
    }

    return lr_model, rf_model, gb_model, tfidf_model, metrics, X_train, y_train


FEATURE_SCHEMA = tuple(sorted(extract_features("")[0].keys()))
lr_model, rf_model, gb_model, tfidf_model, model_metrics, X_train, y_train = train_models(FEATURE_SCHEMA)

# =========================
# UI STYLE
# =========================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
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

/* Force readable text for native Streamlit widgets in the main area,
   regardless of the visitor's OS/browser light-dark preference. */
.main h1, .main h2, .main h3, .main h4, .main h5, .main h6,
[data-testid="stMain"] h1, [data-testid="stMain"] h2, [data-testid="stMain"] h3,
[data-testid="stMain"] h4, [data-testid="stMain"] h5, [data-testid="stMain"] h6 {
    color: #0f172a !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricLabel"],
[data-testid="stMetricDelta"] {
    color: #0f172a !important;
}

.main [data-testid="stCaptionContainer"],
[data-testid="stMain"] [data-testid="stCaptionContainer"] {
    color: #64748b !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] svg {
    color: #0f172a !important;
    fill: #0f172a !important;
}

[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
    color: #1e293b !important;
}

[data-testid="stTabs"] [data-baseweb="tab"] p {
    color: #334155 !important;
}

[data-testid="stDataFrame"] {
    color: #0f172a;
}

.main [data-testid="stWidgetLabel"] p,
[data-testid="stMain"] [data-testid="stWidgetLabel"] p,
.main [data-testid="stFileUploaderDropzoneInstructions"] div,
[data-testid="stMain"] [data-testid="stFileUploaderDropzoneInstructions"] div {
    color: #0f172a !important;
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

.howto-card {
    padding: 22px 26px;
}

.howto-steps {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 16px;
}

.howto-step {
    flex: 1;
    min-width: 220px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 14px 16px;
    font-size: 16px;
    font-weight: 700;
    color: #1e293b;
    line-height: 1.5;
}

.legend-title {
    font-size: 14px;
    font-weight: 800;
    color: #475569;
    margin-bottom: 8px;
}

.legend-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.legend-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 750;
    border: 1px solid transparent;
}
.legend-low      {background:#dcfce7; color:#166534; border-color:#bbf7d0;}
.legend-mid      {background:#fef9c3; color:#854d0e; border-color:#fef08a;}
.legend-high     {background:#ffedd5; color:#9a3412; border-color:#fed7aa;}
.legend-critical {background:#fee2e2; color:#991b1b; border-color:#fecaca;}

.disclaimer-box {
    margin-top: 16px;
    padding: 12px 16px;
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 14px;
    color: #92400e;
    font-size: 13px;
    font-weight: 600;
    line-height: 1.5;
}

.meter-wrap {
    padding: 10px 6px 4px 6px;
}

.meter-title {
    font-size: 13px;
    font-weight: 750;
    color: #475569;
    margin-bottom: 22px;
}

.meter-track {
    position: relative;
    height: 22px;
    border-radius: 999px;
    background: linear-gradient(90deg, #22c55e 0%, #eab308 35%, #f97316 65%, #ef4444 100%);
    border: 1px solid rgba(15, 23, 42, 0.08);
    overflow: visible;
}

.meter-pointer {
    position: absolute;
    top: -3px;
    width: 4px;
    height: 28px;
    background: #0f172a;
    border-radius: 2px;
    transform: translateX(-2px);
    box-shadow: 0 0 0 2px #ffffff;
}

.meter-threshold-line {
    position: absolute;
    top: -6px;
    height: 34px;
    border-left: 2px dashed #334155;
    opacity: 0.85;
}

.meter-value-label {
    position: absolute;
    top: -30px;
    transform: translateX(-50%);
    font-weight: 900;
    font-size: 13px;
    color: #0f172a;
    background: #ffffff;
    padding: 2px 8px;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    white-space: nowrap;
}

.meter-scale {
    display: flex;
    justify-content: space-between;
    margin-top: 8px;
    font-size: 11px;
    color: #94a3b8;
    font-weight: 700;
}

@media (max-width: 640px) {
    .howto-steps { flex-direction: column; }
    .howto-step { min-width: 0; font-size: 15px; }
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

.stButton > button,
[data-testid="stDownloadButton"] button,
[data-testid="stFormSubmitButton"] button {
    border-radius: 18px !important;
    padding: 0.8rem 1rem !important;
    font-weight: 850 !important;
    border: 0 !important;
    background: linear-gradient(135deg, #2563eb, #0f766e) !important;
    color: white !important;
    box-shadow: 0 14px 30px rgba(37,99,235,0.23) !important;
}

.stButton > button:hover,
[data-testid="stDownloadButton"] button:hover,
[data-testid="stFormSubmitButton"] button:hover {
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

.highlight-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 16px 18px;
    line-height: 1.85;
    color: #1e293b;
    font-size: 15px;
    word-wrap: break-word;
}

.hl-urgent, .hl-secret, .hl-money, .hl-threat,
.hl-reward, .hl-pressure, .hl-identity, .hl-link {
    padding: 1px 5px;
    border-radius: 6px;
    font-weight: 750;
    box-decoration-break: clone;
    -webkit-box-decoration-break: clone;
}
.hl-urgent   {background:#fef9c3; color:#854d0e;}
.hl-secret   {background:#fee2e2; color:#991b1b;}
.hl-money    {background:#dbeafe; color:#1e40af;}
.hl-threat   {background:#ffedd5; color:#9a3412;}
.hl-reward   {background:#f3e8ff; color:#6b21a8;}
.hl-pressure {background:#fce7f3; color:#9d174d;}
.hl-identity {background:#ccfbf1; color:#0f766e;}
.hl-link     {background:#e0e7ff; color:#3730a3; text-decoration: underline;}

.glass-card, .metric-card,
.stButton > button, [data-testid="stDownloadButton"] button, [data-testid="stFormSubmitButton"] button {
    transition: transform .2s ease, box-shadow .2s ease;
}

.glass-card:hover {
    box-shadow: 0 24px 60px rgba(15, 23, 42, 0.12);
}

.risk-low, .risk-mid, .risk-high, .risk-critical {
    animation: risk-fade-in .35s ease;
}

@keyframes risk-fade-in {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 900px) {
    .hero-grid { grid-template-columns: 1fr; }
    .hero-title { font-size: 38px; }
    .hero { padding: 26px; }
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .metric-value { font-size: 26px; }
    .logo-title { font-size: 32px; }
}

@media (max-width: 640px) {
    .hero-panel-value { font-size: 28px; }
    .risk-low, .risk-mid, .risk-high, .risk-critical { font-size: 20px; padding: 18px 20px; }
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

    st.radio(
        "🌍 Language / Тіл / Язык",
        LANG_OPTIONS,
        index=LANG_OPTIONS.index(st.session_state.lang),
        horizontal=True,
        key="lang_selector",
        on_change=apply_lang_change,
    )

    st.divider()

    if st.button(T["install_button"], use_container_width=True, key="install_toggle_btn"):
        st.session_state["show_install_instructions"] = not st.session_state.get(
            "show_install_instructions", False
        )

    st.divider()

    MODE_KEYS = ["sms", "call", "file", "batch"]
    mode = st.selectbox(
        T["mode"],
        MODE_KEYS,
        format_func=lambda k: T[k],
        key="mode_select",
        help=T["mode_help"]
    )

    threshold = st.slider(
        T["threshold"], 0.1, 0.9, 0.5, 0.05,
        key="threshold_slider",
        help=T["threshold_help"]
    )

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

    if "main_input_text" not in st.session_state:
        st.session_state["main_input_text"] = next(iter(demo_texts.values()))

    def apply_demo_change():
        st.session_state["main_input_text"] = demo_texts[st.session_state["demo_select"]]

    demo = st.selectbox(
        T["demo"],
        list(demo_texts.keys()),
        key="demo_select",
        on_change=apply_demo_change,
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
    with st.expander(f"🛠️ {T['dev_stack']}"):
        for item in ["Ensemble (LR + RF + GB + TF-IDF/NB)", "Feature Engineering", "Domain Analysis",
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
            <div class="hero-panel-small">Analyzes text, links, pressure words, secret-code requests and suspicious domains using an ensemble of 4 ML models.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# INSTALL INSTRUCTIONS (toggled by the sidebar button)
# =========================
if st.session_state.get("show_install_instructions"):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">📲 {T["install_panel_title"]}</div>', unsafe_allow_html=True)
    st.markdown(f"**{T['install_mobile_title']}**")
    st.markdown(T["install_mobile_android"])
    st.markdown(T["install_mobile_ios"])
    st.markdown(f"**{T['install_extension_title']}**")
    st.markdown(T["install_extension_steps"])
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# STATS + HOW TO USE + COLOR LEGEND — collapsed by default so a first-time
# visitor (especially on mobile) reaches the input box without first
# scrolling past four stat tiles and a 3-step guide. The safety disclaimer
# stays visible outside the expander since it matters even to someone who
# never opens this.
# =========================
_avg_acc = round(sum(model_metrics.values()) / len(model_metrics), 1)

with st.expander(f"🧭 {T['how_to_use']}", expanded=False):
    st.markdown(f"""
    <div class="metrics-bar">
        <div class="metrics-row">
            <div class="metrics-item">
                <div class="metrics-item-val">{len(data)}</div>
                <div class="metrics-item-label">{T['stat_examples']}</div>
            </div>
            <div class="metrics-item">
                <div class="metrics-item-val">24+</div>
                <div class="metrics-item-label">{T['stat_categories']}</div>
            </div>
            <div class="metrics-item">
                <div class="metrics-item-val">{_avg_acc}%</div>
                <div class="metrics-item-label">{T['stat_accuracy']}</div>
            </div>
            <div class="metrics-item">
                <div class="metrics-item-val">3</div>
                <div class="metrics-item-label">{T['stat_languages']}</div>
            </div>
        </div>
    </div>
    <div class="howto-steps">
        <div class="howto-step">{T['step1']}</div>
        <div class="howto-step">{T['step2']}</div>
        <div class="howto-step">{T['step3']}</div>
    </div>
    <div class="legend-title">{T['legend_title']}</div>
    <div class="legend-row">
        <span class="legend-chip legend-low">🟢 {T['low']}</span>
        <span class="legend-chip legend-mid">🟡 {T['mid']}</span>
        <span class="legend-chip legend-high">🟠 {T['high']}</span>
        <span class="legend-chip legend-critical">🔴 {T['critical']}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f'<div class="disclaimer-box">{T["disclaimer"]}</div>', unsafe_allow_html=True)

# =========================
# INPUT
# =========================
left, right = st.columns([2.1, 0.9], gap="large")

with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    if mode == "batch":
        st.markdown(f'<div class="section-title">📊 {T["batch"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-subtitle">{T["batch_subtitle"]}</div>', unsafe_allow_html=True)

        with st.form("batch_form", clear_on_submit=False):
            batch_file = st.file_uploader(T["batch_upload"], type=["csv"], key="batch_file_uploader")
            batch_column = st.text_input(T["batch_column"], value="text", key="batch_column_input")
            batch_go = st.form_submit_button(T["batch_run"], use_container_width=True)

        analyze = False
        input_text = ""
    else:
        st.markdown(f'<div class="section-title">✍️ {T["input_title"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-subtitle">{T["input_subtitle"]}</div>', unsafe_allow_html=True)

        with st.form("analysis_form", clear_on_submit=False):
            uploaded = None
            if mode == "file":
                uploaded = st.file_uploader(T["upload"], type=["txt"], key="txt_file_uploader")

            if uploaded:
                input_text = uploaded.read().decode("utf-8", errors="ignore")
                st.success(T["file_uploaded"])
            else:
                input_text = st.text_area(
                    T["input_label"],
                    key="main_input_text",
                    height=210
                )

            # FIX: Show live character/word count below textarea
            char_count = len(input_text)
            word_count_val = len(input_text.split()) if input_text.strip() else 0
            st.caption(f"📝 {T['char_count']}: {char_count} | {T['word_count']}: {word_count_val}")

            analyze = st.form_submit_button(T["analyze"], use_container_width=True)

        batch_go = False
        batch_file = None
        batch_column = "text"
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown(f"""
    <div class="glass-card feature-list">
        <div class="section-title">⚙️ {T['features']}</div>
        <p>🧾 Text analysis</p>
        <p>🌐 Domain analysis</p>
        <p>🧠 Ensemble ML (LR+RF+GB)</p>
        <p>🔍 Explainable result</p>
        <p>🖍️ Highlighted trigger words</p>
        <p>📈 Risk gauge & model charts</p>
        <p>📊 Batch CSV analysis</p>
        <p>📥 Downloadable report & history</p>
        <p>🚨 Real scam scenarios</p>
        <p>⚡ Rule-based risk boost</p>
    </div>
    """, unsafe_allow_html=True)

# =========================
# HISTORY & FEEDBACK INIT
# =========================
if "history" not in st.session_state:
    st.session_state.history = load_history()
if "feedback" not in st.session_state:
    st.session_state.feedback = load_feedback()

# =========================
# BATCH ANALYSIS
# =========================
if mode == "batch" and batch_go:
    if not batch_file:
        st.warning(T["batch_no_file"])
    else:
        try:
            batch_df = pd.read_csv(batch_file)
        except Exception:
            batch_file.seek(0)
            batch_df = pd.read_csv(batch_file, sep=";")

        col = batch_column.strip()
        if col not in batch_df.columns:
            st.warning(T["batch_no_column"])
        else:
            texts = [t for t in batch_df[col].astype(str).fillna("") if t.strip()]
            if not texts:
                st.warning(T["batch_no_column"])
            else:
                # Vectorized: extract all feature rows once, then a single
                # batch predict_proba call per model instead of one call per row.
                feats_list = [extract_features(t)[0] for t in texts]
                X_batch = pd.DataFrame(feats_list)

                lr_p = lr_model.predict_proba(X_batch)[:, 1]
                rf_p = rf_model.predict_proba(X_batch)[:, 1]
                gb_p = gb_model.predict_proba(X_batch)[:, 1]
                tfidf_p = tfidf_model.predict_proba(texts)[:, 1]
                raw_p = (lr_p + rf_p + gb_p + tfidf_p) / 4.0
                boosts = np.array([rule_boost(f) for f in feats_list])
                probs = np.minimum(0.99, raw_p + boosts)

                results = []
                for t, p in zip(texts, probs):
                    risk_lbl, _, em = risk_style(p)
                    verdict = "FRAUD" if p >= threshold else "SAFE"
                    results.append({
                        "Text": t[:100] + ("..." if len(t) > 100 else ""),
                        "Risk %": round(p * 100, 1),
                        "Level": risk_lbl,
                        "Verdict": f"{em} {verdict}",
                    })

                st.markdown(f'<div class="section-title">📊 {T["batch_results"]}</div>', unsafe_allow_html=True)
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True, hide_index=True)

                fraud_n = sum(1 for r in results if "FRAUD" in r["Verdict"])
                safe_n = len(results) - fraud_n

                bc1, bc2 = st.columns([1, 1])
                with bc1:
                    st.markdown(f"**{T['batch_summary']}**")
                    sm1, sm2 = st.columns(2)
                    sm1.metric(T["fraud_count"], fraud_n)
                    sm2.metric(T["safe_count"], safe_n)
                    st.download_button(
                        T["download_csv_results"],
                        results_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"batch_fraud_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with bc2:
                    summary_df = pd.DataFrame(
                        {"Count": [fraud_n, safe_n]},
                        index=[T["fraud_count"], T["safe_count"]],
                    )
                    st.bar_chart(summary_df, height=220)

# =========================
# ANALYSIS
# =========================
if mode != "batch" and analyze:
    if not input_text.strip():
        st.warning(T["no_text"])
    else:
        features, domains = extract_features(input_text)
        X_input = pd.DataFrame([features])

        # Ensemble prediction — average probabilities from all 4 models
        lr_prob = float(lr_model.predict_proba(X_input)[0][1])
        rf_prob = float(rf_model.predict_proba(X_input)[0][1])
        gb_prob = float(gb_model.predict_proba(X_input)[0][1])
        tfidf_prob = float(tfidf_model.predict_proba([input_text])[0][1])
        raw_prob = (lr_prob + rf_prob + gb_prob + tfidf_prob) / 4.0

        boost = rule_boost(features)
        prob = min(0.99, raw_prob + boost)
        pred = int(prob >= threshold)
        risk_label, risk_class, emoji = risk_style(prob)
        explanations = explain(features)

        # A fresh analysis invalidates any deep-analysis result from a
        # previous message, so it doesn't keep showing stale next to a new
        # verdict until the user clicks the deep-analysis button again.
        st.session_state.pop("deep_analysis_result", None)

        # Persist the result in session_state so it stays on screen across
        # reruns triggered by feedback/download/clear-history buttons, which
        # aren't this form's own submit button and would otherwise make the
        # whole results section disappear on the next rerun.
        st.session_state["last_result"] = {
            "input_text": input_text,
            "char_count": char_count,
            "word_count_val": word_count_val,
            "features": features,
            "domains": domains,
            "lr_prob": lr_prob,
            "rf_prob": rf_prob,
            "gb_prob": gb_prob,
            "tfidf_prob": tfidf_prob,
            "raw_prob": raw_prob,
            "boost": boost,
            "prob": prob,
            "pred": pred,
            "threshold": threshold,
            "risk_label": risk_label,
            "risk_class": risk_class,
            "emoji": emoji,
            "explanations": explanations,
            "lang": lang,
        }
        st.session_state.pop("feedback_given", None)

        # Save to history with richer info, persisted to disk
        st.session_state.history.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Risk %": round(prob * 100, 1),
            "Level": risk_label,
            "Verdict": "🚨 FRAUD" if pred == 1 else "✅ SAFE",
            "Features": len(explanations),
            "Text": input_text[:70] + ("..." if len(input_text) > 70 else ""),
        })
        save_history(st.session_state.history)

# =========================
# RESULT DISPLAY (reads from session_state so it survives feedback/history
# button clicks instead of disappearing on the next rerun)
# =========================
if mode != "batch" and "last_result" in st.session_state:
    r = st.session_state["last_result"]
    X_input = pd.DataFrame([r["features"]])

    st.markdown(f'<div class="section-title">📊 {T["result"]}</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-icon">%</div><div class="metric-label">{T["risk"]}</div><div class="metric-value">{r["prob"]*100:.1f}%</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-icon">⚠️</div><div class="metric-label">{T["detected"]}</div><div class="metric-value">{len(r["explanations"])}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-icon">🎚️</div><div class="metric-label">{T["threshold"]}</div><div class="metric-value">{r["threshold"]:.2f}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-icon">🧠</div><div class="metric-label">{T["model"]}</div><div class="metric-value">4-way</div></div>', unsafe_allow_html=True)
    st.caption(f"ℹ️ {T['model_help']}")

    st.markdown(f'<div class="{r["risk_class"]}">{r["emoji"]} {r["risk_label"]}</div>', unsafe_allow_html=True)
    st.progress(float(r["prob"]))

    # Plain-language verdict shown immediately, not buried in a tab
    st.markdown(f'<div class="section-title" style="font-size:19px;">💡 {T["simple_result"]}</div>', unsafe_allow_html=True)
    if r["pred"] == 1:
        st.error(T["bad_advice"])
    else:
        st.success(T["good_advice"])

    already_gave_feedback = st.session_state.get("feedback_given")
    fb_label, fb1, fb2 = st.columns([2, 1, 1])
    fb_label.markdown(
        f"**{T['feedback_thanks'] if already_gave_feedback else T['feedback_prompt']}**"
    )
    if fb1.button(T["feedback_yes"], use_container_width=True, disabled=bool(already_gave_feedback), key="fb_yes"):
        st.session_state.feedback.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Text": r["input_text"][:200],
            "Predicted": "FRAUD" if r["pred"] == 1 else "SAFE",
            "Risk %": round(r["prob"] * 100, 1),
            "UserSaysCorrect": True,
        })
        save_feedback(st.session_state.feedback)
        st.session_state["feedback_given"] = True
        st.rerun()
    if fb2.button(T["feedback_no"], use_container_width=True, disabled=bool(already_gave_feedback), key="fb_no"):
        st.session_state.feedback.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Text": r["input_text"][:200],
            "Predicted": "FRAUD" if r["pred"] == 1 else "SAFE",
            "Risk %": round(r["prob"] * 100, 1),
            "UserSaysCorrect": False,
        })
        save_feedback(st.session_state.feedback)
        st.session_state["feedback_given"] = True
        st.rerun()

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown(render_risk_meter(r["prob"], r["threshold"], T["gauge_title"]), unsafe_allow_html=True)
    with chart_col2:
        st.markdown(f"**{T['model_compare']}**")
        model_df = pd.DataFrame(
            {"Probability (%)": [r["lr_prob"] * 100, r["rf_prob"] * 100, r["gb_prob"] * 100, r["tfidf_prob"] * 100]},
            index=["Logistic Regression", "Random Forest", "Gradient Boosting", "Text Patterns (TF-IDF)"],
        )
        st.bar_chart(model_df, height=220)

        probs_spread = max(r["lr_prob"], r["rf_prob"], r["gb_prob"], r["tfidf_prob"]) - \
            min(r["lr_prob"], r["rf_prob"], r["gb_prob"], r["tfidf_prob"])
        if probs_spread < 0.15:
            agreement_msg = T["agreement_high"]
        elif probs_spread < 0.35:
            agreement_msg = T["agreement_mid"]
        else:
            agreement_msg = T["agreement_low"]
        st.caption(f"**{T['agreement_title']}:** {agreement_msg}")

    tab1, tab2, tab3, tab4 = st.tabs([
        T["explain_tab"],
        f"🧠 {T['vector']}",
        f"📥 {T['report']}",
        f"📜 {T['history']}"
    ])

    with tab1:
        st.subheader(T["why"])
        if r["explanations"]:
            chips = "".join([f'<span class="feature-chip">{e}</span>' for e in r["explanations"]])
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.success(T["no_features"])

        st.subheader(T["highlighted"])
        st.markdown(
            f'<div class="highlight-box">{highlight_text(r["input_text"])}</div>',
            unsafe_allow_html=True
        )

        st.subheader(T["domain"])
        if r["domains"]:
            for d in r["domains"]:
                flags = domain_flags(d)
                flag_str = " | ".join(
                    ("🔴 " if sev == "critical" else "⚠️ ") + label for label, sev in flags
                ) if flags else "✅ No issues found"
                st.markdown(
                    f'<div class="domain-box">🌐 {d}<br><small style="color:#94a3b8">{flag_str}</small></div>',
                    unsafe_allow_html=True
                )
        else:
            st.info(T["no_domain"])

        st.subheader(T["feature_contrib"])
        coef = lr_model.named_steps["clf"].coef_[0]
        feature_names = list(X_input.columns)
        contrib = []
        for name, value, w in zip(feature_names, X_input.iloc[0], coef):
            contrib.append([name, round(float(value), 3), round(float(w), 3), round(float(value) * round(float(w), 3), 3)])

        contrib_df = pd.DataFrame(contrib, columns=["Feature", "Value", "Weight", "Contribution"])
        st.dataframe(
            contrib_df.sort_values("Contribution", ascending=False),
            use_container_width=True,
            hide_index=True
        )

        st.subheader(T["feature_importance"])
        rf_importances = rf_model.named_steps["clf"].feature_importances_
        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": [round(float(i), 4) for i in rf_importances]
        }).sort_values("Importance", ascending=False)
        st.dataframe(imp_df, use_container_width=True, hide_index=True)

        # Words the TF-IDF/Naive Bayes model learned directly from the
        # training text — not from the hand-curated keyword lists above.
        st.subheader(T["learned_words"])
        vectorizer = tfidf_model.named_steps["tfidf"]
        nb = tfidf_model.named_steps["clf"]
        vocab = np.array(vectorizer.get_feature_names_out())
        log_ratio = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
        top_n = min(10, len(vocab))
        top_fraud_idx = np.argsort(log_ratio)[-top_n:][::-1]
        top_safe_idx = np.argsort(log_ratio)[:top_n]

        lw1, lw2 = st.columns(2)
        with lw1:
            st.markdown(f"**{T['learned_fraud_words']}**")
            chips = "".join(
                f'<span class="feature-chip" style="background:#fee2e2;color:#991b1b;border-color:#fecaca;">{w}</span>'
                for w in vocab[top_fraud_idx]
            )
            st.markdown(chips, unsafe_allow_html=True)
        with lw2:
            st.markdown(f"**{T['learned_safe_words']}**")
            chips = "".join(
                f'<span class="feature-chip" style="background:#dcfce7;color:#166534;border-color:#bbf7d0;">{w}</span>'
                for w in vocab[top_safe_idx]
            )
            st.markdown(chips, unsafe_allow_html=True)

    with tab2:
        st.subheader(T["vector"])
        display_df = X_input.T.reset_index()
        display_df.columns = ["Feature", "Value"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.subheader(T["text_stats"])
        ts1, ts2, ts3 = st.columns(3)
        ts1.metric(T["char_count"], r["char_count"])
        ts2.metric(T["word_count"], r["word_count_val"])
        ts3.metric("URLs", len(r["domains"]))

    with tab3:
        report = f"""
AI Fraud Detector Report
========================
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Language: {r["lang"]}

INPUT TEXT:
{r["input_text"]}

ANALYSIS RESULTS:
-----------------
Logistic Regression probability: {r["lr_prob"]*100:.1f}%
Random Forest probability:       {r["rf_prob"]*100:.1f}%
Gradient Boosting probability:   {r["gb_prob"]*100:.1f}%
Text Patterns (TF-IDF) probability: {r["tfidf_prob"]*100:.1f}%
Ensemble average (raw):          {r["raw_prob"]*100:.1f}%
Model agreement spread:          {(max(r["lr_prob"], r["rf_prob"], r["gb_prob"], r["tfidf_prob"]) - min(r["lr_prob"], r["rf_prob"], r["gb_prob"], r["tfidf_prob"]))*100:.1f} pts
Rule boost applied:              +{r["boost"]*100:.1f}%
Final fraud risk:                {r["prob"]*100:.1f}%
Risk level:                      {r["risk_label"]}
Decision threshold:              {r["threshold"]}
Prediction:                      {"FRAUD" if r["pred"] == 1 else "SAFE"}

DETECTED FEATURES ({len(r["explanations"])}):
{chr(10).join("- " + e for e in r["explanations"]) if r["explanations"] else "No strong fraud indicators"}

DETECTED DOMAINS ({len(r["domains"])}):
{chr(10).join("- " + d for d in r["domains"]) if r["domains"] else "No domains"}

TEXT STATS:
Characters: {r["char_count"]}
Words: {r["word_count_val"]}

SECURITY ADVICE:
{T["bad_advice"] if r["pred"] == 1 else T["good_advice"]}
"""
        result_json = json.dumps({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "language": r["lang"],
            "input_text": r["input_text"],
            "probabilities": {
                "logistic_regression": round(r["lr_prob"], 4),
                "random_forest": round(r["rf_prob"], 4),
                "gradient_boosting": round(r["gb_prob"], 4),
                "text_patterns_tfidf": round(r["tfidf_prob"], 4),
                "ensemble_raw": round(r["raw_prob"], 4),
                "rule_boost": round(r["boost"], 4),
                "final": round(r["prob"], 4),
            },
            "model_agreement_spread": round(
                max(r["lr_prob"], r["rf_prob"], r["gb_prob"], r["tfidf_prob"])
                - min(r["lr_prob"], r["rf_prob"], r["gb_prob"], r["tfidf_prob"]),
                4,
            ),
            "threshold": r["threshold"],
            "prediction": "FRAUD" if r["pred"] == 1 else "SAFE",
            "risk_level": r["risk_label"],
            "detected_features": r["explanations"],
            "detected_domains": r["domains"],
        }, ensure_ascii=False, indent=2)

        dl1, dl2 = st.columns(2)
        dl1.download_button(
            T["download"],
            report,
            file_name=f"fraud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            use_container_width=True
        )
        dl2.download_button(
            T["download_json"],
            result_json,
            file_name=f"fraud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

    with tab4:
        if st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True
            )
            hc1, hc2 = st.columns(2)
            hc1.download_button(
                T["download_history"],
                history_df.to_csv(index=False).encode("utf-8"),
                file_name=f"fraud_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            if hc2.button(T["clear_history"], use_container_width=True):
                st.session_state.history = []
                save_history([])
                st.rerun()
        else:
            st.info(T["no_history"])

    # =========================
    # DEEP ANALYSIS (optional, opt-in per click — sends text to Claude)
    # =========================
    # Only rendered at all when a key is actually configured, so the
    # section is fully invisible rather than a dead-end "not set up"
    # button when nobody's configured ANTHROPIC_API_KEY. Reappears on its
    # own, with no code changes, the moment a key is added.
    if get_anthropic_api_key():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">{T["deep_analysis_title"]}</div>', unsafe_allow_html=True)
        st.markdown(T["deep_analysis_intro"])
        st.caption(T["deep_analysis_privacy_note"])

        if st.button(T["deep_analysis_button"], key="deep_analysis_btn"):
            lang_name = {"🇰🇿 KZ": "Kazakh", "🇷🇺 RU": "Russian", "🇬🇧 EN": "English"}.get(lang, "English")
            with st.spinner(T["deep_analysis_loading"]):
                try:
                    st.session_state["deep_analysis_result"] = run_deep_analysis(r["input_text"], lang_name)
                except Exception:
                    st.session_state["deep_analysis_result"] = None
                    st.error(T["deep_analysis_error"])

        if st.session_state.get("deep_analysis_result"):
            st.markdown(f"**{T['deep_analysis_result_title']}**")
            st.info(st.session_state["deep_analysis_result"])
        st.markdown('</div>', unsafe_allow_html=True)

# =========================
# HOW IT WORKS
# =========================
with st.expander(f"📘 {T['how']}"):
    if lang == "🇰🇿 KZ":
        st.write("1. Мәтіннен 20 белгі алынады. 2. Олар сандық векторға айналады. 3. 4 ML модель (соның ішінде мәтіннен сөз үлгілерін үйренетін TF-IDF моделі) ықтималдық есептейді. 4. Ереже күшейткіш қолданылады. 5. Сайт нәтиже мен кеңес береді.")
    elif lang == "🇷🇺 RU":
        st.write("1. Из текста извлекаются 20 признаков. 2. Они превращаются в числовой вектор. 3. 4 модели ML (включая TF-IDF модель, которая учится на словах самого текста) считают вероятность мошенничества. 4. Применяется ансамблирование и правило-буст. 5. Сайт показывает результат и совет по безопасности.")
    else:
        st.write("1. 20 features are extracted from the text. 2. They are converted into a numeric vector. 3. Four ML models (LR, RF, GB, plus a TF-IDF/Naive Bayes model that learns word patterns directly from the text) independently predict fraud probability. 4. Probabilities are averaged and a rule-based boost is applied. 5. The app shows the result, explanation, and safety advice.")

# =========================
# METHODOLOGY / MODEL CARD
# =========================
with st.expander(f"📋 {T['methodology_title']}"):
    fraud_n = sum(1 for _, lbl in data if lbl == 1)
    safe_n = sum(1 for _, lbl in data if lbl == 0)
    avg_acc = round(sum(model_metrics.values()) / len(model_metrics), 1)

    st.markdown(f"#### {T['methodology_dataset_title']}")
    st.markdown(T["methodology_dataset_body"].format(total=len(data), fraud=fraud_n, safe=safe_n))

    st.markdown(f"#### {T['methodology_models_title']}")
    st.markdown(T["methodology_models_body"].format(avg_acc=avg_acc))

    st.markdown(f"#### {T['methodology_limitations_title']}")
    st.markdown(T["methodology_limitations_body"])

    st.markdown(f"#### {T['methodology_ethics_title']}")
    st.markdown(T["methodology_ethics_body"])

st.markdown(f'<div class="footer">{T["footer"]} · Ensemble ML · {datetime.now().year}</div>', unsafe_allow_html=True)
