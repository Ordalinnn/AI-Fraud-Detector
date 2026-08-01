import json
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
    urgent_words, secret_words, money_words, threat_words,
    suspicious_domain_words, suspicious_zones, KNOWN_BRANDS,
    identity_words, reward_words, pressure_phrases,
    extract_urls, get_domain, count_matches,
    brand_impersonation, domain_flags, extract_features,
    rule_boost, risk_level, highlight_text,
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
]

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
            "brand_flag": "Домен белгілі банк/брендке ұқсатылған, бірақ ол емес",
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
            "brand_flag": "Домен маскируется под известный банк/бренд, но им не является",
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
            "brand_flag": "Domain mimics a known bank/brand but isn't the real one",
        }

    # Only explain features that are relevant (non-zero and meaningful)
    irrelevant = {"text_length", "word_count", "avg_word_length"}
    return [labels[k] for k, v in features.items() if v > 0 and k in labels and k not in irrelevant]

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
# IMPACT / STATS BANNER
# =========================
_avg_acc = round(sum(model_metrics.values()) / len(model_metrics), 1)

st.markdown(f"""
<div class="metrics-bar">
    <div class="metrics-row">
        <div class="metrics-item">
            <div class="metrics-item-val">{len(data)}</div>
            <div class="metrics-item-label">{T['stat_examples']}</div>
        </div>
        <div class="metrics-item">
            <div class="metrics-item-val">17+</div>
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
# HOW TO USE (plain-language guide, for any age/skill level)
# =========================
st.markdown(f"""
<div class="glass-card howto-card">
    <div class="section-title">🧭 {T['how_to_use']}</div>
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
    <div class="disclaimer-box">{T['disclaimer']}</div>
</div>
""", unsafe_allow_html=True)

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
