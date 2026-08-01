"""
Pure, Streamlit-independent fraud-detection logic.

This module deliberately has zero dependency on Streamlit (or on anything
that needs a running app/session) so it can be imported directly in unit
tests. app.py imports everything it needs from here; the UI, translations,
and model training stay in app.py.
"""
import re
import html

# =========================
# WORD / DOMAIN LISTS
# =========================
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
    "login", "verify", "secure", "bonus", "gift",
    "account", "support", "confirm", "prize", "payment", "wallet",
    "security", "update", "auth", "free"
]
suspicious_zones = [".xyz", ".top", ".click", ".site", ".online", ".live", ".info", ".icu"]

# Known KZ/RU financial & delivery brands commonly impersonated in phishing
# domains (e.g. "kaspi-login.xyz"). Kept separate from suspicious_domain_words
# above so a *legitimate* bank.kz / kaspi.kz domain isn't itself flagged as
# suspicious just for containing the brand name.
KNOWN_BRANDS = [
    "kaspi.kz", "halykbank.kz", "sberbank.kz", "sberbank.ru",
    "tinkoff.ru", "vtb.ru", "egov.kz", "kazpost.kz", "dhl.com",
]

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
# URL / DOMAIN HELPERS
# =========================
def extract_urls(text):
    """Returns every http(s):// or www. link found in the text, in the order they appear."""
    return re.findall(r"https?://[^\s]+|www\.[^\s]+", text.lower())

def get_domain(url):
    """Strips the protocol/www prefix and any path, leaving just the bare domain (e.g. "kaspi-login.xyz")."""
    url = url.replace("https://", "").replace("http://", "").replace("www.", "")
    return url.split("/")[0]

def count_matches(text, words):
    """Counts how many entries from `words` appear anywhere in `text` (case-insensitive substring match)."""
    text = text.lower()
    return sum(1 for w in words if w in text)

def brand_impersonation(domain):
    """Returns the brand being impersonated if the domain contains a known
    brand's name but isn't that brand's real domain (classic phishing pattern
    like "kaspi-login.xyz"), otherwise None."""
    for brand in KNOWN_BRANDS:
        brand_core = brand.split(".")[0]
        if brand_core in domain and domain != brand and not domain.endswith("." + brand):
            return brand
    return None

def domain_flags(d):
    """Shared list of (label, severity) flags for a single domain, used both
    for feature extraction and for the Domain analysis display."""
    flags = []
    impersonated = brand_impersonation(d)
    if impersonated:
        flags.append((f"Impersonates {impersonated}", "critical"))
    if any(w in d for w in suspicious_domain_words):
        flags.append(("Suspicious keyword", "warn"))
    if len(d) > 20:
        flags.append(("Long domain", "warn"))
    if any(d.endswith(z) for z in suspicious_zones):
        flags.append(("Suspicious TLD", "critical"))
    if any(ch.isdigit() for ch in d):
        flags.append(("Contains digits", "warn"))
    return flags

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(text):
    """Turns raw message text into the 20-value numeric feature vector the
    LR/RF/GB models are trained on: counts of urgency/secrecy/money/threat/
    identity/reward/pressure words, link and domain-reputation flags
    (suspicious keywords, TLD, length, digits, brand impersonation), plus
    basic text statistics (length, word count, punctuation/case usage).
    Returns (features_dict, list_of_domains_found)."""
    text_lower = text.lower()
    urls = extract_urls(text_lower)
    domains = [get_domain(u) for u in urls]

    suspicious_domain = 0
    long_domain = 0
    suspicious_zone = 0
    digit_domain = 0
    brand_flag = 0

    for d in domains:
        if any(w in d for w in suspicious_domain_words):
            suspicious_domain = 1
        if len(d) > 20:
            long_domain = 1
        if any(d.endswith(z) for z in suspicious_zones):
            suspicious_zone = 1
        if any(ch.isdigit() for ch in d):
            digit_domain = 1
        if brand_impersonation(d):
            brand_flag = 1

    words = text_lower.split()
    avg_word_len = sum(len(w) for w in words) / len(words) if words else 0

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
        "brand_flag": brand_flag,
        "digit_count": sum(ch.isdigit() for ch in text_lower),
        "exclamation_count": text_lower.count("!"),
        "uppercase_count": sum(1 for ch in text if ch.isupper()),
        "text_length": len(text),
        "word_count": len(words),
        "avg_word_length": round(avg_word_len, 2),
        "url_count": len(urls),
        "has_multiple_warnings": int(
            count_matches(text_lower, urgent_words) > 0
            and count_matches(text_lower, threat_words) > 0
        ),
    }, domains

# =========================
# RULE-BASED SCORE BOOST
# =========================
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
    if features["brand_flag"]:
        boost += 0.15
    if features["has_multiple_warnings"]:
        boost += 0.08
    if features["url_count"] > 1:
        boost += 0.05
    return min(boost, 0.30)

# =========================
# RISK LEVEL
# =========================
def risk_level(prob):
    """Returns (level_key, css_class, emoji). level_key is looked up in the
    UI's translation dict by the caller — this module has no notion of
    language."""
    if prob < 0.3:
        return "low", "risk-low", "🟢"
    if prob < 0.6:
        return "mid", "risk-mid", "🟡"
    if prob < 0.8:
        return "high", "risk-high", "🟠"
    return "critical", "risk-critical", "🔴"

# =========================
# TEXT HIGHLIGHTING
# =========================
def _compile_word_pattern(words):
    """One alternation regex per category, longest phrases first so they win over substrings."""
    escaped = sorted((re.escape(w) for w in words if w), key=len, reverse=True)
    return re.compile("|".join(escaped), flags=re.IGNORECASE) if escaped else None

HIGHLIGHT_PATTERNS = [
    (_compile_word_pattern(pressure_phrases), "hl-pressure"),
    (_compile_word_pattern(threat_words), "hl-threat"),
    (_compile_word_pattern(secret_words), "hl-secret"),
    (_compile_word_pattern(reward_words), "hl-reward"),
    (_compile_word_pattern(identity_words), "hl-identity"),
    (_compile_word_pattern(urgent_words), "hl-urgent"),
    (_compile_word_pattern(money_words), "hl-money"),
]
LINK_PATTERN = re.compile(r"https?://[^\s]+|www\.[^\s]+", flags=re.IGNORECASE)

def highlight_text(text):
    """Wrap detected trigger words/links in colored <span> tags for display."""
    matches = []
    for pattern, css_class in HIGHLIGHT_PATTERNS:
        if pattern is None:
            continue
        for m in pattern.finditer(text):
            matches.append((m.start(), m.end(), css_class))
    for m in LINK_PATTERN.finditer(text):
        matches.append((m.start(), m.end(), "hl-link"))

    if not matches:
        return html.escape(text).replace("\n", "<br>")

    matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    filtered = []
    last_end = -1
    for start, end, css_class in matches:
        if start >= last_end:
            filtered.append((start, end, css_class))
            last_end = end

    pieces = []
    cursor = 0
    for start, end, css_class in filtered:
        if start > cursor:
            pieces.append(html.escape(text[cursor:start]))
        pieces.append(f'<span class="{css_class}">{html.escape(text[start:end])}</span>')
        cursor = end
    if cursor < len(text):
        pieces.append(html.escape(text[cursor:]))

    return "".join(pieces).replace("\n", "<br>")
