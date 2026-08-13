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
    "карта", "счет", "банк", "ақша", "төле", "оплат", "перевод",
    "баланс", "средства", "деньги", "transfer", "payment", "pay",
    "wallet", "iban"
]
# "оплат" (not "оплата") is deliberate: as a substring-match stem it also
# catches "оплатить"/"оплатите"/"оплачивает"/etc. — Russian verb conjugation
# isn't handled by exact-word matching, so a shorter stem covers the family
# of forms for free. Same reasoning for "pay" covering "pay"/"paying"/"repay".

# Personal "lend/borrow me money" phrasing — distinct from money_words above,
# which is generic banking vocabulary that a hacked-account loan request
# often *doesn't* use at all (no "card"/"account"/"bank", just "can you lend
# me some money, I'll pay you back"). This is the dominant pattern behind
# the "hacked WhatsApp/Telegram, message a relative/friend asking to
# urgently borrow money" scam widely reported in Kazakhstan and the wider
# CIS region: the message comes from what looks like a real contact's own
# account rather than a stranger, so the text itself (not the sender) is
# often the only signal available to catch it. "займ"/"занять" deliberately
# NOT shortened to "заня" — that shorter stem would also match unrelated
# words like "занятость" (employment) and "занятия" (classes/lessons).
loan_words = [
    "займ", "занять", "одолж", "в долг", "взаймы", "до зарплаты",
    "выручи", "перекинь", "закинь", "borrow", "lend me", "spot me",
]

# Message-text language for fake "verification service" scams — e.g. a fake
# marketplace buyer asking a seller to pay a bogus registry/certificate site
# before a deal (see the check-tech-base.ru style scam). Distinct from
# suspicious_domain_words below, which matches against a DOMAIN, not the
# message body.
verification_service_words = [
    "провер", "справк", "сертификат", "реестр", "registry",
    "verif", "certificate",
]
# Short stems again: "провер" covers проверка/проверить/проверьте/проверку,
# and "verif" covers verify/verifying/verification — real phrasing varies
# too much ("verify the item", "verification website", "please verify")
# for exact multi-word phrase matching to hold up.
threat_words = [
    "заблокирована", "удален", "штраф", "угрозой", "бұғатталды", "жабылады",
    "blocked", "suspended", "terminated", "penalty", "freeze",
    "ограничен", "будет закрыт"
]
suspicious_domain_words = [
    "login", "verify", "secure", "bonus", "gift",
    "account", "support", "confirm", "prize", "payment", "wallet",
    "security", "update", "auth", "free", "check", "registry",
]
suspicious_zones = [".xyz", ".top", ".click", ".site", ".online", ".live", ".info", ".icu"]

# Known KZ/RU financial & delivery brands, plus globally impersonated
# platforms (the training data covers PayPal/Amazon/Apple/Google/Temu/Shein
# phishing text, but until now brand_impersonation() only ever checked
# against local bank names, so a phishing domain for these global brands
# went undetected by domain analysis even though the wording was flagged).
# Kept separate from suspicious_domain_words above so a *legitimate*
# bank.kz / kaspi.kz domain isn't itself flagged as suspicious just for
# containing the brand name.
KNOWN_BRANDS = [
    "kaspi.kz", "halykbank.kz", "sberbank.kz", "sberbank.ru",
    "tinkoff.ru", "vtb.ru", "egov.kz", "kazpost.kz", "dhl.com",
    "paypal.com", "amazon.com", "apple.com", "google.com", "microsoft.com",
    "instagram.com", "whatsapp.com", "facebook.com", "netflix.com",
    "temu.com", "shein.com", "aliexpress.com",
]

# Cyrillic characters that render visually identical (or near-identical) to
# a Latin letter in most fonts — the basis of IDN homograph phishing, e.g.
# "kаspi.kz" with a Cyrillic а (U+0430) looks exactly like kaspi.kz to a
# human but is a completely different domain that substring matching alone
# would miss. Mapped to their Latin look-alike so brand comparisons see
# through the disguise.
CONFUSABLES = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ѕ": "s", "һ": "h", "ј": "j",
}

def _deconfuse(domain):
    """Maps look-alike Cyrillic characters in a domain to their Latin
    equivalent so brand comparisons aren't fooled by a homograph domain."""
    return "".join(CONFUSABLES.get(ch, ch) for ch in domain)

def _levenshtein(a, b):
    """Standard edit distance via iterative DP. Domains are short (well
    under 100 chars) so this is cheap; used only for near-miss typosquat
    detection (e.g. "kasp1-login.xyz" or "paypa1.com")."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur_row = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur_row[j] = min(prev_row[j] + 1, cur_row[j - 1] + 1, prev_row[j - 1] + cost)
        prev_row = cur_row
    return prev_row[-1]

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

# Real scams increasingly write a domain as plain text (e.g. "check-tech-
# base.ru") instead of a clickable http(s):// or www. link, specifically to
# dodge filters that only look for literal links. This regex catches those
# bare mentions so they still register as a domain, not just as invisible text.
BARE_DOMAIN_PATTERN = re.compile(
    r"\b[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9-]+)*\."
    r"(?:ru|kz|com|net|org|info|xyz|top|site|online|live|icu|click|by|ua|io|me|co)\b",
    re.IGNORECASE,
)

def extract_bare_domains(text_lower, urls):
    """Finds domain-like mentions with no http(s):// or www. prefix. Full
    URLs are masked out first so their domain isn't double-counted."""
    masked = text_lower
    for u in urls:
        masked = masked.replace(u, " ")
    return BARE_DOMAIN_PATTERN.findall(masked)

def get_domain(url):
    """Strips the protocol/www prefix and any path, leaving just the bare domain (e.g. "kaspi-login.xyz").

    Anchored (^) so a "www." that isn't actually the leading subdomain — e.g.
    the "www." inside "kaspi-www.secure-login.xyz" — isn't silently deleted,
    which would turn it into a different domain than what was actually
    reported. Mirrors detector.js's getDomain(), which uses the same anchors."""
    url = re.sub(r"^https?://", "", url)
    url = re.sub(r"^www\.", "", url)
    return url.split("/")[0]

def count_matches(text, words):
    """Counts how many entries from `words` appear anywhere in `text` (case-insensitive substring match)."""
    text = text.lower()
    return sum(1 for w in words if w in text)

def brand_impersonation(domain):
    """Returns the brand being impersonated if the domain contains a known
    brand's name but isn't that brand's real domain (classic phishing pattern
    like "kaspi-login.xyz"), otherwise None.

    Also catches two patterns plain substring matching misses: homograph
    domains that swap in look-alike Cyrillic characters (deconfused before
    comparison), and single-character typosquats like "kasp1.kz" or
    "paypa1.com" (caught via edit distance against each domain token)."""
    # Checked up front against the RAW (not deconfused) domain, against ALL
    # brands at once: some brand cores repeat across TLDs (sberbank.kz vs
    # sberbank.ru), so a per-brand-loop skip would still flag one as
    # impersonating the other. A homoglyph domain that only equals a brand
    # after deconfusion isn't the real domain — it must NOT be caught here.
    if any(domain == b or domain.endswith("." + b) for b in KNOWN_BRANDS):
        return None
    normalized = _deconfuse(domain)
    for brand in KNOWN_BRANDS:
        brand_core = brand.split(".")[0]
        if brand_core in normalized:
            return brand
        # Short cores (e.g. "vtb", "dhl") are skipped here — a 1-edit match
        # against a 3-letter word is too likely to be a false positive.
        if len(brand_core) >= 5:
            for token in re.split(r"[^a-z0-9]+", normalized):
                if len(token) >= 4 and _levenshtein(token, brand_core) == 1:
                    return brand
    return None

def domain_flags(d):
    """Shared list of (label_key, severity, brand) flags for a single domain,
    used both for feature extraction and for the Domain analysis display.
    label_key is looked up in the UI's translation dict by the caller (with
    `brand` filled into a "{brand}" placeholder for "impersonates") — this
    module has no notion of language, mirroring detector.js's
    domainFlags()/labelKey pattern so the website and browser extension
    show identical wording for the same domain."""
    flags = []
    impersonated = brand_impersonation(d)
    if impersonated:
        flags.append(("impersonates", "critical", impersonated))
    elif not d.isascii():
        # No known brand matched, but non-Latin characters in a domain are
        # themselves a classic IDN homograph phishing signal even when the
        # impersonated brand isn't in KNOWN_BRANDS.
        flags.append(("non_latin_domain", "critical", None))
    if any(w in d for w in suspicious_domain_words):
        flags.append(("suspicious_keyword", "warn", None))
    if len(d) > 20:
        flags.append(("long_domain", "warn", None))
    if any(d.endswith(z) for z in suspicious_zones):
        flags.append(("suspicious_tld", "critical", None))
    if any(ch.isdigit() for ch in d):
        flags.append(("contains_digits", "warn", None))
    return flags

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(text):
    """Turns raw message text into the numeric feature vector the LR/RF/GB
    models are trained on: counts of urgency/secrecy/money/loan-request/
    threat/identity/reward/pressure/verification-service words, link and
    domain-reputation flags (suspicious keywords, TLD, length, digits,
    brand impersonation), plus basic text statistics (length, word count,
    punctuation/case usage). Returns (features_dict, list_of_domains_found)."""
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)
    text_lower = text.lower()
    urls = extract_urls(text_lower)
    bare_domains = extract_bare_domains(text_lower, urls)
    domains = [get_domain(u) for u in urls] + bare_domains
    link_count = len(urls) + len(bare_domains)

    suspicious_domain = 0
    long_domain = 0
    suspicious_zone = 0
    digit_domain = 0
    brand_flag = 0
    homoglyph_domain = 0

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
        if not d.isascii():
            homoglyph_domain = 1

    words = text_lower.split()
    avg_word_len = sum(len(w) for w in words) / len(words) if words else 0

    return {
        "has_link": int(link_count > 0),
        "urgent_count": count_matches(text_lower, urgent_words),
        "secret_count": count_matches(text_lower, secret_words),
        "money_count": count_matches(text_lower, money_words),
        "loan_count": count_matches(text_lower, loan_words),
        "threat_count": count_matches(text_lower, threat_words),
        "identity_count": count_matches(text_lower, identity_words),
        "reward_count": count_matches(text_lower, reward_words),
        "pressure_count": count_matches(text_lower, pressure_phrases),
        "verification_count": count_matches(text_lower, verification_service_words),
        "suspicious_domain": suspicious_domain,
        "long_domain": long_domain,
        "suspicious_zone": suspicious_zone,
        "digit_domain": digit_domain,
        "brand_flag": brand_flag,
        "homoglyph_domain": homoglyph_domain,
        "digit_count": sum(ch.isdigit() for ch in text_lower),
        "exclamation_count": text_lower.count("!"),
        "uppercase_count": sum(1 for ch in text if ch.isupper()),
        "text_length": len(text),
        "word_count": len(words),
        "avg_word_length": round(avg_word_len, 2),
        "url_count": link_count,
        "has_multiple_warnings": int(
            count_matches(text_lower, urgent_words) > 0
            and count_matches(text_lower, threat_words) > 0
        ),
    }, domains

# =========================
# RULE-BASED SCORE BOOST
# =========================
RULE_BOOST_CAP = 0.30

def rule_boost(features):
    """
    Rule-based boost for realistic scam patterns.
    Capped at RULE_BOOST_CAP to prevent runaway scores on safe messages.
    """
    boost = 0.0
    if features["has_link"] and features["secret_count"]:
        boost += 0.15
    if features["urgent_count"] and features["money_count"]:
        boost += 0.12
    if features["loan_count"] and features["urgent_count"]:
        boost += 0.12
    if features["secret_count"] and features["money_count"]:
        boost += 0.15
    if features["threat_count"] and features["has_link"]:
        boost += 0.10
    if features["reward_count"] and features["money_count"]:
        boost += 0.12
    if features["verification_count"] and (features["money_count"] or features["has_link"]):
        boost += 0.15
    if features["pressure_count"]:
        boost += 0.10
    if features["suspicious_zone"] or features["suspicious_domain"]:
        boost += 0.10
    if features["brand_flag"]:
        boost += 0.15
    if features["homoglyph_domain"]:
        boost += 0.15
    if features["has_multiple_warnings"]:
        boost += 0.08
    if features["url_count"] > 1:
        boost += 0.05
    return min(boost, RULE_BOOST_CAP)

# =========================
# RISK LEVEL
# =========================
def risk_level(prob, threshold=0.5):
    """Returns (level_key, css_class, emoji). level_key is looked up in the
    UI's translation dict by the caller — this module has no notion of
    language.

    The "mid"/"high" boundary is pinned to `threshold` (not a fixed 0.6) so
    the colored risk badge can never contradict the binary FRAUD/SAFE
    verdict, which is itself `prob >= threshold`: everything below
    threshold reads as green/yellow (low/mid), everything at or above it
    reads as orange/red (high/critical). At the default threshold of 0.5
    this reduces to low<0.3, mid<0.5, high<0.8, critical>=0.8."""
    low_cut = threshold * 0.6
    high_cut = threshold + (1 - threshold) * 0.6
    if prob < low_cut:
        return "low", "risk-low", "🟢"
    if prob < threshold:
        return "mid", "risk-mid", "🟡"
    if prob < high_cut:
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

# =========================
# CSV EXPORT SAFETY
# =========================
_FORMULA_LEAD_CHARS = ("=", "+", "-", "@", "\t", "\r")

def sanitize_for_csv(value):
    """Neutralizes CSV/spreadsheet formula injection: if a cell's text starts
    with a character Excel/Sheets treats as a formula prefix (=, +, -, @, or
    a leading tab/carriage-return), prefix it with a straight quote so it's
    opened as literal text instead of executed. Analyzed message text ends
    up in exported CSVs (batch results, history) and is entirely
    user-controlled, so this must run on every free-text cell before
    DataFrame.to_csv()."""
    if not isinstance(value, str):
        return value
    if value.startswith(_FORMULA_LEAD_CHARS):
        return "'" + value
    return value

# =========================
# FEEDBACK -> TRAINING DATA
# =========================
def derive_feedback_training_examples(entries, core_texts, max_per_session=20, max_examples=300):
    """Converts "was this correct?" feedback entries (see app.py's
    append_feedback_entry / FEEDBACK-DERIVED TRAINING DATA section) into
    (text, label) training pairs.

    `entries` is the raw list loaded from feedback.json - dicts with Text/
    Predicted/UserSaysCorrect/Session keys. `core_texts` is a set of
    lowercased, stripped texts already in the hand-labeled core dataset,
    used to skip feedback that wouldn't add anything new.

    Feedback is treated as weak/unverified signal, never as ground truth:
    a real scammer has a direct incentive to spam "this was wrong" on
    true-positive detections of their own template, to teach a model that
    retrains on it to wave that template through. So this function:
      - caps how many entries any single session can contribute
        (max_per_session), so one visitor can't dominate the result
      - caps the total number of examples returned (max_examples), so
        crowd-submitted data can't outweigh the curated core dataset
      - drops any message that received contradictory feedback (some
        sessions say it was correctly flagged, others say it wasn't)
        entirely, rather than guessing which side is right
      - skips anything already present in core_texts

    Entries are processed most-recent-last-in/first-out (i.e. `entries` is
    expected in original chronological order, as stored in feedback.json;
    this function walks it in reverse), so that when max_examples trims the
    result, it's the oldest signal that gets dropped, not the newest.
    """
    if not entries:
        return []

    per_session_count = {}
    text_labels = {}    # normalized text -> set of derived labels seen
    text_original = {}  # normalized text -> original-cased text to train on

    for entry in reversed(entries):
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("Text", "")).strip()
        predicted = entry.get("Predicted")
        correct = entry.get("UserSaysCorrect")
        session = entry.get("Session", "")

        if not (3 <= len(text) <= 200):
            continue
        if predicted not in ("FRAUD", "SAFE") or not isinstance(correct, bool):
            continue

        normalized = text.lower()
        if normalized in core_texts:
            continue

        if session:
            per_session_count[session] = per_session_count.get(session, 0) + 1
            if per_session_count[session] > max_per_session:
                continue

        predicted_label = 1 if predicted == "FRAUD" else 0
        derived_label = predicted_label if correct else 1 - predicted_label
        text_labels.setdefault(normalized, set()).add(derived_label)
        text_original.setdefault(normalized, text)

    examples = [
        [text_original[norm], next(iter(labels))]
        for norm, labels in text_labels.items()
        if len(labels) == 1  # drop contradictory feedback outright
    ]
    return examples[:max_examples]
