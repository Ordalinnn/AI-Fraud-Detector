"""
Undoes the tricks scammers use to slip past keyword and n-gram filters.

    "Ваша к@рта зaблoкирoвана, назовите к0д из S M S"   (Latin a/o, @, 0, spaced letters)
 -> "Ваша карта заблокирована, назовите код из SMS"

Applied to every message before ANY model sees it, both when training and
when checking a message, in the web app (fraud_model / fraud_logic), the
Android app (TextNormalizer.kt) and the browser extension (normalize.js).
All three are line-by-line ports of this file; tests/test_text_normalize.py
writes data/normalize_golden.tsv, and the Kotlin and JS tests must turn
every raw text in it into exactly the same normalized text.

Steps, in order:
  1. Unicode NFKC: fullwidth "ＳＭＳ", math-bold "𝐤𝐨𝐝", ligatures -> plain letters.
  2. Drop invisible characters: zero-width spaces/joiners, soft hyphen,
     BOM, and combining marks (e.g. "к̲о̲д" -> "код").
  3. Join letters spelled out one by one: "к о д ы" (4+ letters with
     spaces) or "к.о.д" / "к-о-д" / "к_о_д" / "к*о*д" (3+ letters with
     another separator) -> "коды" / "код". Four for spaces so that
     ordinary short words ("а я в шоке") are never glued together.
  4. Per word (a run of letters, digits and @): if it mixes Cyrillic and
     Latin letters and all its letters of one alphabet are look-alikes,
     rewrite those into the other alphabet - the alphabet of the whole
     message when both are possible ("зaблoкирoвана" -> "заблокирована",
     "kаspi" -> "kaspi"; "WhatsAppқа" is left alone); a "0" that touches a letter and no digit becomes
     the letter "о"/"o" ("к0д" -> "код", but "20000тг" stays); in a
     Cyrillic word an "@" between letters becomes "а" ("к@рта").

Web addresses are NOT protected from step 4 here; fraud_logic keeps reading
domains from the original text, so a look-alike domain ("kаspi.kz" with a
Cyrillic "а") is still flagged as one.
"""
from __future__ import annotations

import unicodedata

# Invisible characters that are removed outright (besides combining marks).
INVISIBLE = {0x00AD, 0x180E, 0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x2060, 0x2061, 0x2062, 0x2063, 0x2064, 0xFEFF}

# Separators for letters spelled out one by one.
SPACED_SEPARATORS = {" ", ".", "-", "_", "*"}

# Latin letter -> Cyrillic letter that looks the same (lower and upper case).
LAT_TO_CYR = {
    "a": "а", "c": "с", "e": "е", "i": "і", "o": "о", "p": "р", "x": "х", "y": "у", "k": "к",
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "I": "І", "K": "К", "M": "М", "O": "О",
    "P": "Р", "T": "Т", "X": "Х", "Y": "У",
}
CYR_TO_LAT = {v: k for k, v in LAT_TO_CYR.items()}


def _is_letter(ch: str) -> bool:
    return unicodedata.category(ch).startswith("L")


def _is_digit(ch: str) -> bool:
    return unicodedata.category(ch) == "Nd"


def _is_cyrillic(ch: str) -> bool:
    return 0x0400 <= ord(ch) <= 0x052F


def _is_latin(ch: str) -> bool:
    o = ord(ch)
    return (0x41 <= o <= 0x5A) or (0x61 <= o <= 0x7A) or (0xC0 <= o <= 0x24F and o not in (0xD7, 0xF7))


def _strip_invisible(chars: list[str]) -> list[str]:
    return [c for c in chars if ord(c) not in INVISIBLE and unicodedata.category(c) != "Mn"]


def _join_spaced(chars: list[str]) -> list[str]:
    out: list[str] = []
    n = len(chars)
    i = 0
    while i < n:
        c = chars[i]
        if (_is_letter(c) and (i == 0 or not (_is_letter(chars[i - 1]) or _is_digit(chars[i - 1])))
                and i + 2 < n and chars[i + 1] in SPACED_SEPARATORS):
            sep = chars[i + 1]
            letters = [c]
            j = i
            while (j + 2 < n and chars[j + 1] == sep and _is_letter(chars[j + 2])
                   and (j + 3 >= n or not (_is_letter(chars[j + 3]) or _is_digit(chars[j + 3])))):
                letters.append(chars[j + 2])
                j += 2
            if len(letters) >= (4 if sep == " " else 3):
                out.extend(letters)
                i = j + 1
                continue
        out.append(c)
        i += 1
    return out


def _all_lookalikes(word: list[str], is_script, table: dict) -> bool:
    return all(c in table for c in word if is_script(c))


def _fix_word(word: list[str], prefer_cyr: bool) -> list[str]:
    cyr = sum(1 for c in word if _is_cyrillic(c))
    lat = sum(1 for c in word if _is_latin(c))
    if cyr and lat:
        # Only a word whose letters from one alphabet are ALL look-alikes is a
        # disguise ("зaблoкирoвана"); a real mixed word keeps its letters
        # ("WhatsAppқа": "қ" has no Latin twin). Which way to convert: the
        # alphabet of the whole message first, so a heavily disguised word
        # ("сodе" in an English message) still goes the right way.
        lat_ok = _all_lookalikes(word, _is_latin, LAT_TO_CYR)
        cyr_ok = _all_lookalikes(word, _is_cyrillic, CYR_TO_LAT)
        if lat_ok and (prefer_cyr or not cyr_ok):
            word = [LAT_TO_CYR.get(c, c) if _is_latin(c) else c for c in word]
            cyr, lat = cyr + lat, 0
        elif cyr_ok:
            word = [CYR_TO_LAT.get(c, c) if _is_cyrillic(c) else c for c in word]
            cyr, lat = 0, cyr + lat
    to_cyr = cyr >= lat if (cyr or lat) else prefer_cyr
    if cyr == 0 and lat == 0:
        return word
    out = list(word)
    n = len(word)
    for k, c in enumerate(word):
        if c != "0" and c != "@":
            continue
        prev = word[k - 1] if k > 0 else ""
        nxt = word[k + 1] if k + 1 < n else ""
        if c == "0":
            touches_digit = (prev and _is_digit(prev)) or (nxt and _is_digit(nxt))
            touches_letter = (prev and _is_letter(prev)) or (nxt and _is_letter(nxt))
            if touches_letter and not touches_digit:
                out[k] = "о" if to_cyr else "o"
        elif to_cyr and prev and nxt and _is_letter(prev) and _is_letter(nxt):
            out[k] = "а"
    return out


def _fix_words(chars: list[str]) -> list[str]:
    prefer_cyr = sum(1 for c in chars if _is_cyrillic(c)) >= sum(1 for c in chars if _is_latin(c))
    out: list[str] = []
    word: list[str] = []
    for c in chars:
        if _is_letter(c) or _is_digit(c) or c == "@":
            word.append(c)
        else:
            if word:
                out.extend(_fix_word(word, prefer_cyr))
                word = []
            out.append(c)
    if word:
        out.extend(_fix_word(word, prefer_cyr))
    return out


def normalize_text(text) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    chars = list(unicodedata.normalize("NFKC", text))
    chars = _strip_invisible(chars)
    chars = _join_spaced(chars)
    chars = _fix_words(chars)
    return "".join(chars)


def is_disguised(text) -> bool:
    """True if the message uses one of the disguises normalize_text undoes
    (shown to the user as a warning sign of its own: ordinary people don't
    write "к0д" or mix alphabets inside a word). Plain typography that NFKC
    also rewrites ("…", "²", "№") does not count - only letters changed by
    NFKC (fullwidth, math-styled), invisible characters, letters spelled
    out one by one, and look-alike / digit swaps inside words."""
    if not text:
        return False
    text = str(text)
    if any(c.isalpha() and unicodedata.normalize("NFKC", c) != c for c in text):
        return True
    chars = list(unicodedata.normalize("NFKC", text))
    stripped = _strip_invisible(chars)
    if stripped != chars:
        return True
    joined = _join_spaced(stripped)
    if joined != stripped:
        return True
    return _fix_words(joined) != joined
