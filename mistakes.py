"""
"The model was wrong" reports: one format everywhere.

The site, the browser extension (mistakes.js) and the Android app
(Mistakes.kt) all export reports as CSV in the format of
data/real/team_real_messages_TEMPLATE.csv:

    text,label,language,scam_type,source,date_collected

where `label` is the CORRECTED answer (the user said the model was wrong).
scripts/import_mistakes.py merges such files from testers into
data/real/collected_mistakes.csv for the evaluation (--real).
"""
from __future__ import annotations

import csv
import io
import re
from datetime import date

CSV_COLUMNS = ["text", "label", "language", "scam_type", "source", "date_collected"]
MAX_TEXT = 2000
_KAZAKH_ONLY = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
_CYRILLIC = re.compile(r"[а-яёА-ЯЁ]")
_LATIN = re.compile(r"[a-zA-Z]")


def detect_language(text: str) -> str:
    """Same rule as fraud_eval.detect_language / mistakes.js / Mistakes.kt."""
    if any(ch in _KAZAKH_ONLY for ch in text):
        return "kk"
    cyr = len(_CYRILLIC.findall(text))
    lat = len(_LATIN.findall(text))
    if cyr == 0 and lat == 0:
        return "other"
    return "ru" if cyr >= lat else "en"


def make_row(text: str, predicted_fraud: bool, prob: float, source: str, today: date | None = None) -> dict:
    t = str(text or "").strip()[:MAX_TEXT]
    return {
        "text": t,
        "label": 0 if predicted_fraud else 1,
        "language": detect_language(t),
        "scam_type": "",
        "source": f"{source}: model said {'fraud' if predicted_fraud else 'safe'} {round((prob or 0) * 100)}%",
        "date_collected": (today or date.today()).isoformat(),
    }


def add_row(rows: list, row: dict) -> list:
    """Newest first, one row per text."""
    if not row["text"]:
        return rows
    return [row] + [r for r in rows if r["text"] != row["text"]]


def _cell(v) -> str:
    s = "" if v is None else str(v)
    if s[:1] in ("=", "+", "-", "@", "\t", "\r"):
        s = "'" + s  # spreadsheet formula guard
    return s


def to_csv(rows: list) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_COLUMNS)
    for r in rows:
        w.writerow([_cell(r.get(c, "")) for c in CSV_COLUMNS])
    return buf.getvalue()


def unguard(text: str) -> str:
    """Undo _cell's formula guard when reading a file back."""
    return text[1:] if len(text) > 1 and text[0] == "'" and text[1] in "=+-@\t\r" else text
