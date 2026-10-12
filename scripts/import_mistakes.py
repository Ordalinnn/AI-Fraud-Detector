"""
Collects "the model was wrong" reports from testers into one file for the
evaluation.

Testers press "This is a mistake" in the extension, the Android app or on
the site, then export a CSV (same format everywhere: mistakes.py). Put the
files into data/real/mistakes/ (not committed: they may hold personal data
until anonymised) and run

    python scripts/import_mistakes.py                 # data/real/mistakes/*.csv
    python scripts/import_mistakes.py a.csv b.csv     # or any files

It writes data/real/collected_mistakes.csv (team template format):
  - one row per message (same text from several testers = one row);
  - a message reported with BOTH labels is dropped and listed (testers
    disagree - a person should decide);
  - messages already in the training data are dropped (the test must stay
    independent).
Then check the model on them:

    python scripts/run_jury_evaluation.py --real data/real/collected_mistakes.csv

These are exactly the messages the model got wrong at least once, so
accuracy on them is LOW by construction - the point is to see which kinds
of messages fail and whether a new model version fixes them.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mistakes  # noqa: E402

IN_DIR = ROOT / "data" / "real" / "mistakes"
OUT = ROOT / "data" / "real" / "collected_mistakes.csv"


def read_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig")
    rows = list(csv.DictReader(text.splitlines(keepends=True)))
    if rows and set(mistakes.CSV_COLUMNS) - set(rows[0].keys()):
        raise SystemExit(f"{path}: expected columns {mistakes.CSV_COLUMNS}, got {list(rows[0].keys())}")
    out = []
    for r in rows:
        t = mistakes.unguard((r.get("text") or "").strip())
        if not t or str(r.get("label")).strip() not in ("0", "1"):
            continue
        out.append({**{c: (r.get(c) or "").strip() for c in mistakes.CSV_COLUMNS}, "text": t,
                    "label": int(r["label"]), "file": path.name})
    return out


def merge(rows: list[dict], training_texts: set[str]) -> tuple[list[dict], list[str], int]:
    by_text = defaultdict(list)
    for r in rows:
        by_text[r["text"]].append(r)
    merged, conflicts, in_training = [], [], 0
    for t, rs in by_text.items():
        labels = {r["label"] for r in rs}
        if len(labels) > 1:
            conflicts.append(t)
            continue
        if t in training_texts:
            in_training += 1
            continue
        first = rs[0]
        merged.append({**{c: first[c] for c in mistakes.CSV_COLUMNS},
                       "language": first["language"] or mistakes.detect_language(t),
                       "source": f"{first['source']} (reported {len(rs)}x)" if len(rs) > 1 else first["source"]})
    return merged, conflicts, in_training


def main(argv: list[str]) -> None:
    files = [Path(a) for a in argv] or sorted(IN_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"no CSV files: put testers' exports into {IN_DIR}")
    rows = [r for f in files for r in read_rows(f)]
    import fraud_eval as fe
    training = set(fe.load_dataset()["text"])
    merged, conflicts, in_training = merge(rows, training)
    OUT.write_text(mistakes.to_csv(merged), encoding="utf-8")
    print(f"{len(files)} files, {len(rows)} reports -> {len(merged)} messages in {OUT.relative_to(ROOT)}")
    print("by language/label:", dict(Counter((r["language"], r["label"]) for r in merged)))
    if in_training:
        print(f"dropped {in_training} already in the training data")
    for t in conflicts:
        print("testers disagree, dropped:", t[:80])


if __name__ == "__main__":
    main(sys.argv[1:])
