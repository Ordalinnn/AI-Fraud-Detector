"""
Splits the real UCI SMS Spam Collection into a small TRAINING sample and a
large HELD-OUT test half. Run once; both outputs are committed so the app
and the evaluation always use the identical split.

    python scripts/prepare_external_data.py

Why: the jury pointed out that synthetic templates don't reproduce real
messages. Measured on real SMS, models trained only on the project's
written examples falsely flagged 12-18% of ordinary real messages (casual
texts like "Realy sorry-i don't recognise this number... who r u please?!").
Adding a small real sample (300 legit + 100 spam) to training cuts that to
well under 1%, and the other half of the corpus - split by template group so
no near-duplicate crosses over - stays untouched as an independent real test.

Source: UCI SMS Spam Collection (Almeida & Gómez Hidalgo, 2011), 5,574 real
English SMS, CC BY 4.0. Its "spam" class mixes premium-rate/prize scams
with plain advertising; we treat it as "suspicious", which is what the app
warns about anyway.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fraud_model import (  # noqa: E402
    REAL_HOLDOUT_PATH, REAL_TRAIN_PATH, REAL_TRAIN_N_HAM, REAL_TRAIN_N_SPAM, template_groups,
)

SRC = ROOT / "data" / "external" / "uci_sms_spam.tsv"
SEED = 0


def main():
    uci = pd.read_csv(SRC, sep="\t", header=None, names=["label_raw", "text"], quoting=3, dtype=str).dropna()
    uci["text"] = uci["text"].str.strip()
    uci = uci[uci["text"] != ""].drop_duplicates("text").reset_index(drop=True)
    uci["label"] = (uci["label_raw"] == "spam").astype(int)
    groups = template_groups(uci["text"].tolist())
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED).split(uci, uci["label"], groups))
    train_half, holdout = uci.iloc[tr], uci.iloc[te]

    sample = pd.concat([
        train_half[train_half["label"] == 0].sample(REAL_TRAIN_N_HAM, random_state=SEED),
        train_half[train_half["label"] == 1].sample(REAL_TRAIN_N_SPAM, random_state=SEED),
    ])
    for frame, path in [(sample, REAL_TRAIN_PATH), (holdout, REAL_HOLDOUT_PATH)]:
        out = frame[["text", "label"]].copy()
        out["source"] = "uci_sms_spam_collection"
        out.to_csv(path, index=False)
        print(f"{path.relative_to(ROOT)}: {len(out)} rows ({int(out['label'].sum())} spam)")


if __name__ == "__main__":
    main()
