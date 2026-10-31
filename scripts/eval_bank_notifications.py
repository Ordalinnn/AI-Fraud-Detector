"""
Does adding genuine-format bank notifications (data/bank/) fix false alarms
on real bank SMS without losing scams?

Compares two systems that differ ONLY in the bank-notification file:
  A  current training data (app.py `data` + 400 real UCI SMS)
  B  the same + data/bank/bank_notifications_train.csv
on
  1. the jury's 282 test messages (same split as scripts/run_jury_evaluation.py;
     each system trained on the other 1 116 written messages + its extras),
  2. data/bank/bank_notifications_test.csv - hand-written notifications and
     bank-style scams in wording that is not in the training templates
     (systems trained on everything, like the live site),
  3. the 2 574 real UCI SMS never used in training (false alarms on ordinary
     messages, share of spam flagged).
Also reports how close each bank test message is to its nearest training
example (char 3-5-gram cosine), so it is visible that the test is not a copy
of the training templates.

Writes reports/bank_notifications/results.json and REPORT_RU.md.
Run:  python scripts/eval_bank_notifications.py   (a few minutes)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import fraud_eval as fe  # noqa: E402
import fraud_model as fm  # noqa: E402

OUT = ROOT / "reports" / "bank_notifications"


def fit(texts, y):
    texts = list(texts)
    return fm.FraudEnsemble().fit(texts, np.asarray(y), fm.featurise(texts), fm.template_groups(texts))


def confusion(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {"TN": int(((y == 0) & (p == 0)).sum()), "FP": int(((y == 0) & (p == 1)).sum()),
            "FN": int(((y == 1) & (p == 0)).sum()), "TP": int(((y == 1) & (p == 1)).sum())}


def summary(c):
    n = sum(c.values())
    prec = c["TP"] / (c["TP"] + c["FP"]) if c["TP"] + c["FP"] else 0.0
    rec = c["TP"] / (c["TP"] + c["FN"]) if c["TP"] + c["FN"] else 0.0
    return {**c, "accuracy": (c["TP"] + c["TN"]) / n, "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def main():
    df = fe.load_dataset()
    texts, y = df["text"].tolist(), df["label"].to_numpy()
    real = fm.load_real_training_sample()
    bank = fm.load_bank_notifications()
    if not bank:
        raise SystemExit("data/bank/bank_notifications_train.csv missing: run scripts/make_bank_notifications.py")
    extras = {"A": real, "B": real + bank}
    out = {"n_written": len(texts), "n_real": len(real), "n_bank_train": len(bank),
           "n_bank_train_fraud": int(sum(l for _, l in bank))}

    # 1. jury split (identical to run_jury_evaluation.section1)
    feats, groups = fm.featurise(texts), fm.template_groups(texts)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=fm.EVAL_TEST_SIZE,
                                    random_state=fm.MODEL_RANDOM_STATE).split(feats, y, groups))
    te_texts = [texts[i] for i in te]
    out["jury_test"] = {}
    for name, ex in extras.items():
        m = fit([texts[i] for i in tr] + [t for t, _ in ex], np.concatenate([y[tr], [l for _, l in ex]]))
        p = (m.score(te_texts)["final"] >= 0.5).astype(int)
        out["jury_test"][name] = summary(confusion(y[te], p))
        print("jury", name, out["jury_test"][name])

    # 2 + 3. production-like models trained on everything
    bank_test = fm.load_bank_notifications_test()
    holdout = fm.load_real_holdout()
    out["bank_test"], out["real_holdout"], rows = {}, {}, {}
    for name, ex in extras.items():
        m = fit(texts + [t for t, _ in ex], np.concatenate([y, [l for _, l in ex]]))
        sb = m.score(bank_test["text"].tolist())["final"]
        pb = (sb >= 0.5).astype(int)
        c = confusion(bank_test["label"], pb)
        out["bank_test"][name] = {"safe_total": c["TN"] + c["FP"], "safe_false_alarms": c["FP"],
                                  "fraud_total": c["TP"] + c["FN"], "fraud_caught": c["TP"]}
        rows[name] = sb
        sh = m.score(holdout["text"].tolist())["final"]
        ph = (sh >= 0.5).astype(int)
        ch = confusion(holdout["label"].astype(int), ph)
        out["real_holdout"][name] = {"ham": ch["TN"] + ch["FP"], "ham_false_alarms": ch["FP"],
                                     "spam": ch["TP"] + ch["FN"], "spam_flagged": ch["TP"]}
        print("bank", name, out["bank_test"][name], "real", out["real_holdout"][name])

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit([t for t, _ in bank] + bank_test["text"].tolist())
    sim = cosine_similarity(vec.transform(bank_test["text"]), vec.transform([t for t, _ in bank])).max(axis=1)
    out["bank_test_nearest_train_similarity"] = {"median": float(np.median(sim)), "max": float(sim.max()),
                                                 "share_at_or_above_0.6": float((sim >= fm.TEMPLATE_SIMILARITY).mean())}
    out["bank_test_messages"] = [
        {"text": t, "label": int(l), "language": lang, "note": note, "A": float(a), "B": float(b), "nearest_train_sim": float(s)}
        for t, l, lang, note, a, b, s in zip(bank_test["text"], bank_test["label"], bank_test["language"],
                                            bank_test["note"], rows["A"], rows["B"], sim)]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("written", OUT / "results.json")


if __name__ == "__main__":
    main()
