"""
Federated learning simulation: can several banks train ONE fraud model
together without sending each other their clients' messages?

Context: in October 2025 ForteBank proposed an industry-wide anti-fraud model
for Kazakhstan's banks built on federated learning (each bank trains locally,
only the training result is shared). This script tests that idea on this
project's data and its strongest single model (character n-grams + logistic
regression, the model the Android app runs).

Setup
-----
* Test: the jury's 282 messages (same split as scripts/run_jury_evaluation.py).
  No bank ever trains on them.
* Training pool: the other written messages + 400 real UCI SMS + bank
  notifications (data/bank/), divided between 3 simulated banks.
* Two ways of dividing it:
    iid      template groups shared out at random: every bank sees every
             kind of scam;
    non_iid  each bank sees only its own third of the scam FAMILIES (bank
             cards, delivery, investment...), as real banks see different
             fraud; ordinary messages are shared out at random.
* Features: hashed character 2-5-grams (HashingVectorizer). Unlike TF-IDF,
  hashing needs no shared vocabulary, so no bank reveals its words to build
  the feature space.
* Model: logistic regression trained by plain gradient descent, written out
  in numpy so every step is visible:
      w <- w - eta * ( X^T (sigma(Xw + b) - y) / n + lambda * w )
* Systems, all with the SAME model and the same number of gradient steps:
    local        each bank alone on its own data (3 models; mean reported)
    federated    FedAvg: every round each bank starts from the shared
                 weights, takes LOCAL_STEPS steps on its own data, sends
                 back only its weights; the server averages them, weighted
                 by each bank's number of messages (McMahan et al., 2017)
    central      all data pooled in one place (what a data-sharing
                 agreement would allow; the upper bound)

Writes reports/federated/results.json, chart.png. Run:
    python scripts/eval_federated.py      (about a minute)
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import fraud_eval as fe  # noqa: E402
import fraud_model as fm  # noqa: E402

OUT = ROOT / "reports" / "federated"
N_BANKS = 3
N_FEATURES = 2 ** 18
ROUNDS = 120
LOCAL_STEPS = 10
ETA = 8.0
LAMBDA = 1e-5
SEEDS = [0, 1, 2, 3, 4]

_spec = importlib.util.spec_from_file_location("jury", ROOT / "scripts" / "run_jury_evaluation.py")
_jury = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_jury)
scam_family = _jury.scam_family


def featurise(texts) -> sparse.csr_matrix:
    hv = HashingVectorizer(analyzer="char_wb", ngram_range=fm.CHAR_NGRAM_RANGE, n_features=N_FEATURES,
                           alternate_sign=False, norm=None, lowercase=True)
    X = hv.transform(fm.prepare_texts(texts)).tocsr().astype(np.float64)
    X.data = 1.0 + np.log(X.data)          # sublinear tf, as in the char model
    return normalize(X)                    # l2 per message


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def gd(X, y, w, b, steps):
    n = X.shape[0]
    for _ in range(steps):
        err = sigmoid(X @ w + b) - y
        w = w - ETA * (X.T @ err / n + LAMBDA * w)
        b = b - ETA * err.mean()
    return w, b


def evaluate(w, b, X, y, mask=None):
    p = (sigmoid(X @ w + b) >= 0.5).astype(int)
    if mask is not None:
        p, y = p[mask], y[mask]
    tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    return {"accuracy": (tp + tn) / len(y), "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "FP": fp, "FN": fn, "TP": tp, "TN": tn}


def split_banks(pool: pd.DataFrame, mode: str, rng: np.random.Generator) -> np.ndarray:
    groups = pool["group"].to_numpy()
    uniq = np.unique(groups)
    by_group = dict(zip(uniq, rng.integers(0, N_BANKS, len(uniq))))
    bank = np.array([by_group[g] for g in groups])
    if mode == "non_iid":
        fams = sorted({f for f in pool["family"] if isinstance(f, str)}, key=lambda f: -(pool["family"] == f).sum())
        order = rng.permutation(len(fams))
        fam_bank = {}
        sizes = np.zeros(N_BANKS)
        for i in order:                      # greedy: biggest-to-smallest family to the emptiest bank
            f = fams[i]
            k = int(np.argmin(sizes))
            fam_bank[f] = k
            sizes[k] += (pool["family"] == f).sum()
        for i, f in enumerate(pool["family"]):
            if isinstance(f, str):           # a scam with a known family; safe messages keep their random bank
                bank[i] = fam_bank[f]
    return bank


def run(seed: int, mode: str, pool, Xp, yp, Xt, yt, test_fam):
    rng = np.random.default_rng(seed)
    bank = split_banks(pool, mode, rng)
    d = Xp.shape[1]
    total_steps = ROUNDS * LOCAL_STEPS
    res = {"bank_sizes": [int((bank == k).sum()) for k in range(N_BANKS)],
           "bank_fraud": [int(yp[bank == k].sum()) for k in range(N_BANKS)]}

    w, b = gd(Xp, yp, np.zeros(d), 0.0, total_steps)
    res["central"] = evaluate(w, b, Xt, yt)
    central = (w, b)

    local = []
    for k in range(N_BANKS):
        m = bank == k
        local.append(gd(Xp[m], yp[m], np.zeros(d), 0.0, total_steps))
    res["local"] = [evaluate(w, b, Xt, yt) for w, b in local]

    gw, gb_ = np.zeros(d), 0.0
    curve = []
    sizes = np.array([(bank == k).sum() for k in range(N_BANKS)], dtype=float)
    for r in range(ROUNDS):
        ws, bs = [], []
        for k in range(N_BANKS):
            m = bank == k
            wk, bk = gd(Xp[m], yp[m], gw.copy(), gb_, LOCAL_STEPS)
            ws.append(wk); bs.append(bk)
        gw = np.average(ws, axis=0, weights=sizes)
        gb_ = float(np.average(bs, weights=sizes))
        curve.append(evaluate(gw, gb_, Xt, yt)["accuracy"])
    res["federated"] = evaluate(gw, gb_, Xt, yt)
    res["federated_curve"] = curve

    # scam families a bank never saw itself (non-IID only)
    if mode == "non_iid":
        unseen = []
        for k in range(N_BANKS):
            seen = {f for f in pool["family"][bank == k] if isinstance(f, str)}
            mask = np.array([isinstance(f, str) and f not in seen for f in test_fam]) & (yt == 1)
            if mask.sum():
                unseen.append({"bank": k, "n": int(mask.sum()),
                               "local_recall": evaluate(*local[k], Xt, yt, mask)["recall"],
                               "federated_recall": evaluate(gw, gb_, Xt, yt, mask)["recall"],
                               "central_recall": evaluate(*central, Xt, yt, mask)["recall"]})
        res["unseen_families"] = unseen
    return res


def main():
    df = fe.load_dataset()
    texts, y = df["text"].tolist(), df["label"].to_numpy()
    feats, groups = fm.featurise(texts), fm.template_groups(texts)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=fm.EVAL_TEST_SIZE,
                                    random_state=fm.MODEL_RANDOM_STATE).split(feats, y, groups))

    extra = fm.extra_training_examples()
    bank_texts = {t for t, _ in fm.load_bank_notifications()}
    pool = pd.DataFrame({
        "text": [texts[i] for i in tr] + [t for t, _ in extra],
        "label": np.concatenate([y[tr], [l for _, l in extra]]).astype(int),
        "family": [scam_family(df["category"].iloc[i]) for i in tr]
                  + [("Банк / карта / колл-центр банка" if (t in bank_texts and l == 1) else None) for t, l in extra],
    })
    pool["group"] = fm.template_groups(pool["text"].tolist())
    Xp, yp = featurise(pool["text"].tolist()), pool["label"].to_numpy().astype(float)
    Xt, yt = featurise([texts[i] for i in te]), y[te]
    test_fam = [scam_family(df["category"].iloc[i]) for i in te]

    out = {"setup": {"n_banks": N_BANKS, "rounds": ROUNDS, "local_steps": LOCAL_STEPS, "eta": ETA,
                     "lambda": LAMBDA, "n_features": N_FEATURES, "seeds": SEEDS,
                     "n_pool": len(pool), "n_test": len(te), "shared_per_round_numbers": N_FEATURES + 1}}
    for mode in ("iid", "non_iid"):
        runs = [run(s, mode, pool, Xp, yp, Xt, yt, test_fam) for s in SEEDS]
        summ = {}
        for sysname in ("central", "federated"):
            for metric in ("accuracy", "recall", "FP"):
                v = [r[sysname][metric] for r in runs]
                summ[f"{sysname}_{metric}"] = [float(np.mean(v)), float(np.std(v))]
        for metric in ("accuracy", "recall", "FP"):
            v = [np.mean([l[metric] for l in r["local"]]) for r in runs]
            summ[f"local_{metric}"] = [float(np.mean(v)), float(np.std(v))]
            v = [min(l[metric] for l in r["local"]) if metric != "FP" else max(l[metric] for l in r["local"]) for r in runs]
            summ[f"local_worst_{metric}"] = [float(np.mean(v)), float(np.std(v))]
        if mode == "non_iid":
            u = [x for r in runs for x in r["unseen_families"]]
            n = sum(x["n"] for x in u)
            for sysname in ("local", "federated", "central"):
                summ[f"unseen_{sysname}_recall"] = float(sum(x[f"{sysname}_recall"] * x["n"] for x in u) / n)
            summ["unseen_n_messages_total"] = n
        summ["curve_mean"] = list(np.mean([r["federated_curve"] for r in runs], axis=0))
        out[mode] = {"summary": summ, "runs": runs}
        print(mode, {k: (round(v[0] * 100, 1) if isinstance(v, list) and k.endswith(("accuracy", "recall")) else v)
                     for k, v in summ.items() if k != "curve_mean"})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    charts(out)
    print("written", OUT / "results.json", OUT / "chart.png")


def charts(out: dict, path: Path = OUT / "chart.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = out["non_iid"]["summary"]
    labels = ["Каждый банк\nсам по себе", "Федеративное\nобучение", "Все данные\nв одном месте"]
    groups = [("Точность на 282 тестовых\nсообщениях", [s["local_accuracy"][0], s["federated_accuracy"][0], s["central_accuracy"][0]]),
              ("Поймано схем, которых банк\nсам никогда не видел", [s["unseen_local_recall"], s["unseen_federated_recall"], s["unseen_central_recall"]])]
    colors = ["#94A3B8", "#1D4ED8", "#0E7490"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=160)
    for ax, (title, vals) in zip(axes, groups):
        bars = ax.bar(range(3), [v * 100 for v in vals], color=colors, width=0.62)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v * 100 + 1.5, f"{v * 100:.1f} %".replace(".", ","),
                    ha="center", fontsize=12, fontweight="bold", color="#0F172A")
        ax.set_xticks(range(3), labels, fontsize=10, color="#334155")
        ax.set_ylim(0, 105)
        ax.set_title(title, fontsize=12, color="#0F172A", loc="left")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", colors="#64748B")
        ax.grid(axis="y", color="#E3E9F2")
        ax.set_axisbelow(True)
    fig.suptitle("3 банка, каждый видит только свои типы мошенничества (среднее по 5 разбиениям)",
                 fontsize=12, color="#475569", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
