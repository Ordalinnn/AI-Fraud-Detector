"""
How the neural member's settings (fraud_model.MLP_*) were chosen.

    python scripts/select_neural_member.py          # all 38 candidates, ~40 min
    python scripts/select_neural_member.py --quick  # 6 candidates, ~5 min

Uses ONLY the training part of the jury split (the same 80/20 grouped split
as scripts/run_jury_evaluation.py) plus the real SMS sample - the jury's
282 test messages are never touched, so the later comparison on them stays
honest.

For every candidate network:
  1. its out-of-fold probability on the training part (group 5-fold);
  2. how much it improves the WHOLE stacked system: the 5 current base
     models' out-of-fold scores + the candidate go into the stacker, which is
     itself cross-validated (5 different second-level splits, averaged).
The pick (k5000_h16_a0.01): among the networks tied at the lowest stacked
log-loss (0.0897-0.0902), the one with the lowest stacked Brier score and
the stronger regularization (tied with the 32-unit version, which is twice
the size); small enough for the extension and the phone page. Writes reports/neural_member/selection.csv (the committed file is the
full 38-candidate run).
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fraud_eval as fe  # noqa: E402
import fraud_model as fm  # noqa: E402

OUT = ROOT / "reports" / "neural_member"
OLD_MEMBERS = ["lr", "rf", "gb", "tfidf", "char"]   # the 5-model system


def candidates(quick: bool) -> dict:
    def make(hidden, alpha, k=None):
        steps = [("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=fm.CHAR_NGRAM_RANGE, min_df=fm.CHAR_MIN_DF,
                                           sublinear_tf=True, lowercase=True))]
        if k:
            steps.append(("select", fm.Chi2TopK(k)))
        steps.append(("clf", MLPClassifier(hidden_layer_sizes=(hidden,), alpha=alpha, max_iter=fm.MLP_MAX_ITER,
                                           random_state=fm.MODEL_RANDOM_STATE)))
        return Pipeline(steps)

    if quick:
        grid_all = [(16, 0.01), (16, 0.03)]
        grid_sel = [(5000, 16, 0.01), (5000, 32, 0.01), (3000, 16, 0.01), (10000, 16, 0.01)]
    else:
        grid_all = list(itertools.product([16, 32, 64], [1e-4, 1e-3, 1e-2])) + \
                   list(itertools.product([8, 16, 32], [0.03, 0.1, 0.3]))
        grid_sel = [(2000, 32, 1e-3), (5000, 32, 1e-3)] + \
                   list(itertools.product([3000, 5000, 10000], [16, 32], [0.003, 0.01, 0.03]))
    c = {f"all_h{h}_a{a:g}": (lambda h=h, a=a: make(h, a)) for h, a in grid_all}
    c.update({f"k{k}_h{h}_a{a:g}": (lambda k=k, h=h, a=a: make(h, a, k)) for k, h, a in grid_sel})
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    warnings.simplefilter("ignore", ConvergenceWarning)
    OUT.mkdir(parents=True, exist_ok=True)

    df = fe.load_dataset()
    feats = fm.featurise(df["text"])
    groups = fm.template_groups(df["text"].tolist())
    y = df["label"].to_numpy()
    tr, _ = next(GroupShuffleSplit(n_splits=1, test_size=fm.EVAL_TEST_SIZE, random_state=fm.MODEL_RANDOM_STATE)
                 .split(feats, y, groups))
    extra = pd.DataFrame(fm.load_real_training_sample(), columns=["text", "label"])
    T = [df["text"][i] for i in tr] + extra["text"].tolist()
    Y = np.concatenate([y[tr], extra["label"].to_numpy().astype(int)])
    G = np.concatenate([groups[tr], groups.max() + 1 + np.arange(len(extra))])
    F = pd.concat([feats.iloc[tr].reset_index(drop=True), fm.featurise(extra["text"]).reset_index(drop=True)],
                  ignore_index=True)
    P = fm.prepare_texts(T)
    folds = list(GroupKFold(n_splits=5).split(T, Y, G))

    # out-of-fold scores of the 5 current members (+ rule boost)
    base = {k: np.zeros(len(T)) for k in OLD_MEMBERS + ["boost"]}
    for a, b in folds:
        m = fm.BaseModels()
        m.models = {k: v for k, v in m.models.items() if k in OLD_MEMBERS}
        saved = fm.TEXT_MODELS
        fm.TEXT_MODELS = ["tfidf", "char"]
        try:
            m.fit([T[i] for i in a], Y[a], F.iloc[a])
            s = m.predict([T[i] for i in b], F.iloc[b])
        finally:
            fm.TEXT_MODELS = saved
        for k in base:
            base[k][b] = s[k]

    def stacked(extra_col=None):
        cols = [fm._logit(base[k]) for k in OLD_MEMBERS] + [base["boost"]]
        if extra_col is not None:
            cols.append(fm._logit(extra_col))
        X = np.column_stack(cols)
        res = []
        for seed in range(5):
            p = np.zeros(len(Y))
            for a, b in GroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X, Y, G):
                p[b] = LogisticRegression(C=fm.STACK_C, max_iter=1000).fit(X[a], Y[a]).predict_proba(X[b])[:, 1]
            res.append([log_loss(Y, np.clip(p, 1e-4, 1 - 1e-4)), brier_score_loss(Y, p),
                        f1_score(Y, p >= .5), accuracy_score(Y, p >= .5)])
        return np.mean(res, axis=0)

    rows = [dict(zip(["model", "stack_logloss", "stack_brier", "stack_f1", "stack_acc"], ["5 models (current)", *map(float, stacked())]))]
    char_err = (base["char"] >= .5) != Y
    for name, make in candidates(args.quick).items():
        o = np.zeros(len(T))
        t0 = time.time()
        for a, b in folds:
            o[b] = make().fit([P[i] for i in a], Y[a]).predict_proba([P[i] for i in b])[:, 1]
        r = dict(zip(["model", "stack_logloss", "stack_brier", "stack_f1", "stack_acc"], [name, *map(float, stacked(o))]))
        r.update(alone_logloss=log_loss(Y, np.clip(o, 1e-4, 1 - 1e-4)), alone_f1=f1_score(Y, o >= .5),
                 corr_with_char=float(np.corrcoef(fm._logit(o), fm._logit(base["char"]))[0, 1]),
                 fixes_char_errors=int((((o >= .5) == Y) & char_err).sum()), fit_seconds=(time.time() - t0) / 5)
        rows.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)

    res = pd.DataFrame(rows).set_index("model").sort_values("stack_logloss")
    res.to_csv(OUT / "selection.csv")
    print(res.round(4).to_string())


if __name__ == "__main__":
    main()
