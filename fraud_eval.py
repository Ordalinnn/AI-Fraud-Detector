"""
Offline evaluation toolkit for the AI Fraud Detector (no Streamlit needed).

Answers the three questions the NIS Taraz jury asked before the defense
(written-project review, 09.10.2026):

  1. On the SAME held-out test, how does the best single model compare with
     the 4-model ensemble and with the ensemble + rule boost?
  2. How does the system do on independent real messages, per language,
     and on scam types it has never seen?
  3. Is the final risk score a calibrated probability, and what does a false
     block cost at different thresholds?

Models come from fraud_model.py, the same module app.py trains with, so the
numbers describe the model users are actually scored against. app.py itself
can't be imported outside `streamlit run`, so its `data` list is read from
the source with `ast` instead of being copied (a copy would silently drift).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from fraud_model import BaseModels, FraudEnsemble, featurise, template_groups  # noqa: F401 (re-exported for scripts)

BASE_DIR = Path(__file__).resolve().parent
APP_PATH = BASE_DIR / "app.py"

SYSTEMS = ["old_lr", "old_rf", "old_gb", "old_tfidf", "old_ensemble", "old_ensemble_rules",
           "lr", "rf", "gb", "tfidf", "char", "final", "baseline"]
SINGLE_MODELS = ["lr", "rf", "gb", "tfidf", "char"]
SYSTEM_NAMES = {
    "old_lr": "Logistic Regression (старая система)",
    "old_rf": "Random Forest (старая система)",
    "old_gb": "Gradient Boosting (старая система)",
    "old_tfidf": "TF-IDF + Naive Bayes (старая система)",
    "lr": "Logistic Regression",
    "rf": "Random Forest",
    "gb": "Gradient Boosting",
    "tfidf": "TF-IDF + Naive Bayes",
    "char": "Символьные n-граммы + LR (новая модель)",
    "old_ensemble": "Старый ансамбль: среднее 4 моделей",
    "old_ensemble_rules": "Старый ансамбль + правила (как в работе)",
    "final": "Новый ансамбль: стекинг 5 моделей + правила",
    "baseline": "Простое правило по ключевым словам",
}


# =========================
# LOADING THE DATASET FROM app.py
# =========================
def _module_assignments(tree):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    yield t.id, node


_CATEGORY_RE = re.compile(r"^\s*#\s*(FRAUD|SAFE)\b[^:]*:\s*(.+?)\s*$")


def load_dataset(app_path: Path = APP_PATH) -> pd.DataFrame:
    """Returns the hand-written training set from app.py's `data` literal as a
    DataFrame with columns text, label, category, language.

    `category` is the nearest preceding "# FRAUD: ..." / "# SAFE: ..." comment
    in app.py (the dataset is organised by scam type with those comments), so
    whole scam types can be held out to simulate never-seen fraud."""
    src = app_path.read_text(encoding="utf-8")
    lines = src.splitlines()
    tree = ast.parse(src)
    data_node = next(node for name, node in _module_assignments(tree) if name == "data")

    # Map each source line -> the most recent category comment above it.
    cat_at_line = {}
    current = "uncategorised"
    for i in range(data_node.lineno, data_node.end_lineno + 1):
        m = _CATEGORY_RE.match(lines[i - 1])
        if m:
            current = f"{m.group(1)}: {_normalise_category(m.group(2))}"
        cat_at_line[i] = current

    rows = []
    for elt in data_node.value.elts:
        text, label = ast.literal_eval(elt)
        rows.append({
            "text": text,
            "label": int(label),
            "category": cat_at_line.get(elt.lineno, "uncategorised"),
            "language": detect_language(text),
        })
    return pd.DataFrame(rows)


def _normalise_category(raw: str) -> str:
    """'More bank/card variants' and 'Bank/card, more bank names' are the
    same scam type written in different sections; strip the 'more ...'
    wording and trailing explanation so held-out categories are real
    scam types, not section headings."""
    s = raw.lower()
    s = re.split(r"\s[—-]\s|\(", s)[0]
    s = re.sub(r"^more\s+", "", s)
    s = re.sub(r",?\s*more\b.*$", "", s)
    s = re.sub(r"\bvariants?\b", "", s)
    return re.sub(r"\s+", " ", s).strip(" ,.")


_KAZAKH_ONLY = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
_CYRILLIC = re.compile(r"[а-яёА-ЯЁ]")
_LATIN = re.compile(r"[a-zA-Z]")


def detect_language(text: str) -> str:
    """Script-based language tag: 'kk' if any Kazakh-specific letter appears,
    else 'ru' for Cyrillic, else 'en' for Latin. Good enough for SMS-length
    text in these three languages; mixed messages count by their
    dominant script."""
    if any(ch in _KAZAKH_ONLY for ch in text):
        return "kk"
    cyr = len(_CYRILLIC.findall(text))
    lat = len(_LATIN.findall(text))
    if cyr == 0 and lat == 0:
        return "other"
    return "ru" if cyr >= lat else "en"


# =========================
# SCORING ONE TRAIN/TEST SPLIT
# =========================
def score_split(df: pd.DataFrame, feats: pd.DataFrame, groups, train_idx, test_idx,
                extra: pd.DataFrame | None = None, extra_feats: pd.DataFrame | None = None) -> dict:
    """Scores test_idx with every system the jury asked to compare, all on the
    identical test messages:

      OLD (exactly the system described in the written work - trained on the
      project's train messages only):
        old_ensemble        plain average of LR, RF, GB, TF-IDF+NB
        old_ensemble_rules  that + capped rule boost (the app as submitted)
      NEW (trained on the project's train messages + the small real sample
      `extra`, if given - every single model sees exactly the same data as
      the ensemble, so the single-vs-ensemble comparison is fair):
        lr / rf / gb / tfidf / char   each single model
        final                         stacked ensemble (rules are one input)
      baseline  "flag if any urgency/secret/threat/reward word" (no training)
    """
    texts = df["text"].tolist()
    y = df["label"].to_numpy()
    groups = np.asarray(groups)
    tr_texts = [texts[i] for i in train_idx]
    te_texts = [texts[i] for i in test_idx]
    te_feats = feats.iloc[test_idx]

    old = BaseModels().fit(tr_texts, y[train_idx], feats.iloc[train_idx])
    so = old.predict(te_texts, te_feats)

    T, Y, F, G = tr_texts, y[train_idx], feats.iloc[train_idx].reset_index(drop=True), groups[train_idx]
    if extra is not None and len(extra):
        T = T + extra["text"].tolist()
        Y = np.concatenate([Y, extra["label"].to_numpy().astype(int)])
        F = pd.concat([F, extra_feats.reset_index(drop=True)], ignore_index=True)
        G = np.concatenate([G, groups.max() + 1 + np.arange(len(extra))])
    ens = FraudEnsemble().fit(T, Y, F, G)
    s = ens.score(te_texts, te_feats)
    for k in ("lr", "rf", "gb", "tfidf"):
        s[f"old_{k}"] = so[k]   # the old system's own members (no real sample)
    s["old_ensemble"] = (so["lr"] + so["rf"] + so["gb"] + so["tfidf"]) / 4.0
    s["old_ensemble_rules"] = np.minimum(0.99, s["old_ensemble"] + so["boost"])
    s["baseline"] = ((te_feats["urgent_count"] + te_feats["secret_count"] + te_feats["threat_count"]
                      + te_feats["reward_count"]) > 0).astype(float).to_numpy()
    s["_weights"] = ens.weights
    # out-of-fold scores for the project's own training messages only
    s["_oof_train"] = {k: v[: len(train_idx)] for k, v in ens.oof_scores_.items()}
    return s


# =========================
# METRICS
# =========================
def metrics(y_true, score, threshold: float = 0.5, binary_score: bool = False) -> dict:
    y_true = np.asarray(y_true)
    pred = (np.asarray(score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    m = {
        "n": int(len(y_true)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else float("nan"),
    }
    if not binary_score and len(set(y_true.tolist())) == 2:
        m["roc_auc"] = roc_auc_score(y_true, score)
        m["brier"] = brier_score_loss(y_true, np.clip(score, 0, 1))
    return m


def mcnemar_exact(y_true, pred_a, pred_b) -> dict:
    """Exact McNemar test: of the messages where exactly one of the two
    models is right, is the split significantly different from 50/50?
    The right test for 'is model A better than model B on the same test
    set' (the two error sets are paired, so a plain two-proportion test
    would be wrong)."""
    y = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y
    b_ok = np.asarray(pred_b) == y
    b = int(np.sum(a_ok & ~b_ok))   # A right, B wrong
    c = int(np.sum(~a_ok & b_ok))   # A wrong, B right
    p = binomtest(b, b + c, 0.5).pvalue if (b + c) else 1.0
    return {"a_only_right": b, "b_only_right": c, "p_value": float(p)}


def bootstrap_diff_ci(y_true, score_a, score_b, metric: str = "accuracy",
                      threshold: float = 0.5, n_boot: int = 2000, seed: int = 0,
                      groups=None) -> tuple[float, float, float]:
    """95% bootstrap CI for metric(A) - metric(B) on the same test messages.
    If `groups` is given, whole template groups are resampled (cluster
    bootstrap) because near-duplicate messages are not independent."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    pa = (np.asarray(score_a) >= threshold).astype(int)
    pb = (np.asarray(score_b) >= threshold).astype(int)
    fn = {"accuracy": accuracy_score, "f1": lambda t, p: f1_score(t, p, zero_division=0),
          "recall": lambda t, p: recall_score(t, p, zero_division=0),
          "precision": lambda t, p: precision_score(t, p, zero_division=0)}[metric]
    point = fn(y, pa) - fn(y, pb)
    if groups is None:
        units = [np.array([i]) for i in range(len(y))]
    else:
        g = np.asarray(groups)
        units = [np.where(g == u)[0] for u in np.unique(g)]
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(units), len(units))
        idx = np.concatenate([units[k] for k in pick])
        diffs.append(fn(y[idx], pa[idx]) - fn(y[idx], pb[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def calibration_table(y_true, prob, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table: for each predicted-risk bin, the mean predicted
    risk vs. the fraction that was actually fraud."""
    y = np.asarray(y_true)
    p = np.clip(np.asarray(prob), 0, 1)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append({
            "bin": f"{edges[b]:.1f}–{edges[b+1]:.1f}",
            "n": int(m.sum()),
            "mean_predicted": float(p[m].mean()) if m.any() else np.nan,
            "fraction_fraud": float(y[m].mean()) if m.any() else np.nan,
        })
    return pd.DataFrame(rows)


def expected_calibration_error(y_true, prob, n_bins: int = 10) -> float:
    t = calibration_table(y_true, prob, n_bins)
    t = t[t["n"] > 0]
    return float(np.sum(t["n"] / t["n"].sum() * np.abs(t["mean_predicted"] - t["fraction_fraud"])))


def threshold_table(y_true, prob, thresholds, cost_fp: float = 1.0, cost_fn: float = 1.0) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        m = metrics(y_true, prob, threshold=t, binary_score=True)
        m["threshold"] = t
        m["cost"] = cost_fp * m["FP"] + cost_fn * m["FN"]
        rows.append(m)
    return pd.DataFrame(rows)
