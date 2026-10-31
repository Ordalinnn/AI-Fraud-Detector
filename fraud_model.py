"""
The trained part of the detector: 6 base models + a learned combiner.

Kept out of app.py (no Streamlit import) so the exact same code is used by
the live app, by the offline jury evaluation (fraud_eval.py /
scripts/run_jury_evaluation.py) and by the unit tests.

WHY A STACKED (LEARNED-WEIGHT) ENSEMBLE INSTEAD OF A PLAIN AVERAGE
-----------------------------------------------------------------
The jury's key question was: "what does the ensemble gain over the best
single model on the same 282 test messages?" Measured honestly, the old
plain average of 4 models gained nothing: TF-IDF + Naive Bayes alone
scored 86.2% accuracy on that test vs. 82.3% for ensemble + rules, and it
won on all 15 folds of a repeated group-aware cross-validation. Averaging
gave the three keyword-feature models (LR/RF/GB, ~80% each) three votes
out of four, so they dragged the strongest model down.

The fix has two parts:
  1. A 5th base model that reads character n-grams (char_wb 2-5) with
     logistic regression. Character n-grams handle Kazakh/Russian word
     endings (картаңыз / карту / карта) and obfuscated spellings
     ("k0d", "sr0chno") that whole-word TF-IDF treats as unrelated words.
  2. A logistic-regression combiner ("stacker") that LEARNS how much to
     trust each model, plus the rule boost as one more input, from
     out-of-fold predictions grouped by template (so it never learns from
     a model's score on messages that model was trained on). Its output is
     a probability fitted with log-loss, which is also what makes the
     final risk percentage calibrated (see the jury's 3rd point).

A 6TH MEMBER: A SMALL NEURAL NETWORK (added 31.10.2026)
---------------------------------------------------------
"mlp": the same character n-grams, the 5,000 most informative of them
(chi-squared), into a one-hidden-layer neural network (16 ReLU units). The
settings were chosen on the training part only (out-of-fold, by how much
they improve the whole stacked system, never on the jury's 282 test
messages): a network on ALL n-grams agreed with the char model 99% of the
time and added nothing, while the selected-n-gram one makes different
mistakes, which is what a stacker can use. Its first-layer weights are
stored as 8-bit integers with one scale per hidden unit (_quantize_mlp), so
the browser extension and the offline phone page carry it for ~0.6 MB and
compute exactly the same numbers. Full comparison with the 5-model system:
reports/neural_member/REPORT_RU.md.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import chi2
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import fraud_logic
from fraud_logic import extract_features, rule_boost
from text_normalize import normalize_text

# =========================
# HYPERPARAMETERS (single source of truth for app + evaluation)
# =========================
MODEL_RANDOM_STATE = 42
LR_C = 1.0
LR_MAX_ITER = 1000
RF_N_ESTIMATORS = 50
GB_N_ESTIMATORS = 50
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 1
# Character model: fixed once from common practice for short multilingual
# texts, NOT tuned on the jury's test set (tuning on the test would make the
# comparison dishonest).
CHAR_NGRAM_RANGE = (2, 5)
CHAR_MIN_DF = 2
CHAR_C = 10.0
# Neural member: chosen out-of-fold on the jury split's TRAINING part only
# (scripts/select_neural_member.py), by the whole stacker's log-loss.
MLP_SELECT_K = 5000
MLP_HIDDEN = 16
MLP_ALPHA = 0.01
MLP_MAX_ITER = 300
STACK_C = 1.0
STACK_FOLDS = 5
TEMPLATE_SIMILARITY = 0.6
EVAL_TEST_SIZE = 0.2

# Real messages added to training (see scripts/prepare_external_data.py).
_DATA_DIR = Path(__file__).resolve().parent / "data" / "external"
REAL_TRAIN_PATH = _DATA_DIR / "real_train_sample.csv"
REAL_HOLDOUT_PATH = _DATA_DIR / "real_holdout.csv"
REAL_TRAIN_N_HAM = 300
REAL_TRAIN_N_SPAM = 100

BASE_MODELS = ["lr", "rf", "gb", "tfidf", "char", "mlp"]
FEATURE_MODELS = ["lr", "rf", "gb"]          # use the 20 hand-engineered features
TEXT_MODELS = ["tfidf", "char", "mlp"]       # read the raw text
STACK_INPUTS = BASE_MODELS + ["boost"]

MODEL_DISPLAY_NAMES = {
    "lr": "Logistic Regression",
    "rf": "Random Forest",
    "gb": "Gradient Boosting",
    "tfidf": "Text Patterns (TF-IDF)",
    "char": "Char n-grams (LR)",
    "mlp": "Neural network (MLP)",
    "average": "Plain average (old)",
    "average_rules": "Plain average + rules (old app)",
    "final": "Stacked ensemble (final)",
}


class Chi2TopK(BaseEstimator, TransformerMixin):
    """Keeps the k columns with the highest chi-squared score against the
    label (like SelectKBest(chi2, k)), but never fails when a small training
    set has fewer than k columns, and breaks ties by column order so the
    choice is reproducible."""

    def __init__(self, k: int = MLP_SELECT_K):
        self.k = k

    def fit(self, X, y):
        scores, _ = chi2(X, y)
        scores = np.nan_to_num(scores, nan=0.0)
        k = min(int(self.k), X.shape[1])
        self.support_ = np.sort(np.argsort(-scores, kind="stable")[:k])
        return self

    def transform(self, X):
        return X[:, self.support_]


def _quantize_mlp(pipe: Pipeline) -> None:
    """Stores the network's first layer (n-grams x hidden units, by far its
    biggest part) as 8-bit integers with one scale per hidden unit, and makes
    the fitted model USE those rounded weights. The exported file then holds
    exactly the numbers Python computes with, so the extension reproduces
    the site's probability. Effect on accuracy: within noise (see the
    report)."""
    clf = pipe.named_steps["clf"]
    w = clf.coefs_[0]
    scale = np.abs(w).max(axis=0) / 127.0
    scale[scale == 0] = 1.0
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    clf.coefs_[0] = q.astype(np.float64) * scale
    clf.w1_int8_ = q
    clf.w1_scale_ = scale


def build_base_models() -> dict:
    rs = MODEL_RANDOM_STATE
    return {
        "lr": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=LR_C, max_iter=LR_MAX_ITER, random_state=rs))]),
        "rf": Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=rs))]),
        "gb": Pipeline([("scaler", StandardScaler()), ("clf", GradientBoostingClassifier(n_estimators=GB_N_ESTIMATORS, random_state=rs))]),
        "tfidf": Pipeline([("tfidf", TfidfVectorizer(ngram_range=TFIDF_NGRAM_RANGE, min_df=TFIDF_MIN_DF, lowercase=True)), ("clf", MultinomialNB())]),
        "char": Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=CHAR_NGRAM_RANGE, min_df=CHAR_MIN_DF, sublinear_tf=True, lowercase=True)),
            ("clf", LogisticRegression(C=CHAR_C, max_iter=2000, random_state=rs)),
        ]),
        # Same n-grams as "char" (identical vectorizer settings, so the
        # extension reuses that vocabulary), 5,000 kept, one hidden layer.
        "mlp": Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=CHAR_NGRAM_RANGE, min_df=CHAR_MIN_DF, sublinear_tf=True, lowercase=True)),
            ("select", Chi2TopK(MLP_SELECT_K)),
            ("clf", MLPClassifier(hidden_layer_sizes=(MLP_HIDDEN,), alpha=MLP_ALPHA, max_iter=MLP_MAX_ITER, random_state=rs)),
        ]),
    }


def load_real_training_sample(path: Path = REAL_TRAIN_PATH) -> list[list]:
    """The small sample of REAL messages (UCI SMS Spam Collection) that is
    added to training, as [text, label] pairs like app.py's `data`. Returns
    [] if the file is missing, so the app still starts without it."""
    if not Path(path).exists():
        return []
    df = pd.read_csv(path, dtype={"text": str})
    df = df.dropna(subset=["text", "label"])
    return [[t, int(label)] for t, label in zip(df["text"], df["label"])]


def load_real_holdout(path: Path = REAL_HOLDOUT_PATH) -> pd.DataFrame:
    """The untouched half of the real corpus, used only for evaluation."""
    return pd.read_csv(path, dtype={"text": str}).dropna(subset=["text", "label"])


def featurise(texts) -> pd.DataFrame:
    return pd.DataFrame([extract_features(t)[0] for t in texts])


def template_groups(texts, similarity_threshold: float = TEMPLATE_SIMILARITY) -> np.ndarray:
    """Groups near-duplicate examples — e.g. the same "your card is blocked,
    confirm at <link>" template repeated with a different bank name — so a
    group-aware split puts every variant of a template on one side only.
    Union-find over pairs with char 3-5-gram TF-IDF cosine >= threshold."""
    texts = list(texts)
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit_transform(texts)
    sim = cosine_similarity(X)
    n = len(texts)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    i_idx, j_idx = np.triu_indices(n, k=1)
    mask = sim[i_idx, j_idx] >= similarity_threshold
    for i, j in zip(i_idx[mask], j_idx[mask]):
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[ri] = rj
    return np.array([find(i) for i in range(n)])


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _stack_matrix(scores: dict) -> np.ndarray:
    cols = [_logit(scores[k]) for k in BASE_MODELS] + [np.asarray(scores["boost"], dtype=float)]
    return np.column_stack(cols)


def prepare_texts(texts) -> list:
    """What the text models (TF-IDF, char n-grams) read: the message with
    disguises undone (text_normalize.py), same in training and checking."""
    if not fraud_logic.NORMALIZE_INPUT:
        return list(texts)
    return [normalize_text(t) for t in texts]


class BaseModels:
    """The 5 base models fitted on one training set."""

    def __init__(self):
        self.models = build_base_models()

    def fit(self, texts, y, feats: pd.DataFrame):
        texts = prepare_texts(texts)
        y = np.asarray(y)
        for k in FEATURE_MODELS:
            self.models[k].fit(feats, y)
        for k in TEXT_MODELS:
            if k == "mlp":
                with warnings.catch_warnings():
                    # small data: adam may stop at max_iter before its tolerance
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    self.models[k].fit(texts, y)
                _quantize_mlp(self.models[k])
            else:
                self.models[k].fit(texts, y)
        self.columns_ = list(feats.columns)
        return self

    def predict(self, texts, feats: pd.DataFrame) -> dict:
        texts = prepare_texts(texts)
        feats = feats[self.columns_]
        out = {k: self.models[k].predict_proba(feats)[:, 1] for k in FEATURE_MODELS}
        for k in TEXT_MODELS:
            out[k] = self.models[k].predict_proba(texts)[:, 1]
        out["boost"] = np.array([rule_boost(r) for r in feats.to_dict("records")], dtype=float)
        return out


class FraudEnsemble:
    """5 base models + logistic-regression stacker.

    fit():  1) group-aware K-fold over the training set -> out-of-fold base
               scores for every message;  2) stacker fitted on those;
            3) base models refitted on the whole training set.
    score(): every base probability, the old plain average (kept so the UI
             and the evaluation can show what changed), the rule boost, and
             "final" = the stacker's probability.
    """

    def __init__(self, folds: int = STACK_FOLDS):
        self.folds = folds

    def fit(self, texts, y, feats: pd.DataFrame | None = None, groups=None):
        texts = list(texts)
        y = np.asarray(y)
        feats = featurise(texts) if feats is None else feats.reset_index(drop=True)
        groups = template_groups(texts) if groups is None else np.asarray(groups)

        oof = {k: np.zeros(len(texts)) for k in STACK_INPUTS}
        for tr, te in GroupKFold(n_splits=self.folds).split(feats, y, groups):
            base = BaseModels().fit([texts[i] for i in tr], y[tr], feats.iloc[tr])
            s = base.predict([texts[i] for i in te], feats.iloc[te])
            for k in STACK_INPUTS:
                oof[k][te] = s[k]
        self.oof_scores_ = oof

        self.stacker = LogisticRegression(C=STACK_C, max_iter=1000).fit(_stack_matrix(oof), y)
        self.base = BaseModels().fit(texts, y, feats)
        return self

    @property
    def weights(self) -> dict:
        """Learned stacker weight per input (on the log-odds scale)."""
        return {k: float(w) for k, w in zip(STACK_INPUTS, self.stacker.coef_[0])}

    def score(self, texts, feats: pd.DataFrame | None = None) -> dict:
        texts = list(texts)
        feats = featurise(texts) if feats is None else feats
        out = self.base.predict(texts, feats)
        out["average"] = np.mean([out[k] for k in BASE_MODELS], axis=0)
        out["final"] = self.stacker.predict_proba(_stack_matrix(out))[:, 1]
        return out


# =========================
# EXPORT FOR THE BROWSER EXTENSION
# =========================
# The extension (browser-extension/ensemble.js) re-implements score() in
# JavaScript from these numbers, so the site and the extension give the
# same probability for the same text. scripts/export_extension_model.py
# writes the file and checks the JavaScript against scikit-learn.
EXPORT_FORMAT = "fraud_ensemble_v1"


def training_fingerprint(texts, y) -> str:
    """Short hash of the exact training set (texts, labels, order). The app
    and the exported extension model carry it, so a test can tell when the
    extension's copy is out of date."""
    import hashlib
    import json
    payload = json.dumps([[str(t), int(label)] for t, label in zip(texts, y)], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _export_tree(tree, value) -> dict:
    t = tree.tree_
    return {
        "feature": [int(f) for f in t.feature],
        "threshold": [float(x) for x in t.threshold],
        "left": [int(x) for x in t.children_left],
        "right": [int(x) for x in t.children_right],
        "value": [float(x) for x in value],
    }


def _export_mlp(mlp: Pipeline, char: Pipeline) -> dict:
    """The neural member. It reads the same TF-IDF vector as "char" (same
    vectorizer settings and training texts, checked here), so the extension
    reuses the char vocabulary's idf values and only needs: the kept
    n-grams, the 8-bit first layer (base64, row per kept n-gram), one scale
    per hidden unit, and the small second layer."""
    import base64
    tf, ctf = mlp.named_steps["tfidf"], char.named_steps["tfidf"]
    assert tf.vocabulary_ == ctf.vocabulary_ and np.array_equal(tf.idf_, ctf.idf_), "mlp must share char's TF-IDF"
    clf = mlp.named_steps["clf"]
    assert clf.activation == "relu" and clf.out_activation_ == "logistic" and len(clf.coefs_) == 2
    inv = {i: g for g, i in tf.vocabulary_.items()}
    support = mlp.named_steps["select"].support_
    q = np.ascontiguousarray(clf.w1_int8_, dtype=np.int8)
    assert np.array_equal(q.astype(np.float64) * clf.w1_scale_, clf.coefs_[0])
    return {"hidden": int(q.shape[1]),
            "ngrams": [inv[int(i)] for i in support],
            "w1_int8": base64.b64encode(q.tobytes()).decode("ascii"),
            "w1_scale": [float(x) for x in clf.w1_scale_],
            "b1": [float(x) for x in clf.intercepts_[0]],
            "w2": [float(x) for x in clf.coefs_[1][:, 0]],
            "b2": float(clf.intercepts_[1][0])}


def export_ensemble(model: FraudEnsemble, fingerprint: str) -> dict:
    """Every number FraudEnsemble.score() uses, as plain JSON."""
    m = model.base.models
    out = {"format": EXPORT_FORMAT, "training_fingerprint": fingerprint,
           "normalize": bool(fraud_logic.NORMALIZE_INPUT), "features": list(model.base.columns_)}

    sc, lr = m["lr"].named_steps["scaler"], m["lr"].named_steps["clf"]
    out["lr"] = {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
                 "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}

    sc, rf = m["rf"].named_steps["scaler"], m["rf"].named_steps["clf"]
    trees = []
    for e in rf.estimators_:
        v = e.tree_.value[:, 0, :]
        # DecisionTreeClassifier.predict_proba divides each leaf by its sum
        p = v[:, 1] / np.where(v.sum(axis=1) == 0, 1.0, v.sum(axis=1))
        trees.append(_export_tree(e, p))
    out["rf"] = {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "trees": trees}

    sc, gb = m["gb"].named_steps["scaler"], m["gb"].named_steps["clf"]
    init = float(gb._raw_predict_init(np.zeros((1, len(sc.mean_)), dtype=np.float32))[0, 0])
    out["gb"] = {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "init": init,
                 "learning_rate": float(gb.learning_rate),
                 "trees": [_export_tree(e, e.tree_.value[:, 0, 0]) for e in gb.estimators_[:, 0]]}

    tf, nb = m["tfidf"].named_steps["tfidf"], m["tfidf"].named_steps["clf"]
    assert tf.analyzer == "word" and tf.norm == "l2" and not tf.sublinear_tf and tf.lowercase
    assert tf.token_pattern == r"(?u)\b\w\w+\b" and tf.ngram_range == TFIDF_NGRAM_RANGE
    delta = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
    out["tfidf"] = {"ngram_range": list(tf.ngram_range),
                    "prior": float(nb.class_log_prior_[1] - nb.class_log_prior_[0]),
                    "vocab": {g: [float(tf.idf_[i]), float(delta[i])] for g, i in sorted(tf.vocabulary_.items())}}

    tf, clf = m["char"].named_steps["tfidf"], m["char"].named_steps["clf"]
    assert tf.analyzer == "char_wb" and tf.norm == "l2" and tf.sublinear_tf and tf.lowercase
    out["char"] = {"ngram_range": list(tf.ngram_range), "intercept": float(clf.intercept_[0]),
                   "vocab": {g: [float(tf.idf_[i]), float(clf.coef_[0][i])] for g, i in sorted(tf.vocabulary_.items())}}

    if "mlp" in m:
        out["mlp"] = _export_mlp(m["mlp"], m["char"])

    w = model.stacker.coef_[0]
    out["stacker"] = {"inputs": list(STACK_INPUTS), "coef": [float(x) for x in w],
                      "intercept": float(model.stacker.intercept_[0])}
    return out
