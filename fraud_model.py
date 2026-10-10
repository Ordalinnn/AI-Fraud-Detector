"""
The trained part of the detector: 5 base models + a learned combiner.

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
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_logic import extract_features, rule_boost

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

BASE_MODELS = ["lr", "rf", "gb", "tfidf", "char"]
FEATURE_MODELS = ["lr", "rf", "gb"]          # use the 20 hand-engineered features
TEXT_MODELS = ["tfidf", "char"]              # read the raw text
STACK_INPUTS = BASE_MODELS + ["boost"]

MODEL_DISPLAY_NAMES = {
    "lr": "Logistic Regression",
    "rf": "Random Forest",
    "gb": "Gradient Boosting",
    "tfidf": "Text Patterns (TF-IDF)",
    "char": "Char n-grams (LR)",
    "average": "Plain average (old)",
    "average_rules": "Plain average + rules (old app)",
    "final": "Stacked ensemble (final)",
}


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


class BaseModels:
    """The 5 base models fitted on one training set."""

    def __init__(self):
        self.models = build_base_models()

    def fit(self, texts, y, feats: pd.DataFrame):
        texts = list(texts)
        y = np.asarray(y)
        for k in FEATURE_MODELS:
            self.models[k].fit(feats, y)
        for k in TEXT_MODELS:
            self.models[k].fit(texts, y)
        self.columns_ = list(feats.columns)
        return self

    def predict(self, texts, feats: pd.DataFrame) -> dict:
        texts = list(texts)
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
