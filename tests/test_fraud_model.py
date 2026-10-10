"""
Tests for fraud_model.py (the stacked ensemble) and fraud_eval.py (the
jury-evaluation toolkit). No Streamlit needed.
"""
import numpy as np
import pytest
from sklearn.model_selection import GroupShuffleSplit

import fraud_eval as fe
import fraud_model as fm


@pytest.fixture(scope="module")
def dataset():
    df = fe.load_dataset()
    feats = fm.featurise(df["text"])
    groups = fm.template_groups(df["text"].tolist())
    return df, feats, groups


# ---------- dataset loading ----------

def test_dataset_is_read_from_app_py(dataset):
    df, _, _ = dataset
    assert len(df) > 1000
    assert set(df["label"].unique()) == {0, 1}
    assert set(df["language"].unique()) <= {"kk", "ru", "en", "other"}
    # every example belongs to a FRAUD:/SAFE: section of app.py
    assert (df["category"] != "uncategorised").mean() > 0.95
    # categories agree with labels
    assert df.loc[df["category"].str.startswith("FRAUD"), "label"].mean() > 0.95


@pytest.mark.parametrize("text,lang", [
    ("Сіздің картаңыз бұғатталды", "kk"),
    ("Ваша карта заблокирована", "ru"),
    ("Your card is blocked", "en"),
    ("12345 !!!", "other"),
])
def test_detect_language(text, lang):
    assert fe.detect_language(text) == lang


# ---------- real messages ----------

def test_real_training_sample_and_holdout_do_not_overlap():
    sample = fm.load_real_training_sample()
    hold = fm.load_real_holdout()
    assert len(sample) == fm.REAL_TRAIN_N_HAM + fm.REAL_TRAIN_N_SPAM
    assert sum(label for _, label in sample) == fm.REAL_TRAIN_N_SPAM
    assert not ({t for t, _ in sample} & set(hold["text"]))
    assert len(hold) > 2000


def test_missing_real_sample_file_is_not_fatal(tmp_path):
    assert fm.load_real_training_sample(tmp_path / "nope.csv") == []


# ---------- the ensemble ----------

def test_ensemble_scores_are_probabilities(dataset):
    df, feats, groups = dataset
    idx = np.arange(0, len(df), 3)  # a third of the data keeps the test fast
    ens = fm.FraudEnsemble(folds=3).fit(df["text"].iloc[idx].tolist(), df["label"].to_numpy()[idx],
                                        feats.iloc[idx], groups[idx])
    msgs = ["Срочно! Ваша карта заблокирована, отправьте код из SMS http://kaspi-login.xyz",
            "Привет, завтра урок математики в 9:00. Не забудь тетрадь."]
    s = ens.score(msgs)
    for k in fm.BASE_MODELS + ["average", "final"]:
        assert s[k].shape == (2,)
        assert np.all((s[k] >= 0) & (s[k] <= 1))
    assert s["final"][0] > 0.5 > s["final"][1]
    assert set(ens.weights) == set(fm.STACK_INPUTS)


def test_stacked_ensemble_beats_old_plain_average_on_jury_split(dataset):
    """Regression guard for the jury's key question: on the same held-out
    grouped test, the new ensemble must not be worse than the old plain
    average of 4 models + rules (it was 82.3% vs ~95% when this was written)."""
    df, feats, groups = dataset
    y = df["label"].to_numpy()
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=fm.EVAL_TEST_SIZE, random_state=fm.MODEL_RANDOM_STATE)
                  .split(feats, y, groups))
    s = fe.score_split(df, feats, groups, tr, te)
    acc_new = fe.metrics(y[te], s["final"])["accuracy"]
    acc_old = fe.metrics(y[te], s["old_ensemble_rules"])["accuracy"]
    assert acc_new > acc_old + 0.05


# ---------- statistics helpers ----------

def test_mcnemar_counts_disagreements_only():
    y = np.array([1, 1, 1, 1, 0, 0])
    a = np.array([1, 1, 1, 1, 0, 0])   # all right
    b = np.array([1, 1, 0, 0, 0, 1])   # 3 wrong
    r = fe.mcnemar_exact(y, a, b)
    assert (r["a_only_right"], r["b_only_right"]) == (3, 0)
    assert 0 < r["p_value"] <= 1


def test_perfectly_calibrated_scores_have_zero_ece():
    y = np.array([0] * 9 + [1] + [0] * 1 + [1] * 9)
    p = np.array([0.1] * 10 + [0.9] * 10)
    assert fe.expected_calibration_error(y, p) == pytest.approx(0.0, abs=1e-9)


def test_threshold_table_cost():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.9])
    t = fe.threshold_table(y, p, [0.5], cost_fp=5, cost_fn=1)
    assert int(t["FP"].iloc[0]) == 1 and int(t["FN"].iloc[0]) == 1
    assert t["cost"].iloc[0] == 6


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    a = np.where(rng.random(200) < 0.9, y, 1 - y).astype(float)
    b = np.where(rng.random(200) < 0.7, y, 1 - y).astype(float)
    point, lo, hi = fe.bootstrap_diff_ci(y, a, b, n_boot=300)
    assert lo <= point <= hi
    assert lo > 0  # a is clearly better
