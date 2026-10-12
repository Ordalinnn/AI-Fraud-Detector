"""User study tooling: materials are consistent and the analysis runs.
(The answers here are SIMULATED only to test the code - real answers come
from the students' study, docs/user_study/answers.csv.)"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STUDY = ROOT / "docs" / "user_study"


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_message_sets_are_balanced_and_not_in_training():
    import fraud_eval as fe
    m = pd.read_csv(STUDY / "messages.csv", dtype=str)
    m["label"] = m["label"].astype(int)
    for s in ("A", "B"):
        d = m[m["set"] == s]
        assert len(d) == 10 and d["label"].sum() == 5
        assert (d["language"] == "kk").sum() == 3
    assert not set(m["text"]) & set(fe.load_dataset()["text"])
    v = pd.read_csv(STUDY / "model_verdicts.csv")
    assert list(v["id"]) == list(m["id"])
    html = (STUDY / "cards.html").read_text(encoding="utf-8")
    assert html.count('class="msg"') == 40  # 20 messages, each with and without the hint


def _simulate(seed=0, n=12):
    rng = np.random.default_rng(seed)
    m = pd.read_csv(STUDY / "messages.csv", dtype=str)
    v = pd.read_csv(STUDY / "model_verdicts.csv").set_index("id")
    rows = []
    for p in range(1, n + 1):
        group = 1 if p % 2 else 2
        order = [("A", "without"), ("B", "with")] if group == 1 else [("B", "without"), ("A", "with")]
        for s, cond in order:
            for _, msg in m[m["set"] == s].iterrows():
                truth = int(msg["label"])
                if cond == "with" and rng.random() < 0.8:
                    ans = int(v.loc[msg["id"], "model_says_fraud"])
                else:
                    ans = truth if rng.random() < 0.7 else 1 - truth
                rows.append({"participant": p, "group": group, "age_group": "50+" if p > 8 else "18-49",
                             "message_id": msg["id"], "condition": cond, "answer": ans,
                             "confidence": int(rng.integers(1, 6))})
    return pd.DataFrame(rows)


def test_analysis_runs_and_counts_correctly(tmp_path):
    a = _simulate()
    path = tmp_path / "answers.csv"
    a.to_csv(path, index=False)
    res = _script("analyze_user_study").main([str(path), "--out", str(tmp_path / "out")])
    assert res["n_participants"] == 12 and res["n_answers"] == 240
    assert (tmp_path / "out" / "REPORT_RU.md").exists() and (tmp_path / "out" / "chart.png").exists()
    # following a model that is right 16/20 times beats guessing at 70%
    assert res["accuracy_with"] > res["accuracy_without"]
    # ... except where the model is wrong: there the hint misleads
    assert res["system_wrong"]["with"] < res["system_wrong"]["without"]
    assert set(res["by_age_group"]) == {"18-49", "50+"}
