"""The browser extension must give the same result as the site.

The extension runs an exported copy of the site's model
(browser-extension/ensemble_model.json, scripts/export_extension_model.py)
and browser-extension/tests/ensemble.test.js proves the JavaScript
reproduces the golden file (ensemble_golden.json) to < 1e-9. These tests
close the loop on the Python side:

  1. the exported model is trained on today's training set (else re-export);
  2. a model trained right now gives the golden numbers;
  3. the real app.py, run headless, gives the golden numbers and shows the
     same percentage - and a feedback.json on the server changes nothing.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import fraud_model as fm
from fraud_logic import risk_percent

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "browser-extension"
MODEL = json.loads((EXT / "ensemble_model.json").read_text(encoding="utf-8"))
GOLDEN = json.loads((EXT / "tests" / "ensemble_golden.json").read_text(encoding="utf-8"))
RERUN = "re-run: python scripts/export_extension_model.py"


def _exporter():
    spec = importlib.util.spec_from_file_location("export_extension_model", ROOT / "scripts" / "export_extension_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fresh_model():
    texts, y = _exporter().training_set()
    return fm.FraudEnsemble().fit(texts, y), fm.training_fingerprint(texts, y)


def test_extension_model_is_trained_on_the_current_training_set():
    texts, y = _exporter().training_set()
    assert MODEL["training_fingerprint"] == fm.training_fingerprint(texts, y), RERUN
    assert GOLDEN["training_fingerprint"] == MODEL["training_fingerprint"], RERUN
    assert MODEL["features"] == list(fm.featurise([""]).columns), RERUN


def test_model_trained_now_gives_the_golden_numbers(fresh_model):
    model, fp = fresh_model
    assert fp == MODEL["training_fingerprint"], RERUN
    texts = GOLDEN["texts"]
    feats = fm.featurise(texts)
    assert feats.to_numpy().tolist() == GOLDEN["x"], RERUN
    s = model.score(texts, feats)
    for k, want in GOLDEN["scores"].items():
        diff = float(np.max(np.abs(np.asarray(s[k]) - np.asarray(want))))
        assert diff < 1e-6, f"{k}: {diff} - {RERUN}"
    assert [risk_percent(float(p)) for p in s["final"]] == GOLDEN["pct"]


def test_exported_numbers_are_the_fresh_model(fresh_model):
    model, fp = fresh_model
    again = fm.export_ensemble(model, fp)
    assert again["stacker"]["coef"] == pytest.approx(MODEL["stacker"]["coef"], abs=1e-9), RERUN
    assert again["lr"]["coef"] == pytest.approx(MODEL["lr"]["coef"], abs=1e-9), RERUN
    assert set(again["tfidf"]["vocab"]) == set(MODEL["tfidf"]["vocab"]), RERUN
    assert set(again["char"]["vocab"]) == set(MODEL["char"]["vocab"]), RERUN
    for a, b in zip(again["rf"]["trees"] + again["gb"]["trees"], MODEL["rf"]["trees"] + MODEL["gb"]["trees"]):
        assert a["feature"] == b["feature"] and a["left"] == b["left"], RERUN


# A few golden texts of each kind, scored by the real site.
SITE_SAMPLE = [0, 4, 9, 11, 14, 17, 19, 22, 26]


def test_the_site_gives_the_extension_numbers(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)
    # Feedback on the server must NOT change the site's model (it used to be
    # trained on it, which the extension could never match).
    (tmp_path / "feedback.json").write_text(json.dumps([
        {"Text": "Привет, урок перенесли на пятницу", "Predicted": "SAFE", "UserSaysCorrect": False, "Session": "s"},
        {"Text": "Ваша карта заблокирована, код из SMS", "Predicted": "FRAUD", "UserSaysCorrect": False, "Session": "s"},
    ], ensure_ascii=False), encoding="utf-8")

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    assert at.session_state  # app started

    texts = [t for t in GOLDEN["texts"] if t.strip()]
    sample = [texts[i] for i in SITE_SAMPLE] + texts[100:103] + texts[-3:]
    for text in sample:
        i = GOLDEN["texts"].index(text)
        at.text_area(key="main_input_text").set_value(text)
        [b for b in at.button if "analysis_form" in (b.key or "")][0].click().run()
        assert not at.exception, at.exception
        r = at.session_state["last_result"]
        assert abs(r["prob"] - GOLDEN["scores"]["final"][i]) < 1e-9, text
        assert risk_percent(r["prob"]) == GOLDEN["pct"][i]
        shown = " ".join(m.value for m in at.markdown)
        assert f'{GOLDEN["pct"][i]}%' in shown, text
        assert f'model {MODEL["training_fingerprint"][:8]}' in " ".join(c.value for c in at.caption)
