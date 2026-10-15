"""
Exports the web app's FULL model (5 base models + stacker, fraud_model.py)
for the browser extension, so the site and the extension give the same
probability for the same text.

    python scripts/export_extension_model.py

Trains exactly what app.py trains (app.py's `data` + the real SMS sample,
same order, same hyperparameters, same random_state), then writes:

  browser-extension/ensemble_model.json        every number score() uses
  static/check/lib/ (scripts/build_web_checker.py) the same model for the
                                               offline phone page
  browser-extension/tests/ensemble_golden.json texts + scikit-learn's
                                               features and probabilities;
                                               tests/ensemble.test.js must
                                               reproduce them (< 1e-6)

Re-run whenever app.py's dataset, fraud_logic.extract_features() or
fraud_model.py changes; tests/test_extension_ensemble.py fails until then.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fraud_eval as fe  # noqa: E402
import fraud_model as fm  # noqa: E402
from fraud_logic import risk_percent  # noqa: E402

MODEL_PATH = ROOT / "browser-extension" / "ensemble_model.json"
GOLDEN_PATH = ROOT / "browser-extension" / "tests" / "ensemble_golden.json"

# Texts where Python and JavaScript string handling differ (Unicode digits,
# whitespace, upper case, word boundaries, rounding ties) - the port has to
# get every one of them right.
TRICKY = [
    "", "   ", "a", "ab", "😀 Ваша карта заблокирована 😀",
    "Сіздің картаңыз бұғатталды!!", "КОД   из\tSMS\n\nhttp://kaspi-login.xyz",
    "İstanbul ÇAĞRI", "URGENT!!! send OTP now http://bit.ly/x",
    "Ваша к@рта зaблoкирoвана, назовите к0д из С М С", "к о д ы", "ＳＭＳ ｋｏｄ", "к​о​д",
    # Unicode digits: superscripts, circled, Arabic-Indic, Devanagari, fullwidth
    "Код ²³¹ ① ⑳ ٣٤٥ ४२ ０１２ 𝟓", "сумма ½ ¾ Ⅻ ↉ 〇",
    # whitespace Python counts and JavaScript does not (and the reverse)
    "срочно\x1cкод\x1dкарта\x1e\x1fбанк", "код\x85sms᠎пароль﻿пин",
    "http://a.ru\x1cкод www.b.kz\x85x ﻿http://c.xyz",
    # upper case without a lower-case pair, circled letters, titlecase
    "ϒ ϔ Ⓐ Ⓑ ǅ ǈ ℌ ℍ 𝐀𝐁", "Σ ΣΑΣ ΌΣΟΣ",
    # word boundaries next to non-Latin letters and marks
    "кkaspi.kz жkaspi-login.xyz _kaspi.kz 1kaspi.kz kaspi.kzж", "й̆срочно е́код",
    "ſecure-login.com Kaspi.kz", "Kaspi.KZ WWW.EGOV.KZ/x HTTPS://PAYPAL.COM.verify.xyz",
    # avg_word_length rounding ties (41/8 = 5.125, 43/8 = 5.375, 5/8)
    "abcde abcde abcde abcde abcde abcde abcde abcdef",
    "abcde abcde abcde abcde abcde abcde abcdef abcdefg",
    "a a a a a a a bb", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa b",
    # tokens TF-IDF must split the same way (\\w\\w+ in Python's sense)
    "snake_case_word x_y 12_34 a1 ٣٤ ²³ éé ⅫⅫ 𝐀𝐁 ok!ok?ok", "don't stop-it e-mail co-op",
]


PCT_PROBES = [0.0, 0.0025, 0.0075, 0.0125, 0.5, 0.50125, 0.12345, 0.87349999, 0.8735, 0.99875,
              0.9989, 0.999, 0.9995, 1.0, 1e-9, 0.333333333, 0.6666666666]


def _attacks():
    spec = importlib.util.spec_from_file_location("eval_obfuscation", ROOT / "scripts" / "eval_obfuscation.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev.ATTACKS


def training_set():
    """Exactly app.py's training_data: `data` + the real SMS sample."""
    df = fe.load_dataset()
    real = fm.load_real_training_sample()
    texts = df["text"].tolist() + [t for t, _ in real]
    y = np.array(df["label"].tolist() + [label for _, label in real])
    return texts, y


def main():
    texts, y = training_set()
    model = fm.FraudEnsemble().fit(texts, y)
    fp = fm.training_fingerprint(texts, y)
    exported = fm.export_ensemble(model, fp)
    MODEL_PATH.write_text(json.dumps(exported, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")

    hold = fm.load_real_holdout()["text"].tolist()
    attacks = _attacks()
    rng = np.random.default_rng(0)
    scams = [t for t, label in zip(texts, y) if label == 1]
    disguised = []
    for k, i in enumerate(rng.choice(len(scams), 150, replace=False)):
        name = list(attacks)[1 + k % (len(attacks) - 1)]
        disguised.append(attacks[name](scams[i]))
    golden_texts = TRICKY + texts + hold + disguised

    feats = fm.featurise(golden_texts)
    s = model.score(golden_texts, feats)
    golden = {
        "training_fingerprint": fp,
        "features": list(feats.columns),
        "texts": golden_texts,
        "x": feats.to_numpy().tolist(),
        "scores": {k: [float(v) for v in s[k]] for k in fm.STACK_INPUTS + ["final"]},
        "pct": [risk_percent(float(v)) for v in s["final"]],
        # rounding ties and the 99.9 cap, where JavaScript's toFixed differs
        "pct_probe": [[p, risk_percent(p)] for p in PCT_PROBES],
    }
    GOLDEN_PATH.write_text(json.dumps(golden, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{MODEL_PATH.relative_to(ROOT)}: {MODEL_PATH.stat().st_size:,} bytes, fingerprint {fp}")
    print(f"{GOLDEN_PATH.relative_to(ROOT)}: {len(golden_texts)} texts")

    # the phone page (static/check/) runs the same model: refresh its copy
    spec = importlib.util.spec_from_file_location("build_web_checker", ROOT / "scripts" / "build_web_checker.py")
    web = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(web)
    web.main(check=False)


if __name__ == "__main__":
    main()
