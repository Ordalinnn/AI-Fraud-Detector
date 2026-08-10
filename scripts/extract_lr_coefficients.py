"""Regenerates the embedded LogisticRegression coefficients used by
browser-extension/detector.js's lrScore().

Why this exists: detector.js can't load scikit-learn or run app.py's full
4-model ensemble (LR + RF + GB + TF-IDF/NB) inside a browser extension, so
it ports only the Logistic Regression member — a StandardScaler +
LogisticRegression pipeline is just a dot product + sigmoid, cheap to
hand-port. This script trains that exact pipeline (same hyperparameters as
app.py's _build_ensemble_pipelines(): C=1.0, max_iter=1000, random_state=42)
on the FULL training dataset embedded in app.py's `data` list, using
fraud_logic.extract_features() so the feature vector is byte-for-byte the
same one the real production model is trained on. It then prints the
feature order, StandardScaler mean_/scale_, and LogisticRegression
coef_/intercept_ as ready-to-paste JS array literals.

app.py's `data` list can't be imported directly without triggering
st.set_page_config() and a live Streamlit script context (slow, and fails
outside `streamlit run`), so this parses just the `data = [...]` literal
out of the source text via the AST module instead of importing app.py.

Run from the repo root: python scripts/extract_lr_coefficients.py
Re-run and re-paste into detector.js whenever fraud_logic.extract_features()
changes shape or app.py's training dataset changes size/content.
"""
import ast
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fraud_logic import extract_features  # noqa: E402

# Must match app.py's _build_ensemble_pipelines() LR hyperparameters exactly,
# so the ported coefficients reflect the same model the real app trains.
LR_C = 1.0
LR_MAX_ITER = 1000
MODEL_RANDOM_STATE = 42


def _extract_dataset(app_py_source):
    marker = "\ndata = ["
    start = app_py_source.index(marker) + 1
    bracket_start = app_py_source.index("[", start)
    depth = 0
    i = bracket_start
    while True:
        if app_py_source[i] == "[":
            depth += 1
        elif app_py_source[i] == "]":
            depth -= 1
            if depth == 0:
                break
        i += 1
    return ast.literal_eval(app_py_source[bracket_start:i + 1])


def _js_array(values):
    def literal(v):
        return f'"{v}"' if isinstance(v, str) else repr(v)
    return "[" + ", ".join(literal(v) for v in values) + "]"


def main():
    app_py_source = (REPO_ROOT / "app.py").read_text(encoding="utf-8")
    data = _extract_dataset(app_py_source)
    print(f"// dataset size: {len(data)} examples", file=sys.stderr)

    rows = [extract_features(text)[0] for text, _label in data]
    labels = [label for _text, label in data]
    X = pd.DataFrame(rows)
    feature_order = list(X.columns)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=LR_C, max_iter=LR_MAX_ITER, random_state=MODEL_RANDOM_STATE)),
    ])
    pipe.fit(X, labels)
    scaler = pipe.named_steps["scaler"]
    clf = pipe.named_steps["clf"]

    def camel(snake):
        head, *rest = snake.split("_")
        return head + "".join(w.capitalize() for w in rest)

    print("const LR_FEATURE_ORDER = " + _js_array([camel(k) for k in feature_order]) + ";")
    print("const LR_MEAN = " + _js_array([round(float(v), 6) for v in scaler.mean_]) + ";")
    print("const LR_SCALE = " + _js_array([round(float(v), 6) for v in scaler.scale_]) + ";")
    print("const LR_COEF = " + _js_array([round(float(v), 6) for v in clf.coef_[0]]) + ";")
    print(f"const LR_INTERCEPT = {round(float(clf.intercept_[0]), 6)!r};")


if __name__ == "__main__":
    main()
