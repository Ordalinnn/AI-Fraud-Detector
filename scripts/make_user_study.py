"""
Prepares the user study "does the system help people spot scams?".

    python scripts/make_user_study.py

Reads docs/user_study/messages.csv (20 messages written for the study, not
in the training data: sets A and B, each 5 scam + 5 ordinary, 7 Russian +
3 Kazakh) and writes:

  docs/user_study/model_verdicts.csv   what the web app's model says about
                                       each message (probability, verdict,
                                       reasons) - the REAL output, mistakes
                                       included
  docs/user_study/cards.html           printable sheets: instructions and
                                       consent, set A/B without and with the
                                       system's hint, answer boxes
  docs/user_study/answers_TEMPLATE.csv where the answers are typed in

Then scripts/analyze_user_study.py computes the result.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fraud_eval as fe  # noqa: E402
import fraud_model as fm  # noqa: E402
import translations  # noqa: E402
from text_normalize import is_disguised  # noqa: E402

DIR = ROOT / "docs" / "user_study"
T = translations.TEXT["🇷🇺 RU"]

# same rule as app.py explain()
IRRELEVANT = {"text_length", "word_count", "avg_word_length"}
MIN_THRESHOLDS = {"digit_count": 8, "exclamation_count": 2, "uppercase_count": 5, "url_count": 2}


def reasons(text: str, feats: dict) -> list[str]:
    out = [T["feat_disguised"]] if is_disguised(text) else []
    out += [T[f"feat_{k}"] for k, v in feats.items()
            if v > 0 and k not in IRRELEVANT and f"feat_{k}" in T and v >= MIN_THRESHOLDS.get(k, 1)]
    return out


def verdicts(msgs: pd.DataFrame) -> pd.DataFrame:
    df = fe.load_dataset()
    real = fm.load_real_training_sample()
    texts = df["text"].tolist() + [t for t, _ in real]
    y = np.concatenate([df["label"].to_numpy(), [label for _, label in real]])
    overlap = set(msgs["text"]) & set(texts)
    if overlap:
        raise SystemExit(f"study messages must not be in the training data: {overlap}")
    model = fm.FraudEnsemble().fit(texts, y)
    feats = fm.featurise(msgs["text"])
    prob = model.score(msgs["text"].tolist(), feats)["final"]
    out = msgs[["id", "set", "label"]].copy()
    out["model_probability"] = np.round(prob, 3)
    out["model_says_fraud"] = (prob >= 0.5).astype(int)
    out["model_correct"] = (out["model_says_fraud"] == out["label"]).astype(int)
    out["reasons"] = [" | ".join(reasons(t, f)) for t, f in zip(msgs["text"], feats.to_dict("records"))]
    return out


CSS = """
body { font-family: Arial, sans-serif; font-size: 11pt; color: #111; max-width: 780px; margin: 24px auto; padding: 0 16px; }
h1 { font-size: 17pt; } h2 { font-size: 13.5pt; margin: 0 0 4px; }
.page { page-break-after: always; break-after: page; padding-bottom: 24px; }
.msg { border: 1px solid #999; border-radius: 8px; padding: 5px 10px; margin: 5px 0; page-break-inside: avoid; }
.num { font-weight: bold; }
.text { margin: 2px 0 5px; white-space: pre-wrap; }
.hint { border-left: 4px solid #555; background: #f3f3f3; padding: 4px 8px; margin: 4px 0 5px; font-size: 10.5pt; }
.ans { font-size: 10.5pt; }
.box { display: inline-block; width: 14px; height: 14px; border: 1.5px solid #222; vertical-align: -2px; margin: 0 4px 0 10px; }
.small { font-size: 11pt; color: #333; }
table { border-collapse: collapse; } td, th { border: 1px solid #999; padding: 4px 8px; }
@media print { body { margin: 0 auto; } }
"""


def answer_line() -> str:
    conf = "".join(f'<span class="box"></span>{i}' for i in range(1, 6))
    return (f'<div class="ans">Это мошенничество? <span class="box"></span>Да <span class="box"></span>Нет'
            f'&nbsp;&nbsp;&nbsp; Уверенность: {conf}</div>')


def sheet(set_name: str, with_hint: bool, msgs: pd.DataFrame, ver: pd.DataFrame) -> str:
    title = f"Набор {set_name} — {'С ПОДСКАЗКОЙ системы' if with_hint else 'БЕЗ подсказки'}"
    parts = [f'<div class="page"><h2>{title}</h2>',
             '<p class="small">Участник №: ______ &nbsp;&nbsp; Группа: ______. '
             'Отметьте для каждого сообщения: мошенничество или нет, и насколько вы уверены '
             '(1 — совсем не уверен, 5 — полностью уверен).'
             + (' Под сообщением — что сказала система AI Fraud Detector. Система может ошибаться: решайте сами.'
                if with_hint else '') + '</p>']
    for k, (_, m) in enumerate(msgs[msgs["set"] == set_name].iterrows(), 1):
        parts.append(f'<div class="msg"><div class="num">{k}. ({m["id"]})</div>'
                     f'<div class="text">{html.escape(m["text"])}</div>')
        if with_hint:
            v = ver[ver["id"] == m["id"]].iloc[0]
            pct = int(round(v["model_probability"] * 100))
            verdict = "⚠️ Похоже на мошенничество" if v["model_says_fraud"] else "✅ Признаков мошенничества мало"
            # reasons only next to a warning, like the app's alert
            why = v["reasons"].split(" | ")[:3] if (v["model_says_fraud"] and isinstance(v["reasons"], str)) else []
            parts.append(f'<div class="hint"><b>Система: {verdict} ({pct}%)</b>'
                         + (f'<br>Почему: {html.escape("; ".join(why))}' if why else '') + '</div>')
        parts.append(answer_line() + '</div>')
    parts.append('</div>')
    return "\n".join(parts)


def instructions() -> str:
    return """<div class="page">
<h1>Исследование: помогает ли AI Fraud Detector распознавать мошенничество</h1>
<h2>Для проводящего (ученика)</h2>
<ol>
<li><b>Участники:</b> 16–20 человек, лучше разного возраста, в том числе старше 50 лет.
Несовершеннолетних — только с согласия родителей. Имена не записываем: только номер участника и возрастную группу
(до 18, 18–49, 50+).</li>
<li><b>Группы:</b> нечётные номера участников — группа 1, чётные — группа 2.
<br>Группа 1: сначала <b>набор A без подсказки</b>, потом <b>набор B с подсказкой</b>.
<br>Группа 2: сначала <b>набор B без подсказки</b>, потом <b>набор A с подсказкой</b>.
<br>Так каждое сообщение проверяется и с подсказкой, и без, а разница между наборами не влияет на результат.</li>
<li>Не подсказывайте и не обсуждайте ответы. После обоих листов можно рассказать правильные ответы — это тоже польза.</li>
<li>Ответы перенесите в <code>answers.csv</code> (образец <code>answers_TEMPLATE.csv</code>):
одна строка = один ответ на одно сообщение. Затем: <code>python scripts/analyze_user_study.py docs/user_study/answers.csv</code></li>
</ol>
<h2>Согласие участника (зачитать)</h2>
<p class="small">«Мы проводим школьное исследование о распознавании мошеннических сообщений. Вам покажут 20 коротких сообщений
и попросят отметить, какие из них мошеннические. Это займёт около 10 минут. Мы не записываем ваше имя и телефон,
результаты будут только в обобщённом виде. Вы можете отказаться в любой момент». Участник согласился: да / нет.</p>
<table><tr><th>№ участника</th><th>Группа</th><th>Возраст: до 18 / 18–49 / 50+</th><th>Согласие</th></tr>
<tr><td>&nbsp;</td><td></td><td></td><td></td></tr></table>
</div>"""


def main():
    msgs = pd.read_csv(DIR / "messages.csv", dtype=str)
    msgs["label"] = msgs["label"].astype(int)
    ver = verdicts(msgs)
    ver.to_csv(DIR / "model_verdicts.csv", index=False)
    body = [instructions()]
    for s in ("A", "B"):
        body.append(sheet(s, False, msgs, ver))
    for s in ("A", "B"):
        body.append(sheet(s, True, msgs, ver))
    page = (f'<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Исследование AI Fraud Detector</title>'
            f'<style>{CSS}</style></head><body>{"".join(body)}</body></html>')
    (DIR / "cards.html").write_text(page, encoding="utf-8")
    template = pd.DataFrame([
        {"participant": 1, "group": 1, "age_group": "18-49", "message_id": "A1", "condition": "without",
         "answer": 1, "confidence": 4},
        {"participant": 1, "group": 1, "age_group": "18-49", "message_id": "B1", "condition": "with",
         "answer": 1, "confidence": 5},
    ])
    template.to_csv(DIR / "answers_TEMPLATE.csv", index=False)
    print(ver.to_string(index=False))
    print(f"model correct on {ver['model_correct'].sum()}/{len(ver)} study messages")


if __name__ == "__main__":
    main()
