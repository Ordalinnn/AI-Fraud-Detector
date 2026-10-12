"""
Idea tested (and NOT shipped): judge the last few messages from one sender
together instead of each message alone, to catch scams split into several
short messages.

    python scripts/eval_conversation_window.py

Same honest set-up as the jury evaluation (group-aware split, models
trained without the test messages). Simulated conversations:

  scam split into pieces   each test scam cut into 5-word messages
  scam after 2 ordinary    2 real ordinary SMS from the same sender, then the scam
  scam alone               the test scam as one message
  ordinary chat (real)     5 real ordinary SMS (UCI hold-out) in a row
  ordinary chat (test)     5 ordinary test messages in a row

Policies (an alert if ANY message in the conversation triggers):
  single   each message alone (what the apps do)
  window   the newest message joined with the 2 before it
  either   single OR window

Writes reports/conversation_window/{results.json, REPORT_RU.md}.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fraud_eval as fe  # noqa: E402
import fraud_model as fm  # noqa: E402

OUT = ROOT / "reports" / "conversation_window"
WINDOW = 3
PIECE_WORDS = 5
POLICIES = ["single", "window", "either"]
SYSTEMS = {"final": "Сайт (ансамбль)", "char": "Телефон и расширение"}


def pieces(text: str, k: int = PIECE_WORDS) -> list[str]:
    w = text.split()
    return [" ".join(w[i:i + k]) for i in range(0, len(w), k)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = fe.load_dataset()
    y = df["label"].to_numpy()
    groups = fm.template_groups(df["text"].tolist())
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=fm.EVAL_TEST_SIZE, random_state=fm.MODEL_RANDOM_STATE)
                  .split(df, y, groups))
    extra = pd.DataFrame(fm.load_real_training_sample(), columns=["text", "label"])
    T = [df["text"].iloc[i] for i in tr] + extra["text"].tolist()
    Y = np.concatenate([y[tr], extra["label"].to_numpy().astype(int)])
    ens = fm.FraudEnsemble().fit(T, Y)

    te_texts = [df["text"].iloc[i] for i in te]
    scams = [t for t, l in zip(te_texts, y[te]) if l == 1]
    test_ham = [t for t, l in zip(te_texts, y[te]) if l == 0]
    hold = fm.load_real_holdout()
    real_ham = [t for t, l in zip(hold["text"], hold["label"]) if l == 0]
    rng = random.Random(0)
    cases = {
        "scam_split": ("Мошенничество, разбитое на сообщения по 5 слов", 1, [pieces(t) for t in scams]),
        "scam_after_ordinary": ("Мошенничество после 2 обычных сообщений", 1, [rng.sample(real_ham, 2) + [t] for t in scams]),
        "scam_alone": ("Мошенничество одним сообщением", 1, [[t] for t in scams]),
        "ordinary_real": ("Обычная переписка: 5 реальных SMS", 0, [rng.sample(real_ham, 5) for _ in range(500)]),
        "ordinary_test": ("Обычная переписка: 5 обычных сообщений теста", 0, [rng.sample(test_ham, 5) for _ in range(500)]),
    }

    # score every distinct text once
    all_texts = set()
    for _, _, convs in cases.values():
        for c in convs:
            all_texts.update(c)
            all_texts.update("\n".join(c[max(0, i - WINDOW + 1):i + 1]) for i in range(len(c)))
    all_texts = sorted(all_texts)
    s = ens.score(all_texts)
    prob = {k: dict(zip(all_texts, s[k])) for k in SYSTEMS}

    res = {"window": WINDOW, "piece_words": PIECE_WORDS, "cases": {}}
    for key, (name, label, convs) in cases.items():
        row = {"name": name, "label": label, "n": len(convs)}
        for sysk in SYSTEMS:
            p = prob[sysk]
            single = np.array([max(p[m] for m in c) for c in convs]) >= 0.5
            window = np.array([max(p["\n".join(c[max(0, i - WINDOW + 1):i + 1])] for i in range(len(c))) for c in convs]) >= 0.5
            row[sysk] = {"single": float(single.mean()), "window": float(window.mean()),
                         "either": float((single | window).mean())}
        res["cases"][key] = row
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    def line(key):
        r = res["cases"][key]
        what = "поймано" if r["label"] == 1 else "ложная тревога"
        return (f"| {r['name']} ({what}) | " + " | ".join(
            f"{r[k][p] * 100:.0f}%" for k in SYSTEMS for p in POLICIES) + " |")

    c = res["cases"]
    md = f"""# Проверка идеи: оценивать несколько сообщений вместе

**Идея.** Мошенники иногда присылают текст по частям: «Мама, это я» → «попал в беду» → «переведи срочно». Каждая часть по отдельности выглядит безобидно. Может быть, лучше оценивать последние {WINDOW} сообщения от одного отправителя вместе?

**Как проверяли.** Тот же честный тест, что и для жюри. Переписки смоделированы: мошеннические сообщения теста разрезаны на части по {PIECE_WORDS} слов, или перед мошенническим сообщением стоят 2 обычных; обычные переписки — по 5 обычных сообщений подряд. Тревога засчитывается, если сработало хотя бы на одном шаге переписки.

| Переписка | Сайт: по одному | Сайт: окно | Сайт: любое | Телефон: по одному | Телефон: окно | Телефон: любое |
|---|---|---|---|---|---|---|
{chr(10).join(line(k) for k in res['cases'])}

## Вывод: идею НЕ внедрили

- **Плюс:** на сообщениях, разбитых на части, «любое из двух» ловит больше: телефон {c['scam_split']['char']['single'] * 100:.0f}% → {c['scam_split']['char']['either'] * 100:.0f}%, сайт {c['scam_split']['final']['single'] * 100:.0f}% → {c['scam_split']['final']['either'] * 100:.0f}%.
- **Минус 1:** только «окно» без проверки по одному пропускает мошенничество после обычных сообщений ({c['scam_after_ordinary']['char']['window'] * 100:.0f}% вместо {c['scam_after_ordinary']['char']['single'] * 100:.0f}% на телефоне): обычный текст «разбавляет» мошеннический.
- **Минус 2:** «любое из двух» добавляет ложные тревоги на обычных переписках: на трудных обычных сообщениях теста сайт {c['ordinary_test']['final']['single'] * 100:.0f}% → {c['ordinary_test']['final']['either'] * 100:.0f}%, телефон {c['ordinary_test']['char']['single'] * 100:.0f}% → {c['ordinary_test']['char']['either'] * 100:.0f}%; на реальных SMS сайт {c['ordinary_real']['final']['single'] * 100:.1f}% → {c['ordinary_real']['final']['either'] * 100:.1f}%.
- Переписки здесь **смоделированы**. Настоящие многошаговые схемы (знакомства, «инвестиции») длятся днями и устроены иначе. Для честной проверки нужны реальные переписки — их можно собирать кнопкой «Это ошибка» (см. `data/real/README.md`).

Выигрыш на разбитых сообщениях есть, но за него платим заметным ростом ложных тревог, а сами переписки искусственные. Поэтому приложения по-прежнему проверяют каждое сообщение отдельно. Это тоже результат: идея проверена до внедрения, а не после.

Перезапуск: `python scripts/eval_conversation_window.py`.
"""
    (OUT / "REPORT_RU.md").write_text(md, encoding="utf-8")
    for k, r in res["cases"].items():
        print(f"{k:22s}", {s_: {p: round(r[s_][p], 3) for p in POLICIES} for s_ in SYSTEMS})


if __name__ == "__main__":
    main()
