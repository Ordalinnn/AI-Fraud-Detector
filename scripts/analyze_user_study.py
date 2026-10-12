"""
Result of the user study: do people spot scams better WITH the system's
hint than without it?

    python scripts/analyze_user_study.py docs/user_study/answers.csv

answers.csv: one row per answer
    participant, group, age_group, message_id, condition (without|with),
    answer (1 = "fraud", 0 = "not fraud"), confidence (1-5)

Each participant judges one set (A or B) without the hint and the other set
with it, so every person is compared with themselves (paired design), and
the counterbalanced order cancels out any difference between the sets.

Writes reports/user_study/{REPORT_RU.md, results.json, chart.png}:
  - accuracy without vs with, per person; mean difference with a 95%
    bootstrap CI and the Wilcoxon signed-rank test;
  - scams caught and false alarms, without vs with;
  - separately on messages where the SYSTEM WAS WRONG (does the hint lead
    people astray? - over-reliance) and where it was right;
  - by age group; confidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
STUDY = ROOT / "docs" / "user_study"
OUT = ROOT / "reports" / "user_study"
MIN_PARTICIPANTS = 8


def load(path: Path) -> pd.DataFrame:
    a = pd.read_csv(path)
    need = {"participant", "group", "age_group", "message_id", "condition", "answer"}
    missing = need - set(a.columns)
    if missing:
        raise SystemExit(f"{path}: missing columns {sorted(missing)}")
    a["condition"] = a["condition"].str.strip().str.lower()
    bad = set(a["condition"]) - {"without", "with"}
    if bad:
        raise SystemExit(f"condition must be 'without' or 'with', got {bad}")
    msgs = pd.read_csv(STUDY / "messages.csv", dtype=str)
    msgs["label"] = msgs["label"].astype(int)
    ver = pd.read_csv(STUDY / "model_verdicts.csv")
    a = a.merge(msgs[["id", "set", "label", "language", "type"]], left_on="message_id", right_on="id", how="left")
    if a["label"].isna().any():
        raise SystemExit(f"unknown message ids: {sorted(a.loc[a['label'].isna(), 'message_id'].unique())}")
    a = a.merge(ver[["id", "model_says_fraud", "model_correct"]], on="id", how="left")
    a["correct"] = (a["answer"].astype(int) == a["label"]).astype(int)
    dup = a.duplicated(["participant", "message_id"])
    if dup.any():
        raise SystemExit(f"a participant answered the same message twice: {a.loc[dup, ['participant', 'message_id']].values.tolist()}")
    return a


def boot_ci(diffs: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(diffs, size=(n, len(diffs)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def rate(df: pd.DataFrame, cond: str, mask=None) -> float | None:
    d = df[df["condition"] == cond]
    if mask is not None:
        d = d[mask(d)]
    return None if d.empty else float(d["correct"].mean())


def analyze(a: pd.DataFrame) -> dict:
    per = a.pivot_table(index="participant", columns="condition", values="correct", aggfunc="mean")
    per = per.dropna()
    n = len(per)
    diffs = (per["with"] - per["without"]).to_numpy()
    res = {"n_participants": n, "n_answers": int(len(a))}
    res["accuracy_without"] = float(per["without"].mean())
    res["accuracy_with"] = float(per["with"].mean())
    res["mean_difference"] = float(diffs.mean()) if n else None
    res["difference_ci95"] = boot_ci(diffs) if n >= 2 else None
    nz = diffs[diffs != 0]
    res["wilcoxon_p"] = float(wilcoxon(nz).pvalue) if len(nz) >= 2 else None
    res["improved_people"] = int((diffs > 0).sum())
    res["same_people"] = int((diffs == 0).sum())
    res["worse_people"] = int((diffs < 0).sum())
    res["scams_caught"] = {c: rate(a, c, lambda d: d["label"] == 1) for c in ("without", "with")}
    res["ordinary_judged_correctly"] = {c: rate(a, c, lambda d: d["label"] == 0) for c in ("without", "with")}
    res["system_right"] = {c: rate(a, c, lambda d: d["model_correct"] == 1) for c in ("without", "with")}
    res["system_wrong"] = {c: rate(a, c, lambda d: d["model_correct"] == 0) for c in ("without", "with")}
    res["by_age_group"] = {
        str(g): {"n": int(d["participant"].nunique()), "without": rate(d, "without"), "with": rate(d, "with")}
        for g, d in a.groupby("age_group")}
    res["by_language"] = {
        str(g): {"without": rate(d, "without"), "with": rate(d, "with")} for g, d in a.groupby("language")}
    if "confidence" in a.columns:
        res["confidence"] = {c: float(a.loc[a["condition"] == c, "confidence"].mean()) for c in ("without", "with")}
    return res


def pct(x):
    return "—" if x is None else f"{x * 100:.0f}%"


def chart(res: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [("Все сообщения", res["accuracy_without"], res["accuracy_with"]),
            ("Мошеннические (поймано)", res["scams_caught"]["without"], res["scams_caught"]["with"]),
            ("Обычные (не спутали)", res["ordinary_judged_correctly"]["without"], res["ordinary_judged_correctly"]["with"]),
            ("Где система права", res["system_right"]["without"], res["system_right"]["with"]),
            ("Где система ошиблась", res["system_wrong"]["without"], res["system_wrong"]["with"])]
    rows = [r for r in rows if r[1] is not None and r[2] is not None]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    y = np.arange(len(rows))[::-1]
    ax.barh(y + 0.2, [r[1] * 100 for r in rows], height=0.38, color="#cbd5e1", label="Без подсказки")
    ax.barh(y - 0.2, [r[2] * 100 for r in rows], height=0.38, color="#0f766e", label="С подсказкой системы")
    for yy, r in zip(y, rows):
        ax.text(r[1] * 100 + 1, yy + 0.2, pct(r[1]), va="center", fontsize=9, color="#475569")
        ax.text(r[2] * 100 + 1, yy - 0.2, pct(r[2]), va="center", fontsize=9, color="#0f766e", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Правильных ответов людей, %")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.18), ncol=2, frameon=False, fontsize=9)
    fig.suptitle(f"Люди с подсказкой системы и без (участников: {res['n_participants']})", fontsize=11, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(out / "chart.png", dpi=150)


def report(res: dict, out: Path):
    lo, hi = res["difference_ci95"] or (None, None)
    p = res["wilcoxon_p"]
    small = res["n_participants"] < MIN_PARTICIPANTS
    ages = "\n".join(f"| {g} | {d['n']} | {pct(d['without'])} | {pct(d['with'])} |" for g, d in res["by_age_group"].items())
    md = f"""# Исследование с людьми: помогает ли система распознавать мошенничество

{"> **Мало участников (" + str(res['n_participants']) + "):** выводы предварительные, нужно не меньше " + str(MIN_PARTICIPANTS) + "–20 человек." + chr(10) if small else ""}
**Дизайн.** {res['n_participants']} участников, у каждого 10 сообщений без подсказки и 10 других с подсказкой системы (наборы A/B меняются местами в двух группах). Сообщения написаны для исследования и не входили в обучение модели. Подсказка — настоящий ответ модели сайта, включая её ошибки.

## Главный результат

| | Без подсказки | С подсказкой |
|---|---|---|
| Правильных ответов | {pct(res['accuracy_without'])} | **{pct(res['accuracy_with'])}** |
| Мошеннических сообщений распознано | {pct(res['scams_caught']['without'])} | {pct(res['scams_caught']['with'])} |
| Обычных сообщений не спутали с мошенничеством | {pct(res['ordinary_judged_correctly']['without'])} | {pct(res['ordinary_judged_correctly']['with'])} |
| На сообщениях, где система права | {pct(res['system_right']['without'])} | {pct(res['system_right']['with'])} |
| На сообщениях, где система **ошиблась** | {pct(res['system_wrong']['without'])} | {pct(res['system_wrong']['with'])} |

Средняя разница: **{'+' if (res['mean_difference'] or 0) >= 0 else ''}{(res['mean_difference'] or 0) * 100:.0f} п.п.** (95% ДИ: {'—' if lo is None else f'{lo * 100:+.0f} … {hi * 100:+.0f}'} п.п.; критерий Уилкоксона p = {'—' if p is None else f'{p:.3f}'}).
С подсказкой лучше стали {res['improved_people']} чел., так же — {res['same_people']}, хуже — {res['worse_people']}.

![](chart.png)

## По возрасту

| Возраст | Участников | Без подсказки | С подсказкой |
|---|---|---|---|
{ages}

## Как читать

- Если ДИ целиком выше нуля и p < 0,05, подсказка действительно помогает, а не случайно.
- Строка «где система ошиблась» показывает **чрезмерное доверие**: если с подсказкой там стало хуже, люди слепо верят системе. Поэтому в интерфейсе важно объяснять причины, а не только процент.
- Перезапуск: `python scripts/analyze_user_study.py docs/user_study/answers.csv`.
"""
    (out / "REPORT_RU.md").write_text(md, encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("answers", type=Path)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    a = load(args.answers)
    res = analyze(a)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    chart(res, args.out)
    report(res, args.out)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return res


if __name__ == "__main__":
    main(sys.argv[1:])
