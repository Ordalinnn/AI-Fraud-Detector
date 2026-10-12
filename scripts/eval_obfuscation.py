"""
How much do scammers' disguises hurt the detector, and how much of that does
text_normalize.py win back?

    python scripts/eval_obfuscation.py

Same honest set-up as the jury evaluation: the jury's group-aware split
(282 test messages, no template on both sides), training data = the
project's train messages + the small real-SMS sample. Each scam message of
the test set is then disguised in several ways a real scammer uses, and we
count how many the detector still catches - once with the normalization
step switched OFF (the system before this change) and once with it ON.

Disguises (deterministic, seeded per message):
  homoglyph   half of the letters that have a twin in the other alphabet are
              swapped (Cyrillic "о" -> Latin "o", ...)
  digits      о/o -> 0 and а/a -> @ in a third of the places
  invisible   a zero-width space inside every word of 4+ letters
  spaced      two longer words spelled out letter by letter: "к о д ы"
  fullwidth   the whole message in fullwidth characters ("ＳＭＳ")
  combo       homoglyph + digits + invisible together
  leet        3 for з/e, 4 for ч/a, 1 for i/l, 6 for б - NOT handled by the
              normalization on purpose (too easy to break real numbers), so
              this row shows an honest remaining gap

Also checked: on the ORIGINAL (undisguised) test messages accuracy and false
alarms must not get worse, and on the real SMS hold-out the false alarm rate
must not get worse.

Writes reports/obfuscation/{results.json, REPORT_RU.md, chart.png}.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fraud_eval as fe  # noqa: E402
import fraud_logic  # noqa: E402
import fraud_model as fm  # noqa: E402
from text_normalize import CYR_TO_LAT, LAT_TO_CYR  # noqa: E402

OUT = ROOT / "reports" / "obfuscation"
SYSTEMS = {"final": "Ансамбль (сайт)", "char": "Модель на устройстве (Android, расширение)"}
ATTACK_NAMES = {
    "original": "Без маскировки",
    "homoglyph": "Латинские буквы вместо русских (o, a, c, p…)",
    "digits": "Цифры и символы вместо букв (0, @)",
    "invisible": "Невидимые символы внутри слов",
    "spaced": "Слова по буквам (к о д ы)",
    "fullwidth": "Полноширинные символы (Unicode)",
    "combo": "Всё вместе",
    "leet": "3/4/1/6 вместо букв (не исправляется)",
}


# ---------- disguises ----------

def _rng(text: str, salt: str) -> random.Random:
    return random.Random(f"{salt}|{text}")


def homoglyph(text, p=0.5, salt="h"):
    r = _rng(text, salt)
    out = []
    for c in text:
        if c in CYR_TO_LAT and r.random() < p:
            out.append(CYR_TO_LAT[c])
        elif c in LAT_TO_CYR and r.random() < p:
            out.append(LAT_TO_CYR[c])
        else:
            out.append(c)
    return "".join(out)


def digits(text, p=0.34, salt="d"):
    r = _rng(text, salt)
    out = []
    for c in text:
        if c in "оoОO" and r.random() < p:
            out.append("0")
        elif c in "аa" and r.random() < p:
            out.append("@")
        else:
            out.append(c)
    return "".join(out)


def invisible(text, salt="i"):
    r = _rng(text, salt)
    words = text.split(" ")
    out = []
    for w in words:
        if sum(ch.isalpha() for ch in w) >= 4:
            k = r.randint(1, len(w) - 1)
            w = w[:k] + "​" + w[k:]
        out.append(w)
    return " ".join(out)


def spaced(text, n=2, salt="s"):
    r = _rng(text, salt)
    words = text.split(" ")
    idx = [i for i, w in enumerate(words) if w.isalpha() and len(w) >= 4]
    for i in r.sample(idx, min(n, len(idx))):
        words[i] = " ".join(words[i])
    return " ".join(words)


def fullwidth(text):
    return "".join(chr(ord(c) + 0xFEE0) if 0x21 <= ord(c) <= 0x7E else ("　" if c == " " else c) for c in text)


LEET = {"з": "3", "З": "3", "e": "3", "E": "3", "ч": "4", "a": "4", "A": "4", "i": "1", "l": "1", "б": "6"}


def leet(text, p=0.5, salt="l"):
    r = _rng(text, salt)
    return "".join(LEET[c] if c in LEET and r.random() < p else c for c in text)


ATTACKS = {
    "original": lambda t: t,
    "homoglyph": homoglyph,
    "digits": digits,
    "invisible": invisible,
    "spaced": spaced,
    "fullwidth": fullwidth,
    "combo": lambda t: invisible(digits(homoglyph(t, p=0.3))),
    "leet": leet,
}


# ---------- evaluation ----------

def run(normalize: bool, df, groups, tr, te, extra, holdout):
    fraud_logic.NORMALIZE_INPUT = normalize
    texts = df["text"].tolist()
    y = df["label"].to_numpy()
    T = [texts[i] for i in tr] + extra["text"].tolist()
    Y = np.concatenate([y[tr], extra["label"].to_numpy().astype(int)])
    G = np.concatenate([np.asarray(groups)[tr], np.asarray(groups).max() + 1 + np.arange(len(extra))])
    ens = fm.FraudEnsemble().fit(T, Y, fm.featurise(T), G)

    te_texts = [texts[i] for i in te]
    yt = y[te]
    fraud_texts = [t for t, l in zip(te_texts, yt) if l == 1]
    res = {}
    for name, attack in ATTACKS.items():
        if name == "original":
            s = ens.score(te_texts)
            res[name] = {k: {"recall": float(((s[k] >= 0.5) & (yt == 1)).sum() / (yt == 1).sum()),
                             "accuracy": float(((s[k] >= 0.5).astype(int) == yt).mean()),
                             "false_alarm_rate": float(((s[k] >= 0.5) & (yt == 0)).sum() / (yt == 0).sum())}
                         for k in SYSTEMS}
        else:
            s = ens.score([attack(t) for t in fraud_texts])
            res[name] = {k: {"recall": float((s[k] >= 0.5).mean())} for k in SYSTEMS}
    sh = ens.score(holdout["text"].tolist())
    ham = holdout["label"].to_numpy() == 0
    res["real_holdout_false_alarms"] = {k: float((sh[k][ham] >= 0.5).mean()) for k in SYSTEMS}
    res["n_fraud_test"] = len(fraud_texts)
    res["n_real_legit"] = int(ham.sum())
    fraud_logic.NORMALIZE_INPUT = True
    return res


def chart(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [a for a in ATTACKS]
    before = [out["before"][a]["final"]["recall"] * 100 for a in names]
    after = [out["after"][a]["final"]["recall"] * 100 for a in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    yy = np.arange(len(names))[::-1]
    ax.barh(yy + 0.2, before, height=0.38, color="#cbd5e1", label="До (без нормализации)")
    ax.barh(yy - 0.2, after, height=0.38, color="#0f766e", label="После")
    for y_, b, a in zip(yy, before, after):
        ax.text(b + 0.8, y_ + 0.2, f"{b:.0f}%", va="center", fontsize=9, color="#475569")
        ax.text(a + 0.8, y_ - 0.2, f"{a:.0f}%", va="center", fontsize=9, color="#0f766e", fontweight="bold")
    ax.set_yticks(yy)
    ax.set_yticklabels([ATTACK_NAMES[a] for a in names], fontsize=9)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Пойманных мошеннических сообщений, % (ансамбль, тест жюри)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Замаскированные мошеннические сообщения: до и после нормализации", fontsize=11, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "chart.png", dpi=150)


def pct(x):
    return f"{x * 100:.1f}%"


def report(out):
    b, a = out["before"], out["after"]
    rows = []
    for name in ATTACKS:
        rows.append(f"| {ATTACK_NAMES[name]} | {pct(b[name]['final']['recall'])} | **{pct(a[name]['final']['recall'])}** "
                    f"| {pct(b[name]['char']['recall'])} | **{pct(a[name]['char']['recall'])}** |")
    o_b, o_a = b["original"], a["original"]
    md = f"""# Маскировка текста: насколько она мешала и что даёт нормализация

Мошенники пишут «к0д», «зaблoкирoвана» латинскими буквами, «к о д ы» по буквам или вставляют невидимые символы, чтобы фильтры не узнали слова. Мы проверили, как это действует на систему, и добавили шаг **нормализации текста** (`text_normalize.py`): перед проверкой маскировка снимается. Шаг одинаковый на сайте, в Android-приложении и в расширении.

**Как проверяли.** Тот же честный тест, что и для жюри: 282 сообщения, шаблоны не пересекаются с обучением. Каждое из {out['before']['n_fraud_test']} мошеннических сообщений теста замаскировали шестью способами (детерминированно, с фиксированным seed) и посчитали, сколько система всё равно поймала: без нормализации (как было) и с ней. Модели обучены заново в обоих вариантах на одних и тех же данных.

## Сколько мошеннических сообщений поймано

| Маскировка | Сайт: до | Сайт: после | Телефон/расширение: до | Телефон/расширение: после |
|---|---|---|---|---|
{chr(10).join(rows)}

![](chart.png)

## Не стало ли хуже на обычных сообщениях

| | До | После |
|---|---|---|
| Точность на исходных 282 сообщениях (сайт) | {pct(o_b['final']['accuracy'])} | {pct(o_a['final']['accuracy'])} |
| Ложные тревоги на обычных сообщениях теста (сайт) | {pct(o_b['final']['false_alarm_rate'])} | {pct(o_a['final']['false_alarm_rate'])} |
| Ложные тревоги на {out['before']['n_real_legit']} реальных обычных SMS (сайт) | {pct(b['real_holdout_false_alarms']['final'])} | {pct(a['real_holdout_false_alarms']['final'])} |
| Ложные тревоги на реальных обычных SMS (телефон/расширение) | {pct(b['real_holdout_false_alarms']['char'])} | {pct(a['real_holdout_false_alarms']['char'])} |

## Что это значит и ограничения

- Маскировку проверяли на **наших же** сообщениях, замаскированных программой. Настоящие мошенники могут маскировать иначе; строка «3/4/1/6 вместо букв» специально оставлена неисправленной: замена цифр на буквы ломала бы настоящие суммы и коды, поэтому этот приём остаётся слабым местом.
- Нормализация не трогает ссылки: поддельный адрес «kаspi.kz» с русской «а» по-прежнему распознаётся как подделка.
- Перезапуск: `python scripts/eval_obfuscation.py` (около {out['runtime_seconds'] / 60:.0f} мин).
"""
    (OUT / "REPORT_RU.md").write_text(md, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = fe.load_dataset()
    y = df["label"].to_numpy()
    groups = fm.template_groups(df["text"].tolist())
    fraud_logic.NORMALIZE_INPUT = True
    feats = fm.featurise(df["text"])
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=fm.EVAL_TEST_SIZE, random_state=fm.MODEL_RANDOM_STATE)
                  .split(feats, y, groups))
    extra = pd.DataFrame(fm.load_real_training_sample(), columns=["text", "label"])
    holdout = fm.load_real_holdout()
    out = {"before": run(False, df, groups, tr, te, extra, holdout),
           "after": run(True, df, groups, tr, te, extra, holdout)}
    out["runtime_seconds"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    chart(out)
    report(out)
    for name in ATTACKS:
        print(f"{name:10s} final {out['before'][name]['final']['recall']:.3f} -> {out['after'][name]['final']['recall']:.3f}"
              f" | char {out['before'][name]['char']['recall']:.3f} -> {out['after'][name]['char']['recall']:.3f}")
    print("orig acc", out["before"]["original"]["final"]["accuracy"], out["after"]["original"]["final"]["accuracy"])
    print("holdout FA", out["before"]["real_holdout_false_alarms"], out["after"]["real_holdout_false_alarms"])


if __name__ == "__main__":
    main()
