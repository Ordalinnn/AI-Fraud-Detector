"""
Reproducible answers to the jury's three pre-defense requests
(NIS Taraz written-project review, 09.10.2026).

    python scripts/run_jury_evaluation.py            # full run (~3-5 min)
    python scripts/run_jury_evaluation.py --quick    # fewer repeats, for a smoke test
    python scripts/run_jury_evaluation.py --real data/real/my_messages.csv

Writes everything to reports/jury_evaluation/:
    results.json   every number used in the report
    *.csv          the tables
    *.png          the charts

Sections
  1  Same 282 test messages: every single model vs. old ensemble vs. old
     ensemble + rules vs. new stacked ensemble; exact McNemar tests and
     cluster-bootstrap 95% CIs; then 3x repeated group 5-fold CV to show the
     result is not a property of one lucky split.
  2  Per-language metrics; leave-one-scam-family-out (never-seen fraud types);
     independent REAL messages (UCI SMS Spam Collection, 5,574 real English
     SMS) and, if provided, the team's own collected KZ/RU messages.
  3  Calibration (reliability, Brier, ECE) of old vs. new risk score and the
     cost of false blocking at every threshold.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fraud_eval as fe  # noqa: E402
import fraud_model as fm  # noqa: E402

OUT = ROOT / "reports" / "jury_evaluation"
COMPARE = fe.SYSTEMS
OLD_MEMBERS = ["old_lr", "old_rf", "old_gb", "old_tfidf"]


def r4(x):
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else round(float(x), 4)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, dict):
        return {k: r4(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r4(v) for v in x]
    return x


def metrics_table(y, scores: dict, keys) -> pd.DataFrame:
    rows = []
    for k in keys:
        m = fe.metrics(y, scores[k], binary_score=(k == "baseline"))
        m["system"] = k
        m["name"] = fe.SYSTEM_NAMES[k]
        rows.append(m)
    return pd.DataFrame(rows).set_index("system")


# =========================
# SCAM FAMILIES for the never-seen-type test
# =========================
FAMILY_RULES = [
    ("Банк / карта / колл-центр банка", r"bank|card|callback|call us back|nfc|kyc"),
    ("Доставка / госорганы / родственник в беде", r"deliver|government|relative|courier|grandparent|family|voice-cloning|lost my phone|school-emergency"),
    ("Инвестиции / крипта / работа", r"invest|crypto|pig butchering|job|task-based|mule|certification|deepfake|recruiter"),
    ("Техподдержка / устройство", r"tech|device locked|verify you're human|ransomware"),
    ("Романтические / знакомства", r"romance|dating|widow"),
    ("QR-коды / штрафы / платные дороги", r"qr|toll|traffic"),
    ("Аккаунты / SIM / мессенджеры", r"sim swap|social media|messenger|vote for|account recovery"),
    ("Призы / налоги / наследство / лотереи", r"tax|inheritance|lottery|sweepstakes|rewards|irs|social security|ssa"),
    ("Подделка брендов / подписки", r"brand|subscription|trial|renewal|storefront|premium-rate"),
    ("Бизнес-почта / счета / зарплата", r"business email|bec|payroll|invoice|vendor"),
    ("Аренда / маркетплейсы / покупки", r"rental|landlord|marketplace|puppy|vacation|apartment|platform|timeshare"),
    ("Долги / суды / коммуналка / благотворительность", r"debt|court|jury|law firm|garnishment|utility|charity|loan|credit repair|unemployment|overpayment|extortion|sextortion|solar|warranty|wangiri|mystery-shopper|check-cashing"),
]


def scam_family(category: str) -> str | None:
    if not category.startswith("FRAUD"):
        return None
    c = category.split(":", 1)[1].lower()
    if "english examples" in c or "kazakh examples" in c:
        return None  # language sections mixing many types: always kept in training
    for name, pat in FAMILY_RULES:
        if re.search(pat, c):
            return name
    return "Прочее"


# =========================
# SECTION 1
# =========================
def section1(df, feats, groups, extra, extra_feats, n_repeats: int, out: dict):
    y = df["label"].to_numpy()
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=fm.EVAL_TEST_SIZE, random_state=fm.MODEL_RANDOM_STATE)
                  .split(feats, y, groups))
    s = fe.score_split(df, feats, groups, tr, te, extra, extra_feats)
    yt = y[te]
    tab = metrics_table(yt, s, COMPARE)
    tab.to_csv(OUT / "1_same_test_comparison.csv")

    # Best single model chosen WITHOUT looking at the test: highest F1 on the
    # training set's own out-of-fold predictions (what a careful engineer
    # would pick before ever touching the 282 test messages).
    oof = s["_oof_train"]
    y_tr = y[tr]
    oof_f1 = {k: fe.metrics(y_tr, oof[k])["f1"] for k in fe.SINGLE_MODELS}
    best_single_by_train = max(oof_f1, key=oof_f1.get)
    # ...and, to be conservative, also the single model that happens to be
    # best ON the test (this favours the single model).
    best_single_on_test = tab.loc[fe.SINGLE_MODELS, "f1"].idxmax()

    pred = {k: (s[k] >= 0.5).astype(int) for k in COMPARE}
    g_te = np.asarray(groups)[te]
    tests = {}
    for a, b in [("final", best_single_on_test), ("final", best_single_by_train), ("final", "old_ensemble_rules"),
                 ("old_ensemble_rules", "old_tfidf"), ("old_ensemble", "old_ensemble_rules")]:
        key = f"{a}_vs_{b}"
        if key in tests:
            continue
        mc = fe.mcnemar_exact(yt, pred[a], pred[b])
        acc_ci = fe.bootstrap_diff_ci(yt, s[a], s[b], "accuracy", groups=g_te)
        f1_ci = fe.bootstrap_diff_ci(yt, s[a], s[b], "f1", groups=g_te)
        tests[key] = {"a": a, "b": b, "mcnemar": mc, "accuracy_diff_ci95": acc_ci, "f1_diff_ci95": f1_ci}

    out["section1_same_test"] = {
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "test_fraud": int(yt.sum()), "test_safe": int(len(yt) - yt.sum()),
        "n_template_groups": int(len(set(groups.tolist()))),
        "table": r4(tab.drop(columns="name").to_dict(orient="index")),
        "best_single_by_train_oof": best_single_by_train,
        "train_oof_f1": r4(oof_f1),
        "best_single_on_test": best_single_on_test,
        "stacker_weights": r4(s["_weights"]),
        "paired_tests": r4(tests),
    }
    print(tab[["TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"]].round(4).to_string())
    print("paired tests:", json.dumps(r4(tests), ensure_ascii=False)[:600])

    # Repeated group CV: same folds for every system, so differences are paired.
    rows = []
    pooled = None
    for seed in range(n_repeats):
        pooled_seed = {k: np.zeros(len(df)) for k in COMPARE}
        for f, (a, b) in enumerate(GroupKFold(n_splits=5, shuffle=True, random_state=seed).split(feats, y, groups)):
            sc = fe.score_split(df, feats, groups, a, b, extra, extra_feats)
            for k in COMPARE:
                m = fe.metrics(y[b], sc[k], binary_score=(k == "baseline"))
                m.update(system=k, seed=seed, fold=f)
                rows.append(m)
                pooled_seed[k][b] = sc[k]
        if seed == 0:
            pooled = pooled_seed
    cv = pd.DataFrame(rows)
    cv.to_csv(OUT / "1_repeated_cv_folds.csv", index=False)
    agg = cv.groupby("system")[["accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "brier"]].agg(["mean", "std"])
    agg = agg.loc[COMPARE]
    agg.to_csv(OUT / "1_repeated_cv_summary.csv")
    f1w = cv.pivot_table(index=["seed", "fold"], columns="system", values="f1")
    accw = cv.pivot_table(index=["seed", "fold"], columns="system", values="accuracy")
    best_single_cv = agg.loc[fe.SINGLE_MODELS, ("f1", "mean")].idxmax()
    out["section1_repeated_cv"] = {
        "n_folds_total": int(len(f1w)),
        "summary": r4({k: {m: {"mean": agg.loc[k, (m, "mean")], "std": agg.loc[k, (m, "std")]}
                           for m in ["accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "brier"]} for k in COMPARE}),
        "best_single_cv": best_single_cv,
        "final_beats_best_single_f1_folds": int((f1w["final"] > f1w[best_single_cv]).sum()),
        "final_ties_best_single_f1_folds": int((f1w["final"] == f1w[best_single_cv]).sum()),
        "old_tfidf_beats_old_ensemble_rules_f1_folds": int((f1w["old_tfidf"] > f1w["old_ensemble_rules"]).sum()),
        "old_best_member_cv": agg.loc[OLD_MEMBERS, ("f1", "mean")].idxmax(),
        "final_beats_old_ensemble_rules_f1_folds": int((f1w["final"] > f1w["old_ensemble_rules"]).sum()),
        "mean_acc_gain_final_vs_best_single": float((accw["final"] - accw[best_single_cv]).mean()),
        "mean_acc_gain_final_vs_old": float((accw["final"] - accw["old_ensemble_rules"]).mean()),
    }
    print(agg.round(3))
    return s, te, pooled


# =========================
# SECTION 2
# =========================
def section2(df, feats, groups, extra, extra_feats, pooled, s_jury, te_jury, real_csv, out: dict):
    y = df["label"].to_numpy()
    res = {}

    # 2a. per language - on the jury test and on pooled out-of-fold scores
    lang_rows = []
    for scope, idx, scores in [("jury_test_282", te_jury, s_jury), ("pooled_cv_1398", np.arange(len(df)), pooled)]:
        sub = df.iloc[idx]
        for lang in ["kk", "ru", "en"]:
            m_idx = np.where(sub["language"].to_numpy() == lang)[0]
            if len(m_idx) == 0:
                continue
            for k in ["old_ensemble_rules", "final", "char", "tfidf"]:
                sc = (scores[k] if scope == "jury_test_282" else scores[k][idx])[m_idx]
                m = fe.metrics(sub["label"].to_numpy()[m_idx], sc)
                m.update(scope=scope, language=lang, system=k, n_fraud=int(sub["label"].to_numpy()[m_idx].sum()))
                lang_rows.append(m)
    lang_df = pd.DataFrame(lang_rows)
    lang_df.to_csv(OUT / "2_per_language.csv", index=False)
    res["per_language"] = r4(lang_df.drop(columns=["roc_auc", "brier"], errors="ignore").to_dict(orient="records"))
    print(lang_df[lang_df.scope == "pooled_cv_1398"][["language", "system", "n", "accuracy", "recall", "fpr", "f1"]].round(3))

    # 2b. leave-one-scam-family-out: train with NO example of that family
    # (nor any message from a template group that contains one), then measure
    # how much of that unseen family is still caught, and the false-alarm
    # rate on all safe messages that were held out along with it.
    fam = df["category"].map(scam_family)
    families = [f for f in fam.dropna().unique()]
    fam_rows = []
    g = np.asarray(groups)
    for f in sorted(families):
        fam_idx = np.where(fam.to_numpy() == f)[0]
        if len(fam_idx) < 8:
            continue
        leak_groups = set(g[fam_idx].tolist())
        test_mask = np.isin(g, list(leak_groups))
        tr = np.where(~test_mask)[0]
        te = fam_idx  # only the unseen fraud family
        # No real UCI sample here: its spam includes prize scams, which
        # would leak the "unseen" family. Pure project data on both sides.
        sc = fe.score_split(df, feats, groups, tr, te)
        row = {"family": f, "n_test": int(len(te)), "n_train": int(len(tr))}
        for k in ["old_ensemble_rules", "final", "char", "tfidf", "baseline"]:
            row[f"recall_{k}"] = float(np.mean(sc[k] >= 0.5))
        fam_rows.append(row)
        print(f"  unseen family {f}: n={len(te)} old={row['recall_old_ensemble_rules']:.2f} new={row['recall_final']:.2f}")
    fam_df = pd.DataFrame(fam_rows)
    fam_df.to_csv(OUT / "2_unseen_scam_families.csv", index=False)
    w = fam_df["n_test"]
    res["unseen_families"] = {
        "rows": r4(fam_df.to_dict(orient="records")),
        "weighted_recall": r4({k: float(np.average(fam_df[f"recall_{k}"], weights=w))
                               for k in ["old_ensemble_rules", "final", "char", "tfidf", "baseline"]}),
        "n_families": int(len(fam_df)),
    }

    # 2c. independent REAL messages: the held-out half of the UCI corpus,
    # never used in training. OLD = trained on the project data only (as
    # submitted); NEW = project data + the small real sample, as in the app.
    T_all = df["text"].tolist() + extra["text"].tolist()
    Y_all = np.concatenate([y, extra["label"].to_numpy().astype(int)])
    F_all = pd.concat([feats, extra_feats], ignore_index=True)
    G_all = np.concatenate([groups, groups.max() + 1 + np.arange(len(extra))])
    full = fm.FraudEnsemble().fit(T_all, Y_all, F_all, G_all)
    old_full = fm.BaseModels().fit(df["text"].tolist(), y, feats)
    real = {}

    def score_old_new(texts):
        f = fm.featurise(texts)
        sn = full.score(texts, f)
        so = old_full.predict(texts, f)
        sn["old_ensemble_rules"] = np.minimum(0.99, (so["lr"] + so["rf"] + so["gb"] + so["tfidf"]) / 4.0 + so["boost"])
        return sn

    hold = fm.load_real_holdout()
    sc = score_old_new(hold["text"].tolist())
    yl = hold["label"].to_numpy().astype(int)
    ham = yl == 0
    r = {"n": int(len(hold)), "n_ham": int(ham.sum()), "n_spam": int((~ham).sum()),
         "n_real_in_training": int(len(extra))}
    for k in ["old_ensemble_rules", "final"]:
        p = sc[k] >= 0.5
        r[k] = {
            "false_block_rate_on_real_legit": float(p[ham].mean()),
            "false_blocks_per_1000_legit": float(p[ham].mean() * 1000),
            "flag_rate_on_real_spam": float(p[~ham].mean()),
            "roc_auc_spam_vs_ham": float(fe.roc_auc_score(yl, sc[k])),
            "metrics": fe.metrics(yl, sc[k]),
        }
    for k, name in [("old_ensemble_rules", "old"), ("final", "new")]:
        fp_idx = np.where(ham & (sc[k] >= 0.5))[0]
        r[f"examples_false_flags_{name}"] = [
            {"text": hold["text"].iloc[i][:160], "risk": round(float(sc[k][i]), 3)}
            for i in fp_idx[np.argsort(-sc[k][fp_idx])][:6]
        ]
    # the motivating example from the report
    probe = ["I cant pick the phone right now. Pls send a message"]
    ps = score_old_new(probe)
    r["probe_example"] = {"text": probe[0], "old": float(ps["old_ensemble_rules"][0]), "new": float(ps["final"][0])}
    real["uci_holdout_en"] = r
    print("UCI real holdout:", json.dumps(r4({k: v for k, v in r.items() if not k.startswith("examples")}), ensure_ascii=False)[:900])
    if real_csv:
        rc = pd.read_csv(real_csv)
        need = {"text", "label"}
        if not need <= set(rc.columns):
            raise SystemExit(f"{real_csv} needs columns {need}")
        rc = rc.dropna(subset=["text", "label"])
        rc["language"] = rc.get("language", rc["text"].map(fe.detect_language))
        sc = score_old_new(rc["text"].tolist())
        rows = []
        for lang, sub in [("all", rc)] + list(rc.groupby("language")):
            ii = sub.index.to_numpy()
            pos = rc.index.get_indexer(ii)
            for k in ["old_ensemble_rules", "final"]:
                m = fe.metrics(sub["label"].astype(int).to_numpy(), sc[k][pos], binary_score=len(set(sub["label"])) < 2)
                m.update(language=lang, system=k)
                rows.append(m)
        real["team_collected"] = r4(rows)
        pd.DataFrame(rows).to_csv(OUT / "2_team_real_messages.csv", index=False)
    res["real_messages"] = r4(real)
    out["section2"] = res
    return full


# =========================
# SECTION 3
# =========================
def section3(df, pooled, out: dict):
    y = df["label"].to_numpy()
    res = {}
    for k in ["old_ensemble_rules", "final"]:
        p = pooled[k]
        res[k] = {
            "brier": float(fe.brier_score_loss(y, np.clip(p, 0, 1))),
            "ece_10_bins": fe.expected_calibration_error(y, p),
            "reliability": fe.calibration_table(y, p).to_dict(orient="records"),
        }
    thr = np.round(np.arange(0.1, 0.91, 0.05), 2)
    tables = {}
    for k in ["old_ensemble_rules", "final"]:
        t = fe.threshold_table(y, pooled[k], thr)
        t["false_blocks_per_1000_legit"] = t["fpr"] * 1000
        t["missed_fraud_per_100_fraud"] = (1 - t["recall"]) * 100
        tables[k] = t
        t.to_csv(OUT / f"3_thresholds_{k}.csv", index=False)
    # Cost scenarios: cost of one missed scam = 1. A false block costs
    #  0.2 in "warning" mode (user just sees a yellow/orange banner),
    #  1   when both errors are equally bad,
    #  5   in "auto-block" mode (a real message from a bank/relative is hidden).
    # For a calibrated probability, decision theory gives the optimal
    # threshold directly: t* = c_fp / (c_fp + c_fn).
    scen = []
    for name, c_fp in [("warning", 0.2), ("equal", 1.0), ("auto_block", 5.0)]:
        for k in ["old_ensemble_rules", "final"]:
            t = fe.threshold_table(y, pooled[k], thr, cost_fp=c_fp, cost_fn=1.0)
            best = t.loc[t["cost"].idxmin()]
            scen.append({
                "scenario": name, "cost_false_block": c_fp, "system": k,
                "theory_threshold": round(c_fp / (c_fp + 1.0), 3),
                "best_threshold": float(best["threshold"]),
                "cost_at_best": float(best["cost"]),
                "cost_at_0.5": float(t.loc[np.isclose(t["threshold"], 0.5), "cost"].iloc[0]),
                "FP_at_best": int(best["FP"]), "FN_at_best": int(best["FN"]),
                "recall_at_best": float(best["recall"]), "fpr_at_best": float(best["fpr"]),
            })
    scen_df = pd.DataFrame(scen)
    scen_df.to_csv(OUT / "3_cost_scenarios.csv", index=False)
    res["thresholds"] = {k: r4(v[["threshold", "TN", "FP", "FN", "TP", "precision", "recall", "fpr",
                                  "false_blocks_per_1000_legit", "missed_fraud_per_100_fraud"]].to_dict(orient="records"))
                         for k, v in tables.items()}
    res["cost_scenarios"] = r4(scen_df.to_dict(orient="records"))
    out["section3"] = r4(res)
    print(scen_df.round(3))
    return tables


# =========================
# CHARTS
# =========================
def charts(out, pooled, df, tables):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1e293b", "#64748b", "#e2e8f0"
    old_c, new_c, single_c = "#94a3b8", "#0f766e", "#cbd5e1"
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "font.family": "DejaVu Sans"})

    # 1. same-test comparison (horizontal so long labels never overlap)
    tab = out["section1_same_test"]["table"]
    order = ["baseline", "old_lr", "old_rf", "old_gb", "old_tfidf", "old_ensemble", "old_ensemble_rules", "char", "final"]
    labels = {"baseline": "Правило по ключевым словам",
              "old_lr": "Logistic Regression", "old_rf": "Random Forest", "old_gb": "Gradient Boosting",
              "old_tfidf": "TF-IDF + Naive Bayes", "old_ensemble": "Ансамбль: среднее 4 моделей",
              "old_ensemble_rules": "Ансамбль + правила (в работе)",
              "char": "Символьные n-граммы (новая модель)", "final": "Новый ансамбль: стекинг"}
    colors = [new_c if k == "final" else "#5eead4" if k == "char" else old_c if k.startswith("old_ensemble") else single_c
              for k in order]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    vals = [tab[k]["accuracy"] * 100 for k in order]
    ypos = np.arange(len(order))[::-1]
    ax.barh(ypos, vals, color=colors, height=0.68)
    for yv, v in zip(ypos, vals):
        ax.text(v + 0.4, yv, f"{v:.1f}%", va="center", color=ink, fontsize=10)
    ax.set_yticks(ypos, [labels[k] for k in order], fontsize=10)
    ax.set_xlim(50, 102)
    ax.set_xlabel("Accuracy на тех же 282 тестовых сообщениях, %")
    ax.axhline(1.5, color=grid, lw=1)
    ax.text(101.5, 1.55, "после исправлений ↓", color=muted, fontsize=9, va="bottom", ha="right")
    ax.text(101.5, 8.45, "система из письменной работы ↓", color=muted, fontsize=9, va="bottom", ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, color=grid); ax.set_axisbelow(True)
    ax.set_title("Одна и та же тестовая выборка (282 сообщения)", color=ink, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "chart_1_same_test.png", dpi=180); plt.close(fig)

    # 3a. reliability diagram
    y = df["label"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.plot([0, 1], [0, 1], color=grid, lw=1.5, label="Идеальная калибровка")
    for k, c, lab in [("old_ensemble_rules", old_c, "Старый риск"), ("final", new_c, "Новый риск")]:
        t = fe.calibration_table(y, pooled[k])
        t = t[t["n"] >= 5]
        ece = fe.expected_calibration_error(y, pooled[k])
        ax.plot(t["mean_predicted"], t["fraction_fraud"], "o-", color=c, lw=2, label=f"{lab} (ECE {ece*100:.1f}%)")
    ax.set_xlabel("Предсказанный риск"); ax.set_ylabel("Реальная доля мошенничества")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Калибровка риска (1398 сообщений, вне обучения)", color=ink, loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "chart_3_calibration.png", dpi=180); plt.close(fig)

    # 3b. threshold trade-off for the new model
    t = tables["final"]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(t["threshold"], t["false_blocks_per_1000_legit"], "o-", color="#b45309", lw=2, label="Ложные блокировки на 1000 обычных сообщений")
    ax.plot(t["threshold"], t["missed_fraud_per_100_fraud"] * 10, "s-", color=new_c, lw=2, label="Пропущенные мошенничества на 1000 мошеннических")
    ax.set_xlabel("Порог"); ax.set_ylabel("Ошибок на 1000 сообщений")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=grid); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Цена ошибки при разных порогах (новая модель)", color=ink, loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "chart_3_thresholds.png", dpi=180); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 CV repeat instead of 3")
    ap.add_argument("--real", type=str, default=None, help="CSV with columns text,label[,language] of real messages")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df = fe.load_dataset()
    feats = fm.featurise(df["text"])
    groups = fm.template_groups(df["text"].tolist())
    out = {"dataset": {"n": int(len(df)), "fraud": int(df["label"].sum()), "safe": int((df["label"] == 0).sum()),
                       "languages": df["language"].value_counts().to_dict(),
                       "fraud_by_language": df[df.label == 1]["language"].value_counts().to_dict()}}
    print(out["dataset"])

    extra = pd.DataFrame(fm.load_real_training_sample(), columns=["text", "label"])
    extra_feats = fm.featurise(extra["text"])
    out["dataset"]["real_messages_added_to_training"] = int(len(extra))
    s_jury, te_jury, pooled = section1(df, feats, groups, extra, extra_feats, 1 if args.quick else 3, out)
    full = section2(df, feats, groups, extra, extra_feats, pooled, s_jury, te_jury, args.real, out)
    out["full_model_stacker_weights"] = r4(full.weights)
    tables = section3(df, pooled, out)
    charts(out, pooled, df, tables)
    out["runtime_seconds"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done in {out['runtime_seconds']}s -> {OUT}")


if __name__ == "__main__":
    main()
