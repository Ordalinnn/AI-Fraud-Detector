"""
Measures voice mode on the team's own recordings: how well speech is
turned into text, and - what actually matters - whether the fraud verdict
on the transcript matches the verdict on the true text.

    python scripts/eval_speech.py                      # data/voice/manifest.csv
    python scripts/eval_speech.py --manifest path.csv --systems baseline improved

Manifest columns: file, text, label, language
    file      audio file, relative to the manifest
    text      exactly what was said (reference transcript)
    label     1 = fraud, 0 = ordinary
    language  kk / ru / en

Systems compared (same recordings, same fraud model):
    baseline        voice mode v1: Whisper "small", no vocabulary hint, no loudness fix
    improved        current speech.py: Kazakh model for kk, vocabulary hint, loudness fix
    improved_noprompt  as improved but without the vocabulary hint - isolates
                    whether the hint causes false alarms on ordinary calls

Outputs reports/speech_evaluation/: per_recording.csv, summary.csv, summary.md
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fraud_eval as fe  # noqa: E402
import fraud_model as fm  # noqa: E402
import speech  # noqa: E402

OUT = ROOT / "reports" / "speech_evaluation"

SYSTEMS = {
    "baseline": dict(model=lambda lang: "small", use_prompt=False, normalize=False),
    "improved": dict(model=speech.model_for_language, use_prompt=True, normalize=True),
    "improved_noprompt": dict(model=speech.model_for_language, use_prompt=False, normalize=True),
}


# ---------- text metrics ----------

def normalise_text(t: str) -> str:
    t = unicodedata.normalize("NFC", str(t)).lower().replace("ё", "е")
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> float:
    r, h = normalise_text(ref).split(), normalise_text(hyp).split()
    return edit_distance(r, h) / max(len(r), 1)


def cer(ref: str, hyp: str) -> float:
    """Character error rate - fairer than WER for Kazakh, where one wrong
    suffix makes the whole (long, agglutinated) word count as an error."""
    r, h = normalise_text(ref).replace(" ", ""), normalise_text(hyp).replace(" ", "")
    return edit_distance(list(r), list(h)) / max(len(r), 1)


def prompt_leak(ref: str, hyp: str, language: str) -> bool:
    """True if the transcript contains a vocabulary-hint word that was never
    said - the failure mode where the hint itself creates a false alarm."""
    said = set(normalise_text(ref).split())
    heard = set(normalise_text(hyp).split())
    hint = set(normalise_text(speech.PROMPTS.get(language, "")).split())
    return bool((heard & hint) - said)


# ---------- main ----------

def train_fraud_model():
    df = fe.load_dataset()
    real = fm.load_real_training_sample()
    texts = df["text"].tolist() + [t for t, _ in real]
    y = np.concatenate([df["label"].to_numpy(), [label for _, label in real]])
    return fm.FraudEnsemble().fit(texts, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data" / "voice" / "manifest.csv"))
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS))
    args = ap.parse_args()

    manifest = Path(args.manifest)
    rec = pd.read_csv(manifest).dropna(subset=["file", "text", "label", "language"])
    if rec.empty:
        raise SystemExit(f"{manifest} has no recordings yet - see data/voice/README.md")
    rec["label"] = rec["label"].astype(int)
    OUT.mkdir(parents=True, exist_ok=True)

    print("training fraud model...")
    fraud = train_fraud_model()
    ref_scores = fraud.score(rec["text"].tolist())["final"]

    models, rows = {}, []
    for system in args.systems:
        cfg = SYSTEMS[system]
        for i, r in rec.iterrows():
            spec = cfg["model"](r["language"])
            if spec not in models:
                print(f"loading Whisper '{spec}'...")
                models[spec] = speech.load_model(spec)
            audio = (manifest.parent / r["file"]).read_bytes()
            t = time.time()
            out = speech.transcribe(models[spec], audio, language=r["language"],
                                    use_prompt=cfg["use_prompt"], normalize=cfg["normalize"])
            rows.append({
                "system": system, "model": spec, "file": r["file"], "language": r["language"],
                "label": r["label"], "reference": r["text"], "transcript": out["text"],
                "seconds_audio": out["duration"], "seconds_compute": time.time() - t,
                "wer": wer(r["text"], out["text"]), "cer": cer(r["text"], out["text"]),
                "prompt_leak": prompt_leak(r["text"], out["text"], r["language"]) if cfg["use_prompt"] else False,
                "risk_reference": float(ref_scores[rec.index.get_loc(i)]),
            })
            print(f"[{system}] {r['file']}: CER {rows[-1]['cer']:.2f}  {out['text'][:70]!r}")

    res = pd.DataFrame(rows)
    res["risk_transcript"] = fraud.score(res["transcript"].fillna("").tolist())["final"]
    res["verdict_ok"] = (res["risk_transcript"] >= 0.5).astype(int) == res["label"]
    res["verdict_same_as_reference"] = (res["risk_transcript"] >= 0.5) == (res["risk_reference"] >= 0.5)
    res.to_csv(OUT / "per_recording.csv", index=False)

    groups = []
    for (system, lang), g in list(res.groupby(["system", "language"])) + [((s, "all"), res[res.system == s]) for s in args.systems]:
        legit = g[g.label == 0]
        groups.append({
            "system": system, "language": lang, "recordings": len(g),
            "WER": g["wer"].mean(), "CER": g["cer"].mean(),
            "fraud_verdict_correct": g["verdict_ok"].mean(),
            "same_verdict_as_true_text": g["verdict_same_as_reference"].mean(),
            "false_alarms_on_ordinary": (legit["risk_transcript"] >= 0.5).mean() if len(legit) else np.nan,
            "prompt_leaks": int(g["prompt_leak"].sum()),
            "compute_per_audio_second": g["seconds_compute"].sum() / max(g["seconds_audio"].sum(), 1e-9),
        })
    summ = pd.DataFrame(groups)
    summ.to_csv(OUT / "summary.csv", index=False)

    md = ["# Voice mode on real recordings", "",
          f"{len(rec)} recordings ({rec['language'].value_counts().to_dict()}), "
          f"{int(rec['label'].sum())} fraud / {int((rec['label'] == 0).sum())} ordinary.", "",
          "| System | Language | Recordings | WER | CER | Fraud verdict correct | Same verdict as true text | False alarms on ordinary | Hint leaks |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, s in summ.iterrows():
        fa = "—" if pd.isna(s["false_alarms_on_ordinary"]) else f"{s['false_alarms_on_ordinary']:.0%}"
        md.append(f"| {s['system']} | {s['language']} | {s['recordings']} | {s['WER']:.0%} | {s['CER']:.0%} | "
                  f"{s['fraud_verdict_correct']:.0%} | {s['same_verdict_as_true_text']:.0%} | {fa} | {s['prompt_leaks']} |")
    (OUT / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
