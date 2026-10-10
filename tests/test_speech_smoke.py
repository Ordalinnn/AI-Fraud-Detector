"""
End-to-end check of voice mode with the REAL Whisper models on synthetic
speech (espeak-ng). Downloads ~0.6 GB, so it only runs when
RUN_SPEECH_SMOKE=1 - in CI it's the separate `speech-smoke` job.

espeak-ng is a robotic voice (its Kazakh especially), so this checks that
the pipeline works end to end and compares the old and new settings on the
same audio; real-world accuracy needs real recordings
(scripts/eval_speech.py). Results are emitted as GitHub annotations so
they're readable from the PR's checks page.
"""
import importlib.util
import os
import resource
import shutil
import subprocess
import time
from pathlib import Path

import pytest

import speech

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SPEECH_SMOKE") != "1" or shutil.which("espeak-ng") is None,
    reason="set RUN_SPEECH_SMOKE=1 and install espeak-ng to run",
)

_spec = importlib.util.spec_from_file_location(
    "eval_speech", Path(__file__).resolve().parent.parent / "scripts" / "eval_speech.py")
es = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(es)

PHRASES = [
    # (id, espeak voice / language, label, text)
    ("ru_fraud", "ru", 1, "Здравствуйте, это служба безопасности банка. Ваша карта заблокирована. "
                          "Назовите код из СМС, чтобы отменить перевод."),
    ("ru_ok", "ru", 0, "Привет, я буду через десять минут. Урок математики перенесли на пятницу."),
    ("kk_fraud", "kk", 1, "Сәлеметсіз бе, бұл банктің қауіпсіздік қызметі. Сіздің картаңыз бұғатталды. "
                          "Аударымды тоқтату үшін смс кодты айтыңыз."),
    ("kk_ok", "kk", 0, "Сәлем, мен он минуттан кейін келемін. Сабақ жұмаға ауыстырылды."),
]
SYSTEMS = {
    "baseline": dict(model=lambda lang: "small", use_prompt=False, normalize=False),
    "improved": dict(model=speech.model_for_language, use_prompt=True, normalize=True),
}


def gh_annotation(level, title, msg):
    msg = str(msg).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"\n::{level} title={title}::{msg}")


@pytest.fixture(scope="module")
def models():
    cache = {}

    def get(spec):
        if spec not in cache:
            t = time.time()
            try:
                cache[spec] = speech.load_model(spec)
            except Exception as exc:
                gh_annotation("error", f"load {spec} failed", repr(exc))
                raise
            gh_annotation("notice", f"loaded {spec}", f"{time.time() - t:.1f}s")
        return cache[spec]
    return get


@pytest.fixture(scope="module")
def audio(tmp_path_factory):
    d = tmp_path_factory.mktemp("speech")
    out = {}
    for pid, voice, _, text in PHRASES:
        wav = d / f"{pid}.wav"
        subprocess.run(["espeak-ng", "-v", voice, "-s", "140", "-w", str(wav), text], check=True)
        out[pid] = wav.read_bytes()
    return out


@pytest.mark.parametrize("pid,lang,label,text", PHRASES, ids=[p[0] for p in PHRASES])
def test_old_vs_new_settings(models, audio, pid, lang, label, text):
    results = {}
    for name, cfg in SYSTEMS.items():
        spec = cfg["model"](lang)
        t = time.time()
        try:
            r = speech.transcribe(models(spec), audio[pid], language=lang,
                                  use_prompt=cfg["use_prompt"], normalize=cfg["normalize"])
        except Exception as exc:
            gh_annotation("error", f"{pid} {name} failed", repr(exc))
            raise
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        results[name] = r
        leak = es.prompt_leak(text, r["text"], lang) if cfg["use_prompt"] else False
        gh_annotation("notice", f"{pid} {name}",
                      f"model={spec} CER={es.cer(text, r['text']):.2f} WER={es.wer(text, r['text']):.2f} "
                      f"leak={leak} {time.time() - t:.1f}s RSS={rss:.0f}MB: {r['text']!r}")
        assert r["text"]
        if label == 0:
            assert not leak, f"vocabulary hint leaked into an ordinary call: {r['text']!r}"
