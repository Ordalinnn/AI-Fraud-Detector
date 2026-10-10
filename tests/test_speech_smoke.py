"""
End-to-end check of voice mode with the REAL Whisper model and synthetic
speech (espeak-ng). Downloads the model (~500 MB for "small"), so it only
runs when RUN_SPEECH_SMOKE=1 - in CI it's the separate `speech-smoke` job.
Prints transcription time and memory so the team can judge whether the
deployment server is big enough.
"""
import os
import resource
import shutil
import subprocess
import time

import pytest

import speech

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SPEECH_SMOKE") != "1" or shutil.which("espeak-ng") is None,
    reason="set RUN_SPEECH_SMOKE=1 and install espeak-ng to run",
)

CASES = [
    # (espeak voice, whisper language, phrase, words that must survive)
    ("ru", "ru", "Здравствуйте, это служба безопасности банка. Ваша карта заблокирована. "
                 "Назовите код из СМС, чтобы отменить перевод.", ["карт", "код"]),
    ("kk", "kk", "Сәлеметсіз бе, бұл банктің қауіпсіздік қызметі. Сіздің картаңыз бұғатталды. "
                 "Аударымды тоқтату үшін смс кодты айтыңыз.", []),
]


@pytest.fixture(scope="module")
def model():
    t = time.time()
    m = speech.load_model(speech.DEFAULT_MODEL_SIZE)
    print(f"\nloaded Whisper '{speech.DEFAULT_MODEL_SIZE}' in {time.time() - t:.1f}s")
    return m


@pytest.mark.parametrize("voice,lang,phrase,must", CASES)
def test_real_transcription(model, tmp_path, voice, lang, phrase, must):
    wav = tmp_path / f"{voice}.wav"
    subprocess.run(["espeak-ng", "-v", voice, "-s", "140", "-w", str(wav), phrase], check=True)
    t = time.time()
    r = speech.transcribe(model, wav.read_bytes(), language=lang)
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[{voice}] {r['duration']:.1f}s audio -> {time.time() - t:.1f}s, peak RSS {rss_mb:.0f} MB: {r['text']!r}")
    assert r["text"]
    low = r["text"].lower()
    for w in must:
        assert w in low, f"expected {w!r} in transcript {r['text']!r}"
