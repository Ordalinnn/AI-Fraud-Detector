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

# espeak-ng is a robotic voice (its Kazakh especially), so this checks that
# the pipeline works end to end, not real-world accuracy - that needs real
# recordings. With "small", espeak's "код" sometimes comes out as "кот".
CASES = [
    # (espeak voice, whisper language, phrase, words that must survive)
    ("ru", "ru", "Здравствуйте, это служба безопасности банка. Ваша карта заблокирована. "
                 "Назовите код из СМС, чтобы отменить перевод.", ["карт", "смс"]),
    ("kk", "kk", "Сәлеметсіз бе, бұл банктің қауіпсіздік қызметі. Сіздің картаңыз бұғатталды. "
                 "Аударымды тоқтату үшін смс кодты айтыңыз.", []),
]


def gh_annotation(level, title, msg):
    """Also emit a GitHub Actions annotation, so results are readable from
    the PR's checks page / API without downloading the full job log."""
    msg = str(msg).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"\n::{level} title={title}::{msg}")


@pytest.fixture(scope="module")
def model():
    t = time.time()
    try:
        m = speech.load_model(speech.DEFAULT_MODEL_SIZE)
    except Exception as exc:
        gh_annotation("error", "Whisper load failed", repr(exc))
        raise
    gh_annotation("notice", "Whisper loaded", f"'{speech.DEFAULT_MODEL_SIZE}' in {time.time() - t:.1f}s")
    return m


@pytest.mark.parametrize("voice,lang,phrase,must", CASES)
def test_real_transcription(model, tmp_path, voice, lang, phrase, must):
    wav = tmp_path / f"{voice}.wav"
    subprocess.run(["espeak-ng", "-v", voice, "-s", "140", "-w", str(wav), phrase], check=True)
    t = time.time()
    try:
        r = speech.transcribe(model, wav.read_bytes(), language=lang)
    except Exception as exc:
        gh_annotation("error", f"transcribe {voice} failed", repr(exc))
        raise
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    gh_annotation("notice", f"speech {voice}",
                  f"{r['duration']:.1f}s audio -> {time.time() - t:.1f}s, peak RSS {rss_mb:.0f} MB: {r['text']!r}")
    assert r["text"]
    low = r["text"].lower()
    for w in must:
        assert w in low, f"expected {w!r} in transcript {r['text']!r}"
