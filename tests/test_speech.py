"""
Tests for speech.py (voice mode) and its wiring in app.py. A fake Whisper
model is used so the tests need no model download; the real model is
exercised by the optional `speech-smoke` CI job.
"""
import io
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import speech

pytestmark = pytest.mark.skipif(not speech.is_available(), reason="faster-whisper not installed")

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def make_wav(seconds=1.0, freq=440.0, amplitude=0.3, rate=16_000):
    t = np.arange(int(seconds * rate)) / rate
    pcm = (amplitude * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class FakeModel:
    def __init__(self, text="Ваша карта заблокирована, назовите код из SMS", language="ru"):
        self.text = text
        self.language = language
        self.calls = []

    def transcribe(self, samples, language=None, **kwargs):
        self.calls.append({"n_samples": len(samples), "language": language, **kwargs})
        segs = [SimpleNamespace(text=" " + part) for part in self.text.split(", ")]
        return iter(segs), SimpleNamespace(language=language or self.language, language_probability=0.97)


# ---------- decode ----------

def test_decode_wav_duration_and_rate():
    samples, duration, truncated = speech.decode(make_wav(2.0))
    assert duration == pytest.approx(2.0, abs=0.05)
    assert len(samples) == pytest.approx(2 * speech.SAMPLE_RATE, abs=800)
    assert not truncated


def test_decode_truncates_long_audio():
    samples, duration, truncated = speech.decode(make_wav(3.0), max_seconds=1.0)
    assert truncated and duration == pytest.approx(3.0, abs=0.05)
    assert len(samples) == speech.SAMPLE_RATE


@pytest.mark.parametrize("data,reason", [
    (b"definitely not audio" * 10, "decode_failed"),
    (make_wav(1.0, amplitude=0.0), "empty_audio"),
    (b"\0" * (speech.MAX_AUDIO_BYTES + 1), "too_large"),
])
def test_decode_rejects_bad_input(data, reason):
    with pytest.raises(speech.TranscriptionError) as e:
        speech.decode(data)
    assert str(e.value) == reason


# ---------- transcribe ----------

def test_transcribe_joins_segments_and_passes_language():
    model = FakeModel()
    r = speech.transcribe(model, make_wav(1.5), language="kk")
    assert r["text"] == "Ваша карта заблокирована назовите код из SMS"
    assert model.calls[0]["language"] == "kk"
    assert model.calls[0]["vad_filter"] is True
    assert r["duration"] == pytest.approx(1.5, abs=0.05)


def test_auto_language_means_detection():
    model = FakeModel()
    r = speech.transcribe(model, make_wav(1.0), language="auto")
    assert model.calls[0]["language"] is None
    assert r["language"] == "ru"


# ---------- app wiring ----------

def test_voice_mode_transcript_lands_in_text_box_and_is_analysed(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(speech, "load_model", lambda size=None: FakeModel())

    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    at.selectbox(key="mode_select").set_value("voice").run()
    assert not at.exception, at.exception

    at.file_uploader(key="voice_file_uploader").set_value(("call.wav", make_wav(1.0), "audio/wav"))
    at.run()
    at.button(key="voice_transcribe_btn").click().run()
    assert not at.exception, at.exception
    assert "заблокирована" in at.text_area(key="main_input_text").value

    submit = [b for b in at.button if "analysis_form" in b.key][0]
    submit.click().run()
    assert not at.exception, at.exception
    assert at.session_state["last_result"]["prob"] > 0.5


def test_only_one_transcription_runs_at_a_time():
    with speech.transcription_slot(timeout=0):
        with pytest.raises(speech.TranscriptionError, match="busy"):
            with speech.transcription_slot(timeout=0):
                pass
    with speech.transcription_slot(timeout=0):  # released again afterwards
        pass


def test_voice_mode_tells_user_to_wait_when_another_transcription_runs(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from translations import get_translations

    monkeypatch.chdir(tmp_path)
    model = FakeModel()
    monkeypatch.setattr(speech, "load_model", lambda size=None: model)
    monkeypatch.setattr(speech, "TRANSCRIPTION_WAIT_SECONDS", 0.1)

    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    at.selectbox(key="mode_select").set_value("voice").run()
    at.file_uploader(key="voice_file_uploader").set_value(("call.wav", make_wav(1.0), "audio/wav"))
    at.run()
    with speech.transcription_slot(timeout=0):  # someone else is transcribing
        at.button(key="voice_transcribe_btn").click().run()
    assert not at.exception, at.exception
    assert not model.calls
    busy = get_translations(at.session_state["lang"])["voice_busy"]
    assert [e.value for e in at.error] == [busy]


def test_transcription_settings_keep_cpu_use_short():
    model = FakeModel()
    speech.transcribe(model, make_wav(1.0), language="ru")
    assert model.calls[0]["beam_size"] == speech.BEAM_SIZE == 1
    assert speech.CPU_THREADS == 2
    assert speech.MAX_AUDIO_SECONDS == 2 * 60


# ---------- improvements: model choice, prompt, loudness, Kazakh download ----------

def test_kazakh_gets_the_kazakh_model_others_the_general_one():
    assert speech.model_for_language("kk") == speech.KAZAKH_MODEL
    for lang in ("ru", "en", "auto", None):
        assert speech.model_for_language(lang) == speech.DEFAULT_MODEL_SIZE


def test_vocabulary_hint_is_passed_and_can_be_disabled():
    model = FakeModel()
    speech.transcribe(model, make_wav(1.0), language="ru")
    speech.transcribe(model, make_wav(1.0), language="ru", use_prompt=False)
    speech.transcribe(model, make_wav(1.0), language="auto")
    assert model.calls[0]["initial_prompt"] == speech.PROMPTS["ru"]
    assert model.calls[1]["initial_prompt"] is None
    assert model.calls[2]["initial_prompt"] is None  # no hint when the language is unknown


def test_loudness_normalisation_boosts_quiet_audio_without_clipping():
    quiet = (0.01 * np.sin(np.linspace(0, 200, 16000))).astype(np.float32)
    out = speech.normalize_loudness(quiet)
    assert np.sqrt(np.mean(out ** 2)) > 5 * np.sqrt(np.mean(quiet ** 2))
    assert np.max(np.abs(out)) <= 0.99
    loud = (0.95 * np.sign(np.sin(np.linspace(0, 200, 16000)))).astype(np.float32)
    assert np.max(np.abs(speech.normalize_loudness(loud))) <= 0.99
    silence = np.zeros(1000, dtype=np.float32)
    assert np.array_equal(speech.normalize_loudness(silence), silence)


def _model_tarball(tmp_path):
    import tarfile
    src = tmp_path / "src" / "whisper-base-kk-ct2"
    src.mkdir(parents=True)
    (src / "model.bin").write_bytes(b"weights")
    tgz = tmp_path / "model.tar.gz"
    with tarfile.open(tgz, "w:gz") as tar:
        tar.add(src, arcname="whisper-base-kk-ct2")
    return tgz


def test_kazakh_model_download_is_checksum_verified(tmp_path):
    tgz = _model_tarball(tmp_path)
    url = tgz.as_uri()
    good = speech._sha256(tgz)
    with pytest.raises(speech.TranscriptionError, match="model_checksum"):
        speech.ensure_kazakh_model(tmp_path / "cache_bad", url=url, sha256="0" * 64)
    assert not (tmp_path / "cache_bad" / "whisper-base-kk-ct2").exists()
    path = speech.ensure_kazakh_model(tmp_path / "cache", url=url, sha256=good)
    assert (path / "model.bin").read_bytes() == b"weights"
    # second call uses the cache (would fail if it tried the bad checksum again)
    assert speech.ensure_kazakh_model(tmp_path / "cache", url=url, sha256="0" * 64) == path


def test_speech_metrics():
    import importlib.util as u
    spec = u.spec_from_file_location("eval_speech", Path(__file__).resolve().parent.parent / "scripts" / "eval_speech.py")
    es = u.module_from_spec(spec)
    spec.loader.exec_module(es)
    assert es.wer("Ваша карта заблокирована", "ваша карта заблокирована!") == 0
    assert es.wer("назовите код из смс", "назовите котис смс") == pytest.approx(0.5)
    assert es.cer("код", "кот") == pytest.approx(1 / 3)
    assert es.prompt_leak("привет как дела", "привет kaspi как дела", "ru")
    assert not es.prompt_leak("переведи на kaspi", "переведи на kaspi", "ru")
