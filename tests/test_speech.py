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
