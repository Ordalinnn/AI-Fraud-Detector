"""
Speech-to-text for the app's voice mode (no Streamlit dependency).

Why this exists: phones don't let ordinary apps record or listen to calls
(Android 10+ reserves call audio for system apps, iOS gives none), so the
app can't analyse calls automatically. What it CAN do is take a recording
the user makes themselves - the call on speakerphone recorded with the
browser's microphone, a voice message (WhatsApp/Telegram .ogg/.opus), or a
recording file - turn it into text with Whisper, and pass that text to the
same fraud model as a typed call transcript.

Whisper runs locally inside the app (faster-whisper / CTranslate2, CPU,
int8), so the audio is never sent to a third-party service and is not
stored: it lives in memory only for the duration of one transcription.
"""
from __future__ import annotations

import importlib.util
import io
import os

import numpy as np

SAMPLE_RATE = 16_000
# Whisper "small" is the smallest size that handles Kazakh acceptably;
# "base"/"tiny" are much worse on Kazakh. Override with the WHISPER_MODEL
# environment variable (e.g. "base" if the server runs out of memory).
DEFAULT_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")
MAX_AUDIO_SECONDS = 5 * 60          # longer recordings are cut, with a warning
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # refuse anything bigger before decoding
SUPPORTED_AUDIO_TYPES = ["wav", "mp3", "m4a", "ogg", "opus", "oga", "webm", "aac", "flac", "mp4", "amr", "3gp"]
LANGUAGES = ["auto", "kk", "ru", "en"]


class TranscriptionError(Exception):
    """Raised with a short machine-readable reason: 'too_large',
    'decode_failed', 'empty_audio' or 'unavailable'."""


def is_available() -> bool:
    """True if faster-whisper is installed. The rest of the app works
    without it; voice mode then explains that it's not set up."""
    return importlib.util.find_spec("faster_whisper") is not None


def load_model(size: str = DEFAULT_MODEL_SIZE):
    """Loads (downloading on first use, ~250-500 MB for "small") a Whisper
    model. Callers should cache the result - loading takes seconds."""
    if not is_available():
        raise TranscriptionError("unavailable")
    from faster_whisper import WhisperModel
    return WhisperModel(size, device="cpu", compute_type="int8")


def decode(audio_bytes: bytes, max_seconds: float = MAX_AUDIO_SECONDS) -> tuple[np.ndarray, float, bool]:
    """Decodes any common audio container to 16 kHz mono float32.
    Returns (samples, original_duration_seconds, was_truncated).
    Done before transcription so an over-long recording is cut first
    instead of being fully transcribed and then thrown away."""
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise TranscriptionError("too_large")
    if not is_available():
        raise TranscriptionError("unavailable")
    from faster_whisper import decode_audio
    try:
        samples = decode_audio(io.BytesIO(audio_bytes), sampling_rate=SAMPLE_RATE)
    except Exception as exc:  # PyAV raises many different error types
        raise TranscriptionError("decode_failed") from exc
    duration = len(samples) / SAMPLE_RATE
    if duration < 0.3 or not np.any(np.abs(samples) > 1e-4):
        raise TranscriptionError("empty_audio")
    limit = int(max_seconds * SAMPLE_RATE)
    truncated = len(samples) > limit
    return samples[:limit], duration, truncated


def transcribe(model, audio_bytes: bytes, language: str | None = None,
               max_seconds: float = MAX_AUDIO_SECONDS) -> dict:
    """Audio bytes -> transcript dict:
        text, language, language_probability, duration, truncated.
    `language` is "kk", "ru", "en" or None/"auto" for auto-detection
    (auto-detection sometimes confuses Kazakh with Russian, so the UI lets
    the user pick)."""
    samples, duration, truncated = decode(audio_bytes, max_seconds)
    lang = None if language in (None, "", "auto") else language
    segments, info = model.transcribe(
        samples,
        language=lang,
        beam_size=5,
        vad_filter=True,                  # skip silence / hold music
        condition_on_previous_text=False,  # fewer repetition loops on noisy calls
    )
    text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
    return {
        "text": text.strip(),
        "language": info.language,
        "language_probability": float(info.language_probability),
        "duration": float(duration),
        "truncated": bool(truncated),
    }
