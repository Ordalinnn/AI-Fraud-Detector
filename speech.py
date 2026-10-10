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

import hashlib
import importlib.util
import io
import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path

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

# ---- Kazakh model -------------------------------------------------------
# Stock Whisper is weak on Kazakh. akuzdeuov/whisper-base.kk is Whisper-base
# fine-tuned on ISSAI's Kazakh Speech Corpus 2 (~1000 h), 15.4% WER on the
# KSC2 test set, Apache-2.0. It is Kazakh-only, so it is used only when the
# user picks Kazakh. The CTranslate2 conversion is built by
# .github/workflows/build-kazakh-model.yml and published as a release asset;
# the file is checked against a pinned SHA-256 before it is unpacked.
KAZAKH_MODEL = "kk-ksc2"
KAZAKH_MODEL_URL = os.environ.get(
    "WHISPER_KK_MODEL_URL",
    "https://github.com/Ordalinnn/AI-Fraud-Detector/releases/download/"
    "models-whisper-base-kk-v1/whisper-base-kk-ct2-int8.tar.gz",
)
KAZAKH_MODEL_SHA256 = os.environ.get("WHISPER_KK_MODEL_SHA256", "3839de03315cfdce56377cb9227af0bbf884c8d8bcb565e0663ed5e4cc88f56c")
MODEL_CACHE_DIR = Path(os.environ.get("WHISPER_CACHE_DIR", Path.home() / ".cache" / "ai-fraud-detector"))

# ---- Vocabulary hint ----------------------------------------------------
# Whisper's initial prompt biases it towards these spellings, so e.g.
# "код из СМС" stops coming out as "котис СМС". Kept short and phrased as a
# neutral list: a long fraud-like prompt can leak into the transcript of an
# unclear recording and create a false alarm - scripts/eval_speech.py
# measures exactly that on legitimate recordings (prompt on vs off).
# No Kazakh hint: the Kazakh fine-tune was trained without prompts and a
# prompt makes it output garbage (tested: CER 0.31 -> 0.81-0.93).
PROMPTS = {
    "ru": "Звонок. Kaspi, Halyk, банк, карта, код из СМС, перевод, ЭЦП, eGov.",
    "en": "Phone call. Bank, card, SMS code, transfer, PayPal, Amazon.",
}


class TranscriptionError(Exception):
    """Raised with a short machine-readable reason: 'too_large',
    'decode_failed', 'empty_audio', 'unavailable' or 'model_checksum'."""


def is_available() -> bool:
    """True if faster-whisper is installed. The rest of the app works
    without it; voice mode then explains that it's not set up."""
    return importlib.util.find_spec("faster_whisper") is not None


def model_for_language(language: str | None) -> str:
    """Which model to use: the Kazakh fine-tune for Kazakh, otherwise the
    general multilingual model (Russian, English, auto-detect)."""
    return KAZAKH_MODEL if language == "kk" else DEFAULT_MODEL_SIZE


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_kazakh_model(cache_dir: Path = MODEL_CACHE_DIR, url: str = KAZAKH_MODEL_URL,
                        sha256: str = KAZAKH_MODEL_SHA256) -> Path:
    """Downloads and unpacks the Kazakh model once; returns its directory."""
    target = Path(cache_dir) / "whisper-base-kk-ct2"
    if (target / "model.bin").exists():
        return target
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(url, tmp_path)  # noqa: S310 - fixed https URL
        actual = _sha256(tmp_path)
        if actual != sha256:
            raise TranscriptionError("model_checksum")
        with tarfile.open(tmp_path, "r:gz") as tar:
            tar.extractall(cache_dir, filter="data")  # refuses absolute/.. paths
    finally:
        tmp_path.unlink(missing_ok=True)
    if not (target / "model.bin").exists():
        raise TranscriptionError("model_checksum")
    return target


def load_model(spec: str = DEFAULT_MODEL_SIZE):
    """Loads a Whisper model: a size name ("small", "base"...) downloaded by
    faster-whisper, or KAZAKH_MODEL. Callers should cache the result -
    loading takes seconds and hundreds of MB."""
    if not is_available():
        raise TranscriptionError("unavailable")
    from faster_whisper import WhisperModel
    path = str(ensure_kazakh_model()) if spec == KAZAKH_MODEL else spec
    return WhisperModel(path, device="cpu", compute_type="int8")


def normalize_loudness(samples: np.ndarray, target_rms: float = 0.1, max_gain: float = 20.0) -> np.ndarray:
    """Brings quiet speakerphone recordings up to a standard level (about
    -20 dBFS RMS) without clipping. Gain is capped so pure background noise
    isn't blown up."""
    rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0
    if rms < 1e-6:
        return samples
    gain = min(target_rms / rms, max_gain)
    peak = float(np.max(np.abs(samples))) * gain
    if peak > 0.99:
        gain *= 0.99 / peak
    return (samples * gain).astype(np.float32)


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
               max_seconds: float = MAX_AUDIO_SECONDS, use_prompt: bool = True,
               normalize: bool = True) -> dict:
    """Audio bytes -> transcript dict:
        text, language, language_probability, duration, truncated.
    `language` is "kk", "ru", "en" or None/"auto" for auto-detection
    (auto-detection sometimes confuses Kazakh with Russian, so the UI lets
    the user pick)."""
    samples, duration, truncated = decode(audio_bytes, max_seconds)
    if normalize:
        samples = normalize_loudness(samples)
    lang = None if language in (None, "", "auto") else language
    segments, info = model.transcribe(
        samples,
        language=lang,
        initial_prompt=PROMPTS.get(lang) if use_prompt else None,
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
