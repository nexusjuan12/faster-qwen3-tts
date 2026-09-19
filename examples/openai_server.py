#!/usr/bin/env python3
"""
OpenAI-compatible TTS API server for faster-qwen3-tts.

Exposes POST /v1/audio/speech compatible with OpenAI's TTS API, enabling
integration with OpenWebUI, llama-swap, and other OpenAI-compatible clients.

Usage:
    pip install "faster-qwen3-tts[demo]"

    # Single default voice:
    python examples/openai_server.py \\
        --ref-audio voice.wav --ref-text "Reference transcription" \\
        --language English

    # Multiple named voices from a JSON config:
    python examples/openai_server.py --voices voices.json

    # Custom model and port:
    python examples/openai_server.py \\
        --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \\
        --ref-audio voice.wav --ref-text "transcript" \\
        --port 8000

Voices config (voices.json):
    {
        "alloy": {"ref_audio": "voice.wav", "ref_text": "...", "language": "English"},
        "echo":  {"ref_audio": "voice2.wav", "ref_text": "...", "language": "English"}
    }

API usage:
    curl -s http://localhost:8000/v1/audio/speech \\
        -H "Content-Type: application/json" \\
        -d '{"model": "tts-1", "input": "Hello!", "voice": "alloy", "response_format": "wav"}' \\
        --output speech.wav
"""
import argparse
import asyncio
import io
import json
import logging
import os
import queue
import re
import struct
import sys
import threading
import uuid
from pathlib import Path
from typing import AsyncGenerator, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

app = FastAPI(title="faster-qwen3-tts OpenAI-compatible API")

tts_model = None
voices: dict = {}
default_voice: Optional[str] = None
SAMPLE_RATE = 24000  # updated once the model loads
BACKEND = "torch"
_model_lock = threading.Lock()  # prevent concurrent GPU inference
_voice_registry_lock = threading.RLock()
MANAGED_VOICE_DIR: Optional[Path] = None
MANAGED_VOICE_REGISTRY: Optional[Path] = None
managed_voice_metadata: dict[str, dict] = {}
_VOICE_ID_RE = re.compile(r"^voice-[0-9a-f-]{36}$")
_ALLOWED_VOICE_SUFFIXES = {".wav", ".mp3", ".ogg", ".flac"}
_MAX_VOICE_SAMPLE_BYTES = 25 * 1024 * 1024
_MIN_VOICE_SAMPLE_SECONDS = 2
_MAX_VOICE_SAMPLE_SECONDS = 180

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "alloy"
    response_format: str = "wav"  # wav | pcm | mp3
    speed: float = 1.0           # accepted but not yet applied


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _to_pcm16(pcm: np.ndarray) -> bytes:
    """Convert float32 numpy array to raw 16-bit little-endian PCM bytes."""
    return np.clip(pcm * 32768, -32768, 32767).astype(np.int16).tobytes()


def _wav_header(sample_rate: int, data_len: int = 0xFFFFFFFF) -> bytes:
    """Build a WAV header.  Use data_len=0xFFFFFFFF for streaming (unknown size)."""
    n_channels = 1
    bits = 16
    byte_rate = sample_rate * n_channels * bits // 8
    block_align = n_channels * bits // 8
    riff_size = 0xFFFFFFFF if data_len == 0xFFFFFFFF else 36 + data_len
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", riff_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, n_channels, sample_rate,
                          byte_rate, block_align, bits))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_len))
    return buf.getvalue()


def _to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 numpy array to a complete WAV file in memory."""
    raw = _to_pcm16(pcm)
    return _wav_header(sample_rate, len(raw)) + raw


def _to_mp3_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 numpy array to MP3 bytes (requires pydub + ffmpeg)."""
    try:
        from pydub import AudioSegment
    except ImportError:
        raise HTTPException(
            status_code=400,
            detail="response_format='mp3' requires pydub: pip install pydub",
        )
    segment = AudioSegment(
        _to_pcm16(pcm),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )
    buf = io.BytesIO()
    segment.export(buf, format="mp3")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Voice resolution
# ---------------------------------------------------------------------------


def resolve_voice(voice_name: str) -> dict:
    """Return voice config dict or fall back to default, else raise 400."""
    if voice_name in voices:
        return voices[voice_name]
    if default_voice and default_voice in voices:
        logger.warning(
            "Voice %r not configured; falling back to default voice %r",
            voice_name,
            default_voice,
        )
        return voices[default_voice]
    raise HTTPException(
        status_code=400,
        detail=(
            f"Voice {voice_name!r} is not configured. "
            f"Available voices: {list(voices.keys())}"
        ),
    )


def _managed_audio_path(voice_id: str) -> Path:
    if MANAGED_VOICE_DIR is None:
        raise HTTPException(status_code=503, detail="Managed voice storage is unavailable")
    return MANAGED_VOICE_DIR / f"{voice_id}.wav"


def _persist_managed_voices() -> None:
    """Atomically persist only provider-managed profiles, never built-in voices."""
    if MANAGED_VOICE_REGISTRY is None:
        raise RuntimeError("Managed voice registry is not configured")
    temporary = MANAGED_VOICE_REGISTRY.with_suffix(".tmp")
    temporary.write_text(json.dumps(managed_voice_metadata, indent=2, sort_keys=True))
    os.chmod(temporary, 0o600)
    temporary.replace(MANAGED_VOICE_REGISTRY)


def _load_managed_voices() -> None:
    """Restore managed profiles whose canonical audio files still exist."""
    if MANAGED_VOICE_DIR is None or MANAGED_VOICE_REGISTRY is None:
        return
    MANAGED_VOICE_DIR.mkdir(parents=True, exist_ok=True)
    if not MANAGED_VOICE_REGISTRY.is_file():
        return
    try:
        stored = json.loads(MANAGED_VOICE_REGISTRY.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable managed voice registry: %s", exc)
        return
    if not isinstance(stored, dict):
        logger.warning("Ignoring invalid managed voice registry")
        return
    for voice_id, metadata in stored.items():
        if not _VOICE_ID_RE.fullmatch(voice_id) or not isinstance(metadata, dict):
            continue
        audio_path = _managed_audio_path(voice_id)
        reference_text = str(metadata.get("reference_text", "")).strip()
        if not audio_path.is_file() or not reference_text:
            continue
        managed_voice_metadata[voice_id] = {
            "display_name": str(metadata.get("display_name", voice_id))[:100],
            "reference_text": reference_text[:2000],
            "language": str(metadata.get("language", "English"))[:40],
        }
        voices[voice_id] = {
            "ref_audio": str(audio_path),
            "ref_text": managed_voice_metadata[voice_id]["reference_text"],
            "language": managed_voice_metadata[voice_id]["language"],
        }
    logger.info("Restored %d managed voice profile(s)", len(managed_voice_metadata))


def _remove_managed_voice_cache(audio_path: Path) -> None:
    """Remove this profile's local GGML speaker cache when it can be identified."""
    if BACKEND != "ggml" or tts_model is None:
        return
    try:
        from faster_qwen3_tts.ggml_backend import _load_ref_audio_24k

        with _model_lock:
            reference_audio = _load_ref_audio_24k(audio_path, append_silence=True)
            cache_key, _metadata = tts_model._voice_ref_cache_key(reference_audio, append_silence=True)
            for cache_path in tts_model._voice_ref_paths(cache_key):
                cache_path.unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("Could not remove cached reference for %s: %s", audio_path.name, exc)


# ---------------------------------------------------------------------------
# Streaming helper: run sync generator in a background thread
# ---------------------------------------------------------------------------


def _max_generation_frames(text: str) -> int:
    """Bound runaway synthesis while leaving normal responses enough headroom."""
    word_count = len(re.findall(r"\S+", text))
    return min(640, max(96, 48 + word_count * 4))


async def _stream_chunks(voice_cfg: dict, text: str) -> AsyncGenerator[bytes, None]:
    """
    Run generate_voice_clone_streaming in a background thread and yield
    raw PCM bytes for each chunk as they arrive.
    """
    q: queue.Queue = queue.Queue()
    _DONE = object()

    def producer():
        try:
            with _model_lock:
                for chunk, _sr, _timing in tts_model.generate_voice_clone_streaming(
                    text=text,
                    language=voice_cfg.get("language", "Auto"),
                    ref_audio=voice_cfg["ref_audio"],
                    ref_text=voice_cfg.get("ref_text", ""),
                    chunk_size=voice_cfg.get("chunk_size", 12),
                    max_new_tokens=_max_generation_frames(text),
                    # qwentts.cpp streams natively but does not support the
                    # Torch backend's step-by-step text feeding switch.
                    non_streaming_mode=False if BACKEND == "torch" else True,
                ):
                    q.put(chunk)
        except Exception as exc:
            q.put(exc)
        finally:
            q.put(_DONE)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    loop = asyncio.get_event_loop()
    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is _DONE:
            break
        if isinstance(item, Exception):
            raise item
        yield _to_pcm16(item)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": tts_model is not None,
        "backend": BACKEND,
        "sample_rate": SAMPLE_RATE,
        "voices": list(voices),
    }


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="'input' text is empty")

    voice_cfg = resolve_voice(req.voice)
    fmt = req.response_format.lower()

    _CONTENT_TYPES = {
        "wav": "audio/wav",
        "pcm": "audio/pcm",
        "mp3": "audio/mpeg",
    }
    if fmt not in _CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"response_format {fmt!r} not supported. Use: wav, pcm, mp3",
        )
    content_type = _CONTENT_TYPES[fmt]

    # --- MP3: generate all audio, then encode (non-streaming) ---
    if fmt == "mp3":
        loop = asyncio.get_event_loop()

        def _generate():
            with _model_lock:
                return tts_model.generate_voice_clone(
                    text=req.input,
                    language=voice_cfg.get("language", "Auto"),
                    ref_audio=voice_cfg["ref_audio"],
                    ref_text=voice_cfg.get("ref_text", ""),
                    max_new_tokens=_max_generation_frames(req.input),
                )

        audio_arrays, sr = await loop.run_in_executor(None, _generate)
        audio = audio_arrays[0] if audio_arrays else np.zeros(1, dtype=np.float32)
        return Response(content=_to_mp3_bytes(audio, sr), media_type=content_type)

    # --- WAV / PCM: stream chunks as they are generated ---
    async def audio_stream():
        if fmt == "wav":
            yield _wav_header(SAMPLE_RATE)  # stream with unknown data length
        async for raw_chunk in _stream_chunks(voice_cfg, req.input):
            yield raw_chunk

    return StreamingResponse(audio_stream(), media_type=content_type)


@app.post("/v1/voices", status_code=201)
async def create_voice(
    sample: UploadFile = File(...),
    reference_text: str = Form(...),
    display_name: str = Form(...),
    language: str = Form("English"),
):
    """Register a consented voice-clone sample for private provider use."""
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if MANAGED_VOICE_DIR is None:
        raise HTTPException(status_code=503, detail="Managed voice storage is unavailable")
    reference_text = reference_text.strip()
    display_name = display_name.strip()
    language = language.strip() or "English"
    if not reference_text or not display_name:
        raise HTTPException(status_code=400, detail="display_name and reference_text are required")
    if len(reference_text) > 2000 or len(display_name) > 100 or len(language) > 40:
        raise HTTPException(status_code=400, detail="Voice profile metadata is too long")
    suffix = Path(sample.filename or "").suffix.lower()
    if suffix not in _ALLOWED_VOICE_SUFFIXES:
        raise HTTPException(status_code=400, detail="Supported sample formats: WAV, MP3, OGG, FLAC")

    voice_id = f"voice-{uuid.uuid4()}"
    temporary_source = MANAGED_VOICE_DIR / f".{voice_id}{suffix}"
    canonical_audio = _managed_audio_path(voice_id)
    total = 0
    try:
        with temporary_source.open("wb") as destination:
            while chunk := await sample.read(1024 * 1024):
                total += len(chunk)
                if total > _MAX_VOICE_SAMPLE_BYTES:
                    raise HTTPException(status_code=413, detail="Voice sample exceeds 25 MB")
                destination.write(chunk)
        if not total:
            raise HTTPException(status_code=400, detail="Voice sample is empty")

        from pydub import AudioSegment

        audio = AudioSegment.from_file(temporary_source)
        duration_seconds = len(audio) / 1000
        if not _MIN_VOICE_SAMPLE_SECONDS <= duration_seconds <= _MAX_VOICE_SAMPLE_SECONDS:
            raise HTTPException(
                status_code=400,
                detail=f"Voice samples must be {_MIN_VOICE_SAMPLE_SECONDS}-{_MAX_VOICE_SAMPLE_SECONDS} seconds",
            )
        audio.set_channels(1).set_frame_rate(SAMPLE_RATE).export(canonical_audio, format="wav")
        os.chmod(canonical_audio, 0o600)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Could not prepare managed voice sample")
        raise HTTPException(status_code=400, detail="Voice sample could not be decoded") from exc
    finally:
        temporary_source.unlink(missing_ok=True)

    metadata = {
        "display_name": display_name,
        "reference_text": reference_text,
        "language": language,
    }
    try:
        with _voice_registry_lock:
            managed_voice_metadata[voice_id] = metadata
            voices[voice_id] = {
                "ref_audio": str(canonical_audio),
                "ref_text": reference_text,
                "language": language,
            }
            _persist_managed_voices()
    except Exception:
        canonical_audio.unlink(missing_ok=True)
        with _voice_registry_lock:
            managed_voice_metadata.pop(voice_id, None)
            voices.pop(voice_id, None)
        raise
    return {"id": voice_id, "display_name": display_name, "language": language}


@app.delete("/v1/voices/{voice_id}", status_code=204)
async def delete_voice(voice_id: str):
    """Remove a provider-managed voice profile and its local cache material."""
    if not _VOICE_ID_RE.fullmatch(voice_id):
        raise HTTPException(status_code=404, detail="Voice profile not found")
    with _voice_registry_lock:
        if voice_id not in managed_voice_metadata:
            raise HTTPException(status_code=404, detail="Voice profile not found")
        audio_path = _managed_audio_path(voice_id)
        _remove_managed_voice_cache(audio_path)
        managed_voice_metadata.pop(voice_id, None)
        voices.pop(voice_id, None)
        _persist_managed_voices()
    audio_path.unlink(missing_ok=True)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="OpenAI-compatible TTS server for faster-qwen3-tts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model",
        default=os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
        help="HuggingFace model ID or local path (default: Qwen/Qwen3-TTS-12Hz-1.7B-Base)",
    )
    p.add_argument(
        "--voices",
        default=os.environ.get("QWEN_TTS_VOICES"),
        metavar="FILE",
        help="JSON file mapping voice names to {ref_audio, ref_text, language}",
    )
    p.add_argument(
        "--ref-audio",
        default=os.environ.get("QWEN_TTS_REF_AUDIO"),
        metavar="FILE",
        help="Reference audio file when --voices is not used",
    )
    p.add_argument(
        "--ref-text",
        default=os.environ.get("QWEN_TTS_REF_TEXT", ""),
        help="Transcript of --ref-audio",
    )
    p.add_argument(
        "--language",
        default=os.environ.get("QWEN_TTS_LANGUAGE", "Auto"),
        help="Target language (English, French, Auto, …) when --voices is not used",
    )
    p.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    p.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
    p.add_argument(
        "--backend",
        choices=["torch", "ggml"],
        default=os.environ.get("QWEN_TTS_BACKEND", "torch"),
        help="Inference backend (default: torch)",
    )
    p.add_argument(
        "--quant",
        default=os.environ.get("QWEN_TTS_GGML_QUANT", "BF16"),
        help="GGUF quantization when --backend=ggml (default: BF16)",
    )
    p.add_argument(
        "--qwentts-no-fa",
        action="store_true",
        default=os.environ.get("QWEN_TTS_QWENTTS_NO_FA", "").lower() in {"1", "true", "yes"},
        help="Disable qwentts.cpp flash-attention kernels",
    )
    p.add_argument(
        "--qwentts-clamp-fp16",
        action="store_true",
        default=os.environ.get("QWEN_TTS_QWENTTS_CLAMP_FP16", "").lower() in {"1", "true", "yes"},
        help="Enable qwentts.cpp FP16 clamping",
    )
    p.add_argument(
        "--qwentts-ref-cache-dir",
        default=os.environ.get("QWEN_TTS_QWENTTS_REF_CACHE_DIR"),
        help="Directory for cached GGML .spk/.rvq voice references",
    )
    p.add_argument(
        "--managed-voice-dir",
        default=os.environ.get("QWEN_TTS_MANAGED_VOICE_DIR", "~/.local/share/bizarre-tts/voices"),
        help="Private directory for provider-managed voice samples and registry",
    )
    return p.parse_args()


def main():
    global tts_model, voices, default_voice, SAMPLE_RATE, BACKEND
    global MANAGED_VOICE_DIR, MANAGED_VOICE_REGISTRY

    args = _parse_args()

    # Build voice registry
    if args.voices:
        with open(args.voices) as f:
            voices = json.load(f)
        default_voice = next(iter(voices))
        logger.info("Loaded %d voice(s) from %s", len(voices), args.voices)
    elif args.ref_audio:
        voices = {
            "default": {
                "ref_audio": args.ref_audio,
                "ref_text": args.ref_text,
                "language": args.language,
            }
        }
        default_voice = "default"
        logger.info("Using single voice from --ref-audio: %s", args.ref_audio)
    else:
        print(
            "ERROR: provide --ref-audio <file> or --voices <config.json>",
            file=sys.stderr,
        )
        sys.exit(1)

    MANAGED_VOICE_DIR = Path(args.managed_voice_dir).expanduser().resolve()
    MANAGED_VOICE_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(MANAGED_VOICE_DIR, 0o700)
    MANAGED_VOICE_REGISTRY = MANAGED_VOICE_DIR / "voices.json"
    _load_managed_voices()

    from faster_qwen3_tts import FasterQwen3TTS

    BACKEND = args.backend
    logger.info("Loading model %s on %s using %s …", args.model, args.device, args.backend)
    model_kwargs = {"device": args.device, "backend": args.backend}
    if args.backend == "ggml":
        model_kwargs.update(
            {
                "quant": args.quant,
                "qwentts_use_fa": not args.qwentts_no_fa,
                "qwentts_clamp_fp16": args.qwentts_clamp_fp16,
                "qwentts_ref_cache_dir": args.qwentts_ref_cache_dir,
            }
        )
    else:
        model_kwargs["dtype"] = torch.bfloat16
    tts_model = FasterQwen3TTS.from_pretrained(args.model, **model_kwargs)
    SAMPLE_RATE = tts_model.sample_rate
    logger.info("Model ready. Sample rate: %d Hz", SAMPLE_RATE)
    logger.info("Server listening on http://%s:%d", args.host, args.port)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
