#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Minimal web page + JSON/WAV API for OmniVoice (Sawti).

A single HTML page (``static/index.html``) talks to a small FastAPI app:

- ``GET  /``               the page
- ``GET  /api/health``     model status, whether an API key is required
- ``GET  /api/languages``  supported languages (id + display name)
- ``POST /api/tts``        multipart form -> ``audio/wav``

Compared with ``omnivoice-demo`` this server is safer by default: it binds to
``127.0.0.1``, can require an API key, caps text / upload sizes, requires an
explicit consent flag for voice cloning, and runs one generation at a time.

Usage:
    omnivoice-web --model k2-fsa/OmniVoice --port 8002
    omnivoice-web --host 0.0.0.0 --api-key "$SAWTI_API_KEY"   # LAN, protected
"""

import argparse
import io
import logging
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

logger = logging.getLogger("omnivoice.web")

STATIC_DIR = Path(__file__).parent / "static"

MAX_TEXT_CHARS = 2000
MAX_INSTRUCT_CHARS = 200
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MIN_REF_SECONDS = 1.0
MAX_REF_SECONDS = 30.0
ALLOWED_AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
MODES = {"auto", "clone", "design"}


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _decode_audio(raw: bytes) -> tuple[np.ndarray, int]:
    """Decode uploaded bytes to a mono float32 (1, T) array and sample rate."""
    buf = io.BytesIO(raw)
    try:
        data, sr = sf.read(buf, dtype="float32", always_2d=True)
        data = data.T
    except Exception:
        # MP3/M4A need librosa (audioread + ffmpeg).
        import librosa

        buf.seek(0)
        data, sr = librosa.load(buf, sr=None, mono=False)
        if data.ndim == 1:
            data = data[np.newaxis, :]
    if data.shape[0] > 1:
        data = np.mean(data, axis=0, keepdims=True)
    return data.astype(np.float32), int(sr)


def _to_wav_bytes(audio: np.ndarray, sampling_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sampling_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _parse_float(value: Optional[str], name: str, lo: float, hi: float):
    if value is None or str(value).strip() == "":
        return None
    try:
        v = float(value)
    except ValueError:
        raise ApiError(400, "invalid_param", f"'{name}' must be a number.")
    if not (lo <= v <= hi):
        raise ApiError(400, "invalid_param", f"'{name}' must be between {lo} and {hi}.")
    return v


def create_app(
    model,
    api_key: Optional[str] = None,
    asr_enabled: bool = True,
    languages: Optional[list] = None,
    busy_timeout: float = 120.0,
) -> FastAPI:
    """Build the FastAPI app around an already-loaded ``OmniVoice`` model."""
    app = FastAPI(title="Sawti TTS", docs_url=None, redoc_url=None)
    gpu_lock = threading.Lock()
    lang_list = languages if languages is not None else _default_languages()

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return JSONResponse(
            status_code=exc.status, content={"error": exc.code, "message": exc.message}
        )

    def _check_key(request: Request):
        if not api_key:
            return
        given = request.headers.get("x-api-key", "")
        if not secrets.compare_digest(given.encode(), api_key.encode()):
            raise ApiError(401, "unauthorized", "Missing or invalid API key.")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "auth_required": bool(api_key),
            "asr_enabled": asr_enabled,
            "sampling_rate": model.sampling_rate,
            "busy": gpu_lock.locked(),
            "limits": {
                "max_text_chars": MAX_TEXT_CHARS,
                "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
                "ref_seconds": [MIN_REF_SECONDS, MAX_REF_SECONDS],
            },
        }

    @app.get("/api/languages")
    def get_languages():
        return {"languages": lang_list}

    # Sync handler: FastAPI runs it in a worker thread, so the event loop
    # stays responsive while the GPU is busy.
    @app.post("/api/tts")
    def tts(
        request: Request,
        text: str = Form(...),
        mode: str = Form("auto"),
        language: Optional[str] = Form(None),
        instruct: Optional[str] = Form(None),
        ref_text: Optional[str] = Form(None),
        consent: Optional[str] = Form(None),
        speed: Optional[str] = Form(None),
        duration: Optional[str] = Form(None),
        num_step: Optional[str] = Form(None),
        ref_audio: Optional[UploadFile] = File(None),
    ):
        _check_key(request)

        text = (text or "").strip()
        if not text:
            raise ApiError(400, "empty_text", "Please enter the text to synthesize.")
        if len(text) > MAX_TEXT_CHARS:
            raise ApiError(
                413, "text_too_long", f"Text exceeds {MAX_TEXT_CHARS} characters."
            )
        if mode not in MODES:
            raise ApiError(400, "invalid_mode", f"mode must be one of {sorted(MODES)}.")

        kw: Dict[str, Any] = {"text": text}

        lang = (language or "").strip()
        if lang and lang.lower() != "auto":
            if lang not in {item["id"] for item in lang_list}:
                raise ApiError(
                    400, "invalid_language", f"Unsupported language '{lang}'."
                )
            kw["language"] = lang

        spd = _parse_float(speed, "speed", 0.5, 1.5)
        dur = _parse_float(duration, "duration", 0.5, 120.0)
        steps = _parse_float(num_step, "num_step", 4, 64)
        if dur is not None:
            kw["duration"] = dur
        elif spd is not None and spd != 1.0:
            kw["speed"] = spd

        ref = None
        if mode == "design":
            instruct = (instruct or "").strip()
            if not instruct:
                raise ApiError(400, "missing_instruct", "Describe the voice to design.")
            if len(instruct) > MAX_INSTRUCT_CHARS:
                raise ApiError(
                    400, "instruct_too_long", "Voice description is too long."
                )
            kw["instruct"] = instruct
        elif mode == "clone":
            if str(consent).lower() not in {"true", "1", "yes", "on"}:
                raise ApiError(
                    403,
                    "consent_required",
                    "Confirm you have the speaker's consent before cloning a voice.",
                )
            if ref_audio is None or not ref_audio.filename:
                raise ApiError(
                    400, "missing_ref_audio", "Upload a reference audio file."
                )
            ext = Path(ref_audio.filename).suffix.lower()
            if ext not in ALLOWED_AUDIO_EXT:
                raise ApiError(
                    415,
                    "unsupported_audio",
                    f"Unsupported audio type '{ext}'. "
                    f"Use one of {sorted(ALLOWED_AUDIO_EXT)}.",
                )
            raw = ref_audio.file.read(MAX_UPLOAD_BYTES + 1)
            if len(raw) > MAX_UPLOAD_BYTES:
                raise ApiError(413, "audio_too_large", "Reference audio exceeds 10 MB.")
            try:
                wav, sr = _decode_audio(raw)
            except Exception:
                raise ApiError(
                    415, "undecodable_audio", "Could not read the audio file."
                )
            seconds = wav.shape[-1] / sr
            if seconds < MIN_REF_SECONDS or seconds > MAX_REF_SECONDS:
                raise ApiError(
                    400,
                    "bad_ref_length",
                    f"Reference audio is {seconds:.1f}s; use "
                    f"{MIN_REF_SECONDS:.0f}-{MAX_REF_SECONDS:.0f}s (3-10s is best).",
                )
            ref_text = (ref_text or "").strip() or None
            if ref_text is None and not asr_enabled:
                raise ApiError(
                    400,
                    "missing_ref_text",
                    "Auto-transcription is disabled; enter the reference transcript.",
                )
            ref = (wav, sr, ref_text)

        if not gpu_lock.acquire(timeout=busy_timeout):
            raise ApiError(503, "busy", "The server is busy, please retry shortly.")
        started = time.perf_counter()
        try:
            from omnivoice import OmniVoiceGenerationConfig

            if steps is not None:
                kw["generation_config"] = OmniVoiceGenerationConfig(num_step=int(steps))
            if ref is not None:
                wav, sr, rtext = ref
                kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                    ref_audio=(wav, sr), ref_text=rtext
                )
            audio = model.generate(**kw)[0]
        except ValueError as e:
            raise ApiError(400, "generation_rejected", str(e))
        except Exception:
            logger.exception("Generation failed")
            raise ApiError(500, "generation_failed", "Speech generation failed.")
        finally:
            gpu_lock.release()
        elapsed = time.perf_counter() - started

        sr_out = model.sampling_rate
        audio_seconds = len(audio) / sr_out
        # Log metadata only: the text may contain customer data.
        logger.info(
            "tts mode=%s lang=%s chars=%d audio=%.2fs elapsed=%.2fs",
            mode,
            kw.get("language", "auto"),
            len(text),
            audio_seconds,
            elapsed,
        )
        return Response(
            content=_to_wav_bytes(audio, sr_out),
            media_type="audio/wav",
            headers={
                "X-Audio-Seconds": f"{audio_seconds:.2f}",
                "X-Elapsed-Seconds": f"{elapsed:.2f}",
                "Cache-Control": "no-store",
            },
        )

    return app


def _default_languages() -> list:
    from omnivoice.utils.lang_map import LANG_NAME_TO_ID, lang_display_name

    seen = {}
    for name, lid in LANG_NAME_TO_ID.items():
        seen.setdefault(lid, lang_display_name(name))
    return sorted(
        ({"id": lid, "name": name} for lid, name in seen.items()),
        key=lambda x: x["name"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnivoice-web",
        description="Serve a simple OmniVoice web page and TTS API.",
    )
    parser.add_argument(
        "--model",
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--device", default=None, help="Device to use. Auto-detected if not specified."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1). Use 0.0.0.0 only with --api-key.",
    )
    parser.add_argument("--port", type=int, default=8002, help="Port (default: 8002).")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("SAWTI_API_KEY"),
        help="Require this key in the X-API-Key header (env: SAWTI_API_KEY).",
    )
    parser.add_argument(
        "--no-asr",
        action="store_true",
        default=False,
        help="Skip loading Whisper; the reference transcript becomes mandatory.",
    )
    parser.add_argument(
        "--asr-model",
        default="openai/whisper-large-v3-turbo",
        help="ASR model path or HuggingFace repo id.",
    )
    return parser


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args = build_parser().parse_args(argv)

    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.api_key:
        logger.warning(
            "Binding to %s without --api-key: anyone on the network can clone voices.",
            args.host,
        )

    import torch
    import uvicorn

    from omnivoice import OmniVoice
    from omnivoice.utils.common import get_best_device

    device = args.device or get_best_device()
    logger.info("Loading model from %s, device=%s ...", args.model, device)
    model = OmniVoice.from_pretrained(
        args.model,
        device_map=device,
        dtype=torch.float16,
        load_asr=not args.no_asr,
        asr_model_name=args.asr_model,
    )

    app = create_app(model, api_key=args.api_key, asr_enabled=not args.no_asr)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
