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

"""Tests for the simple web server (omnivoice-web), using a stub model so no
weights or GPU are needed."""

import io

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from omnivoice.web.server import create_app

SR = 24000
LANGS = [{"id": "arb", "name": "Standard Arabic"}, {"id": "en", "name": "English"}]


class StubModel:
    sampling_rate = SR

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def create_voice_clone_prompt(self, ref_audio, ref_text=None):
        wav, sr = ref_audio
        return {"samples": wav.shape[-1], "sr": sr, "ref_text": ref_text}

    def generate(self, **kw):
        self.calls.append(kw)
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        return [np.zeros(SR, dtype=np.float32)]


def _wav(seconds, sr=16000):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(seconds * sr), dtype=np.float32), sr, format="WAV")
    return buf.getvalue()


@pytest.fixture
def model():
    return StubModel()


@pytest.fixture
def client(model):
    return TestClient(create_app(model, languages=LANGS))


def test_index_and_health(client):
    assert "Sawti" in client.get("/").text
    h = client.get("/api/health").json()
    assert h["auth_required"] is False and h["sampling_rate"] == SR


def test_auto_mode_returns_wav(client, model):
    r = client.post("/api/tts", data={"text": "مرحبا بكم", "language": "arb"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    data, sr = sf.read(io.BytesIO(r.content))
    assert sr == SR and len(data) == SR
    assert model.calls[0]["language"] == "arb"


def test_design_mode_passes_instruct(client, model):
    r = client.post(
        "/api/tts",
        data={"text": "Hello", "mode": "design", "instruct": "female, low pitch"},
    )
    assert r.status_code == 200
    assert model.calls[0]["instruct"] == "female, low pitch"


def test_clone_mode_with_consent(client, model):
    r = client.post(
        "/api/tts",
        data={"text": "Hello", "mode": "clone", "consent": "true", "ref_text": "hi"},
        files={"ref_audio": ("ref.wav", _wav(4), "audio/wav")},
    )
    assert r.status_code == 200
    prompt = model.calls[0]["voice_clone_prompt"]
    assert prompt["ref_text"] == "hi" and prompt["sr"] == 16000


def test_duration_overrides_speed(client, model):
    client.post("/api/tts", data={"text": "x", "speed": "1.2", "duration": "5"})
    assert model.calls[0]["duration"] == 5.0 and "speed" not in model.calls[0]


@pytest.mark.parametrize(
    "data,files,status,code",
    [
        ({"text": "  "}, None, 400, "empty_text"),
        ({"text": "a" * 2001}, None, 413, "text_too_long"),
        ({"text": "x", "mode": "sing"}, None, 400, "invalid_mode"),
        ({"text": "x", "language": "klingon"}, None, 400, "invalid_language"),
        ({"text": "x", "speed": "3"}, None, 400, "invalid_param"),
        ({"text": "x", "speed": "fast"}, None, 400, "invalid_param"),
        ({"text": "x", "mode": "design"}, None, 400, "missing_instruct"),
        ({"text": "x", "mode": "clone"}, None, 403, "consent_required"),
        (
            {"text": "x", "mode": "clone", "consent": "true"},
            None,
            400,
            "missing_ref_audio",
        ),
        (
            {"text": "x", "mode": "clone", "consent": "true"},
            {"ref_audio": ("ref.exe", b"MZ", "application/octet-stream")},
            415,
            "unsupported_audio",
        ),
        (
            {"text": "x", "mode": "clone", "consent": "true"},
            {"ref_audio": ("ref.wav", b"not audio", "audio/wav")},
            415,
            "undecodable_audio",
        ),
        (
            {"text": "x", "mode": "clone", "consent": "true"},
            {"ref_audio": ("ref.wav", _wav(0.3), "audio/wav")},
            400,
            "bad_ref_length",
        ),
        (
            {"text": "x", "mode": "clone", "consent": "true"},
            {"ref_audio": ("ref.wav", _wav(45), "audio/wav")},
            400,
            "bad_ref_length",
        ),
    ],
)
def test_rejections(client, model, data, files, status, code):
    r = client.post("/api/tts", data=data, files=files)
    assert r.status_code == status
    assert r.json()["error"] == code
    assert model.calls == []


def test_clone_requires_ref_text_without_asr():
    model = StubModel()
    c = TestClient(create_app(model, asr_enabled=False, languages=LANGS))
    r = c.post(
        "/api/tts",
        data={"text": "x", "mode": "clone", "consent": "true"},
        files={"ref_audio": ("ref.wav", _wav(4), "audio/wav")},
    )
    assert r.status_code == 400 and r.json()["error"] == "missing_ref_text"


def test_api_key_enforced():
    model = StubModel()
    c = TestClient(create_app(model, api_key="s3cret", languages=LANGS))
    assert c.get("/api/health").json()["auth_required"] is True
    assert c.post("/api/tts", data={"text": "x"}).status_code == 401
    r = c.post("/api/tts", data={"text": "x"}, headers={"X-API-Key": "wrong"})
    assert r.status_code == 401
    r = c.post("/api/tts", data={"text": "x"}, headers={"X-API-Key": "s3cret"})
    assert r.status_code == 200


def test_generation_failure_hides_internal_error():
    c = TestClient(create_app(StubModel(fail=True), languages=LANGS))
    r = c.post("/api/tts", data={"text": "x"})
    assert r.status_code == 500
    assert r.json()["error"] == "generation_failed"
    assert "CUDA" not in r.text
