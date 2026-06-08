import json
import time
import wave
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient

from tts_service.app import create_app
from tts_service.local_runner import LocalTTSJobRunner
from tts_service.synthesizer import SynthesisResult


def valid_wav_bytes() -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x00\x00" * 2400)
    return buffer.getvalue()


WAV_BYTES = valid_wav_bytes()


def wait_for_succeeded(client: TestClient, status_url: str, timeout_seconds: float = 2.0):
    deadline = time.time() + timeout_seconds
    last = None
    while time.time() < deadline:
        response = client.get(status_url)
        assert response.status_code == 200
        last = response.json()
        if last["status"] == "succeeded":
            return last
        time.sleep(0.02)
    raise AssertionError(f"job did not succeed in time: {last}")


class FakeSynthesizer:
    def synthesize(self, *, reference_audio_path: Path, output_path: Path, payload):
        assert reference_audio_path.read_bytes() == b"voice-bytes"
        assert payload.text == "Noi dung can doc."
        assert payload.reference_text == "Transcript cua audio mau."
        assert output_path.suffix == ".wav"
        output_path.write_bytes(WAV_BYTES)
        return SynthesisResult(
            metrics={
                "elapsed_seconds": 0.25,
                "wav_seconds": 1.5,
                "rtf": 0.1667,
                "segments": 1,
                "sample_rate": 24000,
            }
        )


def test_submit_poll_and_download_audio(tmp_path):
    runner = LocalTTSJobRunner(
        job_dir=tmp_path,
        synthesizer=FakeSynthesizer(),
    )
    client = TestClient(create_app(runner=runner))

    response = client.post(
        "/v1/tts/jobs",
        data={
            "request": json.dumps(
                {
                    "text": "Noi dung can doc.",
                    "reference_text": "Transcript cua audio mau.",
                }
            )
        },
        files={
            "reference_audio": ("voice.wav", b"voice-bytes", "audio/wav"),
        },
    )

    assert response.status_code == 202
    created = response.json()
    assert created["status"] == "queued"
    assert created["audio_url"] is None

    status = wait_for_succeeded(client, created["status_url"])
    assert status["status"] == "succeeded"
    assert status["audio_url"] == f"/v1/tts/jobs/{created['job_id']}/audio"
    assert status["metrics"]["sample_rate"] == 24000

    audio_response = client.get(status["audio_url"])
    assert audio_response.status_code == 200
    assert audio_response.headers["content-type"].startswith("audio/mpeg")
    assert audio_response.content.startswith((b"ID3", b"\xff"))
    assert audio_response.content != WAV_BYTES


def test_submit_rejects_empty_reference_audio(tmp_path):
    runner = LocalTTSJobRunner(
        job_dir=tmp_path,
        synthesizer=FakeSynthesizer(),
    )
    client = TestClient(create_app(runner=runner))

    response = client.post(
        "/v1/tts/jobs",
        data={
            "request": json.dumps(
                {
                    "text": "Noi dung can doc.",
                    "reference_text": "Transcript cua audio mau.",
                }
            )
        },
        files={
            "reference_audio": ("voice.wav", b"", "audio/wav"),
        },
    )

    assert response.status_code == 422
    assert "reference_audio" in response.json()["detail"]


def test_submit_rejects_invalid_request_json(tmp_path):
    runner = LocalTTSJobRunner(
        job_dir=tmp_path,
        synthesizer=FakeSynthesizer(),
    )
    client = TestClient(create_app(runner=runner))

    response = client.post(
        "/v1/tts/jobs",
        data={"request": "{not-json"},
        files={
            "reference_audio": ("voice.wav", b"voice-bytes", "audio/wav"),
        },
    )

    assert response.status_code == 422
    assert "request" in response.json()["detail"]
