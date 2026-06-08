from types import SimpleNamespace

import pytest

from tts_service.modal_runner import ModalTTSJobRunner
from tts_service.models import TTSJobCreatePayload


class FakeWorkerFunction:
    def __init__(self, call):
        self.call = call
        self.spawn_kwargs = None

    def spawn(self, **kwargs):
        self.spawn_kwargs = kwargs
        return self.call


class FakeFunctionCallClass:
    calls = {}

    @classmethod
    def from_id(cls, call_id):
        return cls.calls[call_id]


class PendingCall:
    object_id = "fc-pending"

    def get(self, timeout=0):
        raise TimeoutError

    def get_call_graph(self):
        return [SimpleNamespace(status=SimpleNamespace(name="PENDING"))]


class SucceededCall:
    object_id = "fc-succeeded"

    def get(self, timeout=0):
        return {
            "audio_bytes": b"ID3mp3",
            "output_format": "mp3",
            "metrics": {
                "elapsed_seconds": 0.5,
                "wav_seconds": 2.0,
                "rtf": 0.25,
                "segments": 1,
                "sample_rate": 24000,
            },
        }


def test_submit_spawns_modal_worker_and_returns_call_id():
    call = PendingCall()
    worker = FakeWorkerFunction(call)
    runner = ModalTTSJobRunner(
        worker_function=worker,
        function_call_cls=FakeFunctionCallClass,
    )
    payload = TTSJobCreatePayload.model_validate(
        {
            "text": "Noi dung can doc.",
            "reference_text": "Transcript.",
            "client_request_id": "req-1",
        }
    )

    response = runner.submit(
        reference_audio=b"voice",
        reference_audio_filename="voice.wav",
        payload=payload,
    )

    assert response.job_id == "fc-pending"
    assert response.status == "queued"
    assert response.status_url == "/v1/tts/jobs/fc-pending"
    assert response.client_request_id == "req-1"
    assert worker.spawn_kwargs["reference_audio"] == b"voice"
    assert worker.spawn_kwargs["reference_audio_filename"] == "voice.wav"
    assert worker.spawn_kwargs["payload_data"]["text"] == "Noi dung can doc."


def test_get_job_maps_pending_and_succeeded_calls():
    FakeFunctionCallClass.calls = {
        "fc-pending": PendingCall(),
        "fc-succeeded": SucceededCall(),
    }
    runner = ModalTTSJobRunner(
        worker_function=FakeWorkerFunction(PendingCall()),
        function_call_cls=FakeFunctionCallClass,
    )

    pending = runner.get_job("fc-pending")
    assert pending.status == "running"
    assert pending.audio_url is None

    succeeded = runner.get_job("fc-succeeded")
    assert succeeded.status == "succeeded"
    assert succeeded.audio_url == "/v1/tts/jobs/fc-succeeded/audio"
    assert succeeded.metrics.sample_rate == 24000


def test_get_audio_returns_bytes_only_after_success():
    FakeFunctionCallClass.calls = {
        "fc-pending": PendingCall(),
        "fc-succeeded": SucceededCall(),
    }
    runner = ModalTTSJobRunner(
        worker_function=FakeWorkerFunction(PendingCall()),
        function_call_cls=FakeFunctionCallClass,
    )

    assert runner.get_audio("fc-succeeded") == b"ID3mp3"
    assert runner.get_audio_content_type("fc-succeeded") == "audio/mpeg"
    with pytest.raises(RuntimeError):
        runner.get_audio("fc-pending")
