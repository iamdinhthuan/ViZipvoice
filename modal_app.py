import os
import tempfile
from pathlib import Path
from typing import Any

from tts_service.typing_compat import ensure_typing_extensions_sentinel

ensure_typing_extensions_sentinel()

import modal


APP_NAME = os.getenv("VIZIPVOICE_MODAL_APP_NAME", "vizipvoice-tts")
MODAL_GPU = os.getenv("VIZIPVOICE_MODAL_GPU", "L4")
WORKER_TIMEOUT_SECONDS = int(os.getenv("VIZIPVOICE_MODAL_WORKER_TIMEOUT", "900"))
WORKER_STARTUP_TIMEOUT_SECONDS = int(
    os.getenv("VIZIPVOICE_MODAL_STARTUP_TIMEOUT", "900")
)
WORKER_SCALEDOWN_WINDOW_SECONDS = int(
    os.getenv("VIZIPVOICE_MODAL_SCALEDOWN_WINDOW", "300")
)
WORKER_MIN_CONTAINERS = int(os.getenv("VIZIPVOICE_MODAL_MIN_CONTAINERS", "1"))
WORKER_MAX_CONTAINERS = int(os.getenv("VIZIPVOICE_MODAL_MAX_CONTAINERS", "3"))
WEB_SCALEDOWN_WINDOW_SECONDS = int(os.getenv("VIZIPVOICE_MODAL_WEB_SCALEDOWN_WINDOW", "10"))


app = modal.App(APP_NAME)

web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi>=0.115",
        "pydantic>=2",
        "python-multipart>=0.0.9",
    )
    .add_local_python_source("tts_service")
)

worker_image = modal.Image.from_dockerfile(
    "Dockerfile",
    context_dir=".",
)


def modal_worker_env() -> dict[str, str]:
    keys = [
        "VIZIPVOICE_REPO_ID",
        "VIZIPVOICE_REVISION",
        "VIZIPVOICE_MODEL_DIR",
        "VIZIPVOICE_CHECKPOINT_NAME",
        "VIZIPVOICE_VOCODER_PATH",
        "VIZIPVOICE_DEVICE",
        "VIZIPVOICE_FP16",
        "VIZIPVOICE_NUM_THREADS",
        "HF_HOME",
        "HF_HUB_CACHE",
    ]
    return {key: value for key in keys if (value := os.getenv(key))}


def modal_worker_volumes() -> dict[str, modal.Volume]:
    volume_name = os.getenv("VIZIPVOICE_MODAL_VOLUME")
    if not volume_name:
        return {}
    mount_path = os.getenv("VIZIPVOICE_MODAL_VOLUME_MOUNT", "/models")
    return {
        mount_path: modal.Volume.from_name(
            volume_name,
            create_if_missing=False,
        )
    }


def modal_worker_secrets() -> list[modal.Secret]:
    secret_name = os.getenv("VIZIPVOICE_MODAL_SECRET")
    if not secret_name:
        return []
    return [modal.Secret.from_name(secret_name)]


_SYNTHESIZER = None


def get_modal_synthesizer():
    global _SYNTHESIZER
    if _SYNTHESIZER is None:
        from tts_service.synthesizer import ViZipVoiceSynthesizer

        _SYNTHESIZER = ViZipVoiceSynthesizer.from_env()
    return _SYNTHESIZER


@app.function(
    image=worker_image,
    gpu=MODAL_GPU,
    timeout=WORKER_TIMEOUT_SECONDS,
    startup_timeout=WORKER_STARTUP_TIMEOUT_SECONDS,
    min_containers=WORKER_MIN_CONTAINERS,
    buffer_containers=0,
    max_containers=WORKER_MAX_CONTAINERS,
    scaledown_window=WORKER_SCALEDOWN_WINDOW_SECONDS,
    allow_concurrent_inputs=WORKER_MAX_CONTAINERS,
    env=modal_worker_env(),
    volumes=modal_worker_volumes(),
    secrets=modal_worker_secrets(),
)
def synthesize_tts_job(
    *,
    reference_audio: bytes,
    reference_audio_filename: str | None,
    payload_data: dict[str, Any],
) -> dict[str, Any]:
    from tts_service.local_runner import safe_reference_audio_name
    from tts_service.audio_output import encode_audio_output
    from tts_service.models import TTSJobCreatePayload

    payload = TTSJobCreatePayload.model_validate(payload_data)
    with tempfile.TemporaryDirectory(prefix="vizipvoice_modal_tts_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        reference_audio_path = tmp_dir / safe_reference_audio_name(reference_audio_filename)
        synthesis_wav_path = tmp_dir / "output.wav"
        output_format = payload.output.format
        output_path = tmp_dir / f"output.{output_format}"
        reference_audio_path.write_bytes(reference_audio)

        result = get_modal_synthesizer().synthesize(
            reference_audio_path=reference_audio_path,
            output_path=synthesis_wav_path,
            payload=payload,
        )
        encode_audio_output(synthesis_wav_path, output_path, output_format)
        return {
            "audio_bytes": output_path.read_bytes(),
            "output_format": output_format,
            "metrics": result.metrics,
            "client_request_id": payload.client_request_id,
        }


@app.function(
    image=web_image,
    timeout=150,
    min_containers=0,
    buffer_containers=0,
    scaledown_window=WEB_SCALEDOWN_WINDOW_SECONDS,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def fastapi_app():
    from tts_service.app import create_app
    from tts_service.modal_runner import ModalTTSJobRunner

    runner = ModalTTSJobRunner(
        worker_function=synthesize_tts_job,
        function_call_cls=modal.FunctionCall,
    )
    return create_app(runner=runner)
