import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response
from pydantic import ValidationError

from tts_service.local_runner import LocalTTSJobRunner
from tts_service.models import TTSJobCreatePayload, TTSJobResponse

HTTP_422_UNPROCESSABLE = 422


def default_runner() -> LocalTTSJobRunner:
    return LocalTTSJobRunner(
        job_dir=Path(os.getenv("TTS_JOB_DIR", "/tmp/vizipvoice_tts_jobs")),
    )


def create_app(*, runner: Any | None = None) -> FastAPI:
    app = FastAPI(
        title="ViZipVoice TTS Service",
        version="0.1.0",
    )
    app.state.runner = runner or default_runner()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/v1/tts/jobs",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=TTSJobResponse,
    )
    async def create_tts_job(
        reference_audio: UploadFile = File(...),
        request: str = Form(...),
    ) -> TTSJobResponse:
        payload = parse_request_payload(request)
        audio_bytes = await reference_audio.read()
        if not audio_bytes:
            raise HTTPException(
                status_code=HTTP_422_UNPROCESSABLE,
                detail="reference_audio must not be empty",
            )
        try:
            return app.state.runner.submit(
                reference_audio=audio_bytes,
                reference_audio_filename=reference_audio.filename,
                payload=payload,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=HTTP_422_UNPROCESSABLE,
                detail=str(exc),
            ) from exc

    @app.get("/v1/tts/jobs/{job_id}", response_model=TTSJobResponse)
    def get_tts_job(job_id: str) -> TTSJobResponse:
        try:
            return app.state.runner.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    @app.get("/v1/tts/jobs/{job_id}/audio")
    def get_tts_audio(job_id: str) -> Response:
        try:
            audio = app.state.runner.get_audio(job_id)
            content_type = app.state.runner.get_audio_content_type(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        return Response(content=audio, media_type=content_type)

    return app


def parse_request_payload(raw_request: str) -> TTSJobCreatePayload:
    try:
        data = json.loads(raw_request)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE,
            detail="request must be valid JSON",
        ) from exc
    try:
        return TTSJobCreatePayload.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE,
            detail=exc.errors(),
        ) from exc
