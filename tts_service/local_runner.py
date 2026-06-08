from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import time
from uuid import uuid4

from tts_service.audio_output import content_type_for_format, encode_audio_output
from tts_service.models import (
    TTSJobCreatePayload,
    TTSJobMetrics,
    TTSJobResponse,
    TTSJobStatus,
)
from tts_service.synthesizer import TTSSynthesizer, ViZipVoiceSynthesizer


@dataclass
class LocalJobRecord:
    job_id: str
    payload: TTSJobCreatePayload
    reference_audio_path: Path
    synthesis_wav_path: Path
    output_path: Path
    output_format: str
    status: TTSJobStatus
    created_at: float
    updated_at: float
    metrics: TTSJobMetrics | None = None
    error: str | None = None


class LocalTTSJobRunner:
    def __init__(
        self,
        *,
        job_dir: Path,
        synthesizer: TTSSynthesizer | None = None,
        max_workers: int = 1,
    ) -> None:
        self.job_dir = Path(job_dir)
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self.synthesizer = synthesizer or ViZipVoiceSynthesizer.from_env()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, LocalJobRecord] = {}
        self._lock = Lock()

    def submit(
        self,
        *,
        reference_audio: bytes,
        reference_audio_filename: str | None,
        payload: TTSJobCreatePayload,
    ) -> TTSJobResponse:
        if not reference_audio:
            raise ValueError("reference_audio must not be empty")

        job_id = f"tts_{uuid4().hex}"
        job_path = self.job_dir / job_id
        job_path.mkdir(parents=True, exist_ok=False)
        reference_audio_path = job_path / safe_reference_audio_name(reference_audio_filename)
        synthesis_wav_path = job_path / "output.wav"
        output_format = payload.output.format
        output_path = job_path / f"output.{output_format}"
        reference_audio_path.write_bytes(reference_audio)

        now = time()
        record = LocalJobRecord(
            job_id=job_id,
            payload=payload,
            reference_audio_path=reference_audio_path,
            synthesis_wav_path=synthesis_wav_path,
            output_path=output_path,
            output_format=output_format,
            status=TTSJobStatus.QUEUED,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._jobs[job_id] = record

        response = self._to_response(record)
        self._executor.submit(self.run_job, job_id)
        return response

    def run_job(self, job_id: str) -> None:
        record = self._get_record(job_id)
        self._update_status(job_id, TTSJobStatus.RUNNING)
        try:
            result = self.synthesizer.synthesize(
                reference_audio_path=record.reference_audio_path,
                output_path=record.synthesis_wav_path,
                payload=record.payload,
            )
            encode_audio_output(
                record.synthesis_wav_path,
                record.output_path,
                record.output_format,
            )
            metrics = TTSJobMetrics.model_validate(result.metrics)
            with self._lock:
                record.metrics = metrics
                record.status = TTSJobStatus.SUCCEEDED
                record.updated_at = time()
        except Exception as exc:
            with self._lock:
                record.status = TTSJobStatus.FAILED
                record.error = str(exc)
                record.updated_at = time()

    def get_job(self, job_id: str) -> TTSJobResponse:
        return self._to_response(self._get_record(job_id))

    def get_audio(self, job_id: str) -> bytes:
        record = self._get_record(job_id)
        if record.status != TTSJobStatus.SUCCEEDED:
            raise RuntimeError("job has not succeeded")
        if not record.output_path.is_file():
            raise FileNotFoundError("job audio output is missing")
        return record.output_path.read_bytes()

    def get_audio_content_type(self, job_id: str) -> str:
        record = self._get_record(job_id)
        return content_type_for_format(record.output_format)

    def _get_record(self, job_id: str) -> LocalJobRecord:
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError as exc:
                raise KeyError(f"unknown job_id: {job_id}") from exc

    def _update_status(self, job_id: str, status: TTSJobStatus) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = status
            record.updated_at = time()

    def _to_response(self, record: LocalJobRecord) -> TTSJobResponse:
        audio_url = None
        if record.status == TTSJobStatus.SUCCEEDED:
            audio_url = f"/v1/tts/jobs/{record.job_id}/audio"
        return TTSJobResponse(
            job_id=record.job_id,
            status=record.status,
            status_url=f"/v1/tts/jobs/{record.job_id}",
            audio_url=audio_url,
            metrics=record.metrics,
            error=record.error,
            client_request_id=record.payload.client_request_id,
        )


def safe_reference_audio_name(filename: str | None) -> str:
    name = Path(filename or "reference.wav").name
    if not name or name in {".", ".."}:
        return "reference.wav"
    return name
