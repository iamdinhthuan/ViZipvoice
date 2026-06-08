from typing import Any

from tts_service.audio_output import content_type_for_format
from tts_service.models import (
    TTSJobCreatePayload,
    TTSJobMetrics,
    TTSJobResponse,
    TTSJobStatus,
)


class ModalTTSJobRunner:
    def __init__(
        self,
        *,
        worker_function: Any,
        function_call_cls: Any,
    ) -> None:
        self.worker_function = worker_function
        self.function_call_cls = function_call_cls

    def submit(
        self,
        *,
        reference_audio: bytes,
        reference_audio_filename: str | None,
        payload: TTSJobCreatePayload,
    ) -> TTSJobResponse:
        if not reference_audio:
            raise ValueError("reference_audio must not be empty")
        call = self.worker_function.spawn(
            reference_audio=reference_audio,
            reference_audio_filename=reference_audio_filename,
            payload_data=payload.model_dump(mode="json"),
        )
        return TTSJobResponse(
            job_id=call.object_id,
            status=TTSJobStatus.QUEUED,
            status_url=f"/v1/tts/jobs/{call.object_id}",
            client_request_id=payload.client_request_id,
        )

    def get_job(self, job_id: str) -> TTSJobResponse:
        call = self.function_call_cls.from_id(job_id)
        result = self._try_get_result(call)
        if result["status"] != TTSJobStatus.SUCCEEDED:
            return TTSJobResponse(
                job_id=job_id,
                status=result["status"],
                status_url=f"/v1/tts/jobs/{job_id}",
                error=result.get("error"),
            )

        metrics = TTSJobMetrics.model_validate(result["value"]["metrics"])
        return TTSJobResponse(
            job_id=job_id,
            status=TTSJobStatus.SUCCEEDED,
            status_url=f"/v1/tts/jobs/{job_id}",
            audio_url=f"/v1/tts/jobs/{job_id}/audio",
            metrics=metrics,
            client_request_id=result["value"].get("client_request_id"),
        )

    def get_audio(self, job_id: str) -> bytes:
        call = self.function_call_cls.from_id(job_id)
        result = self._try_get_result(call)
        if result["status"] != TTSJobStatus.SUCCEEDED:
            raise RuntimeError("job has not succeeded")
        return result["value"]["audio_bytes"]

    def get_audio_content_type(self, job_id: str) -> str:
        call = self.function_call_cls.from_id(job_id)
        result = self._try_get_result(call)
        if result["status"] != TTSJobStatus.SUCCEEDED:
            raise RuntimeError("job has not succeeded")
        return content_type_for_format(result["value"].get("output_format", "mp3"))

    def _try_get_result(self, call: Any) -> dict[str, Any]:
        try:
            return {
                "status": TTSJobStatus.SUCCEEDED,
                "value": call.get(timeout=0),
            }
        except TimeoutError:
            return {"status": self._status_from_call_graph(call)}
        except Exception as exc:
            if exc.__class__.__name__ == "OutputExpiredError":
                return {"status": TTSJobStatus.EXPIRED, "error": str(exc)}
            return {"status": TTSJobStatus.FAILED, "error": str(exc)}

    def _status_from_call_graph(self, call: Any) -> TTSJobStatus:
        try:
            graph = call.get_call_graph()
        except Exception:
            return TTSJobStatus.RUNNING

        status_names = {
            getattr(getattr(item, "status", None), "name", "").upper()
            for item in graph
        }
        if status_names & {"FAILURE", "INIT_FAILURE", "TIMEOUT"}:
            return TTSJobStatus.FAILED
        if "TERMINATED" in status_names:
            return TTSJobStatus.CANCELED
        return TTSJobStatus.RUNNING
