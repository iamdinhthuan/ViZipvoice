from enum import Enum
from typing import Literal, Optional

from pydantic import AnyHttpUrl, BaseModel, Field, field_validator


class TTSJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"
    EXPIRED = "expired"


class TTSGenerationOptions(BaseModel):
    num_steps: int = Field(default=16, ge=4, le=64)
    guidance_scale: float = Field(default=1.0, ge=0.0, le=5.0)
    speed: float = Field(default=1.0, ge=0.5, le=1.5)
    t_shift: float = Field(default=0.5, ge=0.1, le=1.0)
    seed: Optional[int] = Field(default=666, ge=-1)
    max_duration_seconds: float = Field(default=100, ge=1, le=300)
    normalize_vietnamese: bool = True
    split_sentences: bool = True
    remove_long_silence: bool = False


class TTSPostprocessOptions(BaseModel):
    crossfade_ms: int = Field(default=80, ge=0, le=300)
    silence_ms: int = Field(default=180, ge=0, le=800)
    fade_in_ms: int = Field(default=20, ge=0, le=500)
    fade_out_ms: int = Field(default=80, ge=0, le=500)


class TTSOutputOptions(BaseModel):
    format: Literal["mp3", "wav"] = "mp3"


class TTSJobCreatePayload(BaseModel):
    text: str
    reference_text: str
    generation: TTSGenerationOptions = Field(default_factory=TTSGenerationOptions)
    postprocess: TTSPostprocessOptions = Field(default_factory=TTSPostprocessOptions)
    output: TTSOutputOptions = Field(default_factory=TTSOutputOptions)
    callback_url: Optional[AnyHttpUrl] = None
    client_request_id: Optional[str] = None

    @field_validator("text", "reference_text")
    @classmethod
    def require_non_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class TTSJobMetrics(BaseModel):
    elapsed_seconds: float = Field(ge=0)
    wav_seconds: float = Field(ge=0)
    rtf: float = Field(ge=0)
    segments: int = Field(ge=0)
    sample_rate: int = Field(default=24000, ge=1)


class TTSJobResponse(BaseModel):
    job_id: str
    status: TTSJobStatus
    status_url: Optional[str] = None
    audio_url: Optional[str] = None
    metrics: Optional[TTSJobMetrics] = None
    error: Optional[str] = None
    client_request_id: Optional[str] = None
