import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from tts_service.models import TTSJobCreatePayload


@dataclass(frozen=True)
class SynthesisResult:
    metrics: dict[str, Any]


class TTSSynthesizer(Protocol):
    def synthesize(
        self,
        *,
        reference_audio_path: Path,
        output_path: Path,
        payload: TTSJobCreatePayload,
    ) -> SynthesisResult:
        ...


class ViZipVoiceSynthesizer:
    def __init__(
        self,
        *,
        repo_id: str | None = None,
        revision: str | None = None,
        model_dir: str | None = None,
        checkpoint_name: str = "latest",
        vocoder_path: str | None = None,
        device: str | None = None,
        use_fp16: bool = True,
        num_threads: int = 1,
    ) -> None:
        self.repo_id = repo_id
        self.revision = revision
        self.model_dir = model_dir
        self.checkpoint_name = checkpoint_name
        self.vocoder_path = vocoder_path
        self.device = device
        self.use_fp16 = use_fp16
        self.num_threads = num_threads
        self._tts = None

    @classmethod
    def from_env(cls) -> "ViZipVoiceSynthesizer":
        return cls(
            repo_id=os.getenv("VIZIPVOICE_REPO_ID") or None,
            revision=os.getenv("VIZIPVOICE_REVISION") or None,
            model_dir=os.getenv("VIZIPVOICE_MODEL_DIR") or None,
            checkpoint_name=os.getenv("VIZIPVOICE_CHECKPOINT_NAME", "latest"),
            vocoder_path=os.getenv("VIZIPVOICE_VOCODER_PATH") or None,
            device=os.getenv("VIZIPVOICE_DEVICE") or None,
            use_fp16=os.getenv("VIZIPVOICE_FP16", "1") != "0",
            num_threads=int(os.getenv("VIZIPVOICE_NUM_THREADS", "1")),
        )

    @property
    def tts(self):
        if self._tts is None:
            from zipvoice.vizipvoice import DEFAULT_REPO_ID, ViZipVoiceTTS

            self._tts = ViZipVoiceTTS(
                repo_id=self.repo_id or DEFAULT_REPO_ID,
                revision=self.revision,
                model_dir=self.model_dir,
                checkpoint_name=self.checkpoint_name,
                vocoder_path=self.vocoder_path,
                device=self.device,
                use_fp16=self.use_fp16,
                num_threads=self.num_threads,
            )
        return self._tts

    def synthesize(
        self,
        *,
        reference_audio_path: Path,
        output_path: Path,
        payload: TTSJobCreatePayload,
    ) -> SynthesisResult:
        generation = payload.generation
        postprocess = payload.postprocess
        raw_metrics = self.tts.synthesize(
            prompt_wav=reference_audio_path,
            prompt_text=payload.reference_text,
            text=payload.text,
            output_path=output_path,
            num_step=generation.num_steps,
            guidance_scale=generation.guidance_scale,
            speed=generation.speed,
            t_shift=generation.t_shift,
            max_duration=generation.max_duration_seconds,
            remove_long_sil=generation.remove_long_silence,
            seed=generation.seed,
            normalize_vietnamese=generation.normalize_vietnamese,
            split_sentences=generation.split_sentences,
            crossfade_ms=postprocess.crossfade_ms,
            silence_ms=postprocess.silence_ms,
            fade_in_ms=postprocess.fade_in_ms,
            fade_out_ms=postprocess.fade_out_ms,
        )
        return SynthesisResult(metrics=normalize_metrics(raw_metrics))


def normalize_metrics(raw_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "elapsed_seconds": float(raw_metrics.get("t", raw_metrics.get("elapsed_seconds", 0.0))),
        "wav_seconds": float(raw_metrics.get("wav_seconds", 0.0)),
        "rtf": float(raw_metrics.get("rtf", 0.0)),
        "segments": int(raw_metrics.get("segments", 0)),
        "sample_rate": int(raw_metrics.get("sample_rate", 24000)),
    }

