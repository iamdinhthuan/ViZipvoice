import json
import logging
import re
import tempfile
import time
from pathlib import Path
from typing import Optional, Union

import safetensors.torch
import torch
import torchaudio
from huggingface_hub import hf_hub_download, list_repo_files
from lhotse.utils import fix_random_seed

from zipvoice.bin.infer_zipvoice import generate_sentence, get_vocoder
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.tokenizer.tokenizer import SimpleTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank

DEFAULT_REPO_ID = "dolly-vn/ViZipvoice"
DEFAULT_CHECKPOINT_NAME = "latest"
CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)\.pt$")
SENTENCE_SPLIT_PATTERN = re.compile(r"[^.?？。]+(?:[.?？。]+|$)", re.S)
PUNCTUATION_NO_SPACE_BEFORE = r",.;:!?…%"
OPENING_QUOTES_AND_BRACKETS = r"\(\[\{«“‘"
CLOSING_QUOTES_AND_BRACKETS = r"\)\]\}»”’"


def _resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _download_model_files(
    repo_id: str,
    revision: Optional[str],
    checkpoint_name: str,
) -> tuple[Path, Path, Path]:
    checkpoint_name = _resolve_hf_checkpoint_name(
        repo_id=repo_id,
        revision=revision,
        checkpoint_name=checkpoint_name,
    )
    checkpoint_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=checkpoint_name,
            revision=revision,
        )
    )
    model_config_path = _download_config_file(
        repo_id=repo_id,
        revision=revision,
    )
    token_file = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename="tokens.txt",
            revision=revision,
        )
    )
    return checkpoint_path, model_config_path, token_file


def _download_config_file(repo_id: str, revision: Optional[str]) -> Path:
    last_error = None
    for filename in ("config.json", "model.json"):
        try:
            return Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    revision=revision,
                )
            )
        except Exception as exc:
            last_error = exc
    raise FileNotFoundError("No config.json or model.json file found.") from last_error


def _checkpoint_step(filename: str) -> int:
    match = CHECKPOINT_RE.match(Path(filename).name)
    return int(match.group(1)) if match else -1


def _select_latest_checkpoint(filenames: list[str]) -> str:
    checkpoints = [
        filename for filename in filenames if _checkpoint_step(filename) >= 0
    ]
    if checkpoints:
        return max(checkpoints, key=lambda filename: _checkpoint_step(filename))
    raise FileNotFoundError("No checkpoint-<step>.pt file found.")


def _resolve_hf_checkpoint_name(
    repo_id: str,
    revision: Optional[str],
    checkpoint_name: str,
) -> str:
    if checkpoint_name != "latest":
        return checkpoint_name
    filenames = list_repo_files(repo_id=repo_id, revision=revision)
    return _select_latest_checkpoint(filenames)


def _resolve_local_checkpoint_path(
    model_dir: Path,
    checkpoint_name: str,
) -> Path:
    if checkpoint_name != "latest":
        return model_dir / checkpoint_name
    filenames = [path.name for path in model_dir.iterdir() if path.is_file()]
    return model_dir / _select_latest_checkpoint(filenames)


def _resolve_local_config_path(model_dir: Path) -> Path:
    for filename in ("config.json", "model.json"):
        config_path = model_dir / filename
        if config_path.is_file():
            return config_path
    raise FileNotFoundError(f"No config.json or model.json file found in {model_dir}")


def cleanup_vietnamese_spacing(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(
        rf"\s+([{re.escape(PUNCTUATION_NO_SPACE_BEFORE)}])",
        r"\1",
        text,
    )
    text = re.sub(
        rf"\s+([{CLOSING_QUOTES_AND_BRACKETS}])",
        r"\1",
        text,
    )
    text = re.sub(
        rf"([{OPENING_QUOTES_AND_BRACKETS}])\s+",
        r"\1",
        text,
    )
    text = re.sub(
        rf"([{re.escape(PUNCTUATION_NO_SPACE_BEFORE)}])"
        rf"([^\s{CLOSING_QUOTES_AND_BRACKETS}])",
        r"\1 \2",
        text,
    )
    return text.strip()


def normalize_vietnamese_text(text: str, enabled: bool = True) -> str:
    if not enabled:
        return text.strip()

    try:
        from soe_vinorm import normalize_text
    except ImportError as exc:
        raise RuntimeError(
            "Vietnamese normalization requires soe-vinorm. "
            "Install it with `pip install soe-vinorm`."
        ) from exc

    return cleanup_vietnamese_spacing(normalize_text(text))


def split_text_into_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    sentences = [
        match.group(0).strip()
        for match in SENTENCE_SPLIT_PATTERN.finditer(text)
        if match.group(0).strip()
    ]
    return sentences or [text]


def count_sentence_words(sentence: str) -> int:
    return len(re.findall(r"\w+", sentence, flags=re.UNICODE))


def get_sentence_inference_params(
    sentence: str,
    base_num_step: int,
    base_speed: float,
) -> tuple[int, float, int]:
    word_count = count_sentence_words(sentence)
    if word_count == 1:
        return max(base_num_step, 24), 0.6, word_count
    if 2 <= word_count <= 4:
        return base_num_step, 0.8, word_count
    return base_num_step, base_speed, word_count


def match_audio_channels(
    first: torch.Tensor,
    second: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if first.shape[0] == second.shape[0]:
        return first, second
    if first.shape[0] == 1:
        return first.expand(second.shape[0], -1), second
    if second.shape[0] == 1:
        return first, second.expand(first.shape[0], -1)

    channels = min(first.shape[0], second.shape[0])
    return first[:channels], second[:channels]


def append_with_crossfade(
    first: torch.Tensor,
    second: torch.Tensor,
    crossfade_samples: int,
) -> torch.Tensor:
    first, second = match_audio_channels(first, second)
    fade_len = min(crossfade_samples, first.shape[1], second.shape[1])
    if fade_len <= 0:
        return torch.cat([first, second], dim=1)

    fade_out = torch.linspace(
        1.0,
        0.0,
        fade_len,
        dtype=first.dtype,
        device=first.device,
    ).unsqueeze(0)
    fade_in = torch.linspace(
        0.0,
        1.0,
        fade_len,
        dtype=second.dtype,
        device=second.device,
    ).unsqueeze(0)
    overlap = first[:, -fade_len:] * fade_out + second[:, :fade_len] * fade_in
    return torch.cat([first[:, :-fade_len], overlap, second[:, fade_len:]], dim=1)


def apply_fade(audio: torch.Tensor, fade_in_samples: int, fade_out_samples: int) -> torch.Tensor:
    if audio.numel() == 0:
        return audio

    audio = audio.clone()
    if fade_in_samples > 0:
        fade_len = min(fade_in_samples, audio.shape[1])
        fade = torch.linspace(
            0.0,
            1.0,
            fade_len,
            dtype=audio.dtype,
            device=audio.device,
        ).unsqueeze(0)
        audio[:, :fade_len] *= fade

    if fade_out_samples > 0:
        fade_len = min(fade_out_samples, audio.shape[1])
        fade = torch.linspace(
            1.0,
            0.0,
            fade_len,
            dtype=audio.dtype,
            device=audio.device,
        ).unsqueeze(0)
        audio[:, -fade_len:] *= fade

    return audio


def postprocess_audio_segments(
    segment_paths: list[Path],
    output_path: Path,
    sampling_rate: int,
    crossfade_ms: int,
    silence_ms: int,
    fade_in_ms: int,
    fade_out_ms: int,
) -> None:
    if not segment_paths:
        raise RuntimeError("No generated audio segments to postprocess.")

    crossfade_samples = int(sampling_rate * max(crossfade_ms, 0) / 1000)
    silence_samples = int(sampling_rate * max(silence_ms, 0) / 1000)
    fade_in_samples = int(sampling_rate * max(fade_in_ms, 0) / 1000)
    fade_out_samples = int(sampling_rate * max(fade_out_ms, 0) / 1000)

    combined = None
    for index, segment_path in enumerate(segment_paths):
        audio, sr = torchaudio.load(str(segment_path))
        if sr != sampling_rate:
            audio = torchaudio.functional.resample(audio, sr, sampling_rate)

        if index < len(segment_paths) - 1 and silence_samples > 0:
            silence = torch.zeros(
                audio.shape[0],
                silence_samples,
                dtype=audio.dtype,
                device=audio.device,
            )
            audio = torch.cat([audio, silence], dim=1)

        if combined is None:
            combined = audio
        else:
            combined = append_with_crossfade(combined, audio, crossfade_samples)

    combined = apply_fade(
        combined,
        fade_in_samples=fade_in_samples,
        fade_out_samples=fade_out_samples,
    )
    combined = combined.clamp(min=-1.0, max=1.0).cpu()
    torchaudio.save(str(output_path), combined, sampling_rate)


def wav_seconds(path: Union[str, Path]) -> float:
    info = torchaudio.info(str(path))
    return float(info.num_frames) / float(info.sample_rate)


class ViZipVoiceTTS:
    """Small wrapper for Vietnamese ZipVoice inference.

    The wrapper downloads model files from Hugging Face by default, builds the
    ZipVoice model with the Vietnamese character tokenizer, and exposes a
    single synthesize method.
    """

    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        revision: Optional[str] = None,
        model_dir: Optional[Union[str, Path]] = None,
        checkpoint_name: str = DEFAULT_CHECKPOINT_NAME,
        vocoder_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        use_fp16: bool = True,
        num_threads: int = 1,
    ) -> None:
        try:
            torch.set_num_threads(num_threads)
            torch.set_num_interop_threads(num_threads)
        except RuntimeError:
            logging.debug("PyTorch thread settings were already initialized.")

        self.repo_id = repo_id
        self.revision = revision
        self.device = _resolve_device(device)
        self.use_fp16 = bool(use_fp16 and self.device.type == "cuda")

        if model_dir is None:
            checkpoint_path, model_config_path, token_file = _download_model_files(
                repo_id=repo_id,
                revision=revision,
                checkpoint_name=checkpoint_name,
            )
        else:
            model_dir = Path(model_dir)
            checkpoint_path = _resolve_local_checkpoint_path(
                model_dir=model_dir,
                checkpoint_name=checkpoint_name,
            )
            model_config_path = _resolve_local_config_path(model_dir)
            token_file = model_dir / "tokens.txt"

        self.checkpoint_path = Path(checkpoint_path)
        self.model_config_path = Path(model_config_path)
        self.token_file = Path(token_file)
        self._validate_model_files()

        with self.model_config_path.open("r", encoding="utf-8") as f:
            self.model_config = json.load(f)

        self.tokenizer = SimpleTokenizer(token_file=str(self.token_file))
        self.model = ZipVoice(
            **self.model_config["model"],
            vocab_size=self.tokenizer.vocab_size,
            pad_id=self.tokenizer.pad_id,
        )
        self._load_checkpoint()
        self.model.to(self.device)
        self.model.eval()

        self.feature_extractor = VocosFbank()
        self.vocoder = get_vocoder(str(vocoder_path) if vocoder_path else None)
        self.vocoder.to(self.device)
        self.vocoder.eval()
        self.sampling_rate = int(self.model_config["feature"]["sampling_rate"])

        logging.info(
            "Loaded ViZipVoice from %s on %s | fp16 autocast: %s",
            self.checkpoint_path,
            self.device,
            self.use_fp16,
        )

    def _validate_model_files(self) -> None:
        missing = [
            path
            for path in [self.checkpoint_path, self.model_config_path, self.token_file]
            if not path.is_file()
        ]
        if missing:
            missing_text = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"Missing ViZipVoice model file(s): {missing_text}")

    def _load_checkpoint(self) -> None:
        suffix = self.checkpoint_path.suffix.lower()
        if suffix == ".safetensors":
            safetensors.torch.load_model(self.model, str(self.checkpoint_path))
        elif suffix == ".pt":
            load_checkpoint(
                filename=self.checkpoint_path,
                model=self.model,
                strict=True,
            )
        else:
            raise ValueError(f"Unsupported checkpoint format: {self.checkpoint_path}")

    @torch.inference_mode()
    def synthesize(
        self,
        prompt_wav: Union[str, Path],
        prompt_text: str,
        text: str,
        output_path: Union[str, Path] = "output.wav",
        num_step: int = 16,
        guidance_scale: float = 1.0,
        speed: float = 1.0,
        t_shift: float = 0.5,
        target_rms: float = 0.1,
        feat_scale: float = 0.1,
        max_duration: float = 100,
        remove_long_sil: bool = False,
        seed: Optional[int] = 666,
        normalize_vietnamese: bool = True,
        split_sentences: bool = True,
        crossfade_ms: int = 80,
        silence_ms: int = 180,
        fade_in_ms: int = 20,
        fade_out_ms: int = 80,
    ) -> dict:
        if seed is not None and seed >= 0:
            fix_random_seed(int(seed))

        prompt_text = normalize_vietnamese_text(
            prompt_text,
            enabled=normalize_vietnamese,
        )
        text = normalize_vietnamese_text(
            text,
            enabled=normalize_vietnamese,
        )
        target_sentences = split_text_into_sentences(text) if split_sentences else [text]
        if not target_sentences:
            raise ValueError("No valid text to synthesize.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        segment_paths = []
        segment_metrics = []
        segment_settings = []
        start_time = time.time()
        with tempfile.TemporaryDirectory(
            prefix=f"{output_path.stem}_segments_",
            dir=str(output_path.parent),
        ) as segment_dir_name:
            segment_dir = Path(segment_dir_name)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=self.use_fp16,
            ):
                for index, sentence in enumerate(target_sentences, start=1):
                    sentence_num_step, sentence_speed, word_count = (
                        get_sentence_inference_params(
                            sentence=sentence,
                            base_num_step=int(num_step),
                            base_speed=float(speed),
                        )
                    )
                    segment_path = segment_dir / f"segment_{index:03d}.wav"
                    metrics = generate_sentence(
                        save_path=str(segment_path),
                        prompt_text=prompt_text,
                        prompt_wav=str(prompt_wav),
                        text=sentence,
                        model=self.model,
                        vocoder=self.vocoder,
                        tokenizer=self.tokenizer,
                        feature_extractor=self.feature_extractor,
                        device=self.device,
                        num_step=sentence_num_step,
                        guidance_scale=float(guidance_scale),
                        speed=sentence_speed,
                        t_shift=float(t_shift),
                        target_rms=float(target_rms),
                        feat_scale=float(feat_scale),
                        sampling_rate=self.sampling_rate,
                        max_duration=float(max_duration),
                        remove_long_sil=bool(remove_long_sil),
                    )
                    segment_paths.append(segment_path)
                    segment_metrics.append(metrics)
                    segment_settings.append(
                        {
                            "index": index,
                            "word_count": word_count,
                            "speed": sentence_speed,
                            "num_step": sentence_num_step,
                            "text": sentence,
                        }
                    )

            postprocess_audio_segments(
                segment_paths=segment_paths,
                output_path=output_path,
                sampling_rate=self.sampling_rate,
                crossfade_ms=int(crossfade_ms),
                silence_ms=int(silence_ms),
                fade_in_ms=int(fade_in_ms),
                fade_out_ms=int(fade_out_ms),
            )

        elapsed = time.time() - start_time
        audio_seconds = wav_seconds(output_path)
        t_no_vocoder = sum(item.get("t_no_vocoder", 0.0) for item in segment_metrics)
        t_vocoder = sum(item.get("t_vocoder", 0.0) for item in segment_metrics)
        rtf = elapsed / audio_seconds if audio_seconds else 0.0
        return {
            "t": elapsed,
            "t_no_vocoder": t_no_vocoder,
            "t_vocoder": t_vocoder,
            "wav_seconds": audio_seconds,
            "rtf": rtf,
            "rtf_no_vocoder": t_no_vocoder / audio_seconds if audio_seconds else 0.0,
            "rtf_vocoder": t_vocoder / audio_seconds if audio_seconds else 0.0,
            "segments": len(segment_paths),
            "segment_settings": segment_settings,
            "segment_metrics": segment_metrics,
            "crossfade_ms": int(crossfade_ms),
            "silence_ms": int(silence_ms),
            "fade_in_ms": int(fade_in_ms),
            "fade_out_ms": int(fade_out_ms),
        }
