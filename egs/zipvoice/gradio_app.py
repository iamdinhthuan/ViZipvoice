#!/usr/bin/env python3
import argparse
import csv
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gradio as gr
import torch
import torchaudio
from lhotse.utils import fix_random_seed

from zipvoice.bin.infer_zipvoice import generate_sentence, get_vocoder
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.tokenizer.tokenizer import SimpleTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank
from zipvoice.vizipvoice import normalize_vietnamese_text

try:
    import soundfile as sf
except Exception:
    sf = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
SENTENCE_SPLIT_PATTERN = re.compile(r"[^.?？。]+(?:[.?？。]+|$)", re.S)


@dataclass(frozen=True)
class PresetPrompt:
    label: str
    cut_id: str
    text: str
    audio_path: str
    duration: Optional[float]


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)\.pt$", path.name)
    if match:
        return int(match.group(1))
    return -1


def find_latest_checkpoint(exp_dir: Path) -> Path:
    checkpoints = sorted(
        exp_dir.glob("checkpoint-*.pt"),
        key=lambda p: (checkpoint_step(p), p.stat().st_mtime),
    )
    if checkpoints:
        return checkpoints[-1]

    fallback = sorted(exp_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if fallback:
        return fallback[-1]

    raise FileNotFoundError(f"No checkpoint found in {exp_dir}")


def audio_duration(path: str) -> Optional[float]:
    if sf is None:
        return None
    try:
        info = sf.info(path)
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return None


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


def load_presets(train_tsv: Path, preset_count: int) -> list[PresetPrompt]:
    if not train_tsv.is_file():
        raise FileNotFoundError(f"Train TSV not found: {train_tsv}")

    presets = []
    with train_tsv.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue

            cut_id, text, audio_path = parts
            if not Path(audio_path).is_file():
                continue

            duration = audio_duration(audio_path)
            duration_label = f"{duration:.1f}s" if duration is not None else "audio"
            preview = text if len(text) <= 74 else f"{text[:71]}..."
            label = f"{len(presets) + 1:02d}. {duration_label} | {preview}"
            presets.append(
                PresetPrompt(
                    label=label,
                    cut_id=cut_id,
                    text=text,
                    audio_path=audio_path,
                    duration=duration,
                )
            )
            if len(presets) >= preset_count:
                break

    if not presets:
        raise RuntimeError(f"No valid preset audio found in {train_tsv}")
    return presets


def make_preset_label(
    index: int,
    text: str,
    duration: Optional[float],
    speaker: Optional[str] = None,
) -> str:
    duration_label = f"{duration:.1f}s" if duration is not None else "audio"
    text_preview = text if len(text) <= 58 else f"{text[:55]}..."
    if speaker:
        speaker_preview = speaker if len(speaker) <= 32 else f"{speaker[:29]}..."
        return f"{index:02d}. {duration_label} | {speaker_preview} | {text_preview}"
    return f"{index:02d}. {duration_label} | {text_preview}"


def load_sidecar_ref_presets(
    ref_audio_dir: Path,
    preset_count: int,
) -> list[PresetPrompt]:
    presets = []
    audio_paths = [
        path
        for path in sorted(ref_audio_dir.iterdir())
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    for audio_path in audio_paths[:preset_count]:
        text_path = audio_path.with_suffix(".txt")
        if not text_path.is_file():
            continue

        text = text_path.read_text(encoding="utf-8").strip()
        duration = audio_duration(str(audio_path))
        presets.append(
            PresetPrompt(
                label=audio_path.stem,
                cut_id=audio_path.stem,
                text=text,
                audio_path=str(audio_path),
                duration=duration,
            )
        )

    return presets


def load_ref_presets(
    ref_audio_dir: Path,
    preset_count: int,
    metadata_path: Optional[Path] = None,
) -> list[PresetPrompt]:
    ref_audio_dir = Path(ref_audio_dir)
    if not ref_audio_dir.is_dir():
        raise FileNotFoundError(f"Ref audio dir not found: {ref_audio_dir}")

    sidecar_presets = load_sidecar_ref_presets(ref_audio_dir, preset_count)
    if sidecar_presets:
        return sidecar_presets

    metadata_path = Path(metadata_path) if metadata_path else ref_audio_dir / "metadata_audio_multispeaker.csv"
    candidates = []

    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                rel_audio_path = (row.get("audio") or "").strip()
                text = (row.get("text") or "").strip()
                speaker = (row.get("speak") or "").strip()
                if not rel_audio_path:
                    continue

                audio_path = ref_audio_dir / rel_audio_path
                if not audio_path.is_file():
                    continue

                candidates.append((audio_path, text, speaker))
    else:
        for audio_path in sorted(ref_audio_dir.rglob("*")):
            if audio_path.suffix.lower() not in AUDIO_EXTENSIONS or not audio_path.is_file():
                continue

            candidates.append((audio_path, "", ""))

    presets = []
    for audio_path, text, speaker in candidates[-preset_count:]:
        duration = audio_duration(str(audio_path))
        label = make_preset_label(
            index=len(presets) + 1,
            text=text or audio_path.stem,
            duration=duration,
            speaker=speaker,
        )
        presets.append(
            PresetPrompt(
                label=label,
                cut_id=audio_path.stem,
                text=text,
                audio_path=str(audio_path),
                duration=duration,
            )
        )

    if not presets:
        raise RuntimeError(f"No valid ref audio found in {ref_audio_dir}")
    return presets


def resolve_ref_audio_dir(
    exp_dir: Path,
    ref_audio_dir: Optional[str],
) -> Optional[Path]:
    if ref_audio_dir:
        return Path(ref_audio_dir)

    candidates = [exp_dir / "audio", Path("audio")]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def build_model(
    model_config_path: Path,
    checkpoint_path: Path,
    token_file: Path,
    device: torch.device,
) -> tuple[ZipVoice, SimpleTokenizer, dict]:
    with model_config_path.open("r", encoding="utf-8") as f:
        model_config = json.load(f)

    tokenizer = SimpleTokenizer(token_file=str(token_file))
    model = ZipVoice(
        **model_config["model"],
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
    )
    load_checkpoint(filename=str(checkpoint_path), model=model, strict=True)
    model.to(device)
    model.eval()
    return model, tokenizer, model_config


class ZipVoiceGradio:
    def __init__(self, args: argparse.Namespace):
        self.exp_dir = Path(args.exp_dir)
        self.checkpoint_path = find_latest_checkpoint(self.exp_dir)
        self.model_config_path = Path(args.model_config or self.exp_dir / "model.json")
        self.token_file = Path(args.token_file or self.exp_dir / "tokens.txt")
        self.results_dir = Path(args.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        ref_audio_dir = resolve_ref_audio_dir(self.exp_dir, args.ref_audio_dir)
        if ref_audio_dir is not None and ref_audio_dir.is_dir():
            ref_metadata = Path(args.ref_metadata) if args.ref_metadata else None
            self.presets = load_ref_presets(
                ref_audio_dir=ref_audio_dir,
                preset_count=args.preset_count,
                metadata_path=ref_metadata,
            )
            self.preset_source = str(ref_audio_dir)
        else:
            self.presets = load_presets(Path(args.train_tsv), args.preset_count)
            self.preset_source = str(args.train_tsv)
        self.presets_by_label = {preset.label: preset for preset in self.presets}

        self.device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        self.fp16 = bool(args.fp16 and self.device.type == "cuda")

        logging.info("Loading checkpoint: %s", self.checkpoint_path)
        self.model, self.tokenizer, self.model_config = build_model(
            model_config_path=self.model_config_path,
            checkpoint_path=self.checkpoint_path,
            token_file=self.token_file,
            device=self.device,
        )
        self.feature_extractor = VocosFbank()
        self.vocoder = get_vocoder(args.vocoder_path).to(self.device)
        self.vocoder.eval()
        self.sampling_rate = int(self.model_config["feature"]["sampling_rate"])

        logging.info("Device: %s | fp16 autocast: %s", self.device, self.fp16)
        logging.info("Loaded %d preset prompts from %s", len(self.presets), self.preset_source)

    @property
    def default_preset(self) -> PresetPrompt:
        return self.presets[0]

    @property
    def allowed_paths(self) -> list[str]:
        preset_dirs = {str(Path(item.audio_path).resolve().parent) for item in self.presets}
        preset_dirs.add(str(self.results_dir.resolve()))
        return sorted(preset_dirs)

    def select_preset(self, label: str):
        preset = self.presets_by_label[label]
        return preset.audio_path, preset.text

    def generate(
        self,
        preset_label: str,
        prompt_audio: Optional[str],
        prompt_text: str,
        text: str,
        num_step: int,
        guidance_scale: float,
        speed: float,
        t_shift: float,
        max_duration: int,
        seed: int,
        remove_long_sil: bool,
        split_sentences: bool,
        crossfade_ms: int,
        silence_ms: int,
        fade_in_ms: int,
        fade_out_ms: int,
    ):
        if not text or not text.strip():
            raise gr.Error("Nhập text cần sinh trước khi chạy inference.")

        preset = self.presets_by_label[preset_label]
        prompt_wav = prompt_audio or preset.audio_path
        prompt_text = normalize_vietnamese_text(prompt_text.strip() or preset.text)
        if not Path(prompt_wav).is_file():
            raise gr.Error(f"Không tìm thấy prompt audio: {prompt_wav}")

        seed = int(seed)
        if seed >= 0:
            fix_random_seed(seed)

        target_text = normalize_vietnamese_text(text.strip())
        target_sentences = (
            split_text_into_sentences(target_text) if split_sentences else [target_text]
        )
        if not target_sentences:
            raise gr.Error("Không tách được câu hợp lệ từ text cần sinh.")

        run_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        output_path = self.results_dir / f"zipvoice_{run_id}.wav"
        segment_dir = self.results_dir / "segments" / run_id
        segment_dir.mkdir(parents=True, exist_ok=True)
        segment_paths = []
        segment_settings = []
        start_time = time.time()

        autocast_enabled = self.fp16
        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=autocast_enabled,
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
                    generate_sentence(
                        save_path=str(segment_path),
                        prompt_text=prompt_text,
                        prompt_wav=prompt_wav,
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
                        sampling_rate=self.sampling_rate,
                        max_duration=int(max_duration),
                        remove_long_sil=bool(remove_long_sil),
                    )
                    segment_paths.append(segment_path)
                    segment_settings.append(
                        f"{index}:{word_count}w/speed={sentence_speed:g}/step={sentence_num_step}"
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
        status = (
            f"Checkpoint: {self.checkpoint_path.name}\n"
            f"Device: {self.device}\n"
            f"FP16 inference: {'on (CUDA autocast)' if self.fp16 else 'off'}\n"
            f"Prompt: {Path(prompt_wav).name}\n"
            f"Segments: {len(target_sentences)}\n"
            f"Segment settings: {', '.join(segment_settings)}\n"
            f"Crossfade: {int(crossfade_ms)} ms\n"
            f"Silence: {int(silence_ms)} ms\n"
            f"Fade in/out: {int(fade_in_ms)} / {int(fade_out_ms)} ms\n"
            f"Elapsed: {elapsed:.2f}s"
        )
        return str(output_path), status


def build_demo(app: ZipVoiceGradio) -> gr.Blocks:
    checkpoint_info = (
        f"Checkpoint: {app.checkpoint_path} | "
        f"Device: {app.device} | "
        f"FP16: {'on' if app.fp16 else 'off'}"
    )

    with gr.Blocks(title="ZipVoice Vietnamese Inference") as demo:
        gr.Markdown(f"### ZipVoice Vietnamese Inference\n{checkpoint_info}")

        with gr.Row():
            with gr.Column(scale=1):
                preset = gr.Dropdown(
                    choices=[item.label for item in app.presets],
                    value=app.default_preset.label,
                    label="Ref audio",
                )
                prompt_audio = gr.Audio(
                    value=app.default_preset.audio_path,
                    type="filepath",
                    label="Prompt audio",
                )
                prompt_text = gr.Textbox(
                    value=app.default_preset.text,
                    lines=4,
                    label="Prompt text",
                )

            with gr.Column(scale=1):
                text = gr.Textbox(
                    value="Xin chào, đây là bản thử nghiệm tiếng Việt của ZipVoice.",
                    lines=5,
                    label="Text cần sinh",
                )
                with gr.Row():
                    num_step = gr.Slider(4, 64, value=16, step=1, label="Steps")
                    guidance_scale = gr.Slider(0.0, 5.0, value=1.0, step=0.1, label="Guidance")
                with gr.Row():
                    speed = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Speed")
                    t_shift = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="T-shift")
                with gr.Row():
                    max_duration = gr.Slider(10, 200, value=100, step=5, label="Max duration")
                    seed = gr.Number(value=666, precision=0, label="Seed (-1 random)")
                remove_long_sil = gr.Checkbox(value=False, label="Remove long silence")
                split_sentences = gr.Checkbox(value=True, label="Split sentences")
                with gr.Row():
                    crossfade_ms = gr.Slider(0, 300, value=80, step=10, label="Crossfade ms")
                    silence_ms = gr.Slider(0, 800, value=180, step=10, label="Silence ms")
                with gr.Row():
                    fade_in_ms = gr.Slider(0, 500, value=20, step=10, label="Fade in ms")
                    fade_out_ms = gr.Slider(0, 500, value=80, step=10, label="Fade out ms")
                generate_btn = gr.Button("Generate", variant="primary")

        with gr.Row():
            output_audio = gr.Audio(type="filepath", label="Output")
            status = gr.Textbox(lines=6, label="Status")

        preset.change(
            fn=app.select_preset,
            inputs=[preset],
            outputs=[prompt_audio, prompt_text],
        )
        generate_btn.click(
            fn=app.generate,
            inputs=[
                preset,
                prompt_audio,
                prompt_text,
                text,
                num_step,
                guidance_scale,
                speed,
                t_shift,
                max_duration,
                seed,
                remove_long_sil,
                split_sentences,
                crossfade_ms,
                silence_ms,
                fade_in_ms,
                fade_out_ms,
            ],
            outputs=[output_audio, status],
        )

    return demo


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", default="exp/zipvoice_vi_simple_finetune")
    parser.add_argument("--train-tsv", default="data/raw/custom_train.tsv")
    parser.add_argument(
        "--ref-audio-dir",
        default=None,
        help="Directory containing ref audio. Defaults to <exp-dir>/audio, then ./audio.",
    )
    parser.add_argument("--ref-metadata", default=None)
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--token-file", default=None)
    parser.add_argument("--results-dir", default="results/gradio")
    parser.add_argument("--vocoder-path", default=None)
    parser.add_argument("--preset-count", type=int, default=30)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--fp16", type=int, default=1)
    return parser


def main():
    torch.set_num_threads(1)
    args = get_parser().parse_args()
    app = ZipVoiceGradio(args)
    demo = build_demo(app)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=app.allowed_paths,
    )


if __name__ == "__main__":
    main()
