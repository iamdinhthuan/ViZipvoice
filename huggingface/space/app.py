import json
import logging
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import gradio as gr
import torch
from huggingface_hub import hf_hub_download, list_repo_files

try:
    import spaces
except ImportError:
    class _SpacesFallback:
        @staticmethod
        def GPU(*decorator_args, **decorator_kwargs):
            if decorator_args and callable(decorator_args[0]) and not decorator_kwargs:
                return decorator_args[0]

            def decorator(fn):
                return fn

            return decorator

    spaces = _SpacesFallback()


MODEL_REPO_ID = os.getenv("VIZIPVOICE_MODEL_REPO", "contextboxai/ViZipvoice")
GITHUB_REPO_URL = os.getenv(
    "VIZIPVOICE_GITHUB_REPO",
    "https://github.com/iamdinhthuan/ViZipvoice.git",
)
CODE_DIR = Path(os.getenv("VIZIPVOICE_CODE_DIR", "/tmp/ViZipvoice"))
OUTPUT_DIR = Path(os.getenv("VIZIPVOICE_OUTPUT_DIR", "/tmp/vizipvoice_outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEMO_TEXT = (
    "Chiến tranh luôn là một chủ đề nặng nề, nhưng cũng rất cần được nhắc đến "
    "để con người hiểu rõ hơn giá trị của hòa bình. Khi một cuộc chiến xảy ra, "
    "những gì bị phá hủy không chỉ là nhà cửa, đường sá, trường học hay bệnh viện. "
    "Điều đau lòng nhất chính là sinh mạng con người, là những gia đình bị chia cắt, "
    "là những đứa trẻ phải lớn lên trong sợ hãi, và là những vùng đất từng yên bình "
    "bỗng trở nên hoang tàn."
)
PREFERRED_REFS = ["Đinh-Quyết", "Nhã-Uyên", "MC"]
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass(frozen=True)
class RefPrompt:
    label: str
    audio_path: str
    text: str


def ensure_code_repo() -> None:
    refresh_code = os.getenv("VIZIPVOICE_REFRESH_CODE", "1") != "0"
    if (CODE_DIR / "zipvoice" / "vizipvoice.py").is_file() and not refresh_code:
        return

    if CODE_DIR.exists():
        shutil.rmtree(CODE_DIR)

    logging.info("Cloning ViZipVoice code from %s", GITHUB_REPO_URL)
    subprocess.run(
        ["git", "clone", "--depth", "1", GITHUB_REPO_URL, str(CODE_DIR)],
        check=True,
    )


ensure_code_repo()
sys.path.insert(0, str(CODE_DIR))

from zipvoice.vizipvoice import ViZipVoiceTTS  # noqa: E402


def sorted_ref_audio_files() -> list[str]:
    files = list_repo_files(repo_id=MODEL_REPO_ID, repo_type="model")
    audio_files = [
        filename
        for filename in files
        if filename.startswith("audio/")
        and Path(filename).suffix.lower() in AUDIO_EXTENSIONS
    ]

    def sort_key(filename: str) -> tuple[int, str]:
        stem = Path(filename).stem
        if stem in PREFERRED_REFS:
            return (PREFERRED_REFS.index(stem), stem)
        return (len(PREFERRED_REFS), stem)

    return sorted(audio_files, key=sort_key)


@lru_cache(maxsize=1)
def load_ref_prompts() -> tuple[RefPrompt, ...]:
    prompts = []
    for audio_file in sorted_ref_audio_files():
        audio_path = Path(
            hf_hub_download(
                repo_id=MODEL_REPO_ID,
                repo_type="model",
                filename=audio_file,
            )
        )
        text_file = str(Path(audio_file).with_suffix(".txt"))
        text_path = Path(
            hf_hub_download(
                repo_id=MODEL_REPO_ID,
                repo_type="model",
                filename=text_file,
            )
        )
        text = text_path.read_text(encoding="utf-8").strip()
        prompts.append(
            RefPrompt(
                label=audio_path.stem,
                audio_path=str(audio_path),
                text=text,
            )
        )

    if not prompts:
        raise RuntimeError(f"No reference audio found in {MODEL_REPO_ID}/audio")
    return tuple(prompts)


@lru_cache(maxsize=1)
def get_tts() -> ViZipVoiceTTS:
    logging.info("Loading ViZipVoice model from %s", MODEL_REPO_ID)
    return ViZipVoiceTTS(repo_id=MODEL_REPO_ID, checkpoint_name="latest", num_threads=1)


if os.getenv("VIZIPVOICE_PRELOAD_MODEL", "1") != "0":
    get_tts()


def refs_by_label() -> dict[str, RefPrompt]:
    return {item.label: item for item in load_ref_prompts()}


def default_ref() -> RefPrompt:
    prompts = load_ref_prompts()
    for item in prompts:
        if item.label == "Đinh-Quyết":
            return item
    return prompts[0]


def select_ref(label: str) -> tuple[str, str]:
    item = refs_by_label()[label]
    return item.audio_path, item.text


@spaces.GPU(duration=60)
def generate(
    ref_label: str,
    prompt_audio: Optional[str],
    prompt_text: str,
    text: str,
    num_step: int,
    guidance_scale: float,
    speed: float,
    t_shift: float,
    max_duration: int,
    seed: int,
    normalize_vietnamese: bool,
    split_sentences: bool,
    remove_long_sil: bool,
    crossfade_ms: int,
    silence_ms: int,
    fade_in_ms: int,
    fade_out_ms: int,
) -> tuple[str, str]:
    if not text or not text.strip():
        raise gr.Error("Nhập text cần sinh trước khi chạy inference.")

    ref = refs_by_label()[ref_label]
    prompt_wav = prompt_audio or ref.audio_path
    final_prompt_text = prompt_text.strip() or ref.text
    output_path = OUTPUT_DIR / f"vizipvoice_{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"

    start = time.time()
    try:
        metrics = get_tts().synthesize(
            prompt_wav=prompt_wav,
            prompt_text=final_prompt_text,
            text=text.strip(),
            output_path=output_path,
            num_step=int(num_step),
            guidance_scale=float(guidance_scale),
            speed=float(speed),
            t_shift=float(t_shift),
            max_duration=float(max_duration),
            remove_long_sil=bool(remove_long_sil),
            seed=int(seed),
            normalize_vietnamese=bool(normalize_vietnamese),
            split_sentences=bool(split_sentences),
            crossfade_ms=int(crossfade_ms),
            silence_ms=int(silence_ms),
            fade_in_ms=int(fade_in_ms),
            fade_out_ms=int(fade_out_ms),
        )
    except Exception as exc:
        logging.exception("ViZipVoice inference failed")
        raise gr.Error(str(exc)) from exc

    elapsed = time.time() - start
    device = str(get_tts().device)
    status = {
        "checkpoint": Path(get_tts().checkpoint_path).name,
        "device": device,
        "prompt": Path(prompt_wav).name,
        "segments": metrics.get("segments"),
        "wav_seconds": round(metrics.get("wav_seconds", 0.0), 3),
        "rtf": round(metrics.get("rtf", 0.0), 4),
        "elapsed_seconds": round(elapsed, 3),
        "segment_settings": metrics.get("segment_settings", []),
    }
    return str(output_path), json.dumps(status, ensure_ascii=False, indent=2)


def build_app() -> gr.Blocks:
    ref = default_ref()
    choices = [item.label for item in load_ref_prompts()]

    with gr.Blocks(title="ViZipVoice") as demo:
        gr.Markdown(
            "# ViZipVoice\n"
            "[Model](https://huggingface.co/contextboxai/ViZipvoice) · "
            "[GitHub](https://github.com/iamdinhthuan/ViZipvoice)"
        )

        with gr.Row():
            with gr.Column(scale=1):
                ref_label = gr.Dropdown(
                    choices=choices,
                    value=ref.label,
                    label="Ref audio",
                )
                prompt_audio = gr.Audio(
                    value=ref.audio_path,
                    type="filepath",
                    label="Prompt audio",
                )
                prompt_text = gr.Textbox(
                    value=ref.text,
                    lines=4,
                    label="Prompt text",
                )

            with gr.Column(scale=1):
                text = gr.Textbox(
                    value=DEMO_TEXT,
                    lines=8,
                    label="Text",
                )
                generate_btn = gr.Button("Generate", variant="primary")

                with gr.Accordion("Advanced", open=False):
                    with gr.Row():
                        num_step = gr.Slider(4, 64, value=16, step=1, label="Steps")
                        guidance_scale = gr.Slider(
                            0.0,
                            5.0,
                            value=1.0,
                            step=0.1,
                            label="Guidance",
                        )
                    with gr.Row():
                        speed = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Speed")
                        t_shift = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="T-shift")
                    with gr.Row():
                        max_duration = gr.Slider(
                            10,
                            200,
                            value=100,
                            step=5,
                            label="Max duration",
                        )
                        seed = gr.Number(value=666, precision=0, label="Seed")
                    normalize_vietnamese = gr.Checkbox(
                        value=True,
                        label="Vietnamese normalization",
                    )
                    split_sentences = gr.Checkbox(value=True, label="Split sentences")
                    remove_long_sil = gr.Checkbox(value=False, label="Remove long silence")
                    with gr.Row():
                        crossfade_ms = gr.Slider(
                            0,
                            300,
                            value=80,
                            step=10,
                            label="Crossfade ms",
                        )
                        silence_ms = gr.Slider(
                            0,
                            800,
                            value=180,
                            step=10,
                            label="Silence ms",
                        )
                    with gr.Row():
                        fade_in_ms = gr.Slider(
                            0,
                            500,
                            value=20,
                            step=10,
                            label="Fade in ms",
                        )
                        fade_out_ms = gr.Slider(
                            0,
                            500,
                            value=80,
                            step=10,
                            label="Fade out ms",
                        )

        with gr.Row():
            output_audio = gr.Audio(type="filepath", label="Output")
            status = gr.Textbox(lines=12, label="Status")

        ref_label.change(
            fn=select_ref,
            inputs=[ref_label],
            outputs=[prompt_audio, prompt_text],
        )
        generate_btn.click(
            fn=generate,
            inputs=[
                ref_label,
                prompt_audio,
                prompt_text,
                text,
                num_step,
                guidance_scale,
                speed,
                t_shift,
                max_duration,
                seed,
                normalize_vietnamese,
                split_sentences,
                remove_long_sil,
                crossfade_ms,
                silence_ms,
                fade_in_ms,
                fade_out_ms,
            ],
            outputs=[output_audio, status],
        )

    return demo


demo = build_app()

if __name__ == "__main__":
    demo.queue(max_size=8).launch()
