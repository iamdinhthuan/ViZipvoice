import shutil
import subprocess
from pathlib import Path


def content_type_for_format(output_format: str) -> str:
    if output_format == "mp3":
        return "audio/mpeg"
    if output_format == "wav":
        return "audio/wav"
    raise ValueError(f"unsupported output format: {output_format}")


def encode_audio_output(wav_path: Path, output_path: Path, output_format: str) -> None:
    if output_format == "wav":
        if wav_path != output_path:
            shutil.copyfile(wav_path, output_path)
        return
    if output_format != "mp3":
        raise ValueError(f"unsupported output format: {output_format}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(wav_path),
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "128k",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to encode mp3 output") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to encode mp3 output: {stderr}") from exc
