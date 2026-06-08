from pathlib import Path
import re


def minimum_version_for(requirements_path: Path, package_name: str) -> tuple[int, ...] | None:
    pattern = re.compile(rf"^\s*{re.escape(package_name)}\s*>=\s*([0-9]+(?:\.[0-9]+)*)\s*$")
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            return tuple(int(part) for part in match.group(1).split("."))
    return None


def has_requirement(requirements_path: Path, package_name: str) -> bool:
    pattern = re.compile(rf"^\s*{re.escape(package_name)}(?:\s|[<>=!~].*)?$")
    return any(
        pattern.match(line)
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
    )


def test_worker_runtime_pins_typing_extensions_for_sentinel_import():
    repo_root = Path(__file__).resolve().parents[2]
    version = minimum_version_for(
        repo_root / "requirements-tts-runtime.txt",
        "typing_extensions",
    )

    assert version is not None, (
        "requirements-tts-runtime.txt must explicitly pin typing_extensions after "
        "the TTS packages so Modal worker cold starts keep typing_extensions.Sentinel"
    )
    assert version >= (4, 15, 0)


def test_worker_runtime_includes_import_time_zipvoice_dependencies():
    repo_root = Path(__file__).resolve().parents[2]
    runtime_requirements = repo_root / "requirements-tts-runtime.txt"

    assert has_requirement(runtime_requirements, "tensorboard")


def test_worker_runtime_includes_audio_decode_dependencies():
    repo_root = Path(__file__).resolve().parents[2]
    runtime_requirements = repo_root / "requirements-tts-runtime.txt"

    assert has_requirement(runtime_requirements, "torchcodec")
