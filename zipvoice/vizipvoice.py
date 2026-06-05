import json
import logging
from pathlib import Path
from typing import Optional, Union

import safetensors.torch
import torch
from huggingface_hub import hf_hub_download
from lhotse.utils import fix_random_seed

from zipvoice.bin.infer_zipvoice import generate_sentence, get_vocoder
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.tokenizer.tokenizer import SimpleTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank

DEFAULT_REPO_ID = "dolly-vn/ViZipvoice"


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
    checkpoint_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=checkpoint_name,
            revision=revision,
        )
    )
    model_config_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename="model.json",
            revision=revision,
        )
    )
    token_file = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename="tokens.txt",
            revision=revision,
        )
    )
    return checkpoint_path, model_config_path, token_file


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
        checkpoint_name: str = "model.pt",
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
            checkpoint_path = model_dir / checkpoint_name
            model_config_path = model_dir / "model.json"
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
    ) -> dict:
        if seed is not None and seed >= 0:
            fix_random_seed(int(seed))

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=self.use_fp16,
        ):
            return generate_sentence(
                save_path=str(output_path),
                prompt_text=prompt_text,
                prompt_wav=str(prompt_wav),
                text=text,
                model=self.model,
                vocoder=self.vocoder,
                tokenizer=self.tokenizer,
                feature_extractor=self.feature_extractor,
                device=self.device,
                num_step=int(num_step),
                guidance_scale=float(guidance_scale),
                speed=float(speed),
                t_shift=float(t_shift),
                target_rms=float(target_rms),
                feat_scale=float(feat_scale),
                sampling_rate=self.sampling_rate,
                max_duration=float(max_duration),
                remove_long_sil=bool(remove_long_sil),
            )
