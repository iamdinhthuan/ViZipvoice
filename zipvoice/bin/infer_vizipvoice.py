#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

from zipvoice.vizipvoice import DEFAULT_REPO_ID, ViZipVoiceTTS


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Vietnamese ZipVoice inference wrapper.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--checkpoint-name", default="model.pt")
    parser.add_argument("--vocoder-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fp16", type=int, default=1)
    parser.add_argument("--num-thread", type=int, default=1)

    parser.add_argument("--prompt-wav", required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--res-wav-path", default="output.wav")

    parser.add_argument("--num-step", type=int, default=16)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--t-shift", type=float, default=0.5)
    parser.add_argument("--target-rms", type=float, default=0.1)
    parser.add_argument("--feat-scale", type=float, default=0.1)
    parser.add_argument("--max-duration", type=float, default=100)
    parser.add_argument("--remove-long-sil", action="store_true")
    parser.add_argument("--seed", type=int, default=666)
    return parser


def main() -> None:
    args = get_parser().parse_args()

    tts = ViZipVoiceTTS(
        repo_id=args.repo_id,
        revision=args.revision,
        model_dir=args.model_dir,
        checkpoint_name=args.checkpoint_name,
        vocoder_path=args.vocoder_path,
        device=args.device,
        use_fp16=bool(args.fp16),
        num_threads=args.num_thread,
    )
    metrics = tts.synthesize(
        prompt_wav=args.prompt_wav,
        prompt_text=args.prompt_text,
        text=args.text,
        output_path=args.res_wav_path,
        num_step=args.num_step,
        guidance_scale=args.guidance_scale,
        speed=args.speed,
        t_shift=args.t_shift,
        target_rms=args.target_rms,
        feat_scale=args.feat_scale,
        max_duration=args.max_duration,
        remove_long_sil=args.remove_long_sil,
        seed=args.seed,
    )

    logging.info("Saved to: %s", Path(args.res_wav_path))
    logging.info(
        "RTF: %.4f | wav seconds: %.2f",
        metrics["rtf"],
        metrics["wav_seconds"],
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        force=True,
    )
    main()
