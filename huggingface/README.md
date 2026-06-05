---
license: apache-2.0
language:
- vi
library_name: pytorch
pipeline_tag: text-to-speech
tags:
- text-to-speech
- zero-shot-tts
- voice-cloning
- vietnamese
- zipvoice
base_model: k2-fsa/ZipVoice
---

# ViZipVoice

ViZipVoice is a Vietnamese fine-tuned ZipVoice model for zero-shot text-to-speech and voice cloning. Provide a short Vietnamese prompt audio plus its transcript, then synthesize new Vietnamese speech in the same voice style.

- GitHub code: https://github.com/iamdinhthuan/ViZipvoice
- Hugging Face model: https://huggingface.co/dolly-vn/ViZipvoice
- Base project: https://github.com/k2-fsa/ZipVoice
- Base paper: https://arxiv.org/abs/2506.13053

## Files

- `model.pt`: latest `checkpoint-680000.pt`, inference-only state dict saved in FP16.
- `model.json`: ZipVoice model config.
- `tokens.txt`: Vietnamese character tokenizer with 244 tokens.
- `vizipvoice.py`: convenience wrapper example mirrored from the GitHub repo.

The model runs at `24 kHz` and uses `charactr/vocos-mel-24khz` as the default vocoder.

## Install

```bash
git clone https://github.com/iamdinhthuan/ViZipvoice.git
cd ViZipvoice

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD:$PYTHONPATH"
```

## CLI Usage

```bash
python3 -m zipvoice.bin.infer_vizipvoice \
  --prompt-wav prompt.wav \
  --prompt-text "Xin chào, đây là giọng mẫu của tôi." \
  --text "ViZipVoice có thể tổng hợp giọng nói tiếng Việt từ một đoạn mẫu ngắn." \
  --res-wav-path output.wav
```

The command downloads model files from this Hugging Face repo by default.

## Python Wrapper

```python
from zipvoice.vizipvoice import ViZipVoiceTTS

tts = ViZipVoiceTTS()
metrics = tts.synthesize(
    prompt_wav="prompt.wav",
    prompt_text="Xin chào, đây là giọng mẫu của tôi.",
    text="Đây là câu tiếng Việt được sinh bởi ViZipVoice.",
    output_path="output.wav",
)
print(metrics)
```

## Local Model Usage

```bash
huggingface-cli download dolly-vn/ViZipvoice \
  --local-dir models/ViZipvoice \
  --local-dir-use-symlinks False

python3 -m zipvoice.bin.infer_vizipvoice \
  --model-dir models/ViZipvoice \
  --prompt-wav prompt.wav \
  --prompt-text "Transcript của prompt." \
  --text "Nội dung cần đọc." \
  --res-wav-path output.wav
```

## Tips

- Use a clean single-speaker prompt, ideally around `3-10` seconds.
- Make sure `prompt_text` exactly matches `prompt.wav`.
- Add punctuation to long text so the model can split and pace the speech naturally.
- For very short text, try `--speed 0.7` or `--num-step 24`.
- The model is intended for Vietnamese. Out-of-vocabulary characters may be skipped by the character tokenizer.

## Responsible Use

This model can clone voices from short audio prompts. Use it only with voices you own or have explicit permission to use. Do not use it for impersonation, fraud, harassment, misinformation, or other harmful content.

## License

The codebase inherits the Apache License 2.0 license from ZipVoice. Please also credit the original ZipVoice project when reusing this model or code.
