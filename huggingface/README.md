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

Vietnamese zero-shot TTS / voice cloning fine-tuned from [ZipVoice](https://github.com/k2-fsa/ZipVoice).

- GitHub: https://github.com/iamdinhthuan/ViZipvoice
- Model repo: https://huggingface.co/contextboxai/ViZipvoice
- Space: https://huggingface.co/spaces/dinhthuan/ViZipvoice
- Latest checkpoint: `checkpoint-1860000.pt`, FP16 inference state dict
- Training data: about `7000` total hours, including roughly `6500` hours of Vietnamese and `500` hours of English
- Tokenizer: `SimpleTokenizer`, character-level, `244` tokens
- Sample rate: `24 kHz`
- Default vocoder: `charactr/vocos-mel-24khz`

The wrapper loads the largest `checkpoint-<step>.pt` automatically and uses `soe-vinorm` for Vietnamese text normalization.

## Audio Demo

Generated with `checkpoint-1860000.pt`, the current wrapper flow, and the demo text in `demo/demo_text.txt`.

**Đinh-Quyết**

<audio controls src="https://huggingface.co/contextboxai/ViZipvoice/resolve/main/demo/demo_01_%C4%90inh-Quy%E1%BA%BFt.wav"></audio>

[Open audio](https://huggingface.co/contextboxai/ViZipvoice/resolve/main/demo/demo_01_%C4%90inh-Quy%E1%BA%BFt.wav)

**Nhã-Uyên**

<audio controls src="https://huggingface.co/contextboxai/ViZipvoice/resolve/main/demo/demo_02_Nh%C3%A3-Uy%C3%AAn.wav"></audio>

[Open audio](https://huggingface.co/contextboxai/ViZipvoice/resolve/main/demo/demo_02_Nh%C3%A3-Uy%C3%AAn.wav)

**MC**

<audio controls src="https://huggingface.co/contextboxai/ViZipvoice/resolve/main/demo/demo_03_MC.wav"></audio>

[Open audio](https://huggingface.co/contextboxai/ViZipvoice/resolve/main/demo/demo_03_MC.wav)

## Install

```bash
git clone https://github.com/iamdinhthuan/ViZipvoice.git
cd ViZipvoice
pip install -r requirements.txt
export PYTHONPATH="$PWD:$PYTHONPATH"
```

## CLI

```bash
python3 -m zipvoice.bin.infer_vizipvoice \
  --prompt-wav prompt.wav \
  --prompt-text "Xin chào, đây là giọng mẫu của tôi." \
  --text "ViZipVoice có thể tổng hợp giọng nói tiếng Việt từ một đoạn mẫu ngắn." \
  --res-wav-path output.wav
```

The CLI downloads this model repo by default. Use `--model-dir models/ViZipvoice` after downloading files locally.

## Python

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

## Reference Audio

`audio/` contains 30 reference prompts. Each audio file has a sidecar `.txt` transcript with the same basename:

```text
audio/Đinh-Quyết.mp3
audio/Đinh-Quyết.txt
```

Names only keep the audio/person name; the original `lar_*` prefix and `Pro` suffix are removed. The Gradio app reads this sidecar format automatically.

```bash
huggingface-cli download contextboxai/ViZipvoice \
  --local-dir models/ViZipvoice \
  --local-dir-use-symlinks False

python3 egs/zipvoice/gradio_app.py --exp-dir models/ViZipvoice
```

## Inference Flow

The CLI, Python wrapper, and Gradio app use the same default flow:

- normalize Vietnamese text with `soe-vinorm`, then clean spaces around punctuation;
- split long text into sentences;
- for a `1`-word sentence: use at least `24` steps and `speed=0.6`;
- for a `2-4` word sentence: use `speed=0.8`;
- generate each segment separately;
- merge segments with silence, crossfade, fade in, and fade out.

Useful knobs:

```bash
--no-vietnamese-normalize
--no-split-sentences
--crossfade-ms 80
--silence-ms 180
--fade-in-ms 20
--fade-out-ms 80
```

## Files

- `checkpoint-1860000.pt`: latest FP16 checkpoint
- `config.json`, `model.json`: model config
- `tokens.txt`: Vietnamese character tokenizer
- `audio/`: 30 reference audios plus `.txt` transcripts
- `demo/`: regenerated audio demos and `metadata.json`
- `vizipvoice.py`: wrapper mirrored from GitHub

## Responsible Use

This model can clone voices from short audio prompts. Use only voices you own or have explicit permission to use. Do not use it for impersonation, fraud, harassment, misinformation, or other harmful content.

## License

Apache License 2.0. Please also credit the original ZipVoice project.
