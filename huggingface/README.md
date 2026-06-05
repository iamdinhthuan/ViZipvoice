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

- `checkpoint-700000.pt`: latest checkpoint by training step, inference-only state dict saved in FP16.
- `config.json`: model config and Hugging Face download-stats query file.
- `model.json`: ZipVoice model config.
- `tokens.txt`: Vietnamese character tokenizer with 244 tokens.
- `audio/`: 30 reference audio files. Each audio file has a sidecar `.txt` transcript with the same basename.
- `demo/`: generated demo outputs using the latest checkpoint and selected reference audio files.
- `vizipvoice.py`: convenience wrapper example mirrored from the GitHub repo.

The model runs at `24 kHz` and uses `charactr/vocos-mel-24khz` as the default vocoder.
The wrapper automatically selects the largest `checkpoint-<step>.pt` file. The current latest checkpoint is `checkpoint-700000.pt`.

## Training Data & Tokenizer

- Training dataset: approximately `7000` hours of Vietnamese speech.
- Tokenizer: `SimpleTokenizer`, character-level Unicode tokenization.
- Vocabulary: `tokens.txt` contains `244` tokens, including Vietnamese diacritics, letters, digits, punctuation, and `_` padding.
- Text handling: this Vietnamese model does not use phoneme/G2P tokenization; prompt transcript and synthesis text are mapped by character. Out-of-vocabulary characters may be skipped.
- Vietnamese normalization: the wrapper uses `soe-vinorm` by default to expand numbers, dates, units, and abbreviations into TTS-friendly Vietnamese text. It then removes extra spaces around punctuation, e.g. `xin chào , bạn ?` becomes `xin chào, bạn?`.

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
Pass `--no-vietnamese-normalize` to disable `soe-vinorm` and use raw text.
The CLI and Python wrapper use the same inference flow as the Gradio app by default:

- normalize Vietnamese text with `soe-vinorm`;
- split long text into sentences;
- for a `1`-word sentence, use at least `24` steps and `speed=0.6`;
- for a `2-4` word sentence, use `speed=0.8`;
- generate each segment separately, then merge with silence, crossfade, fade in, and fade out.

Postprocessing can be adjusted from the CLI:

```bash
python3 -m zipvoice.bin.infer_vizipvoice \
  --prompt-wav prompt.wav \
  --prompt-text "Transcript của prompt." \
  --text "Nội dung cần đọc." \
  --res-wav-path output.wav \
  --crossfade-ms 80 \
  --silence-ms 180 \
  --fade-in-ms 20 \
  --fade-out-ms 80
```

## Reference Audio

The `audio/` folder contains 30 reference prompts. File names are cleaned from the original source names and do not keep the `Pro` suffix. The prompt transcript is stored next to each audio file:

```text
audio/Đinh-Quyết.mp3
audio/Đinh-Quyết.txt
```

File names only keep the audio/person name and do not keep the original `lar_*` prefix or `Pro` suffix. The Gradio app loads this sidecar format by default when the ref audio directory contains matching `.txt` files.

To run the GitHub Gradio app with this model repo downloaded locally:

```bash
huggingface-cli download dolly-vn/ViZipvoice \
  --local-dir models/ViZipvoice \
  --local-dir-use-symlinks False

python3 egs/zipvoice/gradio_app.py --exp-dir models/ViZipvoice
```

When `--ref-audio-dir` is not provided, the app first looks for `audio/` inside `--exp-dir`.

## Demo Outputs

The `demo/` folder contains generated samples for this text:

```text
Chiến tranh luôn là một chủ đề nặng nề, nhưng cũng rất cần được nhắc đến để con người hiểu rõ hơn giá trị của hòa bình. Khi một cuộc chiến xảy ra, những gì bị phá hủy không chỉ là nhà cửa, đường sá, trường học hay bệnh viện. Điều đau lòng nhất chính là sinh mạng con người, là những gia đình bị chia cắt, là những đứa trẻ phải lớn lên trong sợ hãi, và là những vùng đất từng yên bình bỗng trở nên hoang tàn.
```

Demo files:

- `demo/demo_01_Đinh-Quyết.wav`
- `demo/demo_02_Nhã-Uyên.wav`
- `demo/demo_03_MC.wav`

### Audio Demo

If the embedded player does not render in your browser, use the direct links below each player.

**Đinh-Quyết**

<audio controls src="https://huggingface.co/dolly-vn/ViZipvoice/resolve/main/demo/demo_01_%C4%90inh-Quy%E1%BA%BFt.wav"></audio>

[Open audio](https://huggingface.co/dolly-vn/ViZipvoice/resolve/main/demo/demo_01_%C4%90inh-Quy%E1%BA%BFt.wav)

**Nhã-Uyên**

<audio controls src="https://huggingface.co/dolly-vn/ViZipvoice/resolve/main/demo/demo_02_Nh%C3%A3-Uy%C3%AAn.wav"></audio>

[Open audio](https://huggingface.co/dolly-vn/ViZipvoice/resolve/main/demo/demo_02_Nh%C3%A3-Uy%C3%AAn.wav)

**MC**

<audio controls src="https://huggingface.co/dolly-vn/ViZipvoice/resolve/main/demo/demo_03_MC.wav"></audio>

[Open audio](https://huggingface.co/dolly-vn/ViZipvoice/resolve/main/demo/demo_03_MC.wav)

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

By default, `synthesize()` splits text into sentences, applies the short-text speed/step rules from the Gradio app, and postprocesses generated segments with silence, crossfade, fade in, and fade out.

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
