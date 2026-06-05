# ViZipVoice

ViZipVoice là bản fine-tune tiếng Việt của [ZipVoice](https://github.com/k2-fsa/ZipVoice), hướng tới zero-shot text-to-speech và voice cloning tiếng Việt. Người dùng chỉ cần một đoạn prompt audio ngắn kèm transcript của prompt, sau đó nhập nội dung tiếng Việt cần đọc.

- GitHub: https://github.com/iamdinhthuan/ViZipvoice
- Hugging Face model: https://huggingface.co/dolly-vn/ViZipvoice
- Base project: https://github.com/k2-fsa/ZipVoice
- Base paper: https://arxiv.org/abs/2506.13053

## Model

Model đã được upload tại [dolly-vn/ViZipvoice](https://huggingface.co/dolly-vn/ViZipvoice). Repo Hugging Face chứa:

- `model.pt`: checkpoint mới nhất `checkpoint-680000.pt`, đã tách phần inference và lưu FP16.
- `model.json`: cấu hình ZipVoice dùng để dựng model.
- `tokens.txt`: tokenizer ký tự tiếng Việt, `244` token.
- `README.md`: model card, trỏ ngược về repo GitHub này.
- `vizipvoice.py`: wrapper mẫu cho người muốn tải file từ Hugging Face.

Model sử dụng sample rate `24 kHz` và vocoder mặc định `charactr/vocos-mel-24khz`.

## Cài đặt

```bash
git clone https://github.com/iamdinhthuan/ViZipvoice.git
cd ViZipvoice

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
export PYTHONPATH="$PWD:$PYTHONPATH"
```

Nếu cài `piper_phonemize` bị lỗi, chạy lại với link wheel trong `requirements.txt`:

```bash
pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html
```

## Chạy nhanh bằng CLI

Chuẩn bị:

- `prompt.wav`: audio giọng mẫu, nên sạch, ít nhiễu, khoảng `3-10` giây.
- `prompt_text`: transcript đúng với nội dung trong `prompt.wav`.
- `text`: câu tiếng Việt muốn sinh.

```bash
python3 -m zipvoice.bin.infer_vizipvoice \
  --prompt-wav prompt.wav \
  --prompt-text "Xin chào, đây là giọng mẫu của tôi." \
  --text "ViZipVoice có thể tổng hợp giọng nói tiếng Việt từ một đoạn mẫu ngắn." \
  --res-wav-path output.wav
```

Mặc định lệnh trên sẽ tải model từ [dolly-vn/ViZipvoice](https://huggingface.co/dolly-vn/ViZipvoice) và cache bằng `huggingface_hub`.

Một số tham số hay dùng:

```bash
python3 -m zipvoice.bin.infer_vizipvoice \
  --prompt-wav prompt.wav \
  --prompt-text "Transcript của prompt." \
  --text "Nội dung cần đọc." \
  --res-wav-path output.wav \
  --num-step 16 \
  --guidance-scale 1.0 \
  --speed 1.0 \
  --seed 666
```

## Dùng bằng Python wrapper

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

Wrapper mặc định:

- tải model từ `dolly-vn/ViZipvoice`;
- dùng `SimpleTokenizer` đúng với `tokens.txt` tiếng Việt;
- tự chọn `cuda`, `mps` hoặc `cpu`;
- bật autocast FP16 khi chạy trên CUDA.

## Chạy với model local

Nếu đã tải model từ Hugging Face về thư mục riêng:

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

## Gợi ý chất lượng

- Prompt audio nên là một người nói, rõ chữ, không nhạc nền, không vang phòng quá mạnh.
- Transcript của prompt phải khớp audio. Sai transcript thường làm giọng clone kém hoặc đọc sai.
- Với câu rất ngắn, có thể giảm `--speed 0.7` hoặc tăng `--num-step 24`.
- Với text dài, nên để dấu câu rõ ràng để model tách đoạn tự nhiên hơn.
- Model mạnh nhất cho tiếng Việt. Ký tự ngoài vocab có thể bị bỏ qua bởi tokenizer ký tự.

## Đạo đức sử dụng

ViZipVoice có khả năng clone giọng từ prompt audio. Chỉ sử dụng giọng nói khi bạn có quyền hoặc được người nói đồng ý. Không dùng model để giả mạo danh tính, lừa đảo, tạo nội dung gây hại, hoặc phát tán thông tin sai lệch.

## License

Code kế thừa ZipVoice và được phát hành theo Apache License 2.0. Vui lòng xem [LICENSE](LICENSE) và ghi nguồn ZipVoice khi sử dụng lại.

## Citation

Nếu dùng nền tảng ZipVoice trong nghiên cứu hoặc sản phẩm, vui lòng cite paper gốc:

```bibtex
@article{zhu2025zipvoice,
  title={ZipVoice: Fast and High-Quality Zero-Shot Text-to-Speech with Flow Matching},
  author={Zhu, Han and others},
  journal={arXiv preprint arXiv:2506.13053},
  year={2025}
}
```
