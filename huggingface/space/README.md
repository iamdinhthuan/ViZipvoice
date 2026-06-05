---
title: ViZipVoice
emoji: "🎙️"
colorFrom: blue
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
models:
- dolly-vn/ViZipvoice
---

# ViZipVoice Space

Interactive demo for [dolly-vn/ViZipvoice](https://huggingface.co/dolly-vn/ViZipvoice).

The app loads the latest `checkpoint-<step>.pt`, uses the 30 reference audios in the model repo, and runs the same inference flow as the GitHub wrapper: Vietnamese normalization, sentence splitting, short-text rules, and segment postprocessing.
