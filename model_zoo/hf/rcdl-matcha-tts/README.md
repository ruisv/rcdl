---
license: apache-2.0
library_name: rcdl
base_model:
- csukuangfj/matcha-icefall-zh-en
pipeline_tag: text-to-speech
language:
- zh
- en
tags:
- rknn
- rockchip
- rk3588
- npu
- edge
- tts
- matcha-tts
- speech
---

# Matcha-TTS zh-en for RK3588: speech synthesis

Compiled RKNN models (`.rknn`) for the **Rockchip RK3588 / RK3588S** NPU, ready
to load: no PyTorch, no ONNX export, no rknn-toolkit2. Two fp16 graphs from
rknn-toolkit2 2.3.2, decoded and measured with
[**RCDL**](https://github.com/ruisv/rcdl). RCDL is a C++17 / Python library for
vision and speech on these boards; the NPU, RGA and VPU run as one zero-copy
pipeline. Together with the
[Vocos vocoder](https://huggingface.co/ruisv/rcdl-vocos), the text goes to
16 kHz audio entirely on the board.

Upstream: the Matcha-TTS zh-en model
([ModelScope](https://modelscope.cn/models/dengcunqin/matcha_tts_zh_en_20251010),
packaged for sherpa-onnx as `matcha-icefall-zh-en`), on the
[Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) architecture.

> [!WARNING]
> **Redistributable per the model card, with one thing to check before a
> commercial build: the training data.** The card declares Apache-2.0 but does
> not say what the model was trained on. The author's earlier zh-en model used a
> recipe built on the Baker corpus (BZNSYP, non-commercial), and whether that
> data is in this one is not stated. See [Licence](#licence).

## Files

| file | what it is | size |
|---|---|---|
| `matcha_zh_en_enc_l128_fp16_rk3588.rknn` | text encoder + duration predictor, up to 128 tokens | 16.7 MB |
| `matcha_zh_en_dec_t512_fp16_rk3588.rknn` | decoder, three flow-matching steps compiled in, up to 512 frames (8.2 s) | 42.5 MB |
| `matcha_zh_en_embedding.npy` | the token embedding table, looked up on the CPU | 1.6 MB |
| `matcha_zh_en_tokens.txt` | token table of the text front end | 21 KB |
| `matcha_zh_en_lexicon.txt` | Chinese lexicon of the text front end | 1.3 MB |

RCDL's front end reproduces sherpa-onnx's for this model, checked id for id on
100 sentences. It normalises punctuation, reads digits out, splits sentences,
and does a longest-match lexicon lookup. Durations, length regulation and the
decoder's starting noise run on the CPU between the two graphs.

## Measured on RK3588S

| | |
|---|---|
| one sentence → 3.3 s of audio, with Vocos | 0.90 s (real-time factor 0.27) |
| the 512-frame decoder, one call | 0.23 s of NPU |
| per-token mel vs the float model | mean abs error 0.05–0.10 log-mel, no duration off by a frame |
| synthesised speech through [`rcdl-paraformer`](https://huggingface.co/ruisv/rcdl-paraformer) | transcribes back to its own text |

librknnrt 2.3.2, NPU driver 0.9.8, measured through RCDL's `TextToSpeech`.

## Use it

On the board (linux-aarch64):

```bash
conda create -n rcdl -c https://mirrors.ruis.ai/conda -c conda-forge rcdl
conda activate rcdl
hf download ruisv/rcdl-matcha-tts --local-dir tts
hf download ruisv/rcdl-vocos --local-dir tts
```

```python
import wave, numpy as np, rcdl

tts = rcdl.TextToSpeech.from_files(
    "tts/matcha_zh_en_enc_l128_fp16_rk3588.rknn", "tts/matcha_zh_en_dec_t512_fp16_rk3588.rknn",
    "tts/matcha_zh_en_embedding.npy", "tts/vocos_16k_t256_fp16_rk3588.rknn",
    "tts/matcha_zh_en_tokens.txt", "tts/matcha_zh_en_lexicon.txt")
pcm = np.asarray(tts.synthesize("今天天气很好，我们一起去公园散步吧。").pcm)
with wave.open("hello.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(tts.sample_rate)
    w.writeframes((np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes())
```

## What to know before deploying

**The decoder has no `Sin` in it, though Matcha does.** The decoder's SnakeBeta
activation is `sin²(αx)`, and RK3588's NPU has no `Sin`. A plain build runs it
on the CPU 18 times per call, which is half of a 955 ms decoder. This build folds
the input onto [0, π/2] with nested `Abs`, which is exact, and evaluates a
polynomial there. The decoder takes 253 ms and its mel is unchanged.

**The decoder's padding mask is fixed.** The published export lets the padding
leak into the speech once the graph has a fixed 512-frame shape, by up to ~10
log-mel. This graph masks its GroupNorm and its attention past the valid frames.

**The token lookup stays on the CPU.** Token ids run to 2190, which fp16 cannot
represent exactly above 2048. That is why the embedding table ships as a `.npy`.

**Latin words are dropped by the front end**, as sherpa-onnx's front end for this
model does. Spell English words in Chinese characters if they must be spoken.

## Licence

The model card declares **Apache-2.0**, and the Matcha-TTS code is **MIT**. The
training data is not disclosed; check it before a commercial build, as the
banner above says. `matcha_zh_en_tokens.txt` and `matcha_zh_en_lexicon.txt` are
the sherpa-onnx packaging of the same model.

**RCDL itself is Apache-2.0 and is unrelated to these terms.** It is a
general-purpose runtime that loads any `.rknn`. The licence above applies to
*these weights and these compiled files* only.

How the build is made, including the `sin²` rewrite, is in RCDL's
[`model_zoo/speech/`](https://github.com/ruisv/rcdl/tree/main/model_zoo/speech).
