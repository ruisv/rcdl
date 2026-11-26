---
license: apache-2.0
library_name: rcdl
pipeline_tag: text-to-speech
tags:
- rknn
- rockchip
- rk3588
- npu
- edge
- vocoder
- vocos
- speech
---

# Vocos 16 kHz for RK3588: mel spectrogram to waveform

A compiled RKNN model (`.rknn`) for the **Rockchip RK3588 / RK3588S** NPU, ready
to load: no ONNX export, no rknn-toolkit2. An fp16 build from rknn-toolkit2
2.3.2, decoded and measured with [**RCDL**](https://github.com/ruisv/rcdl). RCDL
is a C++17 / Python library for vision and speech on these boards; the NPU,
RGA and VPU run as one zero-copy pipeline. It is the vocoder of
[Matcha-TTS zh-en](https://huggingface.co/ruisv/rcdl-matcha-tts).

Upstream: `vocos-16khz-univ`, published with Matcha-TTS zh-en by the same author
([ModelScope](https://modelscope.cn/models/dengcunqin/matcha_tts_zh_en_20251010)),
on the [Vocos](https://github.com/gemelo-ai/vocos) architecture.

> [!TIP]
> **Redistributable, including commercially.** The model card declares
> Apache-2.0 and the Vocos code is MIT. The training data is not disclosed,
> which is a gap rather than a known restriction. See [Licence](#licence).

## Files

| file | what it is | size |
|---|---|---|
| `vocos_16k_t256_fp16_rk3588.rknn` | `mels [1,80,256]` → log-magnitude and raw phase, each `[1,513,256]` | 26.6 MB |

The graph stops at the head. RCDL runs the rest on the CPU: exp and clip,
cos/sin of the phase, and a centred inverse STFT (n_fft 1024, hop 256). It also
cuts a mel of any length into 256-frame windows that overlap by 32 frames on
each side. The receptive field is 27 frames, so the seams are exact.

## Measured on RK3588S

| | |
|---|---|
| 3 s of mel → waveform | 53 ms, 26 ms of it NPU |
| spectral distance to the float model, audible cells | 0.17–0.32 dB |

librknnrt 2.3.2, NPU driver 0.9.8, measured through RCDL's `VocosVocoder`.

## Use it

On the board (linux-aarch64):

```bash
conda create -n rcdl -c https://mirrors.ruis.ai/conda -c conda-forge rcdl
conda activate rcdl
curl -LO https://huggingface.co/ruisv/rcdl-vocos/resolve/main/vocos_16k_t256_fp16_rk3588.rknn
```

```python
import numpy as np, rcdl

voc = rcdl.VocosVocoder("vocos_16k_t256_fp16_rk3588.rknn")
mel = np.load("mel.npy")                 # (80, frames) natural-log mel, 16 kHz, hop 256
pcm = voc.synthesize(mel)                # float32 PCM, 256 samples per frame
```

For speech from text, use `rcdl.TextToSpeech` with
[`ruisv/rcdl-matcha-tts`](https://huggingface.co/ruisv/rcdl-matcha-tts), which
drives this vocoder itself.

## What to know before deploying

**cos/sin stay outside the graph.** Raw phase spans about ±42 rad, which a
reduced-precision graph cannot hold well enough for cos/sin to come out right.
The head's outputs are carried out as they are, and the trigonometry runs in
float on the CPU.

## Licence

The model card declares **Apache-2.0**, and the Vocos code is **MIT**. The
training data is not disclosed.

**RCDL itself is Apache-2.0 and is unrelated to these terms.** It is a
general-purpose runtime that loads any `.rknn`. The licence above applies to
*these weights and this compiled file* only.

How the build is made is in RCDL's
[`model_zoo/speech/`](https://github.com/ruisv/rcdl/tree/main/model_zoo/speech).
