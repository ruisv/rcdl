---
license: mit
library_name: rcdl
pipeline_tag: voice-activity-detection
tags:
- rknn
- rockchip
- rk3588
- npu
- edge
- vad
- silero
- speech
---

# Silero VAD v6 for RK3588: voice activity detection

A compiled RKNN model (`.rknn`) for the **Rockchip RK3588 / RK3588S** NPU, ready
to load: no ONNX export, no rknn-toolkit2. An fp16 build from rknn-toolkit2
2.3.2, decoded and measured with [**RCDL**](https://github.com/ruisv/rcdl). RCDL
is a C++17 / Python library for vision and speech on these boards; the NPU,
RGA and VPU run as one zero-copy pipeline.

Upstream: [Silero VAD](https://github.com/snakers4/silero-vad) v6.2.3, 16 kHz.

> [!TIP]
> **Redistributable, including commercially.** Silero VAD is MIT on the code
> and on the published weights. See [Licence](#licence).

## Files

| file | what it is | size |
|---|---|---|
| `silero_vad_16k_fp16_rk3588.rknn` | one 32 ms chunk → speech probability, LSTM state carried in and out | 0.6 MB |

The graph starts at the magnitude spectrum: `mag [1,129,4]` plus the LSTM state
`h`, `c [1,128]`, giving `prob [1,1]` and the new state. RCDL computes the STFT
on the CPU: 64 samples of context, 512 new and 64 reflected, cut into four
periodic-Hann frames of 256. It also carries the state from chunk to chunk.

## Measured on RK3588S

| | |
|---|---|
| one 32 ms chunk, STFT + NPU + state | 0.83 ms (real-time factor 0.026) |
| probability vs the float model | mean \|Δ\| 0.0010, max 0.015 |
| speech / non-speech decisions vs the float model | 100% agree |

librknnrt 2.3.2, NPU driver 0.9.8, measured through RCDL's `SileroVad` on a
board under its default governors.

## Use it

On the board (linux-aarch64):

```bash
conda create -n rcdl -c https://mirrors.ruis.ai/conda -c conda-forge rcdl
conda activate rcdl
curl -LO https://huggingface.co/ruisv/rcdl-silero-vad/resolve/main/silero_vad_16k_fp16_rk3588.rknn
```

```python
import wave, numpy as np, rcdl

seg = rcdl.VadSegmenter(rcdl.SileroVad("silero_vad_16k_fp16_rk3588.rknn"))
with wave.open("speech.wav") as w:                 # 16 kHz mono PCM16
    pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
for s in seg.segments(pcm):
    print(f"speech {s.start / 16000:.2f} - {s.end / 16000:.2f} s")
```

`seg.feed(block)` takes audio of any length as it arrives and returns the
utterances that closed during that call, so the same object serves a live
microphone. Each segment's `pcm` is ready for speech recognition
([`ruisv/rcdl-paraformer`](https://huggingface.co/ruisv/rcdl-paraformer)).

## What to know before deploying

**The STFT stays outside the graph on purpose.** In the graph, squaring the real
and imaginary parts puts power spectra at a range of about 5000. A
reduced-precision build loses the quiet bins there, and the probability drifts
mid-sentence. Feeding the magnitude avoids that.

**The segmenter's defaults were chosen on noisy speech.** Threshold 0.5, 250 ms
to open, 300 ms to close, 30 ms padding and 20 s at most per segment. These
were tuned on AISHELL sessions at 10 and 0 dB SNR plus MUSAN noise. They give
no false segments on pure noise, and declare the end of speech a median 0.36 s
after the speaker stops. Change them through `rcdl.VadConfig`.

## Licence

MIT, on the Silero VAD code and on the published weights this build is
converted from.

**RCDL itself is Apache-2.0 and is unrelated to these terms.** It is a
general-purpose runtime that loads any `.rknn`. The licence above applies to
*these weights and this compiled file* only.

How the build is made is in RCDL's
[`model_zoo/speech/`](https://github.com/ruisv/rcdl/tree/main/model_zoo/speech).
The input contract is in
[`docs/MODELS.md`](https://github.com/ruisv/rcdl/blob/main/docs/MODELS.md#speech).
