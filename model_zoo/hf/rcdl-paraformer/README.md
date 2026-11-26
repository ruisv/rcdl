---
license: other
license_name: funasr-model-license
license_link: https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE
library_name: rcdl
pipeline_tag: automatic-speech-recognition
language:
- zh
tags:
- rknn
- rockchip
- rk3588
- npu
- edge
- asr
- paraformer
- funasr
- speech
---

# Paraformer-large (Chinese) for RK3588: speech recognition

Compiled RKNN models (`.rknn`) for the **Rockchip RK3588 / RK3588S** NPU, ready
to load: no FunASR, no ONNX export, no rknn-toolkit2. Three fp16 stages from
rknn-toolkit2 2.3.2, decoded and measured with
[**RCDL**](https://github.com/ruisv/rcdl). RCDL is a C++17 / Python library for
vision and speech on these boards; the NPU, RGA and VPU run as one zero-copy
pipeline.

Upstream: FunASR
[`iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404`](https://modelscope.cn/models/iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404),
Paraformer-large, a non-autoregressive model for Mandarin with a vocabulary of
8404 tokens.

> [!TIP]
> **Redistributable, including commercially, with attribution.** The FunASR
> Model License lets you use, copy, modify and share the weights, provided the
> source, the authors and the model's name are kept. See [Licence](#licence).

## Files

| file | what it is | size |
|---|---|---|
| `paraformer_encoder_fp16_rk3588.rknn` | SANM encoder: `[1,400,560]` LFR features → `[1,400,512]` | 318.4 MB |
| `paraformer_predictor_fp16_rk3588.rknn` | CIF predictor: per-frame weights `[1,401]` + hidden `[1,401,512]` | 2.3 MB |
| `paraformer_decoder_fp16_rk3588.rknn` | decoder: context, token count, bias, acoustic embeddings → logits `[1,100,8404]` | 118.3 MB |
| `paraformer_tokens_8404.json` | the vocabulary | 91 KB |
| `paraformer_am.mvn` | CMVN statistics for the front end | 11 KB |

One call handles one utterance of up to 24 s (400 LFR frames) and up to 100
tokens. The pieces between the stages run on the CPU in RCDL: the front end
(Kaldi fbank, LFR 7/6, CMVN), continuous integrate-and-fire between predictor
and decoder, and the argmax decode.

## Measured on RK3588S

| | |
|---|---|
| one 2.7–3.9 s utterance → text | 0.76 s, 0.50 s of it the encoder |
| the four AISHELL-1 reference clips | identical to FunASR's own transcription, character for character |
| a sentence synthesised by [`rcdl-matcha-tts`](https://huggingface.co/ruisv/rcdl-matcha-tts) | transcribes back to its own text |

librknnrt 2.3.2, NPU driver 0.9.8, measured through RCDL's `ParaformerAsr`.
The time is the fixed 24 s window's: a short command costs what a long sentence
costs.

## Use it

On the board (linux-aarch64):

```bash
conda create -n rcdl -c https://mirrors.ruis.ai/conda -c conda-forge rcdl
conda activate rcdl
hf download ruisv/rcdl-paraformer --local-dir paraformer   # or curl each file from resolve/main/
```

```python
import wave, numpy as np, rcdl

asr = rcdl.ParaformerAsr.from_files(
    "paraformer/paraformer_encoder_fp16_rk3588.rknn",
    "paraformer/paraformer_predictor_fp16_rk3588.rknn",
    "paraformer/paraformer_decoder_fp16_rk3588.rknn",
    "paraformer/paraformer_tokens_8404.json", "paraformer/paraformer_am.mvn")
with wave.open("utterance.wav") as w:              # 16 kHz mono PCM16
    pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
r = asr.transcribe(pcm)
print(r.text, "(truncated)" if r.truncated else "")
```

Longer recordings go through a VAD first
([`ruisv/rcdl-silero-vad`](https://huggingface.co/ruisv/rcdl-silero-vad)), one
utterance per call.

## What to know before deploying

**The encoder carries its residual stream at 1/32, and it has to.** Paraformer's
49 SANM blocks add into one residual stream that passes 500 000 in the last
third of the encoder. fp16 stops at 65 504, so a plain fp16 build returns NaN.
CIF then counts zero tokens and every transcript is empty, with no error. Every
block reads the stream through a LayerNorm, which does not care about scale, so
this build scales the stream and every branch added into it by 1/32. The float
output is unchanged bit for bit.

**CIF runs on the CPU so the tail lands at the true end of speech.** The
predictor stops at the per-frame weights. The tail weight (0.45) that lets the
last token fire is added at the last real frame, not at the end of the padding.

**There is no resampler.** The input must already be 16 kHz mono. Audio longer
than 24 s is cut and `truncated` is set on the result.

## Licence

**FunASR Model License 1.1.** You may use, copy, modify and share these weights.
You must attribute the source and authors and keep the model's name: this is
Alibaba's FunASR Paraformer-large, converted for RK3588 by RCDL. The ModelScope
card declares Apache-2.0 for the model. The training data, in-house Mandarin
audio, is not disclosed.

**RCDL itself is Apache-2.0 and is unrelated to these terms.** It is a
general-purpose runtime that loads any `.rknn`. The licence above applies to
*these weights and these compiled files* only.

The export and the conversion are in RCDL's
[`model_zoo/speech/`](https://github.com/ruisv/rcdl/tree/main/model_zoo/speech).
That covers the three-stage split, the residual rescale, and the check against
FunASR.
