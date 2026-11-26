# Speech models: how the RK3588 builds are made

RCDL's voice pipeline runs eight fixed-shape graphs on the NPU: wake word, voice
activity, speech recognition (three stages), and speech synthesis (Matcha
encoder, Matcha decoder, Vocos). This directory turns each one into the fp16
`.rknn` that `scripts/fetch_models.sh` registers. What each model expects at run
time is in [`docs/MODELS.md`](../../docs/MODELS.md#speech).

Everything here runs on an x86 host with rknn-toolkit2 2.3.2. The Paraformer
export also needs FunASR (1.3.x) and PyTorch.

**Published:** the VAD, Paraformer, Matcha and Vocos builds are on Hugging Face
(`ruisv/rcdl-silero-vad`, `rcdl-paraformer`, `rcdl-matcha-tts`, `rcdl-vocos`), and
`scripts/fetch_models.sh` downloads them. **The wake-word build is not
published**, because its licence is unresolved. Build it here. Each model's
terms are in [`docs/MODELS.md`](../../docs/MODELS.md#speech).

## The recipe, model by model

| `.rknn` | ONNX it is built from | RK3588 rewrite | NPU vs float ONNX |
|---|---|---|---|
| `silero_vad_16k_fp16_rk3588` | `silero_vad_16k_mag_slim.onnx` | — | prob cosine ≥ 0.99997; every decision agrees |
| `kws_zipformer_zh_en_3m_c16_fp16_rk3588` | `kws_zipformer_zh_en_3m_c16_bpu.onnx` | `max_rewrite.py` | streaming per-frame cosine min 0.997, mean 0.9995 |
| `matcha_zh_en_enc_l128_fp16_rk3588` | `matcha_enc_L128_slim.onnx` | — | cosine ≥ 0.999999 |
| `matcha_zh_en_dec_t512_fp16_rk3588` | `matcha_dec_T512_slim.onnx` | `sin2_rewrite.py` | mel cosine ≥ 0.9997 |
| `vocos_16k_t256_fp16_rk3588` | `vocos16k_t256_head.onnx` | — | log-magnitude cosine ≥ 0.99999 |
| `paraformer_{encoder,predictor,decoder}_fp16_rk3588` | `export_paraformer.py` | `rescale_residual.py` (encoder) | transcripts equal FunASR's |

The first five ONNX files come from the public conversion recipes in
[bcdl-model-zoo](https://github.com/ruisv/bcdl-model-zoo):
`models/silero_vad`, `models/kws_zipformer`, `models/matcha_tts` and
`models/vocos`. Take the float ONNX each recipe writes (then `onnxslim` where
the recipe says so) and stop there; the rest of those recipes compiles for a
different NPU. Their graph surgery is not specific to that NPU and is needed
here too:
- the STFT and the vocoder's cos/sin are taken out of the graph;
- Matcha's padding mask and token lookup are fixed;
- the wake-word encoder gets a float `pad_mask` and a softplus rewrite.

`matcha_zh_en_embedding.npy` comes from the same Matcha recipe. The wake word's
`kws_zipformer_zh_en_3m_decoder_joiner.npy` and `_tokens.txt` come from its
recipe. The Matcha `tokens.txt` / `lexicon.txt` and Paraformer's `tokens.json`
/ `am.mvn` are the upstream files, renamed as `fetch_models.sh` lists them.

Then, for every graph:

```bash
python max_rewrite.py   kws_zipformer_zh_en_3m_c16_bpu.onnx kws_relu.onnx
python sin2_rewrite.py  matcha_dec_T512_slim.onnx           matcha_dec_sin2.onnx
python conv_fp16.py     <graph>.onnx <name>_fp16_rk3588.rknn
```

Paraformer (FunASR `iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404`):

```bash
python export_paraformer.py <model dir>              # three fixed-shape ONNX stages
python check_paraformer.py  <model dir> <audio dir> ref.npz   # split == FunASR, on the clips
python rescale_residual.py  paraformer_encoder.onnx paraformer_encoder_s32.onnx
python find_overflow.py     paraformer_encoder_s32.onnx inputs.npz   # must report 0 nodes
python conv_fp16.py         paraformer_encoder_s32.onnx paraformer_encoder_fp16_rk3588.rknn
python conv_fp16.py         paraformer_predictor.onnx   paraformer_predictor_fp16_rk3588.rknn
python conv_fp16.py         paraformer_decoder.onnx     paraformer_decoder_fp16_rk3588.rknn
```

`<audio dir>` holds 16 kHz WAVs and a `transcripts.tsv`, such as RCDL's own
`data/audio`. The board tests (`tests/test_*_board_py.py`) are the acceptance
check for every build: each compares the NPU against float references
committed under `data/`, and the ASR and TTS tests close the loop by
transcribing.

## What goes wrong, and how it shows

Each problem below compiles without a warning. Where a build is wrong, it is
wrong quietly.

- **A CPU fallback hides in the profile, not the output.** RK3588's NPU has no
  `Sin`, so Matcha's 18 `sin²(αx)` activations each round-trip to the CPU. That
  was 515 ms of a 955 ms decoder, with correct output. `sin2_rewrite.py` folds
  the input onto [0, π/2] with nested `Abs` and evaluates a polynomial there,
  which brings the decoder to 253 ms. The wake-word encoder's four
  `Max(constant, x)` are the same story at 9 ms per chunk; `max_rewrite.py`
  makes them `Relu`. To find these, open the Engine with
  `init_flags=RKNN_FLAG_COLLECT_PERF_MASK` and read `perf_detail()`: the
  `Target` column says `CPU`.
- **An fp16 overflow shows up as an empty transcript.** Paraformer's residual
  stream passes 500 000, so the encoder returns NaN and CIF counts zero tokens,
  with no error. `find_overflow.py` lists every node of the float graph that
  passes 65 504 on a real input. `rescale_residual.py` carries the stream at
  1/32, which LayerNorm makes exact.
- **Exporting the stock FunASR graph gives the wrong interface.** Its CIF loop
  and its `arange(max(lengths))` masks are dynamic. `export_paraformer.py`
  stops the predictor at the per-frame weights; CIF runs on the CPU, so the
  tail weight lands at the true end of speech. It also writes the decoder's
  token mask as `clip(count - i, 0, 1)`.
- **The 4-D float inputs are NHWC.** A hand-written check that feeds the
  wake-word caches in ONNX layout gets garbage from the NPU. RCDL's task
  classes transpose; scripts driving the Engine directly must too.
