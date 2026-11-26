# C++ API

```cmake
find_package(rcdl REQUIRED)
target_link_libraries(app PRIVATE rcdl::rcdl)
```

```c++
#include "rcdl/rcdl.h"      // or the individual layer headers
```

Every header carries the reasoning in its comments; this page is the map and the
handful of rules that are easy to get wrong.

---

## The one idea

**A dma-buf fd is the only buffer.** The NPU imports it
(`rknn_create_mem_from_fd`), RGA imports it (`importbuffer_fd`), the VPU decodes
into it and encodes out of it (external `MppBuffer`). So a frame moves from the
decoder to the 2-D engine to the model without ever being copied, and the CPU
only sees it if you ask.

That is why the central type is a *descriptor*, not an owner:

```c++
struct ImageView {           // preproc/image.h
  void* data;                //   CPU mapping, or null — fd-only is normal
  int fd;                    //   dma-buf fd, or -1 for a host buffer
  int width, height;
  int wstride, hstride;      //   PIXELS and ROWS, not bytes; often > width/height
  PixelFormat format;
  std::size_t size;
};
```

`rcdl::Image` owns one (`Image::alloc`), `VideoFrame::view()` describes a
decoded frame, `engineInputView(engine, 0, fmt)` describes the NPU's own input
tensor, and `hostView()` wraps a `cv::Mat`. Anything that takes an `ImageView`
takes all four.

**Always read strides through `effWStride()` / `effHStride()` / `rowBytes()` /
`uvOffset()`.** An NPU input tensor's row stride is aligned up, a VPU frame's is
aligned up, and `strideAlign()` aligns what RCDL allocates — assuming
`stride == width` is the single most common way to get a sheared image.

---

## Layers

| Header | What is in it |
|---|---|
| `core/status.h` | `rcdl::Error`, `RCDL_CHECK`, `RCDL_REQUIRE` |
| `core/dma_buf.h` | `DmaBuf` (dma-heap alloc, `syncStart`/`syncEnd`), `dmaBufSyncStart/End(fd, ...)` |
| `backend/engine.h` | `Engine`, `EngineOptions`, `NpuCore` |
| `backend/engine_pool.h` | `EnginePool`, `EnginePool::Lease` |
| `backend/output_reader.h` | `outputAsFloat`, `dequantizeToFloat`, `sigmoid`, `quantParams` |
| `preproc/image.h` | `PixelFormat`, `ImageView`, `Image`, `engineInputView`, `strideAlign` |
| `preproc/geometry.h` | `LetterboxInfo`, `computeLetterbox`, `YuvRange`, `YuvMatrix`, `YuvColorSpace` |
| `preproc/letterbox.h` | `letterbox` / `resize` / `cvtColor` + `PreprocBackend` |
| `preproc/rga.h`, `preproc/letterbox_cpu.h` | the two backends, if you want one specifically; `rgaDrawRects()` is the box overlay (CPU by default, RGA2 on a dma32 frame) |
| `preproc/fbank.h`, `preproc/istft.h` | speech front ends: Kaldi `Fbank` + `lfrStack` + `Cmvn`, and `Istft` |
| `media/video_codec.h`, `media/video_frame.h`, `media/jpeg_codec.h` | VPU codecs |
| `tasks/*.h` | detection, classification, pose, instance seg, semantic seg, OBB, depth, OCR, embedding; speech: `vad.h`, `kws.h`, `asr.h`, `command.h`, `vocoder.h`, `tts.h` |
| `tracks/byte_tracker.h`, `tracks/reid.h` | ByteTrack + appearance |
| `pipeline/detection_pipeline.h`, `pipeline/async_detection_pipeline.h` | streaming |

---

## The shape of a task head

Every head is two things, and you can use either:

```c++
// 1. a pure function over float buffers — no Engine, no hardware, testable anywhere
std::vector<Detection> decode(const float* data, const std::vector<int>& shape,
                              const DetectConfig& cfg, const LetterboxInfo& lb);

// 2. an Engine-bound class that reads the outputs and calls it
YoloLtrbDetector det(engine);
std::vector<Detection> dets = det.postprocess(lb);
```

The bound class resolves the model's layout **at construction** and throws with
the full output signature printed if the model is not the head it was asked for.
Structure (grids, class count, `reg_max`, channel order, strides) always comes
from the model; the config supplies only thresholds and the two things the file
cannot tell you — channel order and where the activation lives (see
`docs/MODELS.md`).

Reading outputs goes through `outputAsFloat(engine, i, scratch, shape)`: a
packed f32 tensor is returned zero-copy, anything else (the usual int8-affine
RKNN output, fp16, a stride-padded layout) is gathered into `scratch`.

Writing the *input* is the mirror image, with one exception. A quantized model's
input is presented to the runtime as u8 image bytes, which is right for anything
a camera feeds. A model whose input is a computed **map** — XFeat takes an
InstanceNorm output, roughly ±3 — cannot go through that path: it has no negative
range, so half the tensor clips to the zero point and the head still returns
plausible results. Open those with

```c++
rcdl::EngineOptions opts;
opts.float_inputs = {0};            // input 0 is a map, not pixels
rcdl::Engine engine("xfeat_640x480_i8_rk3588.rknn", opts);
rcdl::FeatureExtractor extractor(engine);   // throws if the Engine was not
```

and the head checks `inputType()` rather than trusting the caller.

Operators the runtime does not have are the other half of that story.
`backend/custom_ops.h` registers RCDL's own CPU kernels on every Engine
(`EngineOptions::custom_ops`, on by default), which is what makes a
correlation-based optical-flow model loadable at all — librknnrt 2.3.2 has no
`GridSample` on the NPU *or* on its CPU fallback path, and a model built without
that node declared as a custom operator segfaults inside `rknn_init`.

`FaceRecognizer` is the other shape worth knowing: a head that owns a **host
image transform** because the transform is part of the model contract. An
identity embedding is computed on a five-point similarity warp onto a fixed
template, not on the detector's box, and a box crop returns a perfectly
well-formed vector with half the identity gone. `faceAlignTransform()` exposes
the geometry for a caller that wants to warp elsewhere; the class does it
internally so the template, the output size, the channel order and whether the
model wants bytes or floats cannot drift apart. It reads `inputType(0)` and
feeds u8 or float32-in-0..255 accordingly, which is what lets one class serve a
quantized and a float build.

One head here is **anchor-based**, and it is worth knowing why it is separate.
Everything above decodes anchor-free LTRB, where a cell predicts distances to the
four box edges. A panoptic driving model built on the YOLOv5 backbone predicts,
per prior box, an offset from the cell and a multiplier on that prior's size:

```c++
rcdl::Engine engine("yolop_cut_640_i8_rk3588.rknn");   // 5 outputs
rcdl::AnchorDetectConfig cfg;                          // priors default to YOLOP's
cfg.num_classes = 1;
rcdl::AnchorDetector det(engine, cfg, /*output_base=*/0);
rcdl::SegConfig scfg; scfg.num_classes = 2;
rcdl::Segmenter drivable(engine, scfg, 3), lane(engine, scfg, 4);
// one inference, three decoders reading it
```

The priors are part of the model rather than a tuning knob — a box's size *is*
its prior times a bounded multiplier, so the wrong set gives plausible-looking
boxes of the wrong size and no error. And the model is `_cut` for a reason: the
published export bakes the decode into the graph with `ScatterND`, which compiles
cleanly into a tensor whose objectness and class columns are never written. See
`docs/MODELS.md`.

The opposite case is a head that needs **no** new code: an open-vocabulary
(YOLOE) build is an ordinary LTRB head whose class axis happens to mean words,
because the CLIP text embeddings were folded into the classification convolution
at conversion time. `rcdl::LabelMap` carries the class_id → prompt table and
`requireSize()` checks it against the class count the model declares, which is
the only way to catch a labels file from a different vocabulary: it renames every
detection without moving one.

---

## Detection, end to end

```c++
rcdl::Engine engine("yolov8n_rk3588.rknn");
rcdl::PipelineConfig cfg;
cfg.model_input = rcdl::PixelFormat::RGB888;     // the export's channel order
rcdl::DetectionPipeline pipe(engine, cfg);

for (;;) {
  auto dets = pipe.process(frame.view());        // RGA -> NPU input tensor -> decode
  ...
}
```

`process()` allocates nothing per frame. The letterbox destination *is* the
NPU's input tensor, so one `improcess` crops, scales, converts NV12→RGB888 and
paints the border straight into the buffer the NPU is about to read.

Three cores, results in submission order:

```c++
rcdl::AsyncDetectionPipeline pipe(engine, cfg, {.workers = 3});
pipe.submit(frame.view());                       // letterbox on this thread, then hand off
std::vector<rcdl::Detection> dets;
while (pipe.next(dets)) { ... }
pipe.finish();
```

The letterbox runs on the **calling** thread deliberately: the source frame is
fully consumed when `submit()` returns, so a decoded frame can go back to the
VPU immediately and there is no lifetime rule to get wrong. If you must not hold
a frame while waiting for capacity, use the split form —
`acquireSlot()` / `letterboxIntoSlot()` / `commitSlot()` — which waits for a slot
before you take a frame.

---

## Compressed video in, detections out

`AsyncVideoDetectionPipeline` owns the whole path — VPU decode, RGA letterbox,
NPU inference on N contexts — so the caller only pumps bytes:

```c++
rcdl::VideoDecConfig dec{.codec = rcdl::VideoCodec::H264};
rcdl::AsyncVideoDetectionPipeline pipe(engine, cfg, dec);

std::vector<rcdl::Detection> dets;
while (std::size_t n = fread(buf, 1, sizeof buf, fp)) {
  while (!pipe.submit(buf, n) && !pipe.finished())
    while (pipe.tryNext(dets)) { /* make room, then retry */ }
  while (pipe.tryNext(dets)) { /* ... */ }
}
pipe.finish();
while (pipe.next(dets)) { /* the reorder tail */ }
```

Chunks may be any size — MPP's parser finds the access units — and results come
back in decode order with `lastPtsUs()` / `lastFrameIndex()` beside them, which
is all that ties a result to its frame once the decoded buffer has gone back to
the pool.

**`submit()` returning `false` is back-pressure, and it must not be a wait.**
Every queue in the pipeline is bounded and the last one is emptied only by
`next()`, so a single-threaded caller that blocked inside `submit()` would
deadlock against itself: no input accepted until frames move, no frames until a
context frees, no context until results are drained, and the only thread that
can drain is the one parked in `submit()`. Hence the retry loop above.
`finished()` distinguishes back-pressure from a closed pipeline; a caller that
drains on another thread can pass a negative timeout and block instead.

Measured on the board (1080p H.264 → YOLOv8n, three pinned contexts), over six
runs of a 300-frame clip: **72–97 fps against 23–28 fps for the same stream
through the synchronous pipeline, a 3.1x to 3.7x speed-up**, with the per-frame
detections identical to the synchronous path. The spread is the board's, not the
pipeline's — both figures move together run to run, which is why a range is
quoted rather than the best single number.

---

## Media

```c++
rcdl::VideoDecoder dec({.codec = rcdl::VideoCodec::H264});
dec.feed(bytes.data(), bytes.size());            // false == back-pressure, drain and retry
rcdl::VideoFrame frame;
while (dec.receive(frame, 0)) {
  auto dets = pipe.process(frame.view());        // no copy: RGA reads the VPU's buffer
  enc.feed(frame.view());                        // no copy: the VPU reads it back
}
```

**Release frames promptly.** Every frame you hold is one buffer the decoder
cannot reuse; holding the whole pool stalls decoding. `VideoFrame` is move-only
and returns its buffer in the destructor — let it go out of scope, or
`reset()` it.

Reading a decoded frame with the CPU needs a coherency window, because the VPU
wrote it through an IOMMU without touching the caches:

```c++
const std::uint8_t* y = frame.beginCpuRead();
...
frame.endCpuRead();
```

---

## Cache discipline, in one rule

- CPU wrote it, hardware will read it → **flush** before the op.
- Hardware wrote it, CPU will read it → **invalidate** before reading.
- It only ever moves between hardware units → **neither**, and skipping it is
  the whole point.

`DmaBuf::syncStart/syncEnd` and the free `dmaBufSyncStart/End(fd, ...)` do it.
The RKNN runtime flushes its own I/O tensors around `rknn_run`, and the preproc
wrappers leave fd-backed buffers alone for exactly this reason.

---

## Voice pipeline

Wake word → VAD → recognition → command → spoken reply, every network on the
NPU: [`examples/voice_assistant.cc`](../examples/voice_assistant.cc) is the
whole loop in C++ (WAV in, WAV replies out; with no input it synthesises a
demo session). Audio format, wake-word tokens and tuning are in the
[voice pipeline overview](API.md#voice-pipeline); the classes are below.

## Speech recognition (Paraformer)

`rcdl/preproc/fbank.h` — Kaldi-compatible front end; `rcdl/tasks/asr.h` — CIF,
token decoding and the three-stage `ParaformerAsr`. One 16 kHz mono utterance
(at most `maxSeconds()`, 24 s for the registered build) in, text out.

```cpp
rcdl::Engine enc("models/paraformer_encoder_fp16_rk3588.rknn");
rcdl::Engine pred("models/paraformer_predictor_fp16_rk3588.rknn");
rcdl::Engine dec("models/paraformer_decoder_fp16_rk3588.rknn");
rcdl::ParaformerAsr asr(enc, pred, dec,
                        rcdl::loadTokenList("models/paraformer_tokens_8404.json"),
                        rcdl::loadKaldiCmvn("models/paraformer_am.mvn"));
rcdl::AsrResult r = asr.transcribe(pcm, num_samples);   // float in [-1, 1]
// r.text, r.token_ids, r.feature_frames, r.truncated

rcdl::Fbank fbank;                                         // FbankConfig{} = 16 kHz, 80 mel
std::vector<float> feats;
int frames = fbank.compute(pcm, num_samples, feats);       // [frames, 80]
// pure: lfrStack(feats, frames, dim, lfr_m, lfr_n, out) · parseKaldiCmvn(text)
// pure: cifIntegrate(alphas, hidden, frames, dim, valid_frames, cfg, embeds)
// pure: argmaxTokens(logits, rows, vocab) · tokensToText(ids, vocab)
```

| member | notes |
|---|---|
| `transcribe(pcm, samples)` | front end + 3 NPU stages + CIF. Longer than the window → truncated, `truncated` set |
| `transcribeFeatures(feats, frames)` | start from `[frames, featureDim()]` LFR+CMVN features |
| `features(pcm, samples, out)` | front end only |
| `maxFrames()` · `maxSeconds()` · `featureDim()` · `maxTokens()` · `vocabSize()` | geometry read from the Engines |

`ParaformerAsr` matches the stages by tensor shape rather than index, and refuses
a token list whose size differs from the decoder's class count. `cifIntegrate`
puts `CifConfig::tail_threshold` at frame `valid_frames` — the true end of
speech. A fixed-shape predictor graph can only add it at the end of its padding,
where masking the padding removes it, and any utterance whose last token
needed the tail then loses it.


### Voice activity detection — `rcdl/tasks/vad.h`

```cpp
rcdl::Engine eng("models/silero_vad_16k_fp16_rk3588.rknn");
rcdl::SileroVad vad(eng);                     // stateful: one stream per object
float p = vad.probability(chunk);             // 512 samples (32 ms) -> speech probability
rcdl::VadSegmenter seg(vad);                  // VadConfig{}: 0.5, 250 ms on, 300 ms off
for (auto& s : seg.feed(pcm, n))              // any length; returns the utterances that closed
  asr.transcribe(s.pcm.data(), s.pcm.size()); // s.start / s.end: samples since reset()
for (auto& s : seg.flush()) { /* end of stream */ }
```

| member | notes |
|---|---|
| `SileroVad::probability(chunk)` · `reset()` | `kChunk` = 512 samples; carries LSTM state and the last 64 samples |
| `VadSegmenter::feed(pcm, n)` · `flush()` · `segments(pcm, n)` | `VadSegment{start, end, pcm}`; `segments` = reset + feed + flush |
| `inSpeech()` · `lastProbability()` · `config()` | |
| `VadConfig` | `threshold`, `neg_threshold` (< 0: = threshold), `min_speech_ms`, `min_silence_ms`, `speech_pad_ms`, `max_speech_s` |

The model's STFT (4 Hann frames of 256, hop 128) runs on the CPU and the NPU
takes the magnitude spectrum: squared inside the graph, the power spans ~5000
and a quantized build loses every quiet bin. A segment opens after `min_speech_ms` at or above
`threshold` and closes after `min_silence_ms` below `neg_threshold`; a segment
reaching `max_speech_s` is cut there and a new one starts.


### Keyword spotting (wake word) — `rcdl/tasks/kws.h`

```cpp
rcdl::Engine enc("models/kws_zipformer_zh_en_3m_c16_fp16_rk3588.rknn");
rcdl::KeywordSpotter kws(enc, "models/kws_zipformer_zh_en_3m_decoder_joiner.npy",
                         "models/kws_zipformer_zh_en_3m_tokens.txt");   // KwsConfig{}
kws.setKeywords("n ǐ h ǎo x iǎo ch ē @你好小车\n"
                "x iǎo ch ē x iǎo ch ē :1.5 #0.3 @小车小车\n");    // or loadKeywords(path)
for (const rcdl::KwsDetection& d : kws.feed(pcm, n))     // any length, float in [-1, 1]
  printf("%s at %.2f s\n", d.keyword.c_str(), d.end);
```

| member | notes |
|---|---|
| `setKeywords(text)` · `loadKeywords(path)` · `keywords()` | one line per keyword: model tokens, optional `:boost` and `#threshold`, `@phrase`; resets the stream |
| `feed(pcm, n)` | `KwsDetection{keyword, index, start, end, tokens}`, seconds since `reset()` |
| `flush()` | end of stream: pushes the last partial chunk through with silence, returns what fires, resets |
| `reset()` · `encode(pcm, n, frames)` · `tokenId(tok)` · `vocabSize()` | `encode` returns the encoder output of a whole recording (tests) |
| `KwsConfig` | `keywords_score` (1.0), `keywords_threshold` (0.25), `num_trailing_blanks` (1), `max_active_paths` (4) — sherpa-onnx's defaults |

The encoder runs on the NPU one 160 ms chunk at a time and carries 37 cache
tensors between calls; the decoder, joiner and keyword search (a beam search
over an Aho-Corasick graph of the keywords) run on the CPU and follow
sherpa-onnx's KeywordSpotter step for step — a float transcription of the same
search gave sherpa-onnx's detections on all 87 clips it was checked on. Features are Kaldi fbank
with the Povey window on samples in [-1, 1] (`FbankConfig::window`,
`input_scale`).


### Voice commands — `rcdl/tasks/command.h`

```cpp
rcdl::CommandMatcher m({"前进", "后退", "左转", "右转", "打开灯", "关灯"});  // max_distance 0.5
rcdl::CommandMatch c = m.match(r.text);   // "大开灯" -> index 4, distance 0
if (c.index >= 0) { /* c.command */ }
// pure: toPinyin(text) -> std::vector<std::string>, one toneless syllable per CJK char
```

| member | notes |
|---|---|
| `match(text)` | `CommandMatch{index (-1 = rejected), nearest, distance, command}` |
| `distances(text)` | normalised syllable edit distance to every command |
| `commands()` · `size()` · `maxDistance()` | |

Distance is the syllable edit distance over the command's syllable count; ties
go to the closer length, then the earlier command. Latin words compare only
with Latin words. The pinyin table (U+3400..U+9FFF) is compiled in and generated
by `scripts/gen_pinyin_table.py`.


### Speech synthesis back end — `rcdl/preproc/istft.h`, `rcdl/tasks/vocoder.h`

```cpp
rcdl::Engine eng("models/vocos_16k_t256_fp16_rk3588.rknn");
rcdl::VocosVocoder voc(eng);                       // VocosConfig{}: context 32, pad -16
std::vector<float> pcm;
int n = voc.synthesize(mel, frames, pcm);          // mel [80, frames] row-major -> 16 kHz PCM
// pure: Istft(IstftConfig{}).compute(real, imag, frames, out)        // [frames, 513] each
// pure: Istft::computePolar(mag, cos, sin, frames, out)               // [513, frames] each
```

| member | notes |
|---|---|
| `synthesize(mel, frames, pcm)` | chunks of `chunkFrames()` overlapping by `context`; returns `(frames - 1) * hop` |
| `numMels()` · `chunkFrames()` · `bodyFrames()` · `sampleRate()` | geometry read from the Engine |

The Engine must output `log_mag` and `phase` (`[1, 513, T]`, in that order);
`exp`, the magnitude clip, `cos`, `sin` and the iSTFT run on the CPU. Chunk
seams are exact for `context >= 27`; only the utterance's two ends see
`pad_value` instead of the network's zero padding.


### Speech synthesis — `rcdl/tasks/tts.h`

```cpp
rcdl::Engine enc("models/matcha_zh_en_enc_l128_fp16_rk3588.rknn");
rcdl::Engine dec("models/matcha_zh_en_dec_t512_fp16_rk3588.rknn");
rcdl::Engine voc_eng("models/vocos_16k_t256_fp16_rk3588.rknn");
rcdl::MatchaAcoustic am(enc, dec, "models/matcha_zh_en_embedding.npy");  // MatchaConfig{}
rcdl::VocosVocoder voc(voc_eng);
rcdl::TextToSpeech tts(rcdl::MatchaFrontend("models/matcha_zh_en_tokens.txt",
                                            "models/matcha_zh_en_lexicon.txt"),
                       am, voc);
rcdl::TtsResult r = tts.synthesize("好的，马上出发。");   // r.pcm, 16 kHz
```

| member | notes |
|---|---|
| `MatchaFrontend::encode(text)` | one id vector per sentence; matches sherpa-onnx's front end |
| `MatchaFrontend::readNumbers(text)` | digits read out as characters (static) |
| `MatchaAcoustic::synthesize(ids, n)` | `MatchaResult{mel [80, frames], frames, durations, truncated}` |
| `MatchaAcoustic::seed(s)` · `maxTokens()` · `maxFrames()` · `numMels()` · `vocabSize()` | |
| `TextToSpeech::synthesize(text)` | `TtsResult{pcm, sample_rate, sentences, truncated}`; pauses shortened per `TextToSpeechConfig::silence_scale` (0.2) |
| `scaleSilence(pcm, sample_rate, scale, min_pause_s, level)` | pure; pauses ≥ `min_pause_s` keep `scale` of their length (sherpa-onnx's rule) |

The encoder takes embeddings, not ids: the lookup runs on the CPU from the
`.npy` table (`[vocab, D]`, pre-scaled). Durations are `ceil(exp(logw)) *
length_scale`, regulated on the CPU exactly as Matcha's `generate_path`; the
decoder noise comes from a seeded `std::mt19937`.

---

## Errors

`rcdl::Error` derives from `std::runtime_error` and carries the vendor return
code (`.code()`). `RCDL_CHECK(expr)` wraps a call returning 0-on-success;
`RCDL_REQUIRE(cond, "msg")` is a precondition. librga inverts the convention
(1 == success), so its wrappers normalise before throwing.

---

See `docs/API.md` for Python, `docs/RGA.md` for what the 2-D engine will and
will not do, and `docs/MODELS.md` for the per-model contract.
