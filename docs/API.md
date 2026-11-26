# Python API

```python
import rcdl
```

The compiled core (`rcdl_py`) exchanges raw buffers; the `rcdl` package adds the
numpy shaping. Everything that touches hardware releases the GIL, so a Python
thread pool over several `Engine`s really does run concurrently on the three NPU
cores.

Pixel formats and codecs are lower-case tokens — `"rgb888"`, `"bgr888"`,
`"rgba8888"`, `"bgra8888"`, `"gray8"`, `"nv12"`, `"nv21"`, `"yuv420p"`;
`"h264"`, `"h265"`, `"vp9"`, `"av1"`, `"mjpeg"` — and they **round-trip**: a
value read off an object (`frame.format`, `decoder.codec`) can be passed
straight back in.

---

## Inference

```python
e = rcdl.Engine("models/yolov8n_rk3588.rknn")          # optionally core=rcdl.NpuCore.CORE_0
outs = e.infer(np.zeros(e.input_shape(0), np.uint8))   # -> list of float32 arrays
```

`Engine` binds its I/O tensors once as dma-bufs (`rknn_create_mem` +
`rknn_set_io_mem`) and reuses them, so `infer()` allocates nothing and copies
nothing inside the runtime. Outputs come back dequantized.

| | |
|---|---|
| `Engine(path, core=NpuCore.AUTO, init_flags=0, float_inputs=())` | load a `.rknn` |
| `.dup(core)` | a second context sharing the weights, optionally on another core |
| `.infer(inputs)` / `.run()` / `.output(i)` | run; `set_input` + `run` + `output` for the explicit form |
| `.input_shape(i)` `.output_shape(i)` `.input_dtype(i)` `.output_quant(i)` | introspection |
| `.input_fd(i)` `.output_fd(i)` | the dma-buf fds, for hardware hand-off |
| `.last_run_micros()` `.perf_detail()` `.sdk_version()` `.driver_version()` | diagnostics |

`float_inputs` names inputs whose tensor is a normalised **map** rather than
image bytes — XFeat's InstanceNorm output is the case in this repo. Those are
handed to the runtime as float32; the u8 path a quantized model normally uses has
no negative range, so half of such a map would clip to the zero point and the
model would still return plausible-looking results. Heads that need it check and
refuse rather than run.

Three cores at once — this is the throughput idiom, not `NpuCore.ALL`:

```python
engines = [rcdl.Engine(path, core=c) for c in
           (rcdl.NpuCore.CORE_0, rcdl.NpuCore.CORE_1, rcdl.NpuCore.CORE_2)]
```

---

## Preprocessing

RGA does the work when it can and the CPU when it cannot; the third return value
tells you which ran, which is the first thing to check when a frame is slow.

```python
img, lb, backend = rcdl.letterbox(bgr, 640, 640, src_fmt="bgr888", dst_fmt="rgb888")
img, backend     = rcdl.cvt_color(nv12, "nv12", "bgr888")
lb               = rcdl.compute_letterbox(1280, 720, 640, 640)
rcdl.rga_available(), rcdl.rga_version()
```

`backend` is `"auto"` (default), `"rga"` (raise if the hardware refuses) or
`"cpu"`. For a YUV source or destination, `studio_range=True` (default) and
`matrix="bt601"` (default) or `"bt709"` describe the YUV side — HD video is
usually `matrix="bt709"`. `"auto"` runs BT.709 full range, and any BT.709
RGB → YUV, on the CPU. `lb` is the 7-tuple `(scale, pad_x, pad_y, src_w, src_h, dst_w, dst_h)`
that post-processing inverts. See `docs/RGA.md` for the constraints that decide
the fallback and for how the two backends differ.

---

## Detection

```python
det = rcdl.Engine("models/yolov8n_rk3588.rknn").detector(model_input="rgb888")
for d in rcdl.detect(det, bgr_image):
    print(rcdl.coco_class_name(d.class_id), d.score, d.x1, d.y1, d.x2, d.y2)
```

Boxes come back in **original-image pixels**, already un-letterboxed. The head
layout — grids, class count, DFL `reg_max`, channel order, strides — is read
from the model; a model that does not match raises at construction.

```python
det.head            # "yolo-ltrb" | "single-tensor"
det.head_layout     # what the resolver read out of the model
det.backend         # which preproc ran on the last frame
det.letterbox       # geometry of the last frame
det.profile         # (preproc_ms, infer_ms, postproc_ms, frames), per frame
```

The decoders are also usable as pure functions on numpy arrays, with no Engine —
this is the path the tests pin:

```python
rcdl.decode(tensor, lb, num_classes=80, apply_sigmoid=True)
rcdl.decode_yolo_ltrb(cls_list, box_list, grids, strides, lb, reg_max=16)
rcdl.nms(boxes_n6, iou_thresh=0.45, max_dets=300)
```

---

## Video (VPU)

```python
for frame in rcdl.decode_video("clip.h264", max_frames=300):
    dets = det.process_frame(frame)     # RGA reads the VPU's buffer directly
```

`decode_video` yields `VideoFrame`s that still live in the decoder's dma-bufs,
so **each is only valid until the next iteration** — hand it to
`process_frame`, or copy what you need with `to_numpy()`.

| | |
|---|---|
| `VideoDecoder(codec="h264", format="nv12", external_buffers=True, pool_heap="system")` | `.feed(bytes)` / `.receive(timeout_ms)` / `.flush()`, `.pool_heap` |
| `VideoFrame` | `.width .height .width_stride .height_stride .fd .format .pts_us .below_4g`, `.to_numpy()`, `.letterbox(w, h)`, `.draw_rects(boxes, color, thickness=2, backend="auto")`, `.release()` |
| `VideoEncoder(width, height, codec="h264", bitrate_kbps=4000, rc="cbr")` | `.feed_frame(frame)` (zero copy) / `.feed(array, w, h)` / `.receive()` / `.flush()` / `.extra_data` |
| `JpegEncoder(w, h, quality=80)` / `JpegDecoder()` | `.encode(array)` / `.encode_frame(frame)` / `.decode(bytes)` |

`feed()` returning `False` is **back-pressure, not an error**: drain and retry.

`frame.draw_rects(boxes, colors)` paints box outlines **in place** on the
decoded frame — `boxes` are `(x1, y1, x2, y2)` in frame pixels, `colors` one
`(r, g, b)` or one per box — so the annotated frame can go straight to
`VideoEncoder.feed_frame()` without a copy. The default backend is the CPU
(one map and one cache sync per frame, measured faster than the hardware);
`backend="rga"` draws on the RGA2 core and needs the frames below 4 GB, i.e.
`VideoDecoder(pool_heap="system-dma32")`. Either way the bytes are the same;
see `docs/RGA.md` §3.

```python
dec = rcdl.VideoDecoder(codec="h264")
while not dec.feed(chunk):
    f = dec.receive(5)
    ...
```

Two things the decoder tells you that are worth checking once:
`decoder.using_external_buffers` (frames are in RCDL's own dma-bufs) and
`frame.width_stride` vs `frame.width` — **the VPU pads rows**, and reading them
at the width is the classic way to get a sheared picture. `to_numpy()` removes
the padding unless you pass `keep_stride=True`.

### Compressed video straight to detections

`engine.video_detector()` puts the whole path — VPU decode, RGA letterbox, NPU
inference across three contexts — behind two calls, all of it C++ threads with
the GIL released. Python never touches a frame, so a driver that only pumps
bytes runs at the C++ speed (**72–97 fps on 1080p H.264 → YOLOv8n against 23–28
fps frame-at-a-time, a 3.1–3.7× speed-up over six runs**).

```python
p = engine.video_detector(codec="h264")            # detector kwargs also apply
for chunk in chunks:                               # any size; MPP splits them
    while not p.submit(chunk) and not p.finished:
        while (d := p.try_next()) is not None:     # make room, then retry
            use(d, p.frame_index)
    while (d := p.try_next()) is not None:
        use(d, p.frame_index)
p.finish()
while (d := p.next()) is not None:                 # the reorder tail
    use(d, p.frame_index)
```

`submit()` returning `False` is **back-pressure**, not an error, and it is a
return value rather than a wait on purpose: every queue inside is bounded and
only `next()` empties the last one, so a single-threaded driver that blocked in
`submit()` would deadlock against its own pipeline. Offer the same bytes again
after draining; `.finished` tells you when `False` means "closed" instead.

| | |
|---|---|
| `engine.video_detector(codec="h264", workers=3, queue_depth=2, **detector_kwargs)` | `.submit(bytes, timeout_ms=20)` / `.next()` / `.try_next()` / `.finish()` |
| result metadata | `.frame_index .pts_us .letterbox .finished` |
| stream | `.width .height .frames_decoded .using_external_buffers .workers .head` |
| `.profile` | `(decode_ms, preproc_ms, infer_ms, postproc_ms, frames)`, per frame |

A raw elementary stream carries no timestamps, so `.pts_us` is 0 on one —
`.frame_index` is what identifies a frame there.

---

## Sparse features and matching

```python
e  = rcdl.Engine("models/xfeat_640x480_i8_rk3588.rknn", float_inputs=[0])
ex = e.feature_extractor()                       # config=rcdl.XfeatConfig()

fa = rcdl.extract_features(ex, frame_a)          # BGR uint8 HxWx3
fb = rcdl.extract_features(ex, frame_b)
pairs, cosines = rcdl.match_features(fa, fb)     # (M,2) indices, (M,) scores
```

`fa.xy` is `(N,2)` in the ORIGINAL frame's pixels, `fa.scores` `(N,)`, and
`fa.descriptors` `(N,64)` with L2-normalised rows — so a dot product *is* the
cosine, and `pairs`/`cosines` drop straight into `cv2.findHomography`.

`match_features` is mutual nearest neighbour with a cosine floor
(`min_cossim=0.82`): a pair survives only if each side is the other's best, which
is what makes it usable on repeated texture without a ratio test. It costs
`O(|a|·|b|·64)`, so `XfeatConfig.top_k` (4096 by default) is the knob that
decides whether a pair costs milliseconds or a quarter second — see
`docs/MODELS.md` for the measured trade-off.

The decoder is also available as a pure function on the three raw maps —
`rcdl.decode_xfeat(feats, keypoints, reliability, config, scale_x, scale_y)` —
with `rcdl.xfeat_preprocess(bgr, in_w, in_h)` producing the input the model
wants. `float_inputs=[0]` above is not optional: see the Inference section.

---

## Whole-body pose

```python
det = rcdl.Engine("models/yolov8n_rk3588.rknn").detector()
wb  = rcdl.Engine("models/rtmw_s_133_256x192_fp16_rk3588.rknn").wholebody_estimator()

for d in rcdl.detect(det, frame):
    if rcdl.coco_class_name(d.class_id) != "person":
        continue
    kp = rcdl.estimate_wholebody(wb, frame, (d.x1, d.y1, d.x2, d.y2))   # (133, 3)
    b, e = rcdl.body_part_range(rcdl.BodyPart.LEFT_HAND)                # 91, 112
    fingers = kp[b:e]
```

Top-down: **one inference per person** (~25 ms), so a detector runs first — the
opposite cost model to `pose_estimator()`, which gives 17 joints for everybody in
one pass. Use this one when the face and the fingers matter.

`kp[i]` is `(x, y, score)` in source pixels; a joint below `kpt_thresh` keeps its
score and comes back at `(-1, -1)` rather than as a guess. `rcdl.body_part(i)`
and `rcdl.body_part_range(part)` slice the 133 into body / feet / face / hands,
and the first 17 are the COCO body joints in the usual order.

The pieces are separately available: `rcdl.crop_geometry(x1, y1, x2, y2)` is the
rect the model is actually shown (the box padded by 1.25, then grown to the
model's aspect), and `rcdl.decode_simcc(simcc_x, simcc_y, crop)` turns raw
outputs into keypoints. Both conventions are load-bearing — see `docs/MODELS.md`.

---

## Promptable segmentation

```python
enc = rcdl.Engine("models/edge_sam_3x_encoder_fp16_rk3588.rknn")
dec = rcdl.Engine("models/edge_sam_3x_decoder_fp16_rk3588.rknn")
sam = enc.prompt_segmenter(dec)

sam.set_image(frame.reshape(-1), w, h, "bgr888")   # ~350 ms, once per frame
m = sam.box(x1, y1, x2, y2)                        # ~140 ms per prompt
m = sam.point(cx, cy)                              # or a click
m.mask, m.score, m.bbox, m.area                    # (H,W) uint8 0/1, in SOURCE pixels
every = sam.masks()                                # all four nestings, best first
```

`rcdl.prompt_mask(sam, img, box=(...))` is the one-shot convenience form; it
re-encodes every call, which is exactly what you do **not** want in a loop over
prompts on one frame.

The mask is a plain `(H, W)` uint8 array over the source frame, so it composes
with anything: `img[m.mask.astype(bool)]`, `cv2.findContours`, a paste onto a new
background. Prompts are in source pixels — a detector's box goes in unchanged,
which is the usual way to turn boxes into silhouettes.

Both models are float here on measurement, and the head refuses an int8 encoder
rather than running one; `docs/MODELS.md` has the numbers, including what a
click returns when the encoder is quantized (0.07% of the frame instead of 3.5%).

---

## Optical flow

```python
e   = rcdl.Engine("models/neuflow_v2_512x384_fp16_rk3588.rknn")
est = e.flow_estimator()
field = rcdl.estimate_flow(est, frame_a, frame_b)   # (H, W, 2) float32, source pixels
speed = np.hypot(field[..., 0], field[..., 1])
viz   = rcdl.flow_colorize(field)                   # Middlebury wheel, (H, W, 3) BGR
```

`field[y, x]` is the displacement of that pixel between the two frames, **+u
right, +v down** — the OpenCV convention, so splitting it into x/y maps feeds
`cv2.remap` directly. `rcdl.flow_endpoint_error(a, b)` is the standard metric
between two fields.

This model carries an operator librknnrt does not implement, and the `Engine`
registers RCDL's CPU kernel for it at construction (`custom_ops=True`, the
default). Each such call crosses the CPU/NPU boundary, so a frame takes about
1.4 s: correct, not fast. `docs/MODELS.md` has the measurements and what a build
without the custom-operator declaration does instead (it segfaults `rknn_init`).

The pieces are also available on their own: `rcdl.flow_preprocess(bgr, w, h)`
formats one frame, and `rcdl.decode_flow(tensor, channels_first=True, scale_x=1,
scale_y=1)` turns a raw output tensor into a field.

---

## Super-resolution

```python
sr = rcdl.Engine("models/realesr_general_x4v3_128_fp16_rk3588.rknn").upscaler(overlap=16)
big = rcdl.upscale(sr, frame)                    # BGR uint8 in, BGR uint8 out
sr.scale, sr.tile, sr.last_tile_count            # 4, 128, tiles the last call ran
```

The model upscales one fixed tile; `SuperResolver` cuts the image into
overlapping tiles, runs each, and cross-fades them back together — cost is
linear in `last_tile_count`, and each tile is a whole inference (70–82 ms in
fp16 on this NPU). `overlap=0` butt-joints the tiles, which is only useful for
seeing the seam the cross-fade exists to remove.

`rcdl.plan_tiles(w, h, tile_w, tile_h, overlap)` and `rcdl.tile_weight(i, len,
ramp)` expose the geometry and the fade on their own.

Judge the result by sharpness, not PSNR: this family is trained perceptually and
scores *below* a bicubic resize against the ground truth while looking better.
See `docs/MODELS.md`, which also covers why the int8 build is not the default.

---

## Tracking

```python
cfg = rcdl.ByteTrackConfig()
cfg.track_buffer = 30
tracker = rcdl.ByteTracker(cfg)
for frame in rcdl.decode_video("clip.h264"):
    for t in tracker.update(det.process_frame(frame)):
        print(t.track_id, t.x1, t.y1, t.x2, t.y2)
```

Pass appearance vectors alongside the detections to switch on ReID-gated
association (`update(dets, embeddings)`, one entry per detection, empty entries
meaning geometry-only). `rcdl.reid_preprocess`, `rcdl.normalize_embedding` and
`rcdl.cosine_similarity` are the primitives.

`TrackingPipeline` is detect-and-track as one object, which is what most callers
want — it is `tracker.update(det.process(...))` with the buffers reused:

```python
det = rcdl.Engine("yolov8n_rk3588.rknn")
tracks = det.tracker()                       # geometry only
for frame in rcdl.decode_video("clip.h264"):
    for t in tracks.process_frame(frame):    # zero-copy: RGA reads the VPU buffer
        print(t.track_id, t.x1, t.y1, t.x2, t.y2)
```

Hand it a second Engine holding an appearance model and association gains a ReID
term, which is what holds identities through the occlusions and crossings that
motion alone loses:

```python
reid = rcdl.Engine("osnet_x0_25_msmt17_rk3588.rknn")
tracks = det.tracker(reid=reid, reid_min_score=0.5, reid_max_crops=32)
for t in rcdl.track(tracks, frame_bgr):
    ...
print(tracks.last_embed_count)   # crops embedded on that frame
```

`reid_min_score` and `reid_max_crops` are cost knobs with teeth: the appearance
model runs **once per crop**, so on a crowded frame it, not the detector, sets
the frame time. `last_embed_count` is the term to watch. The crop size comes
from the ReID Engine's own input shape, and crops are squashed to it rather than
letterboxed — see `docs/MODELS.md`.

With ReID on, `process_frame` maps the decoded frame so the CPU can read the
crops; geometry-only tracking never touches it.

---

## Face recognition

```python
det = rcdl.Engine("models/retinaface_rk3588.rknn").face_detector()
rec = rcdl.Engine("models/arcface_r50_112_fp16_rk3588.rknn").face_recognizer()

vecs = []
for f in rcdl.detect_faces(det, frame):
    lm = np.array(f.landmarks, np.float32)          # 5 points, source pixels
    vecs.append(rcdl.embed_face(rec, frame, lm))    # (512,) unit length

same = float(np.dot(vecs[0], vecs[1]))              # cosine: > ~0.5 same person
```

`embed_face` does the **five-point warp itself**, and that is the entry point to
prefer. The crop is the model's contract — template, size, channel order, and
(for a float build) values still scaled 0..255 rather than 0..1 — and a
caller-side warp is where those get lost silently. Measured: the same face
box-cropped instead of aligned scores 0.493 against its aligned self, where
nuisance transforms of the same photo stay above 0.98. See `docs/MODELS.md`.

`rcdl.face_align_transform(landmarks)` is still there for a caller that wants to
warp with `cv2.warpAffine` or RGA, and `recognizer.embed_aligned(crop)` takes the
result. Vectors are unit length, so `rcdl.cosine_similarity` is a dot product and
`EmbeddingBank` indexes them like any other appearance vector.

---

## Open-vocabulary detection

```python
e      = rcdl.Engine("models/yoloe_11s_streetwear_rk3588.rknn")
labels = e.label_map()               # <model>.labels.txt, checked against the model
det    = e.detector(num_classes=len(labels))

for d in rcdl.detect(det, frame):
    print(labels.name(d.class_id), d.score)                    # "sneakers" 0.77
```

There is **no open-vocabulary decoder**, and that is the point. YOLOE's text
comparison happens on the conversion host, where the CLIP text encoder folds one
embedding per prompt into the classification convolution; what reaches the board
is an ordinary LTRB head with one class channel per word, read by the same
`DetectionPipeline` as every other YOLO here.

So the vocabulary is chosen when the model is converted, not when it runs.
`LabelMap` is the only runtime state — and `require_size()` is worth calling,
because a labels file from a different build does not move a box or change a
score, it silently renames every result.

---

## Panoptic driving

```python
e = rcdl.Engine("models/yolop_cut_640_i8_rk3588.rknn")         # 5 outputs

cfg = rcdl.AnchorDetectConfig()                                # priors default to YOLOP's
cfg.num_classes, cfg.conf_thresh = 1, 0.35
det   = e.anchor_detector(cfg, output_base=0)                  # outputs 0,1,2
drive = e.segmenter(num_classes=2, output_index=3)
lane  = e.segmenter(num_classes=2, output_index=4)

drivable = rcdl.segment(drive, frame)                          # ONE inference
vehicles = det.postprocess(drive.letterbox)                    # ... three decoders
lanes    = lane.postprocess(drive.letterbox)
```

One inference feeds three heads, so exactly one of them runs the preprocessing
(`process`) and the other two decode what is already in the engine
(`postprocess`), using the letterbox geometry that inference used.

This is the library's only **anchor-based** detector: a cell predicts, per prior
box, an offset from the cell and a multiplier on that prior's size. The priors
are part of the model — `AnchorDetectConfig` defaults to YOLOP's, and decoding
with the wrong ones gives plausible-looking boxes of the wrong size rather than
an error. `rcdl.decode_yolov5_anchor()` is the Engine-free entry point.

---

## Voice pipeline

Five pieces make a voice-controlled robot, and all of their networks run on
the NPU. RCDL takes and returns 16 kHz PCM; capturing from a microphone and
playing to a speaker stay in your code.

```
 mic ──> KeywordSpotter ──wake──> VadSegmenter ──utterance──> ParaformerAsr ──text──> CommandMatcher
         (always on)               (until silence)             (one pass)              (pinyin match)
                                                                                            │
 speaker <──────────────────────────── TextToSpeech <────────── reply text <────────────────┘
```

| piece | job | NPU | CPU |
|---|---|---|---|
| `KeywordSpotter` | listens for the wake word, all the time | streaming encoder, one 160 ms chunk per call | fbank, decoder + joiner, keyword search |
| `VadSegmenter` (`SileroVad`) | after the wake word, cuts out one whole utterance | Silero, one 32 ms chunk per call | STFT magnitude, segmenting |
| `ParaformerAsr` | utterance → Chinese text | encoder, predictor, decoder | fbank + LFR + CMVN, CIF, argmax |
| `CommandMatcher` | text → one of your commands, or "not a command" | — | pinyin edit distance |
| `TextToSpeech` | reply text → speech | Matcha encoder + decoder, Vocos | text front end, length regulation, iSTFT |

Every network is a fixed-shape fp16 build, and every one of its inputs is a
computed tensor — mels, caches, LSTM state, embeddings, noise — so they are fed
as float32. These networks do not survive int8: the vocoder's phase, the VAD's
quiet bins and the TTS decoder's mel are all lost to it.

**Try it first.** [`examples/voice_assistant.py`](../examples/voice_assistant.py)
(and its C++ twin [`voice_assistant.cc`](../examples/voice_assistant.cc)) runs
the whole loop. With no argument it synthesises a short session in the robot's
own voice — speech without the wake word, two commands, one request that is not
a command — so it runs on a board with no microphone and no recordings:

```bash
python examples/voice_assistant.py            # or: ./build/voice_assistant
python examples/voice_assistant.py my.wav     # your own 16 kHz mono recording
```

It prints each event — the wake word, what was heard, the command it matched —
with the time taken, and writes every reply to a `.wav`.

**Getting the models.** The speech models are converted from their upstream
checkpoints to fp16 `.rknn` with rknn-toolkit2 on an x86 host;
[`docs/MODELS.md`](MODELS.md) lists each one's inputs, outputs and companion
files. Put them in `models/` under the names `scripts/fetch_models.sh --list`
shows. The tests that need a model that is not there skip rather than fail.

The loop is short; the parts that matter:

```python
awake = False
for block in mic_blocks():                      # any length; float32 or int16, 16 kHz mono
    if not awake:
        if kws.feed(block):                     # the wake word
            awake = True; vad.reset()
        continue
    for seg in vad.feed(block):                 # one whole utterance
        m = matcher.match(asr.transcribe(seg.pcm).text)
        play(tts.synthesize(f"好的，{m.command}。" if m else "没听懂，请再说一遍。").pcm)
        awake = False; kws.reset()              # back to sleep on a fresh stream
        break
```

**Audio in.** 16 kHz, mono, float in [-1, 1] or int16 (int16 is rescaled, not
cast). Resample other rates before RCDL (`ffmpeg -i in.wav -ar 16000 -ac 1
out.wav`, or your audio driver). Any block size works. `KeywordSpotter`,
`SileroVad` and `VadSegmenter` keep state between calls, so give each stream its
own objects and feed them in order. `reset()` starts a new stream, and
`flush()` ends one: without it, a wake word or utterance in the last fraction
of a second of a recording is never reported.

**Your wake word.** A wake word is a line of the model's tokens — pinyin
initials and toned finals — so changing it needs no training. Write the
phrases one per line as `你好小车 @你好小车`, then let sherpa-onnx's tool turn
them into tokens:

```bash
pip install sherpa-onnx
sherpa-onnx-cli text2token --tokens models/kws_zipformer_zh_en_3m_tokens.txt \
    --tokens-type ppinyin keywords_raw.txt keywords.txt
# keywords.txt:  n ǐ h ǎo x iǎo ch ē @你好小车
```

Choose three or four syllables that do not occur in everyday speech or in your
commands; a short or common phrase wakes the robot by mistake. English wake
words need the CMU-style lexicon (`en.phone`) from the upstream model package
and `--tokens-type phone+ppinyin`.

**Tuning.**

| if… | change |
|---|---|
| the wake word is missed | lower `KwsConfig.keywords_threshold` (0.25), or add `#0.15` to that line only |
| it wakes on similar phrases | raise the threshold, or `:0.5` to lower that line's boost |
| a command is cut in two at a pause | raise `VadConfig.min_silence_ms` (300) |
| the robot answers too slowly | lower `min_silence_ms` — it is most of the delay: the end of speech is declared about that long after the speaker stops |
| commands are rejected | add the phrasings people actually use as more entries; `CommandMatcher(..., max_distance=0.5)` |
| other speech is taken as a command | lower `max_distance` |
| replies are too fast or slow | `MatchaConfig.length_scale` (> 1 slower) |
| pauses in replies sound long or clipped | `TextToSpeechConfig.silence_scale` (0.2; 1 keeps the model's) |

**Things to know.**

- **The robot hears itself.** Stop feeding the microphone to the VAD and the
  spotter while a reply plays, or use echo cancellation in your audio stack.
- **Chinese.** Recognition and synthesis are Chinese; synthesis drops Latin
  words (write 无线网, not WiFi) and reads digits out (35% → 百分之三十五).
- **Limits.** One utterance is at most 24 s (`VadConfig.max_speech_s` cuts at
  20 s). One reply sentence is at most ~8 s; longer text is split into
  sentences automatically.
- **Measure on your robot.** Before relying on the pipeline, record a hundred
  real commands on your robot, with its motors running, and run them through
  the example.

---

## Speech recognition (Paraformer)

One 16 kHz mono utterance in, text out, in a single forward pass — Paraformer is
non-autoregressive, so there is no decoding loop. The build is three compiled
stages with continuous integrate-and-fire (CIF) on the CPU between the last two:
fbank → LFR → CMVN → **encoder** → **predictor** → CIF → **decoder** → argmax.

```python
asr = rcdl.ParaformerAsr.from_files(
    "models/paraformer_encoder_fp16_rk3588.rknn",
    "models/paraformer_predictor_fp16_rk3588.rknn",
    "models/paraformer_decoder_fp16_rk3588.rknn",
    "models/paraformer_tokens_8404.json",
    "models/paraformer_am.mvn")
r = asr.transcribe(pcm)          # float32 in [-1, 1] (or int16), 16 kHz mono
print(r.text, r.truncated)

# the front end and CIF on their own (no model needed):
feats = rcdl.Fbank().compute(pcm)                  # (frames, 80) Kaldi log-mel
lfr = rcdl.lfr_stack(feats, 7, 6)                  # (ceil(frames/6), 560)
emb, n = rcdl.cif_integrate(alphas, hidden, valid_frames)
```

**One utterance at a time, at most `asr.max_seconds` (24 s).** Longer input is
truncated and `r.truncated` is set — split long audio with a VAD first. There is
no resampler: resample to 16 kHz before calling.

The front end reproduces `torchaudio.compliance.kaldi.fbank` (dither 0) — to
~1e-5 of a float64 reference, ~2e-4 of torchaudio's own float32 — and FunASR's
LFR bit for bit, because a model trained on those features transcribes worse,
silently, on features that are merely similar.

- **`FbankConfig`** — `sample_rate`, `num_mel_bins`, `frame_length_ms`,
  `frame_shift_ms`, `preemphasis`, `remove_dc_offset`, `low_freq`, `high_freq`,
  `input_scale` (float PCM is multiplied by this, 32768 by default, because
  Kaldi works on int16-scale samples; icefall / k2 models want 1), `window`
  (`"hamming"`, torchaudio's and FunASR's, or `"povey"`, Kaldi's default).
- **`Fbank(config=FbankConfig())`** — `.compute(pcm) -> (frames, num_mel_bins)`,
  `.num_frames(samples)`, `.frame_length`, `.frame_shift`, `.fft_size`.
- **`lfr_stack(feats, lfr_m=7, lfr_n=6) -> np.ndarray`** — low-frame-rate
  stacking, bit-exact with FunASR.
- **`Cmvn`** — `dim`, `shift`, `scale`, `.apply(feats)`;
  **`load_kaldi_cmvn(path)`** / **`parse_kaldi_cmvn(text)`** read an `am.mvn`.
- **`CifConfig`** — `threshold` (1.0), `tail_threshold` (0.45), `max_tokens`.
- **`cif_integrate(alphas, hidden, valid_frames, config=CifConfig())`** →
  `((max_tokens, D) embeddings, token count)`. The tail weight is placed at frame
  `valid_frames`, the true end of speech. A fixed-shape predictor graph can only
  put it at the end of its padding, where masking the padding deletes it — and
  then any utterance whose last character needed the tail loses it.
- **`load_token_list(path) -> list[str]`** — a JSON array (`tokens.json`) or one
  token per line; **`argmax_tokens(logits)`**; **`tokens_to_text(ids, vocab)`**
  (drops `<specials>`, glues `@@` pieces, spaces Latin words).
- **`AsrResult`** — `text`, `token_ids`, `feature_frames`, `truncated`.
- **`ParaformerConfig`** — `fbank`, `lfr_m`, `lfr_n`, `cif`.
- **`ParaformerAsr(encoder, predictor, decoder, vocab, cmvn, config=None)`** —
  three `Engine`s; also `ParaformerAsr.from_files(encoder, predictor, decoder,
  tokens, cmvn, config=None)`. `.transcribe(pcm)`,
  `.transcribe_features(feats)`, `.features(pcm)`, `.max_seconds`,
  `.max_frames`, `.feature_dim`, `.max_tokens`, `.vocab_size`. The stages are
  matched by tensor shape, and a token list whose size disagrees with the
  decoder is refused.

### Voice activity detection (Silero VAD)

Finds where speech starts and stops in a 16 kHz stream and hands each utterance
to the recogniser whole. Silero VAD v6 runs on the NPU, one call per 32 ms
chunk; `VadSegmenter` turns its probabilities into utterances.

```python
vad = rcdl.SileroVad("models/silero_vad_16k_fp16_rk3588.rknn")
seg = rcdl.VadSegmenter(vad)                 # VadConfig(): 0.5, 250 ms on, 300 ms off
for block in mic_blocks():                   # float32 or int16, any length
    for s in seg.feed(block):                # utterances that closed in this block
        print(s.start / 16000, asr.transcribe(s.pcm).text)
for s in seg.flush(): ...                    # end of stream

segs = seg.segments(recording)               # a whole recording at once
p = vad.probabilities(recording)             # one probability per 512 samples
```

The `VadConfig` defaults were chosen on noisy sessions (AISHELL-1 utterances
with MUSAN noise at 10 and 0 dB SNR) and on noise alone: no false segments on
pure noise, and the end of speech declared a median 0.36 s after the speaker
stops.

- **`SileroVad(engine_or_path)`** — `.probability(chunk)` (512 samples →
  float), `.probabilities(pcm)` (whole recording from a reset), `.reset()`,
  `.chunk` (512), `.sample_rate` (16000). Stateful: the LSTM state and the
  previous chunk's last 64 samples carry over, so one object reads one stream,
  in order.
- **`VadConfig`** — `threshold` (0.5), `neg_threshold` (−1 = same as
  threshold; lower it for hysteresis), `min_speech_ms` (250), `min_silence_ms`
  (300), `speech_pad_ms` (30), `max_speech_s` (20 — the recogniser's window is
  24 s).
- **`VadSegmenter(vad, config=None)`** — `.feed(pcm)`, `.flush()`,
  `.segments(pcm)`, `.reset()`, `.in_speech`, `.last_probability`. A segment
  opens after `min_speech_ms` at or above `threshold` and closes after
  `min_silence_ms` below `neg_threshold`; the end of speech is therefore
  declared about `min_silence_ms` after the speaker stops.
- **`VadSegment`** — `start`, `end` (samples since the last reset, padding
  included), `pcm`.

The model's STFT runs on the CPU and the NPU takes the magnitude spectrum.
Inside the graph, squaring the real and imaginary parts puts the power at a
range of ~5000, where a quantized build loses every quiet frequency bin and
the probability sags in the middle of a sentence.

### Keyword spotting (wake word)

Listens continuously for a wake word, then the VAD and recogniser take over.
The model is sherpa-onnx's open-vocabulary zipformer transducer (zh-en, 3M
parameters): a wake word is a line of tokens, not a trained model. The encoder
runs on the NPU, one 160 ms chunk per call; the decoder, joiner and keyword
search run on the CPU and follow sherpa-onnx's KeywordSpotter step for step.

```python
kws = rcdl.KeywordSpotter("models/kws_zipformer_zh_en_3m_c16_fp16_rk3588.rknn",
                          "models/kws_zipformer_zh_en_3m_decoder_joiner.npy",
                          "models/kws_zipformer_zh_en_3m_tokens.txt",
                          "n ǐ h ǎo x iǎo ch ē @你好小车\n"
                          "x iǎo ch ē x iǎo ch ē @小车小车\n")
for block in mic_blocks():                   # float32 or int16, any length
    for d in kws.feed(block):
        print(d.keyword, d.start, d.end)     # seconds since reset()
```

A keyword line is the model's tokens — pinyin initials and toned finals for
Chinese, ARPAbet phones for English — then optional `:boost` and
`#threshold` for that line, then `@phrase`. sherpa-onnx's `text2token` tool
(`pip install sherpa-onnx`) writes these lines from plain text.

- **`KeywordSpotter(encoder, decoder_joiner, tokens, keywords=None,
  config=None)`** — `keywords` is the text of the lines or a path to a file.
  `.feed(pcm)`, `.flush()` (end of stream: a wake word in the last ~0.3 s
  is only seen after it), `.set_keywords(text)`, `.load_keywords(path)`, `.keywords`,
  `.reset()`, `.encode(pcm)` (encoder output of a whole recording, for tests),
  `.token_id(token)`. Streaming: one object reads one stream, in order.
- **`KwsConfig`** — `keywords_score` (1.0, the boost per matched token),
  `keywords_threshold` (0.25, the mean per-token probability a detection
  needs), `num_trailing_blanks` (1), `max_active_paths` (4): sherpa-onnx's
  defaults. Raise the threshold for fewer false wake-ups, lower it for more
  recall.
- **`KwsDetection`** — `keyword`, `index` (its line), `start`, `end`,
  `tokens`.

### Voice commands (closed set)

A robot taking spoken commands needs to know *which* command was said, not a
perfect transcript. `CommandMatcher` compares recogniser text with a fixed list
in toneless pinyin, which absorbs homophones (打开灯 → 大开灯, 出发 → 处罚) and
a stray trailing particle (前进 → 前进了), and rejects speech that is not a
command. Pure CPU, a few µs per call.

```python
m = rcdl.CommandMatcher(["前进", "后退", "左转", "右转", "打开灯", "关灯"])
r = m.match(asr.transcribe(pcm).text)    # e.g. "大开灯"
if r:                                    # r.index >= 0
    print(r.index, r.command, r.distance)   # 4 打开灯 0.0
```

- **`to_pinyin(text) -> list[str]`** — one toneless syllable per CJK character
  (most common reading, ü as `v`); an ASCII letter run is one lower-cased token;
  a digit run is read as a Chinese number (`"90"` → `jiu shi`); full-width
  letters/digits fold to ASCII; everything else is dropped.
- **`CommandMatcher(commands, max_distance=0.5)`** — distance is the
  syllable-level edit distance divided by the command's syllable count; ties go
  to the closer length, then the earlier command. Latin words are only ever
  compared with Latin words ("you" is not 右). `.match(text) -> CommandMatch`,
  `.distances(text) -> list[float]`, `.commands`, `.max_distance`, `len()`.
- **`CommandMatch`** — `index` (-1 = rejected), `nearest` (closest command even
  when rejected; -1 for text with no pinyin), `distance`, `command`; truthy when
  accepted. Several phrasings of one intent are several entries.

### Speech synthesis back end (Vocos)

Mel spectrogram in, waveform out: Vocos on the NPU, the inverse STFT on the CPU.
It is the second half of a non-autoregressive TTS — pair it with an acoustic
model that emits 80-bin log-mels at 16 kHz / hop 256, such as Matcha-TTS zh-en.

```python
voc = rcdl.VocosVocoder("models/vocos_16k_t256_fp16_rk3588.rknn")
pcm = voc.synthesize(mel)        # (80, frames) float32 log-mel -> float32 PCM, 16 kHz

# the inverse STFT on its own (no model needed):
ist = rcdl.Istft()               # n_fft 1024, hop 256, centred, periodic Hann
wav = ist.compute(real, imag)    # (frames, 513) each
wav = ist.compute_polar(mag, cos, sin)   # (513, frames) each — Vocos's layout
```

The build takes 256 frames (4.1 s); longer mels are cut into chunks that
overlap by 32 frames and the seams are exact.

- **`IstftConfig`** — `n_fft` (power of two), `hop_length`, `center`.
- **`Istft(config=IstftConfig())`** — `.compute(real, imag)`,
  `.compute_polar(mag, cos, sin)`, `.num_samples(frames)` (`(frames - 1) * hop`
  centred), `.bins`. Matches `torch.istft` with a periodic Hann window; DC and
  Nyquist imaginary parts are ignored.
- **`VocosConfig`** — `stft`, `context` (overlap per side, ≥ 27 for exact
  seams), `pad_value` (log-mel for frames past either end, −16),
  `max_magnitude` (100), `sample_rate`.
- **`VocosVocoder(engine_or_path, config=None)`** — `.synthesize(mel)` takes
  `(num_mels, frames)` or `(1, num_mels, frames)`; `.num_mels`,
  `.chunk_frames`, `.body_frames`, `.sample_rate`. The model must output
  `log_mag` and `phase` (bound by name): `exp`/`cos`/`sin` are done on the CPU,
  because a quantising compiler's `cos`/`sin` ruin the phase.

### Speech synthesis (Matcha-TTS + Vocos)

Chinese text in, 16 kHz speech out, every network on the NPU. Non-autoregressive:
each sentence is one encoder pass, one decoder pass (three flow-matching steps
compiled into it) and one or two vocoder chunks.

```python
tts = rcdl.TextToSpeech.from_files(
    "models/matcha_zh_en_enc_l128_fp16_rk3588.rknn",
    "models/matcha_zh_en_dec_t512_fp16_rk3588.rknn",
    "models/matcha_zh_en_embedding.npy",
    "models/vocos_16k_t256_fp16_rk3588.rknn",
    "models/matcha_zh_en_tokens.txt",
    "models/matcha_zh_en_lexicon.txt")
r = tts.synthesize("好的，马上出发。")      # r.pcm float32, r.sample_rate 16000

# or stage by stage:
fe = rcdl.MatchaFrontend("models/matcha_zh_en_tokens.txt", "models/matcha_zh_en_lexicon.txt")
for ids in fe.encode(text):                  # one id list per sentence
    m = acoustic.synthesize(ids)             # MatchaAcoustic -> m.mel (80, frames)
    pcm = vocoder.synthesize(m.mel)
```

- **`MatchaFrontend(tokens, lexicon, config=MatchaFrontendConfig())`** —
  `.encode(text) -> list[list[int]]`, `.sentences(text)`,
  `MatchaFrontend.read_numbers(text)`, `.vocab_size`, `.lexicon_size`.
  Reproduces sherpa-onnx's lexicon front end for this model id for id
  (punctuation folding, sentence split/merge, longest-match lexicon lookup, the
  closing punctuation token). Digits are read out first (`90` → 九十, `12.6` →
  十二点六, `35%` → 百分之三十五, `2026年` → 二零二六年). Latin words are dropped —
  the lexicon has no English.
- **`MatchaFrontendConfig`** — `min_words` (5), `max_words` (20).
- **`MatchaConfig`** — `noise_scale` (1.0; 0 makes the output deterministic),
  `length_scale` (> 1 is slower), `seed`.
- **`MatchaAcoustic(encoder, decoder, embedding, config=None)`** —
  `.synthesize(ids) -> MatchaResult` (`mel` `(80, frames)`, `frames`,
  `durations`, `truncated`), `.seed(n)`, `.max_tokens` (128), `.max_frames`
  (512 = 8.2 s), `.num_mels`, `.vocab_size`. Longer input is truncated and
  flagged. The token embedding is looked up on the CPU from the `.npy`.
- **`TextToSpeech(frontend, acoustic, vocoder, tts_config=None)`** /
  **`.from_files(..., config=None, tts_config=None)`** — `.synthesize(text) ->
  TtsResult` (`pcm`, `sample_rate`, `sentences`, `truncated`). Sentences are
  synthesised separately and concatenated, then every pause of 0.2 s or more
  keeps `TextToSpeechConfig.silence_scale` of its length (0.2, sherpa-onnx's
  default; 1 keeps the model's pauses, which run long at punctuation).
- **`scale_silence(pcm, sample_rate, scale, min_pause_s=0.2, level=0.01)`** —
  that pause shortening on its own (`|x| <= level` counts as silence); matches
  sherpa-onnx sample for sample.

---

## Errors

Everything raises `RuntimeError` carrying the vendor's message and, for the
preprocessing layer, a description of both buffers:

```
RCDL: RGA letterbox failed: Unsupported function: src unsupport width stride 810,
bgr888 width stride should be 16 aligned! (IM_STATUS -1)
  src 810x1080 BGR888 ws=810 hs=1080 fd=-1
  dst 640x640 RGB888 ws=640 hs=640 fd=31
```

That is deliberate: the failures in this stack are almost always a buffer whose
shape the hardware will not take, so the message names the rule and both shapes.

---

See `docs/CPP_API.md` for the C++ surface, `docs/RGA.md` for the preprocessing
constraints and `docs/MODELS.md` for what each model expects.
