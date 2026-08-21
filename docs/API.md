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
`"cpu"`. `lb` is the 7-tuple `(scale, pad_x, pad_y, src_w, src_h, dst_w, dst_h)`
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
| `VideoDecoder(codec="h264", format="nv12", external_buffers=True)` | `.feed(bytes)` / `.receive(timeout_ms)` / `.flush()` |
| `VideoFrame` | `.width .height .width_stride .height_stride .fd .format .pts_us`, `.to_numpy()`, `.letterbox(w, h)`, `.release()` |
| `VideoEncoder(width, height, codec="h264", bitrate_kbps=4000, rc="cbr")` | `.feed_frame(frame)` (zero copy) / `.feed(array, w, h)` / `.receive()` / `.flush()` / `.extra_data` |
| `JpegEncoder(w, h, quality=80)` / `JpegDecoder()` | `.encode(array)` / `.encode_frame(frame)` / `.decode(bytes)` |

`feed()` returning `False` is **back-pressure, not an error**: drain and retry.

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
