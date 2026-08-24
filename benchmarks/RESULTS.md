# RCDL on-board benchmark

RK3588S · librknnrt 2.3.2 · NPU driver 0.9.8. Regenerate with

```bash
PYTHONPATH=build:python python benchmarks/bench.py --figures \
    --json benchmarks/results.json --markdown benchmarks/RESULTS.md
```

`--figures` also draws one annotated check image per task into
`benchmarks/figures/` (the gallery in the README). They come out of the SAME run
as the numbers, so a figure cannot end up describing a different build from the
row beside it; a task whose figure fails to draw is re-measured without it
rather than dropped from the table.

Every row runs a real `.rknn` from `models/` on a real sample image. **`infer` is
the NPU alone** (the runtime's own `RKNN_QUERY_PERF_RUN`); **`e2e` is the whole
task** — preprocessing, inference and post-processing — because that is what a
caller waits for, and for several heads here the post-processing is the larger
half. Both are medians over five runs after a warm-up, so they are what a steady
stream costs rather than a best case.

**The `result` column is the point.** A model that got faster and stopped finding
the bus should not look like an improvement, so every row states what it actually
produced. A model that is not staged is reported as skipped rather than dropped
(see `scripts/fetch_models.sh`).

<!-- BENCH:BEGIN -->
| task | infer ms | e2e ms | model MB | result |
|---|---|---|---|---|
| det | 19.43 | 47.4 | 4.1 | 1 bus, 4 person |
| det_yolo11 | 32.88 | 79.7 | 4.0 | 1 bus, 4 person |
| det_yolo26 | 29.15 | 61.9 | 4.1 | 1 bus, 4 person |
| cls | 4.76 | 5.5 | 11.4 | 812:0.949, 404:0.011, 627:0.003 |
| cls_yolo26 | 5.37 | 6.0 | 3.5 | 812:0.931, 404:0.003, 867:0.002 |
| instance_seg | 30.54 | 127.3 | 4.5 | 5 instances |
| semantic_seg | 53.93 | 103.6 | 9.1 | 2048x1024 map, 11 classes present |
| semantic_seg_yolo26 | 22.34 | 45.2 | 2.6 | 2048x1024 map, 10 classes present |
| pose | 46.30 | 86.1 | 5.0 | 4 people, 43 joints over 0.5 |
| obb | 29.98 | 70.9 | 4.2 | 33 rotated boxes |
| depth | 258.81 | 283.1 | 28.3 | 810x1080 disparity [0.00,0.90] |
| ocr | 37.22 | 1231.1 | 9.2 | 16 boxes, 15 lines read |
| face | 6.48 | 8.9 | 18.0 | 2 faces, best 0.995 |
| reid | 11.05 | 20.4 | 2.3 | 4 crops, cross-similarity max 0.471 |
| features | 33.51 | 85.6 | 1.3 | 4096+4096 features, 1989 matches (+195 ms to match) |
| superres | 63.11 | 80.4 | 3.5 | 128x128 -> 512x512, 1 tile(s) |
| flow | 1454.03 | 1420.9 | 263.2 | 512x384 field, EPE 0.103 px vs an 8 px shift |
| promptable_seg | 307.06 | 652.3 | 33.3 | box -> 28.7% of the frame @ 0.969 (encode 461 ms + prompt 191 ms) |
| wholebody | 39.59 | 47.4 | 32.1 | 133/133 keypoints over 0.3, one person |
| face_recognition | 27.37 | 31.6 | 83.9 | 4 faces, worst cross-identity similarity 0.072 |
| open_vocab | 54.85 | 112.5 | 10.7 | 80 prompts -> 1 bus, 4 person, 1 stop sign, 1 tie |
| open_vocab_prompts | 38.69 | 87.0 | 10.7 | 6 prompts -> 4 sneakers |
| open_vocab_seg | 63.24 | 186.6 | 11.5 | 80 prompts -> 1 bus, 4 person, 1 stop sign, 1 tie |
| panoptic_drive | 139.23 | 208.8 | 9.8 | 18 vehicles, drivable 21.5%, lane 1.8% of the frame |
<!-- BENCH:END -->

Reading the table:

* **Post-processing dominates more often than inference does.** `instance_seg`
  spends three quarters of its time assembling masks; `features` spends ~190 ms
  matching 4096 against 4096 descriptors, against ~44 ms of NPU; `ocr`'s `e2e` is
  16 crops through the recogniser, one inference each, which is why one number
  there is 30x the other.
* **The two slow rows are slow for stated reasons, not mysteries.** `flow` is
  nine `GridSample` calls on the CPU between NPU subgraphs, each moving the whole
  correlation volume; `promptable_seg` is an encoder that runs once per frame
  (~424 ms) plus a decoder that runs once per prompt (~146 ms), and the split is
  the whole point of that head. Both are documented in
  [`docs/MODELS.md`](../docs/MODELS.md).
* **`wholebody` is per PERSON**, not per frame — top-down, unlike `pose`, which
  answers for everybody in one pass.
* **Newer is not faster on this NPU.** YOLO11n and YOLO26n both infer slower than
  YOLOv8n here while finding the same 1 bus + 4 people; YOLO26's win is in
  post-processing, where it has no DFL to reduce.
* **`panoptic_drive` is ONE inference and three decoders** — an anchor-based
  detector plus two full-frame masks — so `infer` is a single NPU pass and the
  gap to `e2e` covers the anchor decode and both mask projections.
* **`open_vocab` costs what any other detector costs**, and the vocabulary size
  does not change that: YOLOE's text comparison happened at conversion time, so
  the board runs an ordinary LTRB head. The `result` column is the interesting
  part — the six-prompt build finds `sneakers`, which COCO has no class for.
* Model size is the `.rknn` on disk. `flow` is 263 MB because the conversion
  bakes the correlation grids in as constants; it is not a big network.

The accuracy numbers behind the `result` strings — what each build was measured
against, and why several ship float rather than int8 — are in
[`docs/MODELS.md`](../docs/MODELS.md).
