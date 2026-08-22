# RCDL on-board benchmark

RK3588S · librknnrt 2.3.2 · NPU driver 0.9.8. Regenerate with

```bash
PYTHONPATH=build:python python benchmarks/bench.py \
    --json benchmarks/results.json --markdown benchmarks/RESULTS.md
```

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
| det | 21.79 | 45.3 | 4.1 | 1 bus, 4 person |
| det_yolo11 | 32.43 | 77.2 | 4.0 | 1 bus, 4 person |
| det_yolo26 | 36.04 | 78.4 | 4.1 | 1 bus, 4 person |
| cls | 4.69 | 5.2 | 11.4 | 812:0.949, 404:0.011, 627:0.003 |
| cls_yolo26 | 3.60 | 3.7 | 3.5 | 812:0.931, 404:0.003, 867:0.002 |
| instance_seg | 32.84 | 128.8 | 4.5 | 5 instances |
| semantic_seg | 75.07 | 145.6 | 9.1 | 810x1080 map, 9 classes present |
| pose | 38.02 | 82.1 | 5.0 | 4 people, 43 joints over 0.5 |
| obb | 28.59 | 79.5 | 4.2 | 33 rotated boxes |
| depth | 276.51 | 298.0 | 28.3 | 810x1080 disparity [0.00,0.90] |
| ocr | 39.60 | 1108.1 | 9.2 | 16 boxes, 15 lines read |
| face | 5.34 | 9.2 | 18.0 | 2 faces, best 0.995 |
| reid | 13.65 | 15.4 | 2.3 | 4 crops, cross-similarity max 0.471 |
| features | 33.20 | 79.1 | 1.3 | 4096+4096 features, 1989 matches (+199 ms to match) |
| superres | 62.87 | 78.4 | 3.5 | 128x128 -> 512x512, 1 tile(s) |
| flow | 1493.66 | 1504.7 | 263.2 | 512x384 field, EPE 0.103 px vs an 8 px shift |
| promptable_seg | 364.57 | 585.7 | 33.3 | box -> 28.7% of the frame @ 0.969 (encode 438 ms + prompt 147 ms) |
| wholebody | 31.33 | 33.8 | 32.1 | 133/133 keypoints over 0.3, one person |
| face_recognition | 26.73 | 31.1 | 83.9 | 4 faces, worst cross-identity similarity 0.072 |
| open_vocab | 44.60 | 96.4 | 10.7 | 80 prompts -> 1 bus, 4 person, 1 stop sign, 1 tie |
| open_vocab_prompts | 51.59 | 82.1 | 10.7 | 6 prompts -> 4 sneakers |
| panoptic_drive | 110.45 | 180.1 | 9.8 | 18 vehicles, drivable 21.5%, lane 1.8% of the frame |
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
