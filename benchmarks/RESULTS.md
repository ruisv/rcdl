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
| det | 25.15 | 54.0 | 4.1 | 1 bus, 4 person |
| det_yolo11 | 32.87 | 82.0 | 4.0 | 1 bus, 4 person |
| det_yolo26 | 37.63 | 78.5 | 4.1 | 1 bus, 4 person |
| cls | 4.62 | 5.0 | 11.4 | 812:0.949, 404:0.011, 627:0.003 |
| cls_yolo26 | 2.77 | 2.6 | 3.5 | 812:0.931, 404:0.003, 867:0.002 |
| instance_seg | 33.61 | 100.9 | 4.5 | 5 instances |
| semantic_seg | 68.15 | 128.6 | 9.1 | 810x1080 map, 9 classes present |
| pose | 47.02 | 106.2 | 5.0 | 4 people, 43 joints over 0.5 |
| obb | 28.17 | 75.9 | 4.2 | 33 rotated boxes |
| depth | 318.67 | 324.0 | 28.3 | 810x1080 disparity [0.00,0.90] |
| ocr | 35.55 | 1319.3 | 9.2 | 16 boxes, 15 lines read |
| face | 5.54 | 9.3 | 18.0 | 2 faces, best 0.995 |
| reid | 7.16 | 18.5 | 2.3 | 4 crops, cross-similarity max 0.471 |
| features | 42.91 | 96.7 | 1.3 | 4096+4096 features, 1989 matches (+200 ms to match) |
| superres | 62.97 | 79.3 | 3.5 | 128x128 -> 512x512, 1 tile(s) |
| flow | 1683.47 | 1508.9 | 263.2 | 512x384 field, EPE 0.103 px vs an 8 px shift |
| promptable_seg | 263.32 | 587.4 | 33.3 | box -> 28.7% of the frame @ 0.969 (encode 434 ms + prompt 153 ms) |
| wholebody | 33.01 | 37.7 | 32.1 | 133/133 keypoints over 0.3, one person |
| face_recognition | 25.33 | 30.6 | 83.9 | 4 faces, worst cross-identity similarity 0.072 |
| open_vocab | 53.05 | 101.2 | 10.7 | 80 prompts -> 1 bus, 4 person, 1 stop sign, 1 tie |
| open_vocab_prompts | 51.25 | 84.5 | 10.7 | 6 prompts -> 4 sneakers |
| panoptic_drive | 130.94 | 190.1 | 9.8 | 18 vehicles, drivable 21.5%, lane 1.8% of the frame |
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
