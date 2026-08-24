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
| det | 18.40 | 33.3 | 4.1 | 1 bus, 4 person |
| det_yolo11 | 27.70 | 67.9 | 4.0 | 1 bus, 4 person |
| det_yolo26 | 36.88 | 70.5 | 4.1 | 1 bus, 4 person |
| cls | 4.61 | 5.7 | 11.4 | 812:0.949, 404:0.011, 627:0.003 |
| cls_yolo26 | 2.94 | 3.8 | 3.5 | 812:0.931, 404:0.003, 867:0.002 |
| instance_seg | 31.26 | 122.1 | 4.5 | 5 instances |
| semantic_seg | 51.13 | 128.2 | 9.1 | 2048x1024 map, 11 classes present |
| semantic_seg_yolo26 | 19.49 | 45.1 | 2.6 | 2048x1024 map, 10 classes present |
| pose | 35.15 | 79.7 | 5.0 | 4 people, 43 joints over 0.5 |
| obb | 21.67 | 70.0 | 4.2 | 33 rotated boxes |
| depth | 264.08 | 283.5 | 28.3 | 810x1080 disparity [0.00,0.90] |
| ocr | 41.77 | 1313.5 | 9.2 | 16 boxes, 15 lines read |
| ocr_v6 | 62.09 | 1727.5 | 55.3 | 16 boxes, 15 lines read |
| face | 5.46 | 8.2 | 18.0 | 2 faces, best 0.995 |
| reid | 11.86 | 23.0 | 2.3 | 4 crops, cross-similarity max 0.471 |
| features | 43.98 | 95.1 | 1.3 | 4096+4096 features, 1989 matches (+192 ms to match) |
| superres | 63.05 | 80.0 | 3.5 | 128x128 -> 512x512, 1 tile(s) |
| flow | 1404.33 | 1505.8 | 263.2 | 512x384 field, EPE 0.103 px vs an 8 px shift |
| promptable_seg | 263.94 | 555.9 | 33.3 | box -> 28.7% of the frame @ 0.969 (encode 406 ms + prompt 150 ms) |
| wholebody | 35.62 | 38.7 | 32.1 | 133/133 keypoints over 0.3, one person |
| face_recognition | 27.73 | 35.9 | 83.9 | 4 faces, worst cross-identity similarity 0.072 |
| open_vocab | 60.40 | 104.5 | 10.7 | 80 prompts -> 1 bus, 4 person, 1 stop sign, 1 tie |
| open_vocab_prompts | 52.59 | 91.6 | 10.7 | 6 prompts -> 4 sneakers |
| open_vocab_seg | 64.87 | 171.7 | 11.5 | 80 prompts -> 1 bus, 4 person, 1 stop sign, 1 tie |
| panoptic_drive | 142.34 | 225.7 | 9.8 | 18 vehicles, drivable 21.5%, lane 1.8% of the frame |
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
