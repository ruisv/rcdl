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
| det | 22.19 | 51.8 | 4.1 | 1 bus, 4 person |
| det_yolo11 | 31.89 | 76.6 | 4.0 | 1 bus, 4 person |
| det_yolo26 | 37.53 | 79.4 | 4.1 | 1 bus, 4 person |
| cls | 4.89 | 5.1 | 11.4 | 812:0.949, 404:0.011, 627:0.003 |
| cls_yolo26 | 1.93 | 2.2 | 3.5 | 812:0.931, 404:0.003, 867:0.002 |
| instance_seg | 34.49 | 130.6 | 4.5 | 5 instances |
| semantic_seg | 69.30 | 145.3 | 9.1 | 810x1080 map, 9 classes present |
| pose | 44.89 | 82.0 | 5.0 | 4 people, 43 joints over 0.5 |
| obb | 32.35 | 75.4 | 4.2 | 33 rotated boxes |
| depth | 313.94 | 295.7 | 28.3 | 810x1080 disparity [0.00,0.90] |
| ocr | 40.01 | 1346.1 | 9.2 | 16 boxes, 15 lines read |
| face | 6.07 | 9.3 | 18.0 | 2 faces, best 0.995 |
| reid | 8.03 | 16.5 | 2.3 | 4 crops, cross-similarity max 0.471 |
| features | 43.56 | 98.7 | 1.3 | 4096+4096 features, 1989 matches (+190 ms to match) |
| superres | 62.69 | 78.8 | 3.5 | 128x128 -> 512x512, 1 tile(s) |
| flow | 1431.00 | 1452.6 | 263.2 | 512x384 field, EPE 0.103 px vs an 8 px shift |
| promptable_seg | 293.81 | 636.1 | 33.3 | box -> 28.7% of the frame @ 0.969 (encode 438 ms + prompt 198 ms) |
| wholebody | 34.68 | 40.5 | 32.1 | 133/133 keypoints over 0.3, one person |
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
* Model size is the `.rknn` on disk. `flow` is 263 MB because the conversion
  bakes the correlation grids in as constants; it is not a big network.

The accuracy numbers behind the `result` strings — what each build was measured
against, and why several ship float rather than int8 — are in
[`docs/MODELS.md`](../docs/MODELS.md).
