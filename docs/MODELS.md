# Models

RCDL consumes finished `.rknn` files. Conversion from ONNX (rknn-toolkit2, PTQ,
`accuracy_analysis`, hybrid quantization) runs on an **x86 host** and is owned by
a separate model-zoo project — the toolkit is x86-only, and the board carries
only the runtime.

`scripts/fetch_models.sh` stages the registry below into `models/` (gitignored).
Paths come from `scripts/local.env`; anything unset is reported as MISSING rather
than silently skipped.

```bash
scripts/fetch_models.sh --list     # show the registry
scripts/fetch_models.sh            # stage everything it can find
```

## What the decoders need to be told

Two model properties are **not** discoverable from the `.rknn` file and must be
recorded per model. Getting either wrong degrades accuracy silently rather than
failing:

| Property | Where it goes | How to tell |
|---|---|---|
| **Input channel order** (`rgb` vs `bgr`) | `PipelineConfig::model_input` | A property of the export (`--input-order`). The rknn_model_zoo YOLO exports are **RGB**; an OpenCV-native build is usually BGR. Wrong order costs a few points of confidence and shifts marginal detections. |
| **Whether the activation is in the graph** | `apply_sigmoid` / `apply_softmax` | Run the model once and look at the raw head output: values in [0,1] that behave like probabilities mean the export folded the sigmoid in. `looksLikeProbabilities()` in `tasks/classification.h` does this check. |

Everything else — grid sizes, class count, DFL `reg_max`, strides, NCHW vs NHWC,
quantization scale and zero point — is read **from the model** at construction by
`resolveYoloHead()` and its siblings, which throw with the full output signature
printed when a model does not match the head they were asked for. A mismatched
model fails loudly instead of decoding garbage.

## Registry

Measured on RK3588 (librknnrt 2.3.2, driver 0.9.8), converted with
rknn-toolkit2 2.3.2 from the airockchip model zoo.

| Model | Task | Input | Order | Head layout |
|---|---|---|---|---|
| `yolov8n_rk3588.rknn` | detection | 640×640 u8 | rgb | 9 outputs: per scale `box[1,64,H,W]`, `cls[1,80,H,W]`, `sum[1,1,H,W]`, NCHW. **Sigmoid in-graph.** |
| `yolo11n_rk3588.rknn` | detection | 640×640 u8 | rgb | same as yolov8n |
| `yolov8n-seg_rk3588.rknn` | instance seg | 640×640 u8 | rgb | 13 outputs: per scale `box(64)`, `cls(80)`, `sum(1)`, `mc(32)` + `proto[1,32,160,160]`, NCHW |
| `yolov8n-pose_rk3588.rknn` | pose | 640×640 u8 | rgb | 4 outputs: per scale **fused** `[1,65,H,W]` = 64 DFL + 1 class, plus `keypoints[1,17,3,8400]` f16. **Sigmoid applied on the CPU.** |
| `yolov8n-obb_rk3588.rknn` | oriented boxes | 640×640 u8 | rgb | 4 outputs: per scale **fused** `[1,79,H,W]` = 64 DFL + 15 DOTA classes, plus `angle[1,1,8400]` (sigmoid in-graph; decode as `(a − 0.25)·π`). **Class sigmoid on the CPU.** |
| `ppocrv4_det_rk3588.rknn` | text detection | DBNet probability map | — | thresholded map → contours → unclipped quads |
| `ppocrv4_rec_rk3588.rknn` | text recognition | CRNN | — | CTC greedy decode against `data/ppocr_keys_*.txt` |
| `ppseg_rk3588.rknn` | semantic seg | 512×512 u8 | rgb | `[1,19,512,512]` NCHW int8 **logits** (PP-LiteSeg, Cityscapes 19 classes) — argmax on the CPU |
| `retinaface_rk3588.rknn` | face + 5 landmarks | 320×320 u8 | **bgr** | anchor-based SSD head: `boxes[1,4200,4]`, `scores[1,4200,2]` (**softmax in-graph**), `landmarks[1,4200,10]`. 4200 priors = `(40²+20²+10²)×2` from `steps {8,16,32}`, `min_sizes {{16,32},{64,128},{256,512}}`, variances `[0.1, 0.2]` |
| `resnet18_rk3588.rknn` | classification | 224×224 u8 | rgb | `[1,1000]`, softmax on the CPU |

**RetinaFace wants BGR.** The vendor's Python demo converts BGR→RGB before
inference, but the converted `.rknn` does not behave that way: fed RGB it finds
only one of the two faces in `zidane.jpg` (the second peaks at 0.089, below any
usable threshold); fed BGR it finds both at 0.995 and 0.947. It does not error
and the landmarks do not scatter — it just silently loses most of the recall.
This is exactly the class of thing the "input channel order" column exists for.

**Note the pose and OBB exports differ structurally from detection**: they
concatenate the box and class channels into one tensor per scale, and they do
*not* fold the class sigmoid into the graph. That is not a quirk of one export —
it is what the vendor's own reference implementation reads — so the decoders
support both the fused and the separate-branch shapes and choose from the
model's output signature.

## Measured performance

RK3588S, single-frame latency unless stated:

| Workload | Result |
|---|---|
| ResNet-18 int8, one NPU core | 4.0 ms |
| ResNet-18 int8, three pinned contexts | 707 fps aggregate |
| Detection pipeline, ResNet-18 canvas, synchronous | 99 fps |
| Same, `AsyncDetectionPipeline` with 3 workers | 481 fps (4.86×) |
| 1080p H.264 decode | 324 fps |
| 4K H.265 decode | 244 fps |
| 1080p H.264 encode from decoded dma-bufs | 197 fps |
| decode → encode → decode luma PSNR | 47.2 dB |
| JPEG decode (810×1080) | 4.1 ms; matches libjpeg to 0.02 mean luma |
| YOLOv8n 640, full VPU → RGA → NPU → overlay → VPU per frame | 26.9 ms (37.2 fps); post-processing is 4.1 ms of that, down from 29.9 ms before the score-sum pre-filter |
| PP-LiteSeg 2048×1024 → 512×512 → full-res label map | 77 ms (dominated by the CPU upscale of the label map, not the NPU) |

## Adding a model

1. Convert it on the x86 host and copy the `.rknn` to wherever
   `RCDL_CONVERT_MODELS` points.
2. Add a `name|convert|path` line to `REGISTRY` in `scripts/fetch_models.sh`.
3. Add a row to the table above, **including the input order and where the
   activation lives** — that is the information the file itself does not carry.
4. Add a board test that skips cleanly when the model is absent
   (`tests/board_models.py::require_model`), so the suite stays green on a
   machine that has only some of the registry staged.

## A note on class-name tables

`segmentation.h` ships **VOC-21** names and palette, but `ppseg_rk3588.rknn` is a
**Cityscapes-19** model — the label indices are correct either way (argmax is
argmax), but the names and colours will not match that model. Pass your own
table rather than trusting `vocClassName()` for a model that is not VOC-trained.
The same caution applies to `cocoClassName()` for any detector not trained on
COCO, and to the DOTA-15 names for OBB.
