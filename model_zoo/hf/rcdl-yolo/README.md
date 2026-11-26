---
license: agpl-3.0
library_name: rcdl
base_model:
- Ultralytics/YOLOv8
- Ultralytics/YOLO11
- Ultralytics/YOLO26
pipeline_tag: object-detection
tags:
- rknn
- rockchip
- rk3588
- npu
- edge
- yolo
- instance-segmentation
- pose-estimation
- oriented-object-detection
- image-classification
---

# YOLOv8n · YOLO11n · YOLO26n for RK3588: detection, segmentation, pose, oriented boxes, classification

Compiled RKNN models (`.rknn`) for the **Rockchip RK3588 / RK3588S** NPU, ready
to load: no ONNX export, no calibration, no rknn-toolkit2. Every file is an int8
build from rknn-toolkit2 2.3.2, decoded and measured with
[**RCDL**](https://github.com/ruisv/rcdl). RCDL is a C++17 / Python library that
drives the NPU, RGA and VPU on these boards as one zero-copy pipeline.

Upstream: [Ultralytics](https://github.com/ultralytics/ultralytics) YOLOv8n,
YOLO11n and YOLO26n. Training data per task: COCO for detection, segmentation and
pose; DOTA v1 for oriented boxes; ImageNet for classification.

> [!WARNING]
> **AGPL-3.0.** These are Ultralytics models, and the compiled builds carry the
> same licence. Software that ships them or serves them over a network falls
> under AGPL-3.0 too. A closed-source or commercial product needs an
> [Ultralytics Enterprise licence](https://www.ultralytics.com/license). See
> [Licence](#licence).

## Files

| file | task | size |
|---|---|---|
| `yolov8n_rk3588.rknn` | detection, 80 COCO classes | 4.1 MB |
| `yolo11n_rk3588.rknn` | detection, 80 COCO classes | 4.1 MB |
| `yolo26n_rk3588.rknn` | detection, 80 COCO classes, NMS-free head | 4.1 MB |
| `yolov8n-seg_rk3588.rknn` | instance segmentation | 4.5 MB |
| `yolo26n-seg_rk3588.rknn` | instance segmentation | 4.6 MB |
| `yolov8n-pose_rk3588.rknn` | person pose, 17 keypoints | 5.0 MB |
| `yolo26n-pose_rk3588.rknn` | person pose, 17 keypoints | 4.7 MB |
| `yolov8n-obb_rk3588.rknn` | oriented boxes, 15 DOTA classes | 4.2 MB |
| `yolo26n-obb_rk3588.rknn` | oriented boxes, 15 DOTA classes | 4.2 MB |
| `yolo26n-cls_rk3588.rknn` | classification, 1000 ImageNet classes | 3.5 MB |

Each model takes uint8 **RGB** at 640×640 (224×224 for the classifier). The
`/255` is inside the model. RCDL letterboxes the frame into the input tensor on
the RGA and maps the boxes back to source pixels.

## Measured on RK3588S

| model | NPU | whole task | result on the test image |
|---|---|---|---|
| `yolov8n` | 18.4 ms | 33.3 ms | 1 bus, 4 people |
| `yolo11n` | 27.7 ms | 67.9 ms | 1 bus, 4 people |
| `yolo26n` | 36.9 ms | 70.5 ms | 1 bus, 4 people |
| `yolov8n-seg` | 31.3 ms | 122.1 ms | 5 instances |
| `yolov8n-pose` | 35.2 ms | 79.7 ms | 4 people |
| `yolov8n-obb` | 21.7 ms | 70.0 ms | 33 rotated boxes |
| `yolo26n-cls` | 2.9 ms | 3.8 ms | space shuttle, 0.931 |

The table is from RCDL's
[benchmark](https://github.com/ruisv/rcdl/blob/main/benchmarks/RESULTS.md):
librknnrt 2.3.2, NPU driver 0.9.8, one NPU core, median of five runs after a
warm-up. **NPU** is the runtime's own measurement of inference alone. **Whole
task** adds preprocessing and the CPU decode.

**Newer is not faster on this NPU.** YOLO11n and YOLO26n both infer slower than
YOLOv8n here and find the same five objects. YOLO26's gain is on the CPU: its
head has no DFL to reduce.

## Use it

On the board (linux-aarch64):

```bash
conda create -n rcdl -c https://mirrors.ruis.ai/conda -c conda-forge rcdl
conda activate rcdl
curl -LO https://huggingface.co/ruisv/rcdl-yolo/resolve/main/yolov8n_rk3588.rknn
```

```python
import cv2, rcdl

det = rcdl.Engine("yolov8n_rk3588.rknn").detector(model_input="rgb888")
for d in rcdl.detect(det, cv2.imread("bus.jpg")):
    print(rcdl.coco_class_name(d.class_id), f"{d.score:.2f}", d.x1, d.y1, d.x2, d.y2)
```

The other heads, and the three settings YOLO26 needs:

```python
seg  = rcdl.Engine("yolo26n-seg_rk3588.rknn").instance_segmenter()
pose = rcdl.Engine("yolo26n-pose_rk3588.rknn").pose_estimator(
           kpt_decode="cell_relative_whole", kpt_apply_sigmoid=True)
obb  = rcdl.Engine("yolo26n-obb_rk3588.rknn").obb_detector(angle_bias=0.0, angle_scale=1.0)
cls  = rcdl.Engine("yolo26n-cls_rk3588.rknn").classifier(apply_softmax=False)
```

`hf download ruisv/rcdl-yolo --local-dir .` fetches the whole set, which the
second snippet needs.

## What to know before deploying

**The head layout is read from the file.** Grid sizes, class count, whether the
box branch uses DFL (v8, v11) or plain LTRB (v26), channel order and
quantization all come from the model. A file that does not match the decoder it
was handed fails at construction, with its output signature printed. It does
not decode garbage.

**Three YOLO26 conventions are not in the file.** The library default is the v8
convention, and using it on YOLO26 still produces output that looks plausible:

- **Pose:** YOLO26 dropped the doubling in the keypoint offset. Use
  `kpt_decode="cell_relative_whole"`. With the default, joints land outside
  their own person's box.
- **Oriented boxes:** YOLO26 regresses the angle in radians. Use
  `angle_bias=0, angle_scale=1`. The v8 default is `(sigmoid − 0.25)·π`.
- **Classification:** the softmax is in the graph. Use `apply_softmax=False`.

**These are plain int8 builds, and int8 can lower YOLO26's scores.** The
YOLO26 files use rknn-toolkit2's default `normal` calibration. On a separate
YOLO26 model, that recipe lowered class scores to about 0.64× the PyTorch
model's, which costs recall at a fixed confidence threshold. Two changes
recovered almost all of it: `kl_divergence` calibration, and
fp16 for the last two stages of the classification branch plus the box branch.
If recall at your threshold matters, compare scores against the `.pt` on your
own images before choosing the threshold.

**The output layout is not the stock Ultralytics export.** Each scale's box,
class and score branches are separate tensors: the layout from Rockchip's
[rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo). That split keeps
the int8 quantization of the boxes separate from the scores.

- **v8 and v11:** exported and converted the rknn_model_zoo way.
- **YOLO26:** cut at the same place from its one-to-one, NMS-free branch. The
  detection, segmentation, pose and oriented-box builds are calibrated on
  rknn_model_zoo's 20-image COCO subset; the classifier is calibrated on
  ImageNet images.

## Licence

**AGPL-3.0**, inherited from the Ultralytics weights. Redistributing these files
is permitted under its terms. Combining them with your own software brings that
software under the same terms. Ultralytics sells an Enterprise licence for
closed or commercial use.

**RCDL itself is Apache-2.0 and is unrelated to these terms.** It is a
general-purpose runtime that loads any `.rknn`. The licence above applies to
*these weights and these compiled files* only.

How each build was made, and the input contract and decoder settings for every
model, are documented in RCDL's
[`docs/MODELS.md`](https://github.com/ruisv/rcdl/blob/main/docs/MODELS.md). The
licence status of every model RCDL has verified is in
[`model_zoo/`](https://github.com/ruisv/rcdl/tree/main/model_zoo).
