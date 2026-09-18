<h1 align="center">RCDL</h1>
<p align="center"><b>A vision application on Rockchip RK3588 in minutes, with every hardware block doing its job.</b><br>
One C++ / Python API over VPU codecs → RGA preprocessing → NPU inference → post-processing, zero-copy end to end.</p>

<p align="center">
<a href="https://github.com/ruisv/rcdl/releases"><img src="https://img.shields.io/github/v/release/ruisv/rcdl?color=2ea44f" alt="release"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="license"></a>
<img src="https://img.shields.io/badge/C%2B%2B-17-00599C.svg" alt="C++17">
<img src="https://img.shields.io/badge/python-3.9%E2%80%933.14-3776AB.svg" alt="python">
<img src="https://img.shields.io/badge/platform-RK3588%20%C2%B7%20linux--aarch64-0A7BBB.svg" alt="platform">
</p>

<p align="center"><b>English</b> | <a href="README.md">简体中文</a></p>

| | | |
|:--:|:--:|:--:|
| <img src="benchmarks/figures/det.jpg" width="260"> | <img src="benchmarks/figures/instance_seg.jpg" width="260"> | <img src="benchmarks/figures/pose.jpg" width="260"> |
| Detection | Instance segmentation | Pose |
| <img src="benchmarks/figures/depth.jpg" width="260"> | <img src="benchmarks/figures/ocr.jpg" width="260"> | <img src="benchmarks/figures/open_vocab_prompts.jpg" width="260"> |
| Monocular depth | OCR | Open-vocabulary detection |

All 18 tasks are in the [gallery below](#gallery); every picture was produced on the board.

## Why RCDL

Rockchip ships three unrelated C libraries: `librknnrt` (NPU), `librga` (2-D
acceleration) and `MPP` (video codecs). Wiring them into one pipeline that never
copies a frame, and then getting each model's post-processing right, is usually
weeks of work. RCDL is that work, done:

- **Ten lines to a result.** An `Engine` plus a task class. Detection, segmentation, pose, OCR, faces, depth, tracking — 18 tasks work out of the box.
- **Each block does its own job.** The model runs on the NPU, scaling and colour conversion on RGA, codecs on the VPU. The CPU does post-processing, and the fallback when the hardware cannot.
- **Actually zero-copy.** The decoded frame, RGA, the NPU input tensor and the encoder share one dma-buf. There is no `memcpy` between them.
- **All three NPU cores.** The async pipeline takes detection from 99 fps to 481 fps, with results identical to the synchronous one and in order.
- **Reproducible.** Every decoder has a numpy reference test, and 584 board tests cover the full hardware path. The things RK3588's documentation does not tell you — RGA's 4 GB line, two RGA generations that resample differently — are already handled, and written up in [`docs/RGA.md`](docs/RGA.md).

RCDL is built on Rockchip's official runtimes. It does not replace them.

## Install

On the board (linux-aarch64), one command:

```bash
conda create -n rcdl -c https://mirrors.ruis.ai/conda -c conda-forge rcdl
conda activate rcdl
python -c "import rcdl; print(rcdl.__version__, rcdl.rga_version())"
```

That brings the Python bindings, the C++ library and the Rockchip userspace
libraries (`librknnrt` 2.3.2, `librga`, `rockchip-mpp`). The board image only
has to provide the kernel drivers. For C++ only, install `librcdl`.

**Requirements:** an RK3588 / RK3588S board, Linux aarch64, RKNPU driver ≥ 0.9.x,
and your user in the `video` and `render` groups. RK3576 / RK356x are not yet verified.
Building from source, permissions and troubleshooting are in
[`docs/INSTALL.md`](docs/INSTALL.md).

## Quick start

You need a `.rknn` compiled for your SoC — see [Models](#models).

**Find objects in an image (Python)**

```python
import cv2, rcdl

det = rcdl.Engine("yolov8n_rk3588.rknn").detector(model_input="rgb888")
for d in rcdl.detect(det, cv2.imread("bus.jpg")):
    print(rcdl.coco_class_name(d.class_id), f"{d.score:.2f}", d.x1, d.y1, d.x2, d.y2)
```

Boxes come back in original-image pixels. Grids, class count, DFL and channel
layout are read from the model, and a model that does not match raises at
construction instead of quietly decoding garbage.

**Video in, annotated video out, and the frame never leaves the hardware**

```python
enc = None
with open("annotated.h264", "wb") as out:
    for frame in rcdl.decode_video("clip.h264"):             # VPU decode; the frame is a dma-buf
        dets = det.process_frame(frame)                      # RGA writes straight into the NPU input tensor
        frame.draw_rects([(d.x1, d.y1, d.x2, d.y2) for d in dets])
        enc = enc or rcdl.VideoEncoder(width=frame.width, height=frame.height)
        while not enc.feed_frame(frame):                     # VPU encode, reading the same dma-buf
            out.write(enc.receive(5) or b"")
        while (pkt := enc.receive(0)) is not None:
            out.write(pkt)
    while (pkt := enc.flush()) is not None:
        out.write(pkt)
```

For throughput use `engine.video_detector()`: decode, preprocessing and
three-core inference all run in C++ threads, 72–97 fps on 1080p H.264 → YOLOv8n.

**C++**

```cpp
#include "rcdl/rcdl.h"

rcdl::Engine engine("yolov8n_rk3588.rknn");
rcdl::PipelineConfig cfg;
cfg.model_input = rcdl::PixelFormat::RGB888;
rcdl::DetectionPipeline pipe(engine, cfg);

for (const rcdl::Detection& d : pipe.process(frame.view()))   // frame: a decoded frame or any ImageView
  std::printf("%s %.2f\n", rcdl::cocoClassName(d.class_id), d.score);
```

```cmake
find_package(rcdl REQUIRED)
target_link_libraries(app PRIVATE rcdl::rcdl)
```

More in [`examples/`](examples): twenty-odd standalone programs covering
detection, segmentation, depth, SAM, optical flow, super-resolution and the
video pipelines.

## Tasks

| Task | Python entry point | Verified models |
|---|---|---|
| Object detection | `engine.detector()` | YOLOv8 · YOLO11 · YOLO26 |
| Open-vocabulary detection / segmentation | `engine.detector()` · `engine.instance_segmenter()` | YOLOE-11s |
| Classification | `engine.classifier()` | ResNet-18 · YOLO26-cls |
| Instance segmentation | `engine.instance_segmenter()` | YOLOv8-seg · YOLO26-seg |
| Semantic segmentation | `engine.segmenter()` | PP-LiteSeg · YOLO26-sem |
| Promptable segmentation (SAM) | `engine.prompt_segmenter()` | EdgeSAM |
| Human pose, 17 keypoints | `engine.pose_estimator()` | YOLOv8-pose · YOLO26-pose |
| Whole-body pose, 133 keypoints | `engine.wholebody_estimator()` | RTMW-s |
| Oriented boxes | `engine.obb_detector()` | YOLOv8-obb · YOLO26-obb |
| Monocular depth | `engine.depth_estimator()` | Depth-Anything-V2-Small |
| OCR (detection · direction · recognition) | `engine.text_detector()` and friends | PP-OCRv4 · v5 · v6 |
| Face detection + recognition | `engine.face_detector()` · `engine.face_recognizer()` | RetinaFace · ArcFace R50 |
| Image / person embeddings | `engine.embedder()` | SigLIP · OSNet |
| Sparse features and matching | `engine.feature_extractor()` | XFeat |
| Dense optical flow | `engine.flow_estimator()` | NeuFlow v2 |
| ×4 super-resolution | `engine.upscaler()` | Real-ESRGAN Compact |
| Panoptic driving (detection + drivable area + lanes) | `engine.anchor_detector()` · `engine.segmenter()` | YOLOP |
| Multi-object tracking | `engine.tracker()` | ByteTrack + BoT-SORT appearance association |

Every task is an engine-free pure decoder, a numpy test and a board test. The
C++ classes mirror these one to one; see [`docs/CPP_API.md`](docs/CPP_API.md).

## Performance

Measured on RK3588S (librknnrt 2.3.2, driver 0.9.8). The full table and how to
rerun it are in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

| | |
|---|---|
| ResNet-18 int8, one core / three cores | 4.0 ms · **707 fps** |
| Detection pipeline, synchronous / async on three cores | 99 fps · **481 fps** |
| 1080p H.264 decode / 4K H.265 decode | **324 fps** · **244 fps**, every frame a dma-buf |
| 1080p H.264 encode, reading decoded frames directly | **197 fps** |
| Compressed video → detections (1080p, YOLOv8n) | **72–97 fps** |

## Models

RCDL consumes compiled `.rknn` files. This repository does not distribute model weights.

- **Ready-made:** Rockchip's [rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo) has download scripts and conversion examples for the YOLO family, RetinaFace, PP-OCR, PP-LiteSeg, ResNet and more, and its YOLO export layout is the one RCDL's decoders read.
- **Your own:** convert from ONNX with [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) on an x86 host. A `.rknn` is tied to one SoC, and the toolkit version must not be newer than the runtime on the board.
- [`docs/MODELS.md`](docs/MODELS.md) lists, for every verified model, the input format, channel order, whether the activation is inside the graph, and the conversion traps actually hit. Channel order and activation placement cannot be read from a `.rknn`, and getting them wrong only costs accuracy silently.

## How it works

```
                    your application (C++ / Python)
                              ↓
   ┌──────────────────────────────────────────────────────┐
   │  RCDL                                                 │
   │  tasks · tracks · pipeline · media · preproc · backend │
   └──────────────────────────────────────────────────────┘
                              ↓
   librknnrt (RKNPU2)  ·  librga (im2d)  ·  librockchip_mpp (rk_mpi)
                              ↓
        NPU ×3   ·   RGA3 ×2 + RGA2   ·   VPU (decode / encode / JPEG)
```

One dma-buf goes the whole way. The VPU decodes into it, RGA imports it by fd
and crops, scales and converts NV12 → RGB in a single operation whose
destination is the input tensor the NPU already has bound; the same memory,
with boxes drawn on it, then goes to the VPU encoder. Caches are synchronised
only when the CPU touches a buffer, and never on the pure hardware path.

## Documentation

| | |
|---|---|
| [`docs/INSTALL.md`](docs/INSTALL.md) | installation, board requirements, building from source, troubleshooting |
| [`docs/API.md`](docs/API.md) | Python API |
| [`docs/CPP_API.md`](docs/CPP_API.md) | C++ API |
| [`docs/MODELS.md`](docs/MODELS.md) | verified models, each model's input contract, conversion notes |
| [`docs/RGA.md`](docs/RGA.md) | what RGA on RK3588 will and will not do, all of it measured |
| [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) | per-task latency and results, rerunnable on your board |
| [`CHANGELOG.md`](CHANGELOG.md) | release notes |

## Gallery

<details>
<summary>All 18 pictures</summary>

| | | |
|:--:|:--:|:--:|
| <img src="benchmarks/figures/det.jpg" width="250"> | <img src="benchmarks/figures/instance_seg.jpg" width="250"> | <img src="benchmarks/figures/semantic_seg.jpg" width="250"> |
| Detection | Instance segmentation | Semantic segmentation |
| <img src="benchmarks/figures/pose.jpg" width="250"> | <img src="benchmarks/figures/wholebody.jpg" width="250"> | <img src="benchmarks/figures/obb.jpg" width="250"> |
| Pose (17 keypoints) | Whole-body pose (133 keypoints) | Oriented boxes |
| <img src="benchmarks/figures/depth.jpg" width="250"> | <img src="benchmarks/figures/flow.jpg" width="250"> | <img src="benchmarks/figures/superres.jpg" width="250"> |
| Monocular depth | Dense optical flow | ×4 super-resolution |
| <img src="benchmarks/figures/face.jpg" width="250"> | <img src="benchmarks/figures/face_recognition.jpg" width="250"> | <img src="benchmarks/figures/reid.jpg" width="250"> |
| Faces + 5 landmarks | Face recognition | Person ReID |
| <img src="benchmarks/figures/ocr.jpg" width="250"> | <img src="benchmarks/figures/features.jpg" width="250"> | <img src="benchmarks/figures/promptable_seg.jpg" width="250"> |
| OCR | Sparse features + matching | Promptable segmentation |
| <img src="benchmarks/figures/open_vocab_prompts.jpg" width="250"> | <img src="benchmarks/figures/panoptic_drive.jpg" width="250"> | <img src="benchmarks/figures/cls.jpg" width="250"> |
| Open-vocabulary detection (`sneakers`) | Panoptic driving | Classification |

Generated on the board by `benchmarks/bench.py --figures`, in the same run as
the performance table, so the pictures and the numbers describe the same build.

</details>

## Contributing

Issues and pull requests are welcome: bugs, new task decoders, and results from
other Rockchip SoCs are all valuable. Read [`CONTRIBUTING.md`](CONTRIBUTING.md)
first. For a bug report, include the SoC, the output of `model_info`, and a
minimal reproduction.

## Acknowledgements

- **Rockchip** — the RK3588 platform, the RKNPU2 runtime and rknn-toolkit2, RGA, MPP.
- **[BCDL](https://github.com/ruisv/bcdl)** — the sister project for D-Robotics RDK boards, where RCDL's API design and post-processing come from.
- The upstream model projects: Ultralytics YOLO, PaddleOCR, Depth-Anything, EdgeSAM, RTMW, XFeat, NeuFlow, Real-ESRGAN, YOLOP, SigLIP, OSNet, RetinaFace, ArcFace.

## License

[Apache License 2.0](LICENSE). Rockchip's runtime libraries carry their own
licenses, and model weights carry those of their upstreams.
