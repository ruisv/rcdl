<h1 align="center">RCDL</h1>
<p align="center"><b>在 Rockchip RK3588 上，几分钟做出一个跑满硬件的视觉应用。</b><br>
统一的 C++ / Python API：VPU 硬件编解码 → RGA 硬件预处理 → NPU 推理 → 后处理，整条链路零拷贝。</p>

<p align="center">
<a href="https://github.com/ruisv/rcdl/releases"><img src="https://img.shields.io/github/v/release/ruisv/rcdl?color=2ea44f" alt="release"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="license"></a>
<img src="https://img.shields.io/badge/C%2B%2B-17-00599C.svg" alt="C++17">
<img src="https://img.shields.io/badge/python-3.9%E2%80%933.14-3776AB.svg" alt="python">
<img src="https://img.shields.io/badge/platform-RK3588%20%C2%B7%20linux--aarch64-0A7BBB.svg" alt="platform">
</p>

<p align="center"><a href="README.en.md">English</a> | <b>简体中文</b></p>

| | | |
|:--:|:--:|:--:|
| <img src="benchmarks/figures/det.jpg" width="260"> | <img src="benchmarks/figures/instance_seg.jpg" width="260"> | <img src="benchmarks/figures/pose.jpg" width="260"> |
| 检测 | 实例分割 | 姿态 |
| <img src="benchmarks/figures/depth.jpg" width="260"> | <img src="benchmarks/figures/ocr.jpg" width="260"> | <img src="benchmarks/figures/open_vocab_prompts.jpg" width="260"> |
| 单目深度 | OCR | 开放词表检测 |

全部 18 类任务的效果图见[下方图库](#效果图库)，每一张都是板上实跑生成的。

## 为什么用 RCDL

Rockchip 官方给的是三套互不相干的 C 库：`librknnrt`（NPU）、`librga`（2D 加速）、
`MPP`（视频编解码）。把它们拼成一条不拷贝内存的流水线，再写对每个模型的后处理，
通常要几周。RCDL 把这件事做完了：

- **十行代码出结果。** `Engine` 加一个任务类，检测、分割、姿态、OCR、人脸、深度、跟踪等 18 类任务开箱即用。
- **硬件各司其职。** 模型在 NPU，缩放和色彩转换在 RGA，编解码在 VPU。CPU 只做后处理，以及硬件做不了时的回退。
- **真正的零拷贝。** 解码帧、RGA、NPU 输入张量、编码器共用同一块 dma-buf，中间没有 `memcpy`。
- **三个 NPU 核都用上。** 异步流水线让检测吞吐从 99 fps 到 481 fps，结果与同步版逐帧一致且保序。
- **结果可复现。** 每个解码器都有 numpy 参考测试，板上 584 个测试覆盖完整硬件路径。RK3588 上那些文档没写的坑（RGA 的 4 GB 限制、两代 RGA 核缩放结果不一致等）已经替你踩过，见 [`docs/RGA.md`](docs/RGA.md)。

RCDL 构建在 Rockchip 官方运行时之上，不是取代它。

## 安装

在板子上（linux-aarch64）一条命令：

```bash
conda create -n rcdl -c https://mirrors.ruis.ai/conda -c conda-forge rcdl
conda activate rcdl
python -c "import rcdl; print(rcdl.__version__, rcdl.rga_version())"
```

Python 绑定、C++ 库和 Rockchip 用户态库（`librknnrt` 2.3.2、`librga`、`rockchip-mpp`）一起装好，
板子镜像只需要提供内核驱动。只用 C++ 的话装 `librcdl`。

**要求：** RK3588 / RK3588S 开发板，Linux aarch64，RKNPU 驱动 ≥ 0.9.x，当前用户在 `video` 和 `render` 组里。
RK3576 / RK356x 尚未验证。源码构建、权限设置和常见问题见 [`docs/INSTALL.md`](docs/INSTALL.md)。

## 快速上手

需要一个为你的芯片编译好的 `.rknn` 模型，获取方式见[模型](#模型)一节。

**一张图片里找目标（Python）**

```python
import cv2, rcdl

det = rcdl.Engine("yolov8n_rk3588.rknn").detector(model_input="rgb888")
for d in rcdl.detect(det, cv2.imread("bus.jpg")):
    print(rcdl.coco_class_name(d.class_id), f"{d.score:.2f}", d.x1, d.y1, d.x2, d.y2)
```

框已经还原到原图坐标。网格、类别数、DFL、通道布局都从模型里读出来，模型不匹配会在构造时直接报错，而不是默默解出一堆垃圾。

**视频进，带框视频出，帧数据全程不离开硬件**

```python
enc = None
with open("annotated.h264", "wb") as out:
    for frame in rcdl.decode_video("clip.h264"):             # VPU 解码，帧是 dma-buf
        dets = det.process_frame(frame)                      # RGA 直接写进 NPU 输入张量
        frame.draw_rects([(d.x1, d.y1, d.x2, d.y2) for d in dets])
        enc = enc or rcdl.VideoEncoder(width=frame.width, height=frame.height)
        while not enc.feed_frame(frame):                     # VPU 编码，读同一块 dma-buf
            out.write(enc.receive(5) or b"")
        while (pkt := enc.receive(0)) is not None:
            out.write(pkt)
    while (pkt := enc.flush()) is not None:
        out.write(pkt)
```

要吞吐就用 `engine.video_detector()`：解码、预处理、三核推理全在 C++ 线程里，1080p H.264 → YOLOv8n 跑到 72–97 fps。

**C++**

```cpp
#include "rcdl/rcdl.h"

rcdl::Engine engine("yolov8n_rk3588.rknn");
rcdl::PipelineConfig cfg;
cfg.model_input = rcdl::PixelFormat::RGB888;
rcdl::DetectionPipeline pipe(engine, cfg);

for (const rcdl::Detection& d : pipe.process(frame.view()))   // frame：解码帧或任意 ImageView
  std::printf("%s %.2f\n", rcdl::cocoClassName(d.class_id), d.score);
```

```cmake
find_package(rcdl REQUIRED)
target_link_libraries(app PRIVATE rcdl::rcdl)
```

更多示例在 [`examples/`](examples)：检测、分割、深度、SAM、光流、超分、视频流水线等二十多个可独立编译的程序。

## 支持的任务

| 任务 | Python 入口 | 已验证的模型 |
|---|---|---|
| 目标检测 | `engine.detector()` | YOLOv8 · YOLO11 · YOLO26 |
| 开放词表检测 / 分割 | `engine.detector()` · `engine.instance_segmenter()` | YOLOE-11s |
| 分类 | `engine.classifier()` | ResNet-18 · YOLO26-cls |
| 实例分割 | `engine.instance_segmenter()` | YOLOv8-seg · YOLO26-seg |
| 语义分割 | `engine.segmenter()` | PP-LiteSeg · YOLO26-sem |
| 可提示分割（SAM） | `engine.prompt_segmenter()` | EdgeSAM |
| 人体姿态 17 点 | `engine.pose_estimator()` | YOLOv8-pose · YOLO26-pose |
| 全身姿态 133 点 | `engine.wholebody_estimator()` | RTMW-s |
| 旋转框 | `engine.obb_detector()` | YOLOv8-obb · YOLO26-obb |
| 单目深度 | `engine.depth_estimator()` | Depth-Anything-V2-Small |
| OCR（检测 · 方向 · 识别） | `engine.text_detector()` 等 | PP-OCRv4 · v5 · v6 |
| 人脸检测 + 识别 | `engine.face_detector()` · `engine.face_recognizer()` | RetinaFace · ArcFace R50 |
| 图像 / 行人特征 | `engine.embedder()` | SigLIP · OSNet |
| 稀疏特征与匹配 | `engine.feature_extractor()` | XFeat |
| 稠密光流 | `engine.flow_estimator()` | NeuFlow v2 |
| ×4 超分 | `engine.upscaler()` | Real-ESRGAN Compact |
| 全景驾驶（检测 + 可行驶区域 + 车道线） | `engine.anchor_detector()` · `engine.segmenter()` | YOLOP |
| 多目标跟踪 | `engine.tracker()` | ByteTrack + BoT-SORT 外观关联 |

每个任务都是「与引擎无关的纯函数解码器 + numpy 测试 + 板上测试」。C++ 里有一一对应的类，见 [`docs/CPP_API.md`](docs/CPP_API.md)。

## 性能

RK3588S 实测（librknnrt 2.3.2，驱动 0.9.8），完整表格和复现方法见 [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md)。

| | |
|---|---|
| ResNet-18 int8，单核 / 三核并发 | 4.0 ms · **707 fps** |
| 检测流水线，同步 / 异步三核 | 99 fps · **481 fps** |
| 1080p H.264 解码 / 4K H.265 解码 | **324 fps** · **244 fps**，每帧都是 dma-buf |
| 1080p H.264 编码（直接读解码帧） | **197 fps** |
| 压缩视频 → 检测结果（1080p，YOLOv8n） | **72–97 fps** |

## 模型

RCDL 只消费编译好的 `.rknn`，本仓库不分发模型权重。

- **现成的：** Rockchip 的 [rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo) 为 YOLO 系列、RetinaFace、PP-OCR、PP-LiteSeg、ResNet 等提供下载脚本和转换示例，它的 YOLO 导出布局正是 RCDL 解码器读的那种。
- **自己转：** 用 [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) 在 x86 主机上从 ONNX 转换。`.rknn` 和芯片型号绑定，转换用的 toolkit 版本不能高于板上运行时的版本。
- [`docs/MODELS.md`](docs/MODELS.md) 列出了每个已验证模型的输入格式、通道顺序、激活函数在图内还是图外，以及转换时实际踩过的坑。通道顺序和激活位置这两项从 `.rknn` 文件里读不出来，填错只会悄悄掉精度。

## 工作原理

```
                    你的应用（C++ / Python）
                              ↓
   ┌──────────────────────────────────────────────────────┐
   │  RCDL                                                 │
   │  tasks · tracks · pipeline · media · preproc · backend │
   └──────────────────────────────────────────────────────┘
                              ↓
   librknnrt (RKNPU2)  ·  librga (im2d)  ·  librockchip_mpp (rk_mpi)
                              ↓
        NPU ×3   ·   RGA3 ×2 + RGA2   ·   VPU（解码 / 编码 / JPEG）
```

一块 dma-buf 走完全程：VPU 把帧解码进去，RGA 按 fd 导入它，一次操作完成裁剪、缩放、NV12→RGB，
结果直接落在 NPU 已经绑定好的输入张量里；画完框的同一块内存再交给 VPU 编码。
CPU 需要碰缓冲区时才做 cache 同步，纯硬件路径上一次都不做。

## 文档

| | |
|---|---|
| [`docs/INSTALL.md`](docs/INSTALL.md) | 安装、板子要求、源码构建、常见问题 |
| [`docs/API.md`](docs/API.md) | Python API |
| [`docs/CPP_API.md`](docs/CPP_API.md) | C++ API |
| [`docs/MODELS.md`](docs/MODELS.md) | 已验证模型、各模型的输入约定、转换注意事项 |
| [`docs/RGA.md`](docs/RGA.md) | RK3588 上 RGA 能做什么、不能做什么，全部带实测数据 |
| [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) | 各任务延迟与结果，可在你的板子上重跑 |
| [`CHANGELOG.md`](CHANGELOG.md) | 版本记录 |

## 效果图库

<details>
<summary>展开全部 18 张</summary>

| | | |
|:--:|:--:|:--:|
| <img src="benchmarks/figures/det.jpg" width="250"> | <img src="benchmarks/figures/instance_seg.jpg" width="250"> | <img src="benchmarks/figures/semantic_seg.jpg" width="250"> |
| 检测 | 实例分割 | 语义分割 |
| <img src="benchmarks/figures/pose.jpg" width="250"> | <img src="benchmarks/figures/wholebody.jpg" width="250"> | <img src="benchmarks/figures/obb.jpg" width="250"> |
| 姿态（17 点） | 全身姿态（133 点） | 旋转框 |
| <img src="benchmarks/figures/depth.jpg" width="250"> | <img src="benchmarks/figures/flow.jpg" width="250"> | <img src="benchmarks/figures/superres.jpg" width="250"> |
| 单目深度 | 稠密光流 | ×4 超分 |
| <img src="benchmarks/figures/face.jpg" width="250"> | <img src="benchmarks/figures/face_recognition.jpg" width="250"> | <img src="benchmarks/figures/reid.jpg" width="250"> |
| 人脸 + 5 点 | 人脸识别 | 行人 ReID |
| <img src="benchmarks/figures/ocr.jpg" width="250"> | <img src="benchmarks/figures/features.jpg" width="250"> | <img src="benchmarks/figures/promptable_seg.jpg" width="250"> |
| OCR | 稀疏特征 + 匹配 | 可提示分割 |
| <img src="benchmarks/figures/open_vocab_prompts.jpg" width="250"> | <img src="benchmarks/figures/panoptic_drive.jpg" width="250"> | <img src="benchmarks/figures/cls.jpg" width="250"> |
| 开放词表检测（`sneakers`） | 全景驾驶 | 分类 |

这些图由 `benchmarks/bench.py --figures` 在板上生成，和性能表来自同一次运行，
所以图和数字描述的是同一个构建。

</details>

## 参与贡献

欢迎提 issue 和 PR：bug、新的任务解码器、其他 Rockchip 芯片上的验证结果都很有价值。
动手前请看 [`CONTRIBUTING.md`](CONTRIBUTING.md)。报 bug 时请附上芯片型号、`model_info` 的输出和最小复现。

## 致谢

- **Rockchip**：RK3588 平台、RKNPU2 运行时与 rknn-toolkit2、RGA、MPP。
- **[BCDL](https://github.com/ruisv/bcdl)**：D-Robotics RDK 上的姊妹项目，RCDL 的上层 API 设计和后处理算法源自它。
- Ultralytics YOLO、PaddleOCR、Depth-Anything、EdgeSAM、RTMW、XFeat、NeuFlow、Real-ESRGAN、YOLOP、SigLIP、OSNet、RetinaFace、ArcFace 等上游模型项目。

## 许可证

[Apache License 2.0](LICENSE)。Rockchip 的运行时库遵循其各自的许可证；模型权重遵循各自上游的许可证。
