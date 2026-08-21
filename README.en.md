# RCDL — Rockchip RKNPU vision framework

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](CMakeLists.txt)
[![Python](https://img.shields.io/badge/python-3.9%E2%80%933.14-3776AB.svg)](pyproject.toml)
[![Platform](https://img.shields.io/badge/platform-RK3588%20%2F%20RK3576%20%2F%20RK356x%20(aarch64)-0A7BBB.svg)](#requirements)
[![Status](https://img.shields.io/badge/status-early%20development-orange.svg)](docs/ROADMAP.md)

**English** | [简体中文](README.md)

> ### Build an RKNPU vision app in minutes.
> One C++ / Python API from capture → VPU hardware codec → RGA hardware preproc → NPU inference → post-processing.
> **Built on Rockchip's official runtimes (RKNPU2 / RGA / MPP), not replacing them.**

RCDL is the Rockchip counterpart of [BCDL](https://github.com/ruisv/bcdl) (RDK
BPU): the same `Engine` + task-class mental model and zero-copy pipeline idea,
re-based on the RK3588-family NPU / RGA / VPU. **Early development** — see
[`docs/ROADMAP.md`](docs/ROADMAP.md).

Measured on RK3588S (librknnrt 2.3.2 / driver 0.9.8):

| | |
|---|---|
| ResNet-18 int8, one core / three pinned contexts | 4.0 ms · **707 fps** |
| Detection pipeline, sync / async (3 workers, one per core) | 99 fps · **481 fps** (4.86x, results field-identical to sync and in order) |
| 1080p H.264 decode / 4K H.265 decode | **324 fps** · **244 fps**, every frame carrying a dma-buf fd |
| 1080p H.264 encode straight from decoded frames | **197 fps**, round-trip luma PSNR 47.2 dB |
| YOLOv8n / YOLO11n on `bus.jpg` | each independently finds 1 bus + 4 people |

```python
import rcdl, numpy as np

engine = rcdl.Engine("models/resnet18_rk3588.rknn")   # zero-copy I/O, uint8 NHWC in
out = engine.infer(np.zeros(engine.input_shape(0), dtype=np.uint8))[0]   # float32, dequantized
print(out.shape, engine.last_run_micros(), "us on the NPU")
```

```cpp
#include "rcdl/rcdl.h"
rcdl::Engine e("models/resnet18_rk3588.rknn", {rcdl::NpuCore::Core0});
auto e1 = e.dup(rcdl::NpuCore::Core1);   // same weights, second NPU core
e.setInput(0, img, e.inputPackedBytes(0));
e.infer();
std::vector<float> logits = e.outputAsFloat(0);
```

## Principles

- **The NPU runs the model, RGA does the preprocessing, the VPU does the codecs.**
  The CPU only runs post-processing (NMS / DFL / CTC …) and guarded fallbacks.
- **dma-buf is the one shared buffer.** NPU (`rknn_create_mem_from_fd`), RGA
  (`importbuffer_fd`) and VPU (MPP external buffers) import the same memory by
  fd — decode → letterbox → infer → encode without a `memcpy`.
- **Multi-core is first class.** RK3588's three NPU cores via `Engine::dup()` +
  core masks: one context per core for small models, combined masks for big ones.
- **Portable, verifiable post-processing.** Decoders are Engine-free pure
  functions pinned by deterministic numpy tests; board tests cover the hardware path.
- **Publishable as written.** No machine names, paths, private notes or local
  tool configuration; `scripts/check_publishable.sh` scans before each commit.

## Architecture

```
                    your app (C++ / Python)
                              ↓
   ┌──────────────────────────────────────────────────────┐
   │  RCDL                                                 │
   │  tasks · tracks · pipeline · media · preproc · backend │
   └──────────────────────────────────────────────────────┘
                              ↓
        Rockchip runtimes (where the capability comes from)
        librknnrt (RKNPU2) · librga (im2d) · librockchip_mpp (rk_mpi)
                              ↓
              NPU ×3 · RGA3 ×2 + RGA2 · VPU (rkvdec / rkvenc / JPEG)
```

| Dir | Contents | Status |
|---|---|---|
| `core/` | `DmaBuf` (dma-heap RAII + cache sync) · `Status` | ✅ M0 |
| `backend/` | `Engine` (zero-copy I/O · dequant · core masks · dup) · output readers | ✅ M0 |
| `preproc/` | RGA letterbox / resize / cvtColor + CPU fallback | ✅ M1 |
| `media/` | MPP H.264 / H.265 / VP9 / AV1 / JPEG codecs, external buffer group | ✅ M2 |
| `tasks/` | det · cls · pose · instance seg · semantic seg · obb · depth · embedding · ocr · face · sparse features · super-resolution · optical flow · promptable seg · whole-body pose · open-vocab det | ✅ M1 / M4 / M7–M10 |
| `tracks/` | ByteTrack + BoT-SORT appearance association · ReID | ✅ M3 |
| `pipeline/` | sync / async detection (multi-core `EnginePool`) | ✅ M3 |
| `python/` | nanobind bindings (GIL released in infer) | ✅ |

Benchmarks (measured on the board, regenerable): [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

## Quick start

On the board (aarch64 with `librknnrt` / `librga` / `librockchip_mpp`):

```bash
scripts/fetch_sdk.sh                 # RKNPU2 headers (not in the board image) → third_party/rknpu2
scripts/build.sh                     # cmake + ninja → build/
./build/model_info models/resnet18_rk3588.rknn        # I/O signature, runtime/driver versions, latency
./build/npu_bench  models/resnet18_rk3588.rknn 5 0,1,2   # three-core throughput
./build/dma_buf_probe                                   # is dma-heap usable by this user?
PYTHONPATH=build:python python -m pytest tests/ --model models/resnet18_rk3588.rknn
```

From a workstation (edit → sync → build on the board):

```bash
cp scripts/local.env.example scripts/local.env   # your ssh alias (gitignored)
scripts/bootstrap_board.sh    # once: create the rcdl conda env on the board
scripts/sync.sh && scripts/board_build.sh --run   # MODEL=/path/on/board.rknn
```

## Requirements

- An **RK3588 / RK3588S** board (RK3576 / RK356x planned), Linux aarch64, with
  the RKNPU driver (`/dev/rknpu`, ≥ 0.9.x), `/usr/lib/librknnrt.so` (2.3.x), and
  the `librga` + `librockchip_mpp` dev packages (`/usr/include/rga`, `/usr/include/rockchip`).
- dma-heap readable/writable by your user (membership of the `video` group usually does it).
- CMake ≥ 3.18, GCC ≥ 11, Ninja; **nanobind** + NumPy for the Python module.
- OpenCV optional (`RCDL_HAVE_OPENCV`-guarded, hand-written fallbacks).

## Models

RCDL consumes compiled `.rknn` files only. ONNX → `.rknn` (rknn-toolkit2 PTQ,
accuracy analysis, hybrid quantization) happens on an x86 host in a separate
model-zoo project; `scripts/fetch_models.sh` stages models into `models/`.
See [`models/README.md`](models/README.md).

## Docs

| Doc | Covers |
|---|---|
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | BCDL → RCDL stack mapping, milestones, RK3588 hardware notes |
| [`docs/API.md`](docs/API.md) | Python API |
| [`docs/CPP_API.md`](docs/CPP_API.md) | C++ API |
| [`docs/RGA.md`](docs/RGA.md) | What the 2-D engine will and will not do — including three limits that are not in the vendor documentation |
| [`docs/MODELS.md`](docs/MODELS.md) | Model registry, each model's input order and activation placement, measured performance |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Setup, building (on the board), testing, submitting changes |
| [`CHANGELOG.md`](CHANGELOG.md) | Release notes (Keep a Changelog / SemVer) |

## Acknowledgements

- **Rockchip** — the RK3588 platform, RKNPU2 runtime and rknn-toolkit2, RGA, MPP.
- **BCDL / ccdl** — the upper-layer API design and post-processing algorithms this project ports.

## License

[Apache License 2.0](LICENSE). Rockchip's runtime libraries and headers are
under their own licenses and are not redistributed here (`scripts/fetch_sdk.sh`
fetches them from the public repository).
