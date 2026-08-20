# RCDL — Rockchip RKNPU 视觉开发框架

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](CMakeLists.txt)
[![Python](https://img.shields.io/badge/python-3.9%E2%80%933.14-3776AB.svg)](pyproject.toml)
[![Platform](https://img.shields.io/badge/platform-RK3588%20%2F%20RK3576%20%2F%20RK356x%20(aarch64)-0A7BBB.svg)](#运行环境要求)
[![Status](https://img.shields.io/badge/status-early%20development-orange.svg)](docs/ROADMAP.md)

[English](README.en.md) | **简体中文**

> ### 几分钟做出一个 RKNPU 视觉应用。
> 统一的 C++ / Python API，覆盖取流 → VPU 硬件编解码 → RGA 硬件预处理 → NPU 推理 → 后处理整条链路。
> **构建在 Rockchip 官方运行时（RKNPU2 / RGA / MPP）之上，不是取代它。**

RCDL 是 [BCDL](https://github.com/ruisv/bcdl)（RDK BPU）的 Rockchip 对应版：同一套
`Engine` + 任务类的心智模型、同样的零拷贝流水线思路，底层换成 RK3588 系列的
NPU / RGA / VPU。**项目处于早期开发阶段**，目前完成了 M0（骨架 + 推理引擎），
路线图见 [`docs/ROADMAP.md`](docs/ROADMAP.md)。

```python
import rcdl, numpy as np

engine = rcdl.Engine("models/resnet18_rk3588.rknn")   # 零拷贝 I/O，输入 uint8 NHWC
out = engine.infer(np.zeros(engine.input_shape(0), dtype=np.uint8))[0]   # float32 已反量化
print(out.shape, engine.last_run_micros(), "us on the NPU")
```

```cpp
#include "rcdl/rcdl.h"
rcdl::Engine e("models/resnet18_rk3588.rknn", {rcdl::NpuCore::Core0});
auto e1 = e.dup(rcdl::NpuCore::Core1);   // 同一权重，第二个 NPU 核
e.setInput(0, img, e.inputPackedBytes(0));
e.infer();
std::vector<float> logits = e.outputAsFloat(0);
```

## 设计原则

- **模型在 NPU 上跑，预处理在 RGA 上做，编解码在 VPU 上做。** CPU 只做后处理
  （NMS / DFL / CTC …）和受保护的回退路径，不是默认路径。
- **dma-buf 是唯一的共享缓冲。** NPU（`rknn_create_mem_from_fd`）、RGA
  （`importbuffer_fd`）、VPU（MPP 外部 buffer）都按 fd 导入同一块内存，
  解码 → letterbox → 推理 → 编码全程不经过 `memcpy`。
- **多核是一等公民。** RK3588 的 3 个 NPU 核通过 `Engine::dup()` + 核掩码
  并发使用；小模型单核一个 context，大模型合并掩码。
- **后处理可移植、可验证。** 解码器是与 Engine 无关的纯函数，用确定性 numpy
  测试钉死；板端测试在真实 `.rknn` 上验证完整硬件路径。
- **写下来就是可公开的。** 仓库不含机器名、路径、私有笔记或本地工具配置；
  `scripts/check_publishable.sh` 在提交前扫描。

## 架构

```
                    你的应用（C++ / Python）
                              ↓
   ┌──────────────────────────────────────────────────────┐
   │  RCDL                                                 │
   │  tasks · tracks · pipeline · media · preproc · backend │
   └──────────────────────────────────────────────────────┘
                              ↓
        Rockchip 官方运行时（能力的来源）
        librknnrt (RKNPU2) · librga (im2d) · librockchip_mpp (rk_mpi)
                              ↓
              NPU ×3 · RGA3 ×2 + RGA2 · VPU（rkvdec / rkvenc / JPEG）
```

| 目录 | 内容 | 状态 |
|---|---|---|
| `core/` | `DmaBuf`（dma-heap RAII + cache sync）· `Status` | ✅ M0 |
| `backend/` | `Engine`（零拷贝 I/O · 反量化 · 核掩码 · dup）· 输出读取 | ✅ M0 |
| `preproc/` | RGA letterbox / resize / cvtColor + CPU 回退 | M1 |
| `media/` | MPP H.264 / H.265 / JPEG 编解码 | M2 |
| `tasks/` | det · cls · pose · seg · obb · ocr · depth … | M1 / M4 |
| `tracks/` | ByteTrack | M3 |
| `pipeline/` | 同步 / 异步检测 · 跟踪 · 视频端到端 | M3 |
| `python/` | nanobind 绑定（推理时释放 GIL） | ✅ M0（Engine） |

## 快速上手

在板上（aarch64，已带 `librknnrt` / `librga` / `librockchip_mpp`）：

```bash
scripts/fetch_sdk.sh                 # RKNPU2 头文件（板子镜像不带）→ third_party/rknpu2
scripts/build.sh                     # cmake + ninja → build/
./build/model_info models/resnet18_rk3588.rknn        # I/O 签名、运行时/驱动版本、延迟
./build/npu_bench  models/resnet18_rk3588.rknn 5 0,1,2   # 三核并发吞吐
./build/dma_buf_probe                                   # dma-heap 是否对当前用户可用
PYTHONPATH=build:python python -m pytest tests/ --model models/resnet18_rk3588.rknn
```

从工作站迭代（编辑 → 同步 → 板上构建）：

```bash
cp scripts/local.env.example scripts/local.env   # 填 ssh 别名（gitignored）
scripts/bootstrap_board.sh    # 一次性：板上建 rcdl conda 环境（env/environment.yml）
scripts/sync.sh && scripts/board_build.sh --run   # MODEL=/path/on/board.rknn
```

## 运行环境要求

- 一块 **RK3588 / RK3588S**（RK3576 / RK356x 计划中）开发板，Linux（aarch64），
  板上有 RKNPU 驱动（`/dev/rknpu`，≥ 0.9.x）、`/usr/lib/librknnrt.so`（2.3.x）、
  `librga` 与 `librockchip_mpp` 的开发包（`/usr/include/rga`、`/usr/include/rockchip`）。
- dma-heap 对当前用户可读写（通常把用户加进 `video` 组即可）。
- CMake ≥ 3.18、GCC ≥ 11、Ninja；Python 模块需要 **nanobind** + NumPy。
- OpenCV 可选（`RCDL_HAVE_OPENCV` 守卫，有手写回退）。

## 模型

RCDL 只消费编译好的 `.rknn`。ONNX → `.rknn`（rknn-toolkit2 PTQ、精度分析、
混合量化）在 x86 主机上完成，由独立的模型仓负责；`scripts/fetch_models.sh`
负责把模型放进 `models/`。见 [`models/README.md`](models/README.md)。

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | 架构映射（BCDL → RCDL）、里程碑、RK3588 硬件要点 |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | 如何搭建、构建（在板上）、测试并提交改动 |
| [`CHANGELOG.md`](CHANGELOG.md) | 发布说明（Keep a Changelog / SemVer） |

## 致谢

- **Rockchip** —— RK3588 平台、RKNPU2 运行时与 rknn-toolkit2、RGA、MPP。
- **BCDL / ccdl** —— 本项目的上层 API 设计与后处理算法来源。

## 许可证

[Apache License 2.0](LICENSE)。Rockchip 的运行时库与头文件遵循其各自的许可证，
本仓库不再分发它们（`scripts/fetch_sdk.sh` 从公开仓库获取）。
