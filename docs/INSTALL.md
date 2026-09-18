# Installing RCDL

RCDL runs **on the board**. Its libraries talk to the RKNPU, RGA and MPP kernel
drivers, so there is nothing to install on an x86 workstation except an editor.

## What the board has to provide

| | |
|---|---|
| SoC | RK3588 / RK3588S (verified). RK3576 and RK356x are untested |
| OS | Linux, aarch64, glibc ≥ 2.28 |
| Kernel drivers | RKNPU ≥ 0.9.x (a `/dev/dri/renderD*` node), RGA (`/dev/rga`), MPP (`/dev/mpp_service`) |
| dma-heap | `/dev/dma_heap/system` readable and writable by your user |

Verified stack: kernel 6.1, RKNPU driver 0.9.8, librknnrt 2.3.2, Ubuntu 24.04.

Everything in userspace — `librknnrt`, `librga`, `librockchip_mpp`, OpenCV — comes
with the conda packages. A vendor image that already ships them in `/usr/lib` is
fine too; the conda environment's copies take precedence inside the environment.

### Permissions

The device nodes are usually owned by group `video`, and the NPU's render node
by group `render`:

```bash
sudo usermod -aG video,render $USER      # then log out and back in
ls -l /dev/dma_heap/ /dev/rga /dev/mpp_service /dev/dri/renderD*
```

If your image creates `/dev/dma_heap/*` as `root:root`, add a udev rule:

```bash
echo 'SUBSYSTEM=="dma_heap", GROUP="video", MODE="0660"' | sudo tee /etc/udev/rules.d/99-dma-heap.rules
sudo udevadm control --reload && sudo udevadm trigger
```

## conda (recommended)

```bash
conda create -n rcdl -c https://mirrors.ruis.ai/conda -c conda-forge rcdl
conda activate rcdl
python -c "import rcdl; print(rcdl.__version__); print(rcdl.rga_version())"
```

| package | what it is |
|---|---|
| `rcdl` | Python bindings (3.9–3.14); depends on `librcdl` |
| `librcdl` | `librcdl.so`, headers, `find_package(rcdl)` CMake config |
| `librknnrt` · `librga` · `rockchip-mpp` | Rockchip's userspace libraries and headers, pulled in automatically |

To make the channel permanent for an environment:

```bash
conda config --env --add channels conda-forge
conda config --env --add channels https://mirrors.ruis.ai/conda
```

### Using the C++ library

```cmake
cmake_minimum_required(VERSION 3.18)
project(app LANGUAGES CXX)
find_package(rcdl REQUIRED)
add_executable(app main.cc)
target_link_libraries(app PRIVATE rcdl::rcdl)
```

```bash
conda install -c https://mirrors.ruis.ai/conda -c conda-forge librcdl cmake ninja cxx-compiler
cmake -S . -B build -G Ninja -DCMAKE_PREFIX_PATH=$CONDA_PREFIX && cmake --build build
```

## Building from source

On the board:

```bash
git clone https://github.com/ruisv/rcdl.git && cd rcdl
conda env create -f env/environment.yml && conda activate rcdl   # cmake, ninja, nanobind, numpy, OpenCV
scripts/fetch_sdk.sh      # rknn_api.h — most board images ship librknnrt.so without its header
scripts/build.sh          # --no-python, --no-examples, --install PREFIX, --pip
```

The build links `librknnrt` / `librga` / `librockchip_mpp` from the conda
environment when they are installed there, otherwise from the board image
(`/usr/lib`, `/usr/include/rga`, `/usr/include/rockchip`). RGA and MPP are
optional at configure time — check the configure log for
`RGA hardware preproc: enabled` and `MPP video/JPEG codecs: enabled`, because
without them RCDL still builds and quietly runs those stages on the CPU.

Then:

```bash
./build/model_info  model.rknn            # I/O signature, runtime and driver versions, latency
./build/npu_bench   model.rknn 5 0,1,2    # throughput across the three NPU cores
./build/dma_buf_probe                     # can this user allocate from each dma-heap?
./build/rga_probe                         # what each RGA core can reach on this board and kernel
PYTHONPATH=build:python python -m pytest tests/ --model model.rknn
```

Tests that need a model or a video stream skip cleanly when it is not there.

## Troubleshooting

**`open dma-heap failed: Permission denied`** — your user cannot open
`/dev/dma_heap/system`. See [Permissions](#permissions).

**`RGA_MMU unsupported memory larger than 4G` in `dmesg`** — something asked the
RGA2 core to touch memory above 4 GB, which its 32-bit MMU cannot. RCDL's own
paths avoid this (it pins each operation to a core that can reach the buffer).
If you call RGA fills or GRAY8 conversions on your own buffers, allocate them
from the `system-dma32` heap (`DmaBuf::Heap::SystemDma32`). Details and
measurements: [`RGA.md`](RGA.md) §3.

**A model loads but the results are slightly off** — the two properties a
`.rknn` does not carry are the input channel order (`rgb` / `bgr`) and whether
the sigmoid / softmax is inside the graph. [`MODELS.md`](MODELS.md) records both
for every verified model.

**`rknn_init` fails, or the model is rejected** — a `.rknn` is compiled for one
SoC, and it must have been converted with an rknn-toolkit2 no newer than the
runtime on the board. `model_info` prints the runtime and driver versions.

**CPU stages are several times slower than the benchmark table** — under the
`schedutil` governor the CPU clocks down while the pipeline waits on the NPU,
and post-processing then starts at a low clock. For a fair measurement:

```bash
for p in /sys/devices/system/cpu/cpufreq/policy*; do echo performance | sudo tee $p/scaling_governor; done
```

**A frame was unexpectedly slow** — check which preprocessing backend ran:
`det.backend` in Python, `DetectionPipeline::lastBackend()` in C++. `"cpu"` means
RGA declined the request; `rcdl::rgaCanHandle(dst, src, &why)` says why.

**A sheared or striped picture from a decoded frame** — the VPU pads rows.
Read frames at `width_stride`, not `width`, or use `frame.to_numpy()`, which
removes the padding.
