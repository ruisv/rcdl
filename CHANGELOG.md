# Changelog

All notable changes to RCDL are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project skeleton: CMake build (`librcdl.so`, examples, nanobind module,
  `find_package(rcdl)` export), conda env spec, workstation → board workflow
  scripts, leak-scan pre-commit hook.
- `rcdl::Engine` — loads an `.rknn`, binds zero-copy I/O tensors
  (`rknn_create_mem` + `rknn_set_io_mem`), runs inference, dequantizes outputs
  (int8 affine / DFP / fp16 / fp32), pins contexts to NPU cores, duplicates
  contexts (`rknn_dup_context`) for multi-core throughput.
- `rcdl::DmaBuf` — RAII dma-heap buffer with explicit CPU cache sync; the
  shared buffer type for NPU / RGA / VPU zero-copy.
- Examples: `model_info`, `npu_bench`, `dma_buf_probe`.
- Python: `rcdl.Engine` (numpy in/out), `rcdl.DmaBuf`, `rcdl.dequantize`.
