# Contributing to RCDL

Contributions are welcome — bug reports, fixes, new task decoders, benchmarks,
docs. By contributing you agree that your contributions are licensed under the
project's [Apache License 2.0](LICENSE).

## The one hard constraint: build on the board

RCDL links Rockchip's **RKNPU2 runtime** (`librknnrt`), **RGA** (`librga`) and
**MPP** (`librockchip_mpp`), which talk to kernel drivers that exist only on
Rockchip hardware. So the C++ library, the examples and the `rcdl_py` module
**must be built and run on an RK3588 / RK3576 / RK356x board** (aarch64). Editing,
review and the static / numpy tests work anywhere.

Typical loop: edit on a workstation → `scripts/sync.sh` → `scripts/board_build.sh`.
One-time board setup: `scripts/bootstrap_board.sh` (creates the `rcdl` conda env
from `env/environment.yml`); `scripts/fetch_sdk.sh` pulls the RKNPU2 headers the
board image does not ship.

## Layout

```
include/rcdl/, src/   library: core / backend / preproc / media / tasks / tracks / pipeline
python/               nanobind bindings + the pure-Python wrapper
examples/             standalone C++ programs
tests/                pytest (static + numpy decode + on-board end-to-end)
scripts/              sync / build / bootstrap / fetch / leak-scan
docs/                 public docs (ROADMAP, API references as they land)
```

## Conventions

- Headers `.h`, impl `.cc`, namespace `rcdl`. Errors via `RCDL_CHECK(...)` /
  `RCDL_REQUIRE(...)` → `rcdl::Error`.
- Hardware first: NPU for the model, RGA for resize/cvtColor/letterbox, VPU for
  codecs; CPU only for post-processing and as a guarded fallback.
- Every behaviour change comes with a test: pure-numpy for decoder math (runs
  anywhere), plus a board test that skips cleanly when its model is missing.
- Keep the repository publishable: no machine names, IPs, local paths, private
  notes or local tool configuration in committed content or commit messages.
  `scripts/install_hooks.sh` installs a pre-commit scan that enforces this.
- Record user-visible changes under `[Unreleased]` in `CHANGELOG.md`.
- Commit messages: Conventional Commits style (`feat(backend): ...`).

## Reporting bugs

Include the board / SoC, `model_info` output (runtime + driver version), the
model's toolkit version, and a minimal reproduction.
