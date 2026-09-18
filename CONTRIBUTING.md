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

The simplest setup is to work on the board itself: see "Building from source"
in [`docs/INSTALL.md`](docs/INSTALL.md). If you prefer editing on a workstation,
`scripts/sync.sh` + `scripts/board_build.sh` rsync the tree to a board over ssh
and build it there (copy `scripts/local.env.example` to `scripts/local.env` and
fill in your board's ssh alias; `scripts/bootstrap_board.sh` creates the conda
env once).

## Layout

```
include/rcdl/, src/   library: core / backend / preproc / media / tasks / tracks / pipeline
python/               nanobind bindings + the pure-Python wrapper
examples/             standalone C++ programs
tests/                pytest (static + numpy decode + on-board end-to-end)
scripts/              build / fetch the SDK headers / workstation → board sync / git hooks
docs/                 user documentation (install, API references, models, RGA notes)
```

## Conventions

- Headers `.h`, impl `.cc`, namespace `rcdl`. Errors via `RCDL_CHECK(...)` /
  `RCDL_REQUIRE(...)` → `rcdl::Error`.
- Hardware first: NPU for the model, RGA for resize/cvtColor/letterbox, VPU for
  codecs; CPU only for post-processing and as a guarded fallback.
- Every behaviour change comes with a test: pure-numpy for decoder math (runs
  anywhere), plus a board test that skips cleanly when its model is missing.
- Nothing machine-specific in committed content: no hostnames, IPs or local
  paths. `scripts/install_hooks.sh` installs a pre-commit scan for it.
- Record user-visible changes under `[Unreleased]` in `CHANGELOG.md`.
- Commit messages: Conventional Commits style (`feat(backend): ...`).

## Reporting bugs

Include the board / SoC, `model_info` output (runtime + driver version), the
model's toolkit version, and a minimal reproduction.
