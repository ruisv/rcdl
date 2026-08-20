# models/

Compiled `.rknn` models used by the examples, tests and benchmarks. The
directory contents are gitignored — populate it with `scripts/fetch_models.sh`
(sources: an `rknn_model_zoo` checkout for Rockchip's prebuilt models, and the
conversion host for models converted by the rcdl model-zoo recipes).

A `.rknn` is compiled for one SoC (`target_platform=rk3588` / `rk3576` /
`rk3566` …); run the build made for your board. Model provenance, which build
to take, and licenses are listed in `docs/MODELS.md` once models land.
