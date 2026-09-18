# models/

Put your compiled `.rknn` files here. The directory's contents are gitignored,
and the examples, tests and benchmarks look for models in it by default
(override with `RCDL_MODELS`).

RCDL does not distribute model weights. Get prebuilt models and conversion
examples from Rockchip's [rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo),
or convert your own with [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2).

A `.rknn` is compiled for one SoC (`target_platform=rk3588` / `rk3576` /
`rk3566` …), so use the build made for your board. The file names the tests
expect, and how each model has to be fed and decoded, are listed in
[`docs/MODELS.md`](../docs/MODELS.md).
