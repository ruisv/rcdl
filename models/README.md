# models/

Put your compiled `.rknn` files here. The directory's contents are gitignored,
and the examples, tests and benchmarks look for models in it by default
(override with `RCDL_MODELS`).

`scripts/fetch_models.sh` fills it with every build published on Hugging Face,
checking each file against its recorded SHA-256. Name files to fetch only those:

```bash
scripts/fetch_models.sh yolov8n_rk3588.rknn ppocrv6_medium_det_rk3588.rknn
```

Which models are published, under what licence, and why the others are not is
in [`model_zoo/README.md`](../model_zoo/README.md). For anything else, convert
your own with [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2), or
start from Rockchip's [rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo).

A `.rknn` is compiled for one SoC (`target_platform=rk3588` / `rk3576` /
`rk3566` …), so use the build made for your board. The file names the tests
expect, and how each model has to be fed and decoded, are listed in
[`docs/MODELS.md`](../docs/MODELS.md).
