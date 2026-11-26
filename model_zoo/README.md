# Model zoo

RCDL runs compiled `.rknn` files. This directory is about getting them: which
builds are published ready to load, what each one's licence allows, and the
model cards those downloads carry.

Every file here was decoded and checked on an RK3588 board by RCDL's own tests.
Each was built with rknn-toolkit2 2.3.2, so it needs librknnrt 2.3.2 or newer.
How to feed and decode each model, and the conversion traps behind each build,
are in [`docs/MODELS.md`](../docs/MODELS.md).

## Prebuilt models on Hugging Face

| repo | what is in it | licence |
|---|---|---|
| [`ruisv/rcdl-yolo`](https://huggingface.co/ruisv/rcdl-yolo) | YOLOv8n · YOLO11n · YOLO26n: detection, instance segmentation, pose, oriented boxes, classification | **AGPL-3.0** |
| [`ruisv/rcdl-ppocr`](https://huggingface.co/ruisv/rcdl-ppocr) | PP-OCR v4 · v5 · v6: text detection, direction and recognition, with their dictionaries | Apache-2.0 |
| [`ruisv/rcdl-depth-anything-v2-small`](https://huggingface.co/ruisv/rcdl-depth-anything-v2-small) | Depth-Anything-V2-Small, 308×308 and 518×518 | Apache-2.0 |
| [`ruisv/rcdl-siglip-b16`](https://huggingface.co/ruisv/rcdl-siglip-b16) | SigLIP base/16 image tower, plus a COCO-80 text table for zero-shot classification | Apache-2.0 |
| [`ruisv/rcdl-xfeat`](https://huggingface.co/ruisv/rcdl-xfeat) | XFeat sparse features, 640×480 | Apache-2.0 |
| [`ruisv/rcdl-superres`](https://huggingface.co/ruisv/rcdl-superres) | Real-ESRGAN Compact ×4, fp16 and int8 | BSD-3-Clause |

Each repo's model card lists its files, the measured speed and accuracy on the
board, a working snippet and what to know before deploying. The card sources are
in [`hf/`](hf/).

### Getting them

From a checkout of this repository, one command fetches every published file
into `models/` and checks each against its recorded SHA-256:

```bash
scripts/fetch_models.sh                          # everything published
scripts/fetch_models.sh yolov8n_rk3588.rknn      # just the files named
```

Without a checkout, use plain `curl` (or `hf download`):

```bash
curl -LO https://huggingface.co/ruisv/rcdl-yolo/resolve/main/yolov8n_rk3588.rknn
hf download ruisv/rcdl-ppocr --local-dir ppocr
```

Both honour `HF_ENDPOINT`. Where huggingface.co is slow or unreachable, point it
at a mirror, for example `export HF_ENDPOINT=https://hf-mirror.com`.

**Keep the file names.** The tests, examples and benchmarks look models up by
these exact names. The suffixes say which build a file is (`_fp16`, `_i8`,
`_logits`, an input size), and those builds differ in how they must be fed.

## What each verified model's licence allows

The licence on a repository is often not the licence on its weights. A checkpoint
also inherits restrictions from the weights it was fine-tuned from and from the
data it was trained on. The table is grouped by what you may do with the compiled
`.rknn`. Where no upstream source answers the question, the model is not
published: silence is not permission.

| what you may do with the `.rknn` | models | published |
|---|---|---|
| **Redistribute, including commercially** | PP-OCR v4/v5/v6 (Apache-2.0 on code and every model card) · Depth-Anything-V2 **Small** (Apache-2.0 on code and weights) · SigLIP base/16 (Apache-2.0) · XFeat (weights committed in the Apache-2.0 repo) · Real-ESRGAN Compact (BSD-3-Clause, weights released by the copyright holder) | yes |
| **Redistribute under copyleft** | YOLOv8n, YOLO11n, YOLO26n and their seg / pose / obb / cls variants (Ultralytics, **AGPL-3.0**): software that ships them falls under AGPL-3.0; a closed or commercial product needs Ultralytics' Enterprise licence | yes |
| **Not yet checked to the end** | RTMW-s (Apache-2.0, trained on a mix of datasets) · RetinaFace (MIT code, trained on WIDER FACE) · ResNet-18 (ImageNet) | not yet |
| **Non-commercial** | PP-LiteSeg and YOLO26n-sem (trained on **Cityscapes**, which bars commercial use of models trained on it; YOLO26n-sem is also AGPL-3.0) · YOLOP (MIT code, but **BDD100K** limits commercial use) · ArcFace R50 (InsightFace weights are non-commercial research only) · YOLOE-11s (AGPL-3.0, and its vocabulary is embedded with Apple's MobileCLIP text encoder, whose weights are licensed for **non-commercial research only**) | no |
| **Unresolved** | NeuFlow v2 (Apache-2.0 code; trained on FlyingThings3D, whose terms are research-only and silent about trained models) · OSNet (MIT code; every ReID-useful checkpoint is trained on MSMT17 or Market-1501 under academic-only terms) · EdgeSAM (the upstream repository states no licence at all) | no |

Unpublished models stay convertible: the input contract and the conversion notes
for each one are in [`docs/MODELS.md`](../docs/MODELS.md).

**RCDL itself is Apache-2.0, and that does not change any of the above.** It is a
general-purpose runtime that loads any `.rknn`; the terms in this table apply to
the weights and the compiled files.

## Layout

```
model_zoo/
  README.md          this page
  hf/<repo>/README.md  the model card published as that Hugging Face repo's README
  publish_hf.py      uploads a repo's files and card, then checks every byte arrived
```

The list of published files and their SHA-256 sums lives in one place, the
registry in [`scripts/fetch_models.sh`](../scripts/fetch_models.sh), and both the
fetch script and `publish_hf.py` read it.
