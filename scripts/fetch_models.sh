#!/usr/bin/env bash
# Populate models/ with the .rknn files the examples, tests and benchmarks use.
#
# Sources, in order: an rknn_model_zoo checkout (RCDL_SRC_ZOO) for Rockchip's
# prebuilt models, then the conversion host (RCDL_CONVERT_HOST +
# RCDL_CONVERT_MODELS) for models converted by the rcdl model-zoo recipes.
# Paths come from scripts/local.env (gitignored; see local.env.example).
# Anything left unset is reported as MISSING rather than silently skipped.
#
#   scripts/fetch_models.sh            # stage everything known
#   scripts/fetch_models.sh --list     # show the registry
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
[ -f "${PROJ}/scripts/local.env" ] && source "${PROJ}/scripts/local.env"
DEST="${PROJ}/models"
mkdir -p "${DEST}"

# name | source kind | relative path at the source
#   zoo     : ${RCDL_SRC_ZOO}/<path>          (rknn_model_zoo checkout; run its
#                                              download_model.sh first)
#   convert : ${RCDL_CONVERT_HOST}:${RCDL_CONVERT_MODELS}/<path>
# Each model's INPUT ORDER (rgb/bgr) and head family are recorded here so the
# examples and tests do not have to guess; see docs/MODELS.md for the table.
REGISTRY=(
  "yolov8n_rk3588.rknn|convert|yolov8n_rk3588.rknn"
  "yolo11n_rk3588.rknn|convert|yolo11n_rk3588.rknn"
  # YOLO26n: NMS-free head, and — the part that matters to the decoder — NO DFL,
  # so the box branch is 4 channels instead of 64 and postprocessing is half the
  # cost. Exported from the one2one branch (see docs/MODELS.md).
  "yolo26n_rk3588.rknn|convert|yolo26n_rk3588.rknn"
  "yolov8n-pose_rk3588.rknn|convert|yolov8n-pose_rk3588.rknn"
  "yolov8n-seg_rk3588.rknn|convert|yolov8n-seg_rk3588.rknn"
  "yolov8n-obb_rk3588.rknn|convert|yolov8n-obb_rk3588.rknn"
  "ppocrv4_det_rk3588.rknn|convert|ppocrv4_det_rk3588.rknn"
  "ppocrv4_rec_rk3588.rknn|convert|ppocrv4_rec_rk3588.rknn"
  # Text-line direction (0/180). BGR in, 48x192, softmax in the graph. It goes
  # between the detector and the recogniser: a CTC model fed an upside-down line
  # reads out confident nonsense rather than failing. See docs/MODELS.md.
  "ppocr_cls_rk3588.rknn|convert|ppocr_cls_rk3588.rknn"
  # PP-OCRv5 mobile detection. Measures identical to the v4 detector on the
  # sample page (same boxes, same text); kept as the newer alternative and
  # pinned by a parity test. There is no v5 recogniser here on purpose — see
  # docs/MODELS.md for what the board does to it.
  "ppocrv5_det_rk3588.rknn|convert|ppocrv5_det_rk3588.rknn"
  "ppseg_rk3588.rknn|convert|ppseg_rk3588.rknn"
  "retinaface_rk3588.rknn|convert|retinaface_rk3588.rknn"
  "resnet18_rk3588.rknn|convert|resnet18_rk3588.rknn"
  # Monocular depth. The 308x308 build is the one the tests use: it is 2.9x
  # faster than the native 518x518 build AND closer to the fp32 reference
  # (see docs/MODELS.md). The 518 build is kept for the native-resolution case.
  "depth_anything_v2_vits_308_rk3588.rknn|convert|depth_anything_v2_vits_308_rk3588.rknn"
  "depth_anything_v2_vits_rk3588.rknn|convert|depth_anything_v2_vits_rk3588.rknn"
  # Person ReID appearance vectors — what ImageEmbedder/EmbeddingBank and the
  # ReID side of ByteTrack consume. 512-d, crops squashed to 128x256 (NOT
  # letterboxed); see docs/MODELS.md.
  "osnet_x0_25_msmt17_rk3588.rknn|convert|osnet_x0_25_msmt17_rk3588.rknn"
)

if [ "${1:-}" = "--list" ]; then
  printf '%s\n' "${REGISTRY[@]}" | column -t -s'|'
  exit 0
fi

missing=0
for entry in "${REGISTRY[@]}"; do
  IFS='|' read -r name kind rel <<<"${entry}"
  out="${DEST}/${name}"
  if [ -s "${out}" ]; then echo "   ok  ${name}"; continue; fi
  case "${kind}" in
    zoo)
      src="${RCDL_SRC_ZOO:-}/${rel}"
      if [ -n "${RCDL_SRC_ZOO:-}" ] && [ -s "${src}" ]; then
        cp "${src}" "${out}" && echo "  got  ${name}  (zoo)"
      else
        echo "MISSING ${name}  — set RCDL_SRC_ZOO and download it there (${rel})"; missing=$((missing+1))
      fi ;;
    convert)
      if [ -n "${RCDL_CONVERT_HOST:-}" ] && [ -n "${RCDL_CONVERT_MODELS:-}" ]; then
        scp -q "${RCDL_CONVERT_HOST}:${RCDL_CONVERT_MODELS}/${rel}" "${out}" \
          && echo "  got  ${name}  (convert host)" \
          || { echo "MISSING ${name}  — not on the convert host (${rel})"; missing=$((missing+1)); }
      else
        echo "MISSING ${name}  — set RCDL_CONVERT_HOST / RCDL_CONVERT_MODELS"; missing=$((missing+1))
      fi ;;
  esac
done
echo ">> models/: $(ls -1 "${DEST}"/*.rknn 2>/dev/null | wc -l | tr -d ' ') staged, ${missing} missing"
