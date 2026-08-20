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
REGISTRY=(
  "resnet18_rk3588.rknn|zoo|examples/resnet/model/resnet18_rk3588.rknn"
  "yolov8n_rk3588.rknn|zoo|examples/yolov8/model/yolov8n_rk3588.rknn"
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
